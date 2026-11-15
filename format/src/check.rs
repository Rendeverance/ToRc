//! check.rs - full verification: the region hashes, then every tensor's and pack's CRC-32C; for a pipeline shard, the
//! CRCs of the parts it holds; and against the source GGUF, every tensor and piece byte for byte. Big tensors are read
//! a chunk at a time (Flash Next's n-gram table is 28.8 GB).

use crate::convert::{SourceMatrix, source_range};
use crate::read::Model;
use crate::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

// one page under test, so the small test models are written in many chunks
const CHUNK: u64 = if cfg!(test) { 4096 } else { 64 << 20 };

fn fetch(f: &mut File, off: u64, buf: &mut [u8]) -> Result<()> {
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(buf)?;
    Ok(())
}

fn hash_region(f: &mut File, r: &Region) -> Result<[u8; 32]> {
    let mut buf = vec![0u8; CHUNK as usize];
    let mut h = blake3::Hasher::new();
    f.seek(SeekFrom::Start(r.offset))?;
    let mut done = 0u64;
    while done < r.bytes {
        let n = (r.bytes - done).min(CHUNK) as usize;
        f.read_exact(&mut buf[..n])?;
        h.update_rayon(&buf[..n]);
        done += n as u64;
    }
    Ok(*h.finalize().as_bytes())
}

/// A tensor's chunks: (start within the tensor, length), each a whole number of rows and at most about CHUNK.
fn chunks(t: &Tensor) -> impl Iterator<Item = (u64, u64)> {
    let row = matrix_bytes(t.ty, 1, t.dims[0]).filter(|&r| r > 0).unwrap_or(t.bytes.max(1));
    let step = (CHUNK / row).max(1) * row;
    let bytes = t.bytes;
    (0..bytes.div_ceil(step)).map(move |k| (k * step, step.min(bytes - k * step)))
}

#[derive(Default, Debug)]
pub struct CheckReport {
    pub tensors: usize,
    pub packs: usize,
    pub seconds: f64,
}

/// The CRCs of the tensors that `keep_tensor` accepts and of the packs of the layers `keep_layer` accepts.
fn check_crcs(
    m: &Model,
    f: &mut File,
    keep_tensor: impl Fn(&str) -> bool,
    keep_layer: impl Fn(u32) -> bool,
    r: &mut CheckReport,
) -> Result<()> {
    let mut buf = Vec::new();
    for t in m.tensors.iter().filter(|t| keep_tensor(&t.name)) {
        let mut crc = Crc32c::default();
        for (at, n) in chunks(t) {
            buf.resize(n as usize, 0);
            fetch(f, m.resident.offset + t.offset + at, &mut buf)?;
            crc.update(&buf);
        }
        if crc.finish() != t.crc {
            return err(format!("tensor {}: CRC mismatch", t.name));
        }
        r.tensors += 1;
    }
    for x in m.experts.iter().filter(|x| keep_layer(x.layer)) {
        buf.resize(x.pack.bytes as usize, 0);
        for e in 0..x.n_expert {
            fetch(f, m.expert_region.offset + x.pack_at(e), &mut buf)?;
            if crc32c(&buf) != x.crcs[e as usize] {
                return err(format!("layer {} expert {e}: CRC mismatch", x.layer));
            }
            r.packs += 1;
        }
    }
    Ok(())
}

/// The whole file: the resident and expert region hashes, then every CRC.
pub fn check(path: &str) -> Result<CheckReport> {
    let t0 = Instant::now();
    let m = Model::open(path)?;
    let mut f = File::open(path)?;
    if hash_region(&mut f, &m.resident)? != m.resident.hash {
        return err("the resident region does not match its hash");
    }
    if m.expert_region.bytes > 0 && hash_region(&mut f, &m.expert_region)? != m.expert_region.hash {
        return err("the expert region does not match its hash");
    }
    let mut r = CheckReport::default();
    check_crcs(&m, &mut f, |_| true, |_| true, &mut r)?;
    r.seconds = t0.elapsed().as_secs_f64();
    Ok(r)
}

/// A pipeline shard (shard.rs): the CRCs of what a machine running layers a..b holds. The region hashes span the
/// parts a shard leaves out, so they are not checked.
pub fn check_shard(path: &str, a: u32, b: u32, ngram_layer: Option<u32>) -> Result<CheckReport> {
    let t0 = Instant::now();
    let m = Model::open(path)?;
    let mut f = File::open(path)?;
    let mut r = CheckReport::default();
    check_crcs(&m, &mut f, |n| crate::shard::keeps(n, a, b, ngram_layer), |l| l >= a && l < b, &mut r)?;
    r.seconds = t0.elapsed().as_secs_f64();
    Ok(r)
}

/// Every tensor and piece against the GGUF it came from (`gguf` is the first file of a split GGUF): each one's bytes
/// must be what converting its source bytes gives (convert::source_range), byte for byte.
pub fn check_against(path: &str, gguf: &str) -> Result<CheckReport> {
    let t0 = Instant::now();
    let m = Model::open(path)?;
    let g = crate::gguf::Gguf::open(gguf)?;
    let mut f = File::open(path)?;
    let mut srcs = g.files.iter().map(File::open).collect::<std::io::Result<Vec<_>>>()?;
    let mut r = CheckReport::default();
    let (mut ours, mut want, mut tmp) = (Vec::new(), Vec::new(), Vec::new());
    for t in &m.tensors {
        let Some(gt) = g.tensor(&t.name) else { return err(format!("tensor {}: not in the GGUF", t.name)) };
        if t.origin.file != gt.file || t.origin.offset != gt.offset {
            return err(format!("tensor {}: its origin is not the GGUF tensor's place", t.name));
        }
        let sm =
            SourceMatrix { origin: t.origin, ty: t.ty, src_bytes: gt.bytes, src_cols: gt.dims[0], dst_cols: t.dims[0] };
        for (at, n) in chunks(t) {
            ours.resize(n as usize, 0);
            fetch(&mut f, m.resident.offset + t.offset + at, &mut ours)?;
            want.clear();
            want.resize(n as usize, 0);
            source_range(&mut srcs[gt.file as usize], &sm, at, &mut want, &mut tmp)?;
            if ours != want {
                return err(format!("tensor {}: differs from its source near byte {at}", t.name));
            }
        }
        r.tensors += 1;
    }
    let mut pack = Vec::new();
    for x in &m.experts {
        pack.resize(x.pack.bytes as usize, 0);
        for e in 0..x.n_expert {
            fetch(&mut f, m.expert_region.offset + x.pack_at(e), &mut pack)?;
            for pc in x.pack.pieces() {
                let Some(gt) = g.tensor(&pc.name) else { return err(format!("{}: not in the GGUF", pc.name)) };
                // one expert's source matrix: half the rows of a fused tensor
                let fused = pc.origin.changes & (FUSED_FIRST_HALF | FUSED_SECOND_HALF) != 0;
                let src_rows = if fused { gt.dims[1] / 2 } else { gt.dims[1] };
                let src_bytes = matrix_bytes(gt.ty, src_rows, gt.dims[0])
                    .ok_or_else(|| Error(format!("{}: bad type or shape", pc.name)))?;
                let origin = Origin { offset: pc.origin.offset + e as u64 * pc.stride, ..pc.origin };
                let sm = SourceMatrix { origin, ty: pc.ty, src_bytes, src_cols: gt.dims[0], dst_cols: pc.cols as u64 };
                want.clear();
                want.resize(pc.bytes as usize, 0);
                source_range(&mut srcs[pc.origin.file as usize], &sm, 0, &mut want, &mut tmp)?;
                if pc.of(&pack) != want.as_slice() {
                    return err(format!("layer {} expert {e}: {} differs from its source", x.layer, pc.name));
                }
            }
            r.packs += 1;
        }
    }
    r.seconds = t0.elapsed().as_secs_f64();
    Ok(r)
}
