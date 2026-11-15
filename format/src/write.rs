//! write.rs - writing a file. `plan` places every resident tensor and every pack; `write` then streams them into
//! place, the caller filling each one's bytes, and finishes with their CRCs in the directory and the header.
//! Memory use is the directory plus one tensor or one pack at a time.

use crate::*;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};

/// A file's layout: the directory with every offset set (CRCs still zero) and where each region starts.
pub struct Plan {
    pub dir: Directory,
    pub arch: String,
    pub directory: Region,
    pub resident: Region,
    pub experts: Region,
    pub file_bytes: u64,
}

/// Places the tensors in the resident region in the order given, each on a page boundary, and each layer's packs
/// (pack.bytes set by the caller, a whole number of pages) back to back in the expert region, layers in the order
/// given. Pieces must already have their place within the pack.
pub fn plan(mut dir: Directory, arch: &str) -> Result<Plan> {
    let mut at = 0u64;
    for t in dir.tensors.iter_mut() {
        t.offset = at;
        t.crc = 0;
        at = page_up(at + t.bytes);
    }
    let resident_len = at;
    let mut at = 0u64;
    for x in dir.experts.iter_mut() {
        if x.pack.bytes == 0 || x.pack.bytes % PAGE != 0 {
            return err(format!("layer {}: pack bytes must be a whole number of pages", x.layer));
        }
        x.first = at;
        x.crcs = vec![0; x.n_expert as usize];
        at += x.n_expert as u64 * x.pack.bytes;
    }
    let experts_len = at;
    let dir_bytes = dir.encode().len() as u64;
    let directory = Region { offset: HEADER_BYTES, bytes: dir_bytes, hash: [0; 32] };
    let resident = Region { offset: page_up(HEADER_BYTES + dir_bytes), bytes: resident_len, hash: [0; 32] };
    let experts = Region { offset: resident.offset + resident_len, bytes: experts_len, hash: [0; 32] };
    Ok(Plan { dir, arch: arch.into(), file_bytes: experts.offset + experts_len, directory, resident, experts })
}

fn write_at(f: &mut File, off: u64, b: &[u8]) -> Result<()> {
    f.seek(SeekFrom::Start(off))?;
    f.write_all(b)?;
    Ok(())
}

/// Hashes a region as it is written in order, the gaps between pieces counted as the zeros they read as.
struct RegionHash {
    h: blake3::Hasher,
    at: u64,
}

impl RegionHash {
    fn new() -> Self {
        RegionHash { h: blake3::Hasher::new(), at: 0 }
    }
    fn add(&mut self, offset: u64, b: &[u8]) -> Result<()> {
        if offset < self.at {
            return err("region written out of order");
        }
        const Z: [u8; 4096] = [0; 4096];
        while self.at < offset {
            let n = (offset - self.at).min(4096) as usize;
            self.h.update(&Z[..n]);
            self.at += n as u64;
        }
        if b.len() >= 1 << 20 {
            self.h.update_rayon(b);
        } else {
            self.h.update(b);
        }
        self.at += b.len() as u64;
        Ok(())
    }
    fn finish(mut self, bytes: u64) -> Result<[u8; 32]> {
        self.add(bytes, &[])?;
        if self.at != bytes {
            return err("region written past its end");
        }
        Ok(*self.h.finalize().as_bytes())
    }
}

/// A tensor is written this many bytes at a time at most (Flash Next's n-gram table alone is 28.8 GB).
// one page under test, so the small test models are written in many chunks
const CHUNK: u64 = if cfg!(test) { 4096 } else { 64 << 20 };

/// Writes the planned file to `path`. `tensor(i, at, buf)` fills bytes at..at + buf.len() of resident tensor i (buf
/// arrives zeroed and is a whole number of the tensor's rows; a big tensor comes in several calls, in order);
/// `pack(j, e, buf)` fills expert e's pack of EXPERTS record j (buf is the pack's length and arrives zeroed, so the
/// padding between pieces stays zero). Returns the header written.
pub fn write(
    path: &str,
    mut p: Plan,
    mut tensor: impl FnMut(usize, u64, &mut [u8]) -> Result<()>,
    mut pack: impl FnMut(usize, u32, &mut [u8]) -> Result<()>,
) -> Result<Header> {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| Error(format!("cannot create {path}: {e}")))?;
    f.set_len(p.file_bytes)?;

    let mut rh = RegionHash::new();
    let mut buf = Vec::new();
    for i in 0..p.dir.tensors.len() {
        let t = &p.dir.tensors[i];
        let (offset, bytes) = (t.offset, t.bytes);
        let row = matrix_bytes(t.ty, 1, t.dims[0]).filter(|&r| r > 0).unwrap_or(bytes.max(1));
        let step = (CHUNK / row).max(1) * row;
        let mut crc = Crc32c::default();
        let mut at = 0;
        while at < bytes {
            let n = step.min(bytes - at);
            buf.clear();
            buf.resize(n as usize, 0);
            tensor(i, at, &mut buf)?;
            crc.update(&buf);
            write_at(&mut f, p.resident.offset + offset + at, &buf)?;
            rh.add(offset + at, &buf)?;
            at += n;
        }
        p.dir.tensors[i].crc = crc.finish();
    }
    p.resident.hash = rh.finish(p.resident.bytes)?;

    let mut eh = RegionHash::new();
    for j in 0..p.dir.experts.len() {
        let n = p.dir.experts[j].n_expert;
        for e in 0..n {
            let (at, bytes) = (p.dir.experts[j].pack_at(e), p.dir.experts[j].pack.bytes);
            buf.clear();
            buf.resize(bytes as usize, 0);
            pack(j, e, &mut buf)?;
            p.dir.experts[j].crcs[e as usize] = crc32c(&buf);
            write_at(&mut f, p.experts.offset + at, &buf)?;
            eh.add(at, &buf)?;
        }
    }
    p.experts.hash = if p.experts.bytes > 0 { eh.finish(p.experts.bytes)? } else { [0; 32] };

    let db = p.dir.encode();
    if db.len() as u64 != p.directory.bytes {
        return err("the directory changed size between the plan and the write");
    }
    p.directory.hash = *blake3::hash(&db).as_bytes();
    write_at(&mut f, p.directory.offset, &db)?;
    let header = Header {
        file_bytes: p.file_bytes,
        directory: p.directory,
        resident: p.resident,
        experts: p.experts,
        model_id: model_id(&p.directory.hash, &p.resident.hash, &p.experts.hash),
        arch: p.arch,
    };
    write_at(&mut f, 0, &header.encode()?)?;
    f.sync_all()?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small model written and read back: tensors, a layer of three experts, and every check at open.
    #[test]
    fn write_then_read() {
        let dir = std::env::temp_dir().join(format!("torc-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.torc");
        let path = path.to_str().unwrap();
        let piece = |name: &str, rows, cols, at| Piece {
            name: name.into(),
            ty: 8,
            rows,
            cols,
            at,
            bytes: matrix_bytes(8, rows as u64, cols as u64).unwrap(),
            origin: Origin::NONE,
            stride: 0,
        };
        // Q8_0: 256 rows of 64 = 17,408 bytes, 5 pages each
        let gate = piece("blk.0.ffn_gate_exps.weight", 256, 64, 0);
        let up = piece("blk.0.ffn_up_exps.weight", 256, 64, 4096 * 5);
        let down = piece("blk.0.ffn_down_exps.weight", 64, 256, 4096 * 10);
        let pack_bytes = page_up(down.at + down.bytes);
        let d = Directory {
            metadata: 0u64.to_le_bytes().to_vec(),
            source: None,
            tensors: vec![
                Tensor {
                    name: "token_embd.weight".into(),
                    ty: 0,
                    n_dims: 2,
                    dims: [64, 10, 1, 1],
                    bytes: 2560,
                    ..Default::default()
                },
                Tensor {
                    name: "blk.0.ffn_gate_inp.weight".into(),
                    ty: 0,
                    n_dims: 2,
                    dims: [64, 3, 1, 1],
                    bytes: 768,
                    ..Default::default()
                },
                Tensor {
                    name: "blk.0.attn_q.weight".into(),
                    ty: 0,
                    n_dims: 2,
                    dims: [64, 64, 1, 1],
                    bytes: 16384,
                    ..Default::default()
                },
            ],
            experts: vec![Experts {
                layer: 0,
                n_expert: 3,
                n_used: 2,
                d_in: 64,
                width: 256,
                first: 0,
                pack: Pack { bytes: pack_bytes, gate, up, down },
                crcs: vec![],
            }],
        };
        let p = plan(d, "qwen35moe").unwrap();
        let h = write(
            path,
            p,
            |i, _at, b| {
                b.iter_mut().enumerate().for_each(|(k, x)| *x = (i * 31 + k) as u8);
                Ok(())
            },
            |_, e, b| {
                b[..100].fill(e as u8 + 1);
                Ok(())
            },
        )
        .unwrap();
        let m = crate::read::Model::open(path).unwrap();
        // the reader takes arch from the metadata, which this test leaves empty
        assert_eq!(Header { arch: h.arch.clone(), ..m.header.clone() }, h);
        assert_eq!(m.tensors.len(), 3);
        assert!(m.tensors.iter().all(|t| t.offset % PAGE == 0));
        assert_eq!(m.experts[0].pack_at(2), 2 * pack_bytes);
        assert_eq!(
            m.blocks.iter().map(|b| (b.kind, b.experts)).collect::<Vec<_>>(),
            [(BlockKind::AttentionFull, None), (BlockKind::Moe, Some(0))]
        );
        let r = crate::check::check(path).unwrap();
        assert_eq!((r.tensors, r.packs), (3, 3));
        // one byte changed in a pack: the check finds it
        let mut f = OpenOptions::new().write(true).open(path).unwrap();
        write_at(&mut f, m.expert_region.offset + pack_bytes + 7, &[0xEE]).unwrap();
        drop(f);
        assert!(crate::check::check(path).is_err());

        // a damaged file is refused at open, with a reason, never read: each case starts from the good file
        let good = std::fs::read(path).unwrap();
        let bad = dir.join("bad.torc");
        let bad = bad.to_str().unwrap();
        let refused = |b: &[u8]| {
            std::fs::write(bad, b).unwrap();
            crate::read::Model::open(bad).err().map(|e| e.0)
        };
        let with = |at: usize, v: u8| {
            let mut b = good.clone();
            b[at] ^= v;
            b
        };
        // a header rewritten with its checksum made good again, so only the region checks can catch it
        let header = |f: &dyn Fn(&mut Header)| {
            let mut hd = Header::decode(&good[..HEADER_BYTES as usize]).unwrap();
            f(&mut hd);
            let mut b = good.clone();
            b[..HEADER_BYTES as usize].copy_from_slice(&hd.encode().unwrap());
            b
        };
        // each case with the words its own check gives, so a case can't pass on another check's refusal
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("too short", vec![]),
            ("truncated", good[..good.len() - PAGE as usize].to_vec()),
            ("not a ToRc file", with(0, 0x20)),
            ("CRC", with(40, 1)),
            ("does not match its hash", with(m.header.directory.offset as usize + 3, 1)),
            ("lies outside", header(&|hd| hd.experts.bytes += PAGE)),
            ("lies outside", header(&|hd| hd.directory.offset = u64::MAX - PAGE + 1)),
            ("overlap", header(&|hd| hd.resident.offset = hd.directory.offset)),
        ];
        for (why, b) in cases {
            let e = refused(&b).unwrap_or_else(|| panic!("{why}: opened"));
            assert!(e.contains(why), "{why}: refused for another reason: {e}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
