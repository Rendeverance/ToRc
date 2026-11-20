//! read.rs - opening a file: the header and the directory, validated before anything is used. Every size and offset
//! comes from the file, so each is checked against the file's length with overflow-checked arithmetic.

use crate::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// The largest directory read (a model with every record still uses a few MB).
const MAX_DIRECTORY: u64 = 256 << 20;

#[derive(Debug)]
pub struct Model {
    pub path: String,
    pub header: Header,
    pub metadata: Vec<u8>,
    pub source: Option<Source>,
    pub tensors: Vec<Tensor>,
    /// one per MoE layer, in layer order
    pub experts: Vec<Experts>,
    /// found from the tensor names (blocks.rs)
    pub blocks: Vec<Block>,
    pub resident: Region,
    pub expert_region: Region,
}

fn fetch(f: &mut File, off: u64, len: u64) -> Result<Vec<u8>> {
    let mut b = vec![0u8; len as usize];
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(&mut b)?;
    Ok(b)
}

/// offset + bytes, if it fits in `within` without overflow.
fn fits(offset: u64, bytes: u64, within: u64) -> bool {
    offset.checked_add(bytes).is_some_and(|e| e <= within)
}

impl Model {
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The model a converter's plan describes (write::plan), held in memory: the directory, regions and blocks the
    /// written file would have, with CRCs, region hashes and the model id still zero. `path` names the source.
    pub fn from_plan(path: &str, p: crate::write::Plan) -> Model {
        let header = Header {
            file_bytes: p.file_bytes,
            directory: p.directory,
            resident: p.resident,
            experts: p.experts,
            model_id: [0; 32],
            arch: p.arch,
        };
        let blocks = blocks::blocks(&header.arch, &p.dir);
        Model {
            path: path.into(),
            metadata: p.dir.metadata,
            source: p.dir.source,
            tensors: p.dir.tensors,
            experts: p.dir.experts,
            blocks,
            resident: p.resident,
            expert_region: p.experts,
            header,
        }
    }

    pub fn open(path: &str) -> Result<Model> {
        let mut f = File::open(path).map_err(|e| Error(format!("cannot open {path}: {e}")))?;
        let len = f.metadata()?.len();
        let mut header = Header::decode(&fetch(&mut f, 0, HEADER_BYTES.min(len))?)?;
        if header.file_bytes != len {
            return err(format!("{path} is {len} bytes, its header says {} (a truncated copy?)", header.file_bytes));
        }
        let (dr, rr, er) = (header.directory, header.resident, header.experts);
        let mut spans = vec![(0, HEADER_BYTES)];
        for (name, r) in [("directory", dr), ("resident", rr), ("expert", er)] {
            if r.offset % PAGE != 0 || r.offset < HEADER_BYTES || !fits(r.offset, r.bytes, len) {
                return err(format!("the {name} region lies outside the file or off a page boundary"));
            }
            spans.push((r.offset, r.offset + r.bytes));
        }
        spans.sort();
        if spans.windows(2).any(|w| w[1].0 < w[0].1) {
            return err("regions overlap");
        }
        if dr.bytes > MAX_DIRECTORY {
            return err("implausibly large directory");
        }
        let db = fetch(&mut f, dr.offset, dr.bytes)?;
        if *blake3::hash(&db).as_bytes() != dr.hash {
            return err("the directory does not match its hash");
        }
        let d = Directory::decode(&db)?;

        let mut names = std::collections::HashSet::new();
        for t in &d.tensors {
            if !names.insert(t.name.as_str()) {
                return err(format!("tensor {} listed twice", t.name));
            }
            let weights: u64 = t.dims[..t.n_dims as usize].iter().try_fold(1u64, |a, &x| a.checked_mul(x)).unwrap_or(0);
            let want = matrix_bytes(t.ty, weights / t.dims[0].max(1), t.dims[0]);
            if want != Some(t.bytes) || weights == 0 || t.offset % PAGE != 0 || !fits(t.offset, t.bytes, rr.bytes) {
                return err(format!("tensor {}: its type, shape, size or place do not hold", t.name));
            }
        }
        let mut layers = std::collections::HashSet::new();
        for x in &d.experts {
            let l = x.layer;
            if !layers.insert(l) {
                return err(format!("layer {l} has two EXPERTS records"));
            }
            if x.n_expert == 0 || x.n_used == 0 || x.n_used > x.n_expert || x.crcs.len() != x.n_expert as usize {
                return err(format!("layer {l}: bad expert counts"));
            }
            let p = &x.pack;
            if p.bytes == 0 || p.bytes % PAGE != 0 || x.first % PAGE != 0 {
                return err(format!("layer {l}: packs off a page boundary"));
            }
            let total = (x.n_expert as u64).checked_mul(p.bytes);
            if !total.is_some_and(|n| fits(x.first, n, er.bytes)) {
                return err(format!("layer {l}: packs outside the expert region"));
            }
            for (what, pc) in [("gate", &p.gate), ("up", &p.up), ("down", &p.down)] {
                if matrix_bytes(pc.ty, pc.rows as u64, pc.cols as u64) != Some(pc.bytes)
                    || pc.at % PAGE != 0
                    || !fits(pc.at, pc.bytes, p.bytes)
                {
                    return err(format!("layer {l}: the {what} piece's type, shape or place do not hold"));
                }
            }
            // down's rows are the width, or the width rounded up to 256 when only down was widened (qwen4exp)
            let down_ok = p.down.cols == x.width || p.down.cols as u64 == (x.width as u64).div_ceil(256) * 256;
            // down's rows are the embedding width, which d_in is rounded up to 256 when the converter padded the
            // embedding (gpt-oss)
            let rows_ok = p.down.rows == x.d_in || x.d_in as u64 == (p.down.rows as u64).div_ceil(256) * 256;
            if p.gate.rows != x.width
                || p.up.rows != x.width
                || p.gate.cols != x.d_in
                || p.up.cols != x.d_in
                || !rows_ok
                || !down_ok
            {
                return err(format!("layer {l}: the pieces' shapes do not match the layer's widths"));
            }
        }
        header.arch = crate::meta::Meta::parse(&d.metadata)?.str("general.architecture").unwrap_or("").to_string();
        let blocks = blocks::blocks(&header.arch, &d);
        Ok(Model {
            path: path.into(),
            metadata: d.metadata,
            source: d.source,
            tensors: d.tensors,
            experts: d.experts,
            blocks,
            resident: rr,
            expert_region: er,
            header,
        })
    }
}
