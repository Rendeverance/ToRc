//! format - the ToRc model file, layout 3. The spec is FORMAT.md at the workspace root.
//!
//! A file is a 4 KiB header, a directory of records, the resident tensors, and one pack per routed expert (its gate, up
//! and down matrices together, so one expert is one read). Every weight byte comes from a GGUF file or is a zero added
//! to widen a matrix, and the directory records where each one came from.

pub mod blocks;
pub mod check;
pub mod convert;
pub mod crc32c;
pub mod gguf;
pub mod meta;
pub mod read;
pub mod shard;
pub mod write;

pub use blocks::{Block, BlockKind};
pub use crc32c::{Crc32c, crc32c};

use std::fmt;

pub const MAGIC: [u8; 8] = *b"TORC\r\n\x1a\n";
pub const VERSION: u16 = 3;
/// Every region, resident tensor, pack and piece starts on this boundary.
pub const PAGE: u64 = 4096;
pub const HEADER_BYTES: u64 = 4096;

pub const TAG_METADATA: u16 = 1;
pub const TAG_SOURCE: u16 = 2;
pub const TAG_TENSOR: u16 = 3;
pub const TAG_EXPERTS: u16 = 4;
/// A record flag: a reader that does not know the record's tag must refuse the file.
pub const NEEDED: u16 = 1;

/// Origin.file when the bytes have no source file.
pub const NO_SOURCE: u16 = u16::MAX;
/// Change bits of an Origin (FORMAT.md section 5).
pub const ROWS_ADDED: u16 = 1;
pub const ROW_TAILS_ADDED: u16 = 2;
pub const FUSED_FIRST_HALF: u16 = 4;
pub const FUSED_SECOND_HALF: u16 = 8;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

/// The next multiple of PAGE at or after v.
pub fn page_up(v: u64) -> u64 {
    v.div_ceil(PAGE) * PAGE
}

/// A stretch of the file: where it starts, how long it is, and its BLAKE3 (zero when not recorded).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub offset: u64,
    pub bytes: u64,
    pub hash: [u8; 32],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header {
    pub file_bytes: u64,
    pub directory: Region,
    pub resident: Region,
    pub experts: Region,
    /// BLAKE3 of the three region hashes in order: two files with the same id hold the same model
    pub model_id: [u8; 32],
    pub arch: String,
}

/// Where bytes came from: a source file (an index into Source.files, or NO_SOURCE), what was done to them (the
/// change bits), and where they start in that file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Origin {
    pub file: u16,
    pub changes: u16,
    pub offset: u64,
}

impl Origin {
    #[cfg(test)]
    pub const NONE: Origin = Origin { file: NO_SOURCE, changes: 0, offset: 0 };
}

/// A resident tensor. dims are GGUF's (the row length first); offset is within the resident region.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub ty: u32,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub offset: u64,
    pub bytes: u64,
    pub crc: u32,
    pub origin: Origin,
}

/// One expert's matrix inside its pack: rows of `cols` weights, starting `at` bytes into the pack. `name` is the
/// GGUF tensor it came from, and `stride` how far apart consecutive experts' matrices sit there.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Piece {
    pub name: String,
    pub ty: u32,
    pub rows: u32,
    pub cols: u32,
    pub at: u64,
    pub bytes: u64,
    pub origin: Origin,
    pub stride: u64,
}

impl Piece {
    /// The piece's bytes within a pack.
    pub fn of<'a>(&self, pack: &'a [u8]) -> &'a [u8] {
        &pack[self.at as usize..(self.at + self.bytes) as usize]
    }
}

/// The shape of every pack of one MoE layer: its length and its three matrices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pack {
    pub bytes: u64,
    pub gate: Piece,
    pub up: Piece,
    pub down: Piece,
}

impl Pack {
    pub fn pieces(&self) -> [&Piece; 3] {
        [&self.gate, &self.up, &self.down]
    }
}

/// One MoE layer's routed experts: n_expert packs in expert order from `first` (within the expert region), each
/// `pack.bytes` long, with a CRC-32C each.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Experts {
    pub layer: u32,
    pub n_expert: u32,
    pub n_used: u32,
    pub d_in: u32,
    pub width: u32,
    pub first: u64,
    pub pack: Pack,
    pub crcs: Vec<u32>,
}

impl Experts {
    /// Where expert e's pack starts within the expert region.
    pub fn pack_at(&self, e: u32) -> u64 {
        self.first + e as u64 * self.pack.bytes
    }

    /// The bytes of all this layer's packs within the expert region: (start, length).
    pub fn span(&self) -> (u64, u64) {
        (self.first, self.n_expert as u64 * self.pack.bytes)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceFile {
    pub bytes: u64,
    pub hash: [u8; 32],
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Source {
    pub files: Vec<SourceFile>,
    pub converter: String,
}

/// Everything the directory says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Directory {
    pub metadata: Vec<u8>,
    pub source: Option<Source>,
    pub tensors: Vec<Tensor>,
    pub experts: Vec<Experts>,
}

// Encoding

/// Little-endian output.
#[derive(Default)]
pub(crate) struct Out(pub Vec<u8>);

impl Out {
    pub fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    pub fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.bytes(s.as_bytes());
    }
    pub fn origin(&mut self, o: &Origin) {
        self.u64(o.offset);
        self.u32(o.file as u32);
        self.u32(o.changes as u32);
    }
    /// A record: tag, flags, payload length, payload, zeros to a multiple of 8.
    pub fn record(&mut self, tag: u16, flags: u16, payload: &[u8]) {
        self.u16(tag);
        self.u16(flags);
        self.u32(payload.len() as u32);
        self.bytes(payload);
        while !self.0.len().is_multiple_of(8) {
            self.0.push(0);
        }
    }
}

/// Little-endian input with every length checked.
pub(crate) struct In<'a> {
    pub b: &'a [u8],
    pub p: usize,
}

impl<'a> In<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        In { b, p: 0 }
    }
    pub fn left(&self) -> usize {
        self.b.len() - self.p
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.left() < n {
            return err("a directory record runs past its end");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn hash(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    pub fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| Error("a directory string is not UTF-8".into()))
    }
    pub fn origin(&mut self) -> Result<Origin> {
        let offset = self.u64()?;
        let file = self.u32()?;
        let changes = self.u32()?;
        if file > u16::MAX as u32 || changes > u16::MAX as u32 {
            return err("an origin's source file or change bits out of range");
        }
        Ok(Origin { file: file as u16, changes: changes as u16, offset })
    }
}

impl Header {
    /// The header's bytes. arch is not among them: it is the metadata's general.architecture, read with the directory.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut o = Out::default();
        o.bytes(&MAGIC);
        o.u64(self.file_bytes);
        o.u32(VERSION as u32);
        o.u32(3);
        for r in [&self.directory, &self.resident, &self.experts] {
            o.u64(r.offset);
            o.u64(r.bytes);
            o.bytes(&r.hash);
        }
        o.bytes(&self.model_id);
        let mut b = o.0;
        b.resize(HEADER_BYTES as usize, 0);
        let crc = crc32c(&b[..4092]);
        b[4092..].copy_from_slice(&crc.to_le_bytes());
        Ok(b)
    }

    pub fn decode(b: &[u8]) -> Result<Header> {
        if b.len() < HEADER_BYTES as usize {
            return err("too short to hold a ToRc header");
        }
        if b[..8] != MAGIC {
            return err("not a ToRc file");
        }
        let mut i = In::new(&b[8..HEADER_BYTES as usize]);
        let file_bytes = i.u64()?;
        let version = i.u32()?;
        if version != VERSION as u32 {
            return err(format!(
                "a ToRc file of layout {version}: this reader knows layout {VERSION} (convert the \
                model again from its GGUF: torc convert-gguf)"
            ));
        }
        if crc32c(&b[..4092]) != u32::from_le_bytes(b[4092..4096].try_into().unwrap()) {
            return err("the header's CRC does not match it");
        }
        if i.u32()? != 3 {
            return err("the header does not list three regions");
        }
        let mut regions = [Region::default(); 3];
        for r in regions.iter_mut() {
            r.offset = i.u64()?;
            r.bytes = i.u64()?;
            r.hash = i.hash()?;
        }
        let model_id = i.hash()?;
        let [directory, resident, experts] = regions;
        Ok(Header { file_bytes, directory, resident, experts, model_id, arch: String::new() })
    }
}

/// The model id: BLAKE3 of the directory, resident and expert hashes, in order.
pub fn model_id(directory: &[u8; 32], resident: &[u8; 32], experts: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(directory);
    h.update(resident);
    h.update(experts);
    *h.finalize().as_bytes()
}

/// A piece and its role in the pack (0 gate, 1 up, 2 down).
fn piece_out(o: &mut Out, p: &Piece, role: u32) {
    o.str(&p.name);
    o.u64(p.at);
    o.u64(p.bytes);
    o.u64(p.stride);
    o.u32(p.ty);
    o.u32(p.rows);
    o.u32(p.cols);
    o.u32(role);
    o.origin(&p.origin);
}

fn piece_in(i: &mut In, role: u32) -> Result<Piece> {
    let name = i.str()?;
    let at = i.u64()?;
    let bytes = i.u64()?;
    let stride = i.u64()?;
    let ty = i.u32()?;
    let rows = i.u32()?;
    let cols = i.u32()?;
    if i.u32()? != role {
        return err(format!("{name}: a pack's pieces out of order"));
    }
    Ok(Piece { name, ty, rows, cols, at, bytes, origin: i.origin()?, stride })
}

impl Directory {
    pub fn encode(&self) -> Vec<u8> {
        let mut o = Out::default();
        o.record(TAG_METADATA, NEEDED, &self.metadata);
        if let Some(s) = &self.source {
            let mut p = Out::default();
            p.u16(s.files.len() as u16);
            for f in &s.files {
                p.u64(f.bytes);
                p.bytes(&f.hash);
                p.str(&f.name);
            }
            p.str(&s.converter);
            o.record(TAG_SOURCE, 0, &p.0);
        }
        for t in &self.tensors {
            let mut p = Out::default();
            p.str(&t.name);
            p.u64(t.offset);
            p.u64(t.bytes);
            p.u32(t.crc);
            p.u32(t.ty);
            p.u32(t.n_dims);
            for d in &t.dims[..t.n_dims as usize] {
                p.u64(*d);
            }
            p.origin(&t.origin);
            o.record(TAG_TENSOR, NEEDED, &p.0);
        }
        for x in &self.experts {
            let mut p = Out::default();
            p.u64(x.first);
            p.u64(x.pack.bytes);
            for v in [x.layer, x.n_expert, x.n_used, x.d_in, x.width, 3] {
                p.u32(v);
            }
            for (role, pc) in x.pack.pieces().into_iter().enumerate() {
                piece_out(&mut p, pc, role as u32);
            }
            for c in &x.crcs {
                p.u32(*c);
            }
            o.record(TAG_EXPERTS, NEEDED, &p.0);
        }
        o.0
    }

    pub fn decode(b: &[u8]) -> Result<Directory> {
        let mut d = Directory::default();
        let mut seen_meta = false;
        let mut i = In::new(b);
        while i.left() > 0 {
            let tag = i.u16()?;
            let flags = i.u16()?;
            let n = i.u32()? as usize;
            let payload = i.take(n)?;
            i.take((8 - (8 + n) % 8) % 8)?;
            let mut p = In::new(payload);
            match tag {
                TAG_METADATA => {
                    if seen_meta {
                        return err("two METADATA records");
                    }
                    seen_meta = true;
                    d.metadata = payload.to_vec();
                    p.p = n;
                }
                TAG_SOURCE => {
                    let nf = p.u16()?;
                    let mut s = Source::default();
                    for _ in 0..nf {
                        let bytes = p.u64()?;
                        let hash = p.hash()?;
                        s.files.push(SourceFile { bytes, hash, name: p.str()? });
                    }
                    s.converter = p.str()?;
                    d.source = Some(s);
                }
                TAG_TENSOR => {
                    let name = p.str()?;
                    let offset = p.u64()?;
                    let bytes = p.u64()?;
                    let crc = p.u32()?;
                    let ty = p.u32()?;
                    let n_dims = p.u32()?;
                    if !(1..=4).contains(&n_dims) {
                        return err(format!("{name}: {n_dims} dimensions"));
                    }
                    let mut dims = [1u64; 4];
                    for dim in dims.iter_mut().take(n_dims as usize) {
                        *dim = p.u64()?;
                    }
                    d.tensors.push(Tensor { name, ty, n_dims, dims, offset, bytes, crc, origin: p.origin()? });
                }
                TAG_EXPERTS => {
                    let first = p.u64()?;
                    let bytes = p.u64()?;
                    let layer = p.u32()?;
                    let n_expert = p.u32()?;
                    let n_used = p.u32()?;
                    let d_in = p.u32()?;
                    let width = p.u32()?;
                    if p.u32()? != 3 {
                        return err(format!("layer {layer}: a pack of other than three pieces"));
                    }
                    let gate = piece_in(&mut p, 0)?;
                    let up = piece_in(&mut p, 1)?;
                    let down = piece_in(&mut p, 2)?;
                    if p.left() != n_expert as usize * 4 {
                        return err(format!("layer {layer}: the expert CRCs do not match the expert count"));
                    }
                    let crcs = (0..n_expert).map(|_| p.u32()).collect::<Result<Vec<_>>>()?;
                    d.experts.push(Experts {
                        layer,
                        n_expert,
                        n_used,
                        d_in,
                        width,
                        first,
                        pack: Pack { bytes, gate, up, down },
                        crcs,
                    });
                }
                _ if flags & NEEDED != 0 => return err(format!("the file needs directory record {tag}, unknown here")),
                _ => p.p = n,
            }
            if p.left() != 0 {
                return err(format!("directory record {tag} has {} bytes left over", p.left()));
            }
        }
        if !seen_meta {
            return err("no METADATA record");
        }
        Ok(d)
    }
}

/// The block size of a ggml type, in weights and in bytes (ggml's type table), or None if the type is unknown here.
pub fn block_size_of(ty: u32) -> Option<(u64, u64)> {
    Some(match ty {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        29 => (256, 56),
        30 => (1, 2),
        39 => (32, 17),
        42 => (64, 18),
        _ => return None,
    })
}

/// The bytes of `rows` rows of `cols` weights of type ty, or None if ty is unknown, cols is not whole blocks or the
/// size does not fit in 64 bits.
pub fn matrix_bytes(ty: u32, rows: u64, cols: u64) -> Option<u64> {
    let (blk, size) = block_size_of(ty)?;
    if !cols.is_multiple_of(blk) {
        return None;
    }
    rows.checked_mul(cols / blk)?.checked_mul(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(name: &str, at: u64) -> Piece {
        Piece {
            name: name.into(),
            ty: 2,
            rows: 512,
            cols: 2048,
            at,
            bytes: 589_824,
            origin: Origin { file: 0, changes: FUSED_FIRST_HALF, offset: 77 },
            stride: 1 << 20,
        }
    }

    #[test]
    fn header_round_trip_and_crc() {
        let h = Header {
            file_bytes: 1 << 34,
            directory: Region { offset: 4096, bytes: 999, hash: [1; 32] },
            resident: Region { offset: 8192, bytes: 1 << 30, hash: [2; 32] },
            experts: Region { offset: 8192 + (1 << 30), bytes: 1 << 33, hash: [3; 32] },
            model_id: model_id(&[1; 32], &[2; 32], &[3; 32]),
            arch: String::new(),
        };
        let b = h.encode().unwrap();
        assert_eq!(Header::decode(&b).unwrap(), h);
        let mut bad = b.clone();
        bad[300] ^= 1;
        assert!(Header::decode(&bad).unwrap_err().0.contains("CRC"));
        let mut v2 = b.clone();
        v2[16] = 2;
        assert!(Header::decode(&v2).unwrap_err().0.contains("layout 2"));
        let mut other = b;
        other[..8].copy_from_slice(b"NOTTORC!");
        assert!(Header::decode(&other).unwrap_err().0.contains("not a ToRc file"));
    }

    #[test]
    fn directory_round_trip() {
        let d = Directory {
            metadata: vec![0, 0, 0, 0, 0, 0, 0, 0],
            source: Some(Source {
                files: vec![SourceFile { bytes: 5, hash: [9; 32], name: "m.gguf".into() }],
                converter: "torc 2".into(),
            }),
            tensors: vec![Tensor {
                name: "output.weight".into(),
                ty: 14,
                n_dims: 2,
                dims: [2048, 248_320, 1, 1],
                offset: 4096,
                bytes: 3,
                crc: 7,
                origin: Origin { file: 0, changes: 0, offset: 12 },
            }],
            experts: vec![Experts {
                layer: 3,
                n_expert: 2,
                n_used: 1,
                d_in: 2048,
                width: 512,
                first: 0,
                pack: Pack {
                    bytes: 1 << 21,
                    gate: piece("blk.3.ffn_gate_up_exps.weight", 0),
                    up: piece("blk.3.ffn_gate_up_exps.weight", 593_920),
                    down: piece("blk.3.ffn_down_exps.weight", 1 << 20),
                },
                crcs: vec![11, 12],
            }],
        };
        assert_eq!(Directory::decode(&d.encode()).unwrap(), d);
    }

    #[test]
    fn unknown_records() {
        let mut o = Out::default();
        o.record(TAG_METADATA, NEEDED, &[0; 8]);
        o.record(99, 0, &[1, 2, 3]);
        assert!(Directory::decode(&o.0).is_ok());
        o.record(98, NEEDED, &[1]);
        assert!(Directory::decode(&o.0).unwrap_err().0.contains("needs directory record 98"));
    }
}
