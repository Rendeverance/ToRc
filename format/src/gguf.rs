//! gguf.rs - reading a GGUF file's header (ggml's GGUF spec, MIT): its key/value metadata and its tensor list. A split
//! GGUF (parts named `-00001-of-0000N`) opens as a single model: the tensors of all parts, each tagged with its part.

use crate::meta::{Meta, Value};
use crate::{Result, block_size_of, err};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Clone, Debug)]
pub struct GgufTensor {
    pub name: String,
    pub ty: u32,
    pub n_dims: u32,
    pub dims: [u64; 4],
    /// which part of a split GGUF holds it (0 for a single file)
    pub file: u16,
    /// where its bytes start in that part
    pub offset: u64,
    pub bytes: u64,
}

pub struct Gguf {
    /// the parts in order (one path for a single file)
    pub files: Vec<String>,
    pub version: u32,
    pub arch: String,
    /// the first part's key/value block as ToRc keeps it: a u64 count, then the entries as they are
    pub metadata: Vec<u8>,
    pub tensors: Vec<GgufTensor>,
}

/// The fixed size of a GGUF value type, or None for a string or an array.
fn fixed_size(t: u32) -> Option<usize> {
    Some(match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

/// The error a header parse gives when it needs more of the file than it was handed.
const SHORT: &str = "SHORT";

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: u64) -> Result<&'a [u8]> {
        if ((self.buf.len() - self.pos) as u64) < n {
            return err(SHORT);
        }
        let s = &self.buf[self.pos..self.pos + n as usize];
        self.pos += n as usize;
        Ok(s)
    }
    fn le32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn le64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<&'a [u8]> {
        let n = self.le64()?;
        self.bytes(n)
    }
    /// Moves past one value of type t without keeping it.
    fn pass_over(&mut self, t: u32) -> Result<()> {
        match t {
            8 => {
                self.string()?;
            }
            9 => {
                let et = self.le32()?;
                let n = self.le64()?;
                match fixed_size(et) {
                    Some(s) => {
                        self.bytes(n.checked_mul(s as u64).ok_or_else(|| crate::Error("array too long".into()))?)?;
                    }
                    None if et == 8 => {
                        for _ in 0..n {
                            self.string()?;
                        }
                    }
                    None => return err(format!("GGUF array of type {et}")),
                }
            }
            _ => {
                self.bytes(
                    fixed_size(t).ok_or_else(|| crate::Error(format!("GGUF value of unknown type {t}")))? as u64
                )?;
            }
        }
        Ok(())
    }
}

/// One part's header: (version, metadata as ToRc keeps it, tensors with offsets made absolute).
fn parse_part(b: &[u8], file: u16, file_bytes: u64) -> Result<(u32, Vec<u8>, Vec<GgufTensor>)> {
    let mut c = Reader { buf: b, pos: 0 };
    if c.bytes(4)? != b"GGUF" {
        return err("no GGUF magic at the start");
    }
    let version = c.le32()?;
    if !(2..=3).contains(&version) {
        return err(format!("GGUF version {version} (this reader knows 2 and 3)"));
    }
    let n_tensors = c.le64()?;
    let n_kv = c.le64()?;
    let meta_start = c.pos;
    for _ in 0..n_kv {
        c.string()?;
        let t = c.le32()?;
        c.pass_over(t)?;
    }
    let mut metadata = n_kv.to_le_bytes().to_vec();
    metadata.extend_from_slice(&b[meta_start..c.pos]);
    let meta = Meta::parse(&metadata)?;
    let alignment = match meta.get("general.alignment") {
        Some(Value::Int(a)) if *a > 0 && (*a as u64).is_power_of_two() => *a as u64,
        Some(_) => return err("general.alignment is not a power of two"),
        None => 32,
    };
    let mut tensors = Vec::with_capacity(n_tensors.min(1 << 20) as usize);
    for _ in 0..n_tensors {
        let name = String::from_utf8(c.string()?.to_vec()).map_err(|_| crate::Error("tensor name not UTF-8".into()))?;
        let n_dims = c.le32()?;
        if !(1..=4).contains(&n_dims) {
            return err(format!("tensor {name}: {n_dims} dimensions"));
        }
        let mut dims = [1u64; 4];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = c.le64()?;
        }
        let ty = c.le32()?;
        let rel = c.le64()?;
        let (blk, size) = block_size_of(ty).ok_or_else(|| crate::Error(format!("tensor {name}: unknown type {ty}")))?;
        let n = dims
            .iter()
            .try_fold(1u64, |a, &d| a.checked_mul(d))
            .ok_or_else(|| crate::Error("tensor too big".into()))?;
        if dims[0] % blk != 0 {
            return err(format!("tensor {name}: rows of {} are not whole blocks of {blk}", dims[0]));
        }
        let bytes = (n / blk).checked_mul(size).ok_or_else(|| crate::Error("tensor too big".into()))?;
        tensors.push(GgufTensor { name, ty, n_dims, dims, file, offset: rel, bytes });
    }
    let data = (c.pos as u64).div_ceil(alignment) * alignment;
    for t in tensors.iter_mut() {
        t.offset = t
            .offset
            .checked_add(data)
            .filter(|o| o.checked_add(t.bytes).is_some_and(|e| e <= file_bytes))
            .ok_or_else(|| crate::Error(format!("tensor {} lies outside its file", t.name)))?;
    }
    Ok((version, metadata, tensors))
}

/// Reads a part's header, reading more of the file until the header fits (a vocabulary can run to tens of MB).
fn read_part(path: &str, file: u16) -> Result<(u32, Vec<u8>, Vec<GgufTensor>)> {
    let mut f = File::open(path).map_err(|e| crate::Error(format!("cannot open {path}: {e}")))?;
    let len = f.metadata()?.len();
    let mut want = (32u64 << 20).min(len);
    loop {
        let mut b = vec![0u8; want as usize];
        f.seek(SeekFrom::Start(0))?;
        f.read_exact(&mut b)?;
        match parse_part(&b, file, len) {
            Err(e) if e.0 == SHORT && want < len => want = (want * 8).min(len),
            Err(e) if e.0 == SHORT => return err(format!("{path}: the header runs past the end of the file")),
            Err(e) => return err(format!("{path}: {e}")),
            Ok(r) => return Ok(r),
        }
    }
}

/// The parts of a split GGUF named by its first part (`...-00001-of-0000N.gguf`), or just the path.
fn parts_of(path: &str) -> Vec<String> {
    let stem = path.strip_suffix(".gguf").unwrap_or(path);
    let split = || -> Option<(String, usize, usize)> {
        let (head, total) = stem.rsplit_once("-of-")?;
        let (base, no) = head.rsplit_once('-')?;
        let (n, k) = (total.parse::<usize>().ok()?, no.parse::<usize>().ok()?);
        (k == 1 && n > 1 && no.len() == total.len()).then(|| (base.to_string(), n, no.len()))
    };
    match split() {
        Some((base, n, w)) => (1..=n).map(|i| format!("{base}-{i:0w$}-of-{n:0w$}.gguf")).collect(),
        None => vec![path.to_string()],
    }
}

impl Gguf {
    /// Opens a GGUF, or every part of a split one given its first part.
    pub fn open(path: &str) -> Result<Gguf> {
        let files = parts_of(path);
        let (version, metadata, mut tensors) = read_part(&files[0], 0)?;
        for (i, p) in files.iter().enumerate().skip(1) {
            let (_, _, t) = read_part(p, i as u16)?;
            tensors.extend(t);
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(t) = tensors.iter().find(|t| !seen.insert(t.name.clone())) {
            return err(format!("tensor {} appears twice", t.name));
        }
        let arch = Meta::parse(&metadata)?.str("general.architecture").unwrap_or("").to_string();
        Ok(Gguf { files, version, arch, metadata, tensors })
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    /// A GGUF image: the given metadata entries (key, type, value bytes) and tensors (name, type, dims, bytes).
    pub fn image(kv: &[(&str, u32, Vec<u8>)], tensors: &[(&str, u32, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        for (k, t, v) in kv {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&t.to_le_bytes());
            b.extend_from_slice(v);
        }
        let mut at = 0u64;
        for (name, ty, dims, data) in tensors {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&at.to_le_bytes());
            at += (data.len() as u64).div_ceil(32) * 32;
        }
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        for (.., data) in tensors {
            b.extend_from_slice(data);
            while !b.len().is_multiple_of(32) {
                b.push(0);
            }
        }
        b
    }

    pub fn string(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    #[test]
    fn parses_and_finds_parts() {
        let img = image(
            &[("general.architecture", 8, string("qwen35moe")), ("x.n", 4, 7u32.to_le_bytes().to_vec())],
            &[("a.weight", 0, vec![4, 2], vec![1; 32]), ("b.weight", 8, vec![32, 3], vec![2; 102])],
        );
        let (v, meta, t) = super::parse_part(&img, 0, img.len() as u64).unwrap();
        assert_eq!(v, 3);
        assert_eq!(crate::meta::Meta::parse(&meta).unwrap().u32("x.n"), Some(7));
        assert_eq!((t[1].bytes, t[1].dims[0], t[1].dims[1]), (102, 32, 3));
        assert_eq!(&img[t[1].offset as usize..t[1].offset as usize + 4], &[2, 2, 2, 2]);
        assert_eq!(super::parse_part(&img[..60], 0, 60).unwrap_err().0, super::SHORT);
        assert_eq!(
            super::parts_of("m/x-00001-of-00003.gguf"),
            ["m/x-00001-of-00003.gguf", "m/x-00002-of-00003.gguf", "m/x-00003-of-00003.gguf"]
        );
        assert_eq!(super::parts_of("m/x.gguf"), ["m/x.gguf"]);
    }
}
