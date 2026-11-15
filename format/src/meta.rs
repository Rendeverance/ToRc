//! meta.rs - the METADATA section: GGUF's key/value region, byte for byte (a u64 count, then entries of a u64 key
//! length, the key, a u32 type and the value). It is parsed once into typed values, and every length is checked against
//! the section.

use crate::{Result, err};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i128),
    Float(f64),
    Bool(bool),
    Str(String),
    /// the element type, then the elements (strings as Value::Str, numbers as Int / Float)
    Array(u32, Vec<Value>),
}

const U8: u32 = 0;
const I8: u32 = 1;
const U16: u32 = 2;
const I16: u32 = 3;
const U32: u32 = 4;
const I32: u32 = 5;
const F32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const U64: u32 = 10;
const I64: u32 = 11;
const F64: u32 = 12;

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.buf.len() - self.pos < n {
            return err("metadata entry runs past the section");
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn le32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn le64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.le64()?;
        if n > (self.buf.len() - self.pos) as u64 {
            return err("metadata string runs past the section");
        }
        Ok(String::from_utf8_lossy(self.bytes(n as usize)?).into_owned())
    }
    fn value(&mut self, t: u32, in_array: bool) -> Result<Value> {
        let b = |c: &mut Self, n| c.bytes(n);
        Ok(match t {
            U8 => Value::Int(b(self, 1)?[0] as i128),
            I8 => Value::Int(b(self, 1)?[0] as i8 as i128),
            U16 => Value::Int(u16::from_le_bytes(b(self, 2)?.try_into().unwrap()) as i128),
            I16 => Value::Int(i16::from_le_bytes(b(self, 2)?.try_into().unwrap()) as i128),
            U32 => Value::Int(self.le32()? as i128),
            I32 => Value::Int(self.le32()? as i32 as i128),
            U64 => Value::Int(self.le64()? as i128),
            I64 => Value::Int(self.le64()? as i64 as i128),
            F32 => Value::Float(f32::from_le_bytes(b(self, 4)?.try_into().unwrap()) as f64),
            F64 => Value::Float(f64::from_le_bytes(b(self, 8)?.try_into().unwrap())),
            BOOL => Value::Bool(b(self, 1)?[0] != 0),
            STRING => Value::Str(self.string()?),
            ARRAY if !in_array => {
                let et = self.le32()?;
                let n = self.le64()?;
                if n > (self.buf.len() - self.pos) as u64 {
                    return err("metadata array longer than the section");
                }
                let v = (0..n).map(|_| self.value(et, true)).collect::<Result<Vec<_>>>()?;
                Value::Array(et, v)
            }
            _ => return err(format!("metadata value of unknown type {t}")),
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub entries: Vec<(String, Value)>,
}

impl Meta {
    pub fn parse(b: &[u8]) -> Result<Meta> {
        let mut c = Reader { buf: b, pos: 0 };
        let n = c.le64()?;
        if n > b.len() as u64 / 13 {
            return err("implausible metadata entry count");
        }
        let mut entries = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let k = c.string()?;
            let t = c.le32()?;
            entries.push((k, c.value(t, false)?));
        }
        if c.pos != b.len() {
            return err("metadata section has trailing bytes");
        }
        Ok(Meta { entries })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn u32(&self, key: &str) -> Option<u32> {
        match self.get(key)? {
            Value::Int(i) => u32::try_from(*i).ok(),
            Value::Bool(b) => Some(*b as u32),
            _ => None,
        }
    }

    /// An f32 from the metadata: an f32 as stored, or an f64 rounded to f32.
    pub fn f32(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            Value::Float(f) => Some(*f as f32),
            _ => None,
        }
    }

    /// `<arch>.<key>`, the GGUF convention for hyperparameters.
    pub fn arch_key(&self, key: &str) -> String {
        format!("{}.{key}", self.str("general.architecture").unwrap_or(""))
    }

    pub fn arch_u32(&self, key: &str) -> Option<u32> {
        self.u32(&self.arch_key(key))
    }

    pub fn arch_f32(&self, key: &str) -> Option<f32> {
        self.f32(&self.arch_key(key))
    }

    /// Integer arrays (per-layer counts and the like).
    pub fn ints(&self, key: &str) -> Option<Vec<i128>> {
        match self.get(key)? {
            Value::Array(_, v) => v.iter().map(|x| if let Value::Int(i) = x { Some(*i) } else { None }).collect(),
            _ => None,
        }
    }

    /// The length of a string array (the tokenizer's vocabulary), without copying the strings out again.
    pub fn len_of(&self, key: &str) -> Option<usize> {
        match self.get(key)? {
            Value::Array(_, v) => Some(v.len()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(out: &mut Vec<u8>, k: &str, t: u32, v: &[u8]) {
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        out.extend(t.to_le_bytes());
        out.extend(v);
    }

    #[test]
    fn parses_scalars_strings_arrays() {
        let mut b = 4u64.to_le_bytes().to_vec();
        let mut s = 9u64.to_le_bytes().to_vec();
        s.extend(b"qwen35moe");
        kv(&mut b, "general.architecture", STRING, &s);
        kv(&mut b, "qwen35moe.block_count", U32, &41u32.to_le_bytes());
        kv(&mut b, "qwen35moe.attention.layer_norm_rms_epsilon", F32, &1e-6f32.to_le_bytes());
        let mut a = I32.to_le_bytes().to_vec();
        a.extend(3u64.to_le_bytes());
        for x in [1i32, -2, 3] {
            a.extend(x.to_le_bytes());
        }
        kv(&mut b, "x.list", ARRAY, &a);
        let m = Meta::parse(&b).unwrap();
        assert_eq!(m.arch_u32("block_count"), Some(41));
        assert_eq!(m.arch_f32("attention.layer_norm_rms_epsilon"), Some(1e-6));
        assert_eq!(m.ints("x.list"), Some(vec![1, -2, 3]));
        assert!(Meta::parse(&b[..b.len() - 1]).is_err());
    }
}
