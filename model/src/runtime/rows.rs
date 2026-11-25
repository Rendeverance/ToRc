//! rows.rs - a table read one row per token, straight from the model file: Flash Next's n-gram table (tens of GB,
//! 16 rows a token) once a mapless load has dropped the file's mapping (weights.rs). Each row is an unbuffered read of
//! the sectors that hold it, spread over the compute pool (each pool thread keeps its own handle), a chunk's repeated
//! rows read once, and the rows read lately kept in a bounded cache (TORC_ROW_CACHE rows, default 1M: Strata keeps
//! about as many). The bytes are the file's, so the products are the same as through the mapping.

use crate::ecache_own::{ALIGN, open_unbuffered, read_at};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct RowReader {
    path: String,
    /// the file offset of row 0
    base: u64,
    /// bytes a row (at most ALIGN, so a row spans at most two sectors)
    rb: usize,
    cache: Mutex<RowCache>,
}

struct RowCache {
    at: HashMap<u32, usize>,
    bytes: Vec<u8>,
    most: usize,
}

thread_local! {
    /// this thread's unbuffered handle (for the file of that path) and its two-sector buffer
    static HANDLE: RefCell<Option<(String, std::fs::File)>> = const { RefCell::new(None) };
    static SECTORS: RefCell<Sectors> = RefCell::new(Sectors::new());
}

/// Two sectors' worth of ALIGN-aligned bytes, for one row's read.
struct Sectors {
    raw: Vec<u8>,
}

impl Sectors {
    fn new() -> Sectors {
        Sectors { raw: vec![0u8; 3 * ALIGN] }
    }

    fn buf(&mut self) -> &mut [u8] {
        let skew = self.raw.as_ptr().align_offset(ALIGN);
        &mut self.raw[skew..skew + 2 * ALIGN]
    }
}

impl RowReader {
    pub fn new(path: &str, base: u64, rb: usize) -> std::io::Result<RowReader> {
        if rb == 0 || rb > ALIGN {
            return Err(std::io::Error::other(format!("row reader: rows of {rb} bytes")));
        }
        // the file opens here once, so a path that cannot be read fails at load rather than in the first reply
        open_unbuffered(path)?;
        let most = crate::switches::num::<usize>("TORC_ROW_CACHE").unwrap_or(1 << 20);
        Ok(RowReader {
            path: path.to_string(),
            base,
            rb,
            cache: Mutex::new(RowCache { at: HashMap::new(), bytes: Vec::new(), most }),
        })
    }

    /// One row from the file into out (rb bytes), on the calling thread's own handle.
    fn read_row(&self, r: u32, out: &mut [u8]) -> std::io::Result<()> {
        let off = self.base + r as u64 * self.rb as u64;
        let a = off & !(ALIGN as u64 - 1);
        let len = ((off + self.rb as u64 - a) as usize).div_ceil(ALIGN) * ALIGN;
        HANDLE.with(|h| {
            let mut h = h.borrow_mut();
            if h.as_ref().is_none_or(|(p, _)| *p != self.path) {
                *h = Some((self.path.clone(), open_unbuffered(&self.path)?));
            }
            let f = &h.as_ref().expect("opened above").1;
            SECTORS.with(|s| {
                let mut s = s.borrow_mut();
                let buf = s.buf();
                read_at(f, buf.as_mut_ptr(), len, a)?;
                let at = (off - a) as usize;
                out.copy_from_slice(&buf[at..at + self.rb]);
                Ok(())
            })
        })
    }

    /// The bytes of `rows` (rb each, in that order): the cached ones copied, the rest read once each (a row asked for
    /// twice is read once) over the pool, then cached.
    pub fn read(&self, rows: &[u32]) -> std::io::Result<Vec<u8>> {
        let rb = self.rb;
        let mut c = self.cache.lock().expect("row cache");
        let mut miss: Vec<u32> = rows.iter().copied().filter(|r| !c.at.contains_key(r)).collect();
        miss.sort_unstable();
        miss.dedup();
        // a cache that would pass its bound starts again (a chunk's rows are what the next chunk reuses most), and then
        // every row of this call is a miss
        if c.at.len() + miss.len() > c.most {
            c.at.clear();
            c.bytes.clear();
            miss = rows.to_vec();
            miss.sort_unstable();
            miss.dedup();
        }
        if !miss.is_empty() {
            let mut got = vec![0u8; miss.len() * rb];
            let failed = Mutex::new(None);
            let gp = crate::pool::SyncPtr::new(&mut got);
            // dynamic: a read's time depends on the drive, not on its index
            crate::pool::par_for_dynamic(miss.len(), &|i| {
                // SAFETY: item i owns its row's slice of got
                let out = unsafe { gp.slice(i * rb, rb) };
                if let Err(e) = self.read_row(miss[i], out) {
                    *failed.lock().expect("row read error") = Some(e);
                }
            });
            if let Some(e) = failed.into_inner().expect("row read error") {
                return Err(e);
            }
            for (i, &r) in miss.iter().enumerate() {
                let at = c.bytes.len();
                c.bytes.extend_from_slice(&got[i * rb..(i + 1) * rb]);
                c.at.insert(r, at);
            }
        }
        let mut out = Vec::with_capacity(rows.len() * rb);
        for r in rows {
            let at = c.at[r];
            out.extend_from_slice(&c.bytes[at..at + rb]);
        }
        Ok(out)
    }
}
