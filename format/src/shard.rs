//! shard.rs - one machine's part of a model file, moved as a stream (the quick form of a sharded file).
//!
//! `pack` writes only the byte ranges a pipeline machine reads (the header and directory, the resident tensors of its
//! layers and the model-wide ones, and its layers' expert packs). `unpack` writes them into a SPARSE file of the
//! original's full length, so the rest takes no disk space and reads as zeros (it is never read: it is the other
//! layers' tensors and experts). The result opens as the original does (header and directory), and runs the machine's
//! layers with the same bits. This way a 104 GB model fits a machine with 56 GB free, since that machine only reads
//! about 45 GB of it. The stream is "TORCPAK1", file_bytes u64, n u64, then n x (offset u64, len u64), then the ranges'
//! bytes in order.

use crate::{Result, err};
use std::io::{Read, Seek, SeekFrom, Write};

const MAGIC: &[u8; 8] = b"TORCPAK1";

/// Whether a machine running layers a..b holds the resident tensor `name`: a layer's tensor (`blk.<l>.`) only on the
/// machine that runs its layer, the n-gram table only on the machine with the n-gram layer, and every other tensor on
/// every machine.
pub fn keeps(name: &str, a: u32, b: u32, ngram_layer: Option<u32>) -> bool {
    match name.strip_prefix("blk.").and_then(|s| s.split('.').next()).and_then(|s| s.parse::<u32>().ok()) {
        Some(l) => l >= a && l < b,
        None if name == "per_layer_token_embd.weight" => ngram_layer.is_some_and(|l| l >= a && l < b),
        None => true,
    }
}

/// The ranges of a ToRc file a machine running layers a..b reads: the header and directory, every resident tensor it
/// keeps (keeps above), and the packs of its layers' experts. Sorted and merged.
pub fn ranges(m: &crate::read::Model, a: u32, b: u32, ngram_layer: Option<u32>) -> Vec<(u64, u64)> {
    let d = m.header.directory;
    let mut r = vec![(0u64, crate::HEADER_BYTES), (d.offset, d.bytes)];
    for t in m.tensors.iter().filter(|t| keeps(&t.name, a, b, ngram_layer)) {
        r.push((m.resident.offset + t.offset, t.bytes));
    }
    for x in m.experts.iter().filter(|x| x.layer >= a && x.layer < b) {
        let (at, n) = x.span();
        r.push((m.expert_region.offset + at, n));
    }
    merged(r)
}

/// Sorted ranges, with overlapping or touching ones joined.
fn merged(mut r: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    r.sort();
    let mut m: Vec<(u64, u64)> = Vec::with_capacity(r.len());
    for (o, n) in r {
        match m.last_mut() {
            Some((po, pn)) if o <= *po + *pn => *pn = (*pn).max(o + n - *po),
            _ => m.push((o, n)),
        }
    }
    m
}

/// Writes the pack stream of `src`'s ranges into `out`, and returns the bytes of data written.
pub fn pack(src: &str, rs: &[(u64, u64)], out: &mut dyn Write) -> Result<u64> {
    let mut f = std::fs::File::open(src).map_err(|e| crate::Error(format!("cannot open {src}: {e}")))?;
    let fb = f.metadata()?.len();
    out.write_all(MAGIC)?;
    out.write_all(&fb.to_le_bytes())?;
    out.write_all(&(rs.len() as u64).to_le_bytes())?;
    for &(o, n) in rs {
        out.write_all(&o.to_le_bytes())?;
        out.write_all(&n.to_le_bytes())?;
    }
    let mut buf = vec![0u8; 8 << 20];
    let mut total = 0u64;
    for &(o, n) in rs {
        f.seek(SeekFrom::Start(o))?;
        let mut left = n;
        while left > 0 {
            let k = left.min(buf.len() as u64) as usize;
            f.read_exact(&mut buf[..k])?;
            out.write_all(&buf[..k])?;
            left -= k as u64;
        }
        total += n;
    }
    out.flush()?;
    Ok(total)
}

/// Writes a pack stream into a sparse file at `dst`, of the original's length, and returns (file bytes, data bytes).
pub fn unpack(input: &mut dyn Read, dst: &str) -> Result<(u64, u64)> {
    let mut h = [0u8; 24];
    input.read_exact(&mut h)?;
    if &h[..8] != MAGIC {
        return err("not a ToRc pack stream");
    }
    let fb = u64::from_le_bytes(h[8..16].try_into().unwrap());
    let n = u64::from_le_bytes(h[16..24].try_into().unwrap());
    // a pack holds one range per tensor at most, so a count past that is a damaged stream, not a reason to reserve
    // memory for it
    if n > 1 << 20 {
        return err("damaged pack stream: implausible range count");
    }
    let mut rs = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut e = [0u8; 16];
        input.read_exact(&mut e)?;
        rs.push((u64::from_le_bytes(e[..8].try_into().unwrap()), u64::from_le_bytes(e[8..].try_into().unwrap())));
    }
    // every range checked before anything is created, so a damaged stream leaves no file behind
    if rs.iter().any(|&(o, len)| o.checked_add(len).is_none_or(|end| end > fb)) {
        return err("pack range past the file's end");
    }
    // a new file only: pointed at a whole model by mistake, this would otherwise empty it
    let mut f = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(dst)
        .map_err(|e| crate::Error(format!("cannot create {dst}: {e} (unpack writes a new file only)")))?;
    let written = write_ranges(&mut f, input, fb, &rs);
    if written.is_err() {
        // a stream cut short leaves a file of the right size with holes in it: removed, so a second try can start
        drop(f);
        let _ = std::fs::remove_file(dst);
    }
    written
}

/// The ranges' bytes from the stream into `f`, a new sparse file of `fb` bytes.
fn write_ranges(f: &mut std::fs::File, input: &mut dyn Read, fb: u64, rs: &[(u64, u64)]) -> Result<(u64, u64)> {
    set_sparse(f)?;
    f.set_len(fb)?;
    let mut buf = vec![0u8; 8 << 20];
    let mut total = 0u64;
    for &(o, len) in rs {
        f.seek(SeekFrom::Start(o))?;
        let mut left = len;
        while left > 0 {
            let k = left.min(buf.len() as u64) as usize;
            input.read_exact(&mut buf[..k])?;
            f.write_all(&buf[..k])?;
            left -= k as u64;
        }
        total += len;
    }
    f.flush()?;
    Ok((fb, total))
}

/// Marks the file as sparse (on Windows with FSCTL_SET_SPARSE; elsewhere files are sparse wherever they were never
/// written).
fn set_sparse(f: &std::fs::File) -> Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        unsafe extern "system" {
            fn DeviceIoControl(
                h: *mut core::ffi::c_void,
                code: u32,
                inb: *const core::ffi::c_void,
                ins: u32,
                outb: *mut core::ffi::c_void,
                outs: u32,
                ret: *mut u32,
                ov: *mut core::ffi::c_void,
            ) -> i32;
        }
        const FSCTL_SET_SPARSE: u32 = 0x0009_00C4;
        let mut ret = 0u32;
        // SAFETY: a valid file handle, and no buffers
        let ok = unsafe {
            DeviceIoControl(
                f.as_raw_handle() as *mut _,
                FSCTL_SET_SPARSE,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0,
                &mut ret,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return err(format!("cannot make the file sparse: {}", std::io::Error::last_os_error()));
        }
    }
    #[cfg(not(windows))]
    let _ = f;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn ranges_merge() {
        // out of order, overlapping, touching and apart
        let v = vec![(20u64, 5u64), (0, 10), (40, 1), (5, 10), (25, 5)];
        assert_eq!(super::merged(v), vec![(0, 15), (20, 10), (40, 1)]);
    }

    #[test]
    fn damaged_streams_are_errors() {
        let head = |n: u64| [super::MAGIC.as_slice(), &100u64.to_le_bytes(), &n.to_le_bytes()].concat();
        // a range count no pack has: refused before anything is reserved or created
        let r = super::unpack(&mut head(u64::MAX).as_slice(), "never-created.torc");
        assert!(r.is_err() && !std::path::Path::new("never-created.torc").exists());
        // a range whose end overflows: refused, not wrapped round to look in bounds
        let dst = std::env::temp_dir().join(format!("torc-shard-test-{}.torc", std::process::id()));
        let s = [head(1), u64::MAX.to_le_bytes().to_vec(), 2u64.to_le_bytes().to_vec()].concat();
        let r = super::unpack(&mut s.as_slice(), dst.to_str().unwrap());
        let _ = std::fs::remove_file(&dst);
        assert!(r.is_err());
    }
}
