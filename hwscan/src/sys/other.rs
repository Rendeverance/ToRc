//! other.rs - fallback platform layer for non-Windows targets. ToRc runs on Windows (x86_64 and aarch64); this keeps
//! the crate compiling elsewhere with honest "unknown" answers rather than guesses.

use super::{LogicalCpu, RawCache, RawCore, RawDrive, RawModule, RawPnp, RawPowerStatus};
use std::fs::{File, OpenOptions};
use std::path::Path;

pub fn reg_dword_hklm(_subkey: &str, _value: &str) -> Option<u32> {
    None
}

pub fn reg_string_hklm(_subkey: &str, _value: &str) -> Option<String> {
    None
}

pub fn cpu_brand() -> Option<String> {
    None
}

pub fn cpu_nominal_mhz() -> Option<u32> {
    None
}

/// Without topology information every logical CPU is reported as its own core.
pub fn cpu_cores() -> Vec<RawCore> {
    let n = std::thread::available_parallelism().map_or(1, |n| n.get());
    (0..n as u32).map(|bit| RawCore { efficiency_class: 0, logical: vec![LogicalCpu { group: 0, bit }] }).collect()
}

pub fn cpu_caches() -> Vec<RawCache> {
    Vec::new()
}

pub fn pin_current_thread(_cpu: LogicalCpu) -> bool {
    false
}

pub fn mem_total_visible() -> Option<u64> {
    None
}

pub fn memory_modules() -> Result<Vec<RawModule>, String> {
    Err("not implemented on this OS".into())
}

pub fn pnp_devices() -> Result<Vec<RawPnp>, String> {
    Err("not implemented on this OS".into())
}

pub fn drives() -> Vec<RawDrive> {
    Vec::new()
}

/// No cache bypass on this platform yet: results will include OS caching.
pub fn open_unbuffered(path: &Path, write: bool) -> std::io::Result<File> {
    if write { OpenOptions::new().write(true).create_new(true).open(path) } else { File::open(path) }
}

pub fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_at(buf, off)
    }
    #[cfg(not(unix))]
    {
        let _ = (f, buf, off);
        Err(std::io::Error::other("positional read not supported on this OS"))
    }
}

pub fn power_status() -> Option<RawPowerStatus> {
    None
}
