//! other.rs - the platform layer for systems other than Windows. On a Mac the CPU's name and the memory size come from
//! sysctl; the rest, and everything on other systems, answers an honest "unknown" rather than a guess.

use super::{LogicalCpu, RawCache, RawCore, RawDrive, RawModule, RawPnp, RawPowerStatus};
use std::fs::{File, OpenOptions};
use std::path::Path;

pub fn reg_dword_hklm(_subkey: &str, _value: &str) -> Option<u32> {
    None
}

pub fn reg_string_hklm(_subkey: &str, _value: &str) -> Option<String> {
    None
}

/// A value macOS keeps under a sysctl name, as its raw bytes.
#[cfg(target_os = "macos")]
fn sysctl(name: &str) -> Option<Vec<u8>> {
    use std::ffi::{CString, c_char, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }
    let name = CString::new(name).ok()?;
    let mut len = 0usize;
    // SAFETY: a null buffer asks only for the value's length, written to len
    if unsafe { sysctlbyname(name.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) } != 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    // SAFETY: buf holds len bytes, and sysctlbyname writes no more than len and says how many it wrote
    if unsafe { sysctlbyname(name.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) } != 0 {
        return None;
    }
    buf.truncate(len);
    Some(buf)
}

#[cfg(target_os = "macos")]
pub fn cpu_brand() -> Option<String> {
    let raw = sysctl("machdep.cpu.brand_string")?;
    let brand = String::from_utf8_lossy(&raw).trim_end_matches('\0').trim().to_string();
    (!brand.is_empty()).then_some(brand)
}

#[cfg(not(target_os = "macos"))]
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

#[cfg(target_os = "macos")]
pub fn mem_total_visible() -> Option<u64> {
    let raw: [u8; 8] = sysctl("hw.memsize")?.try_into().ok()?;
    Some(u64::from_ne_bytes(raw))
}

#[cfg(not(target_os = "macos"))]
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
