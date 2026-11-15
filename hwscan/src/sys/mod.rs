//! sys - platform layer: the only place that talks to the OS. Everything above this module is platform-neutral data and
//! arithmetic.

use std::alloc::{Layout, alloc_zeroed, dealloc};

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use self::windows::*;

#[cfg(not(windows))]
mod other;
#[cfg(not(windows))]
pub use self::other::*;

/// One logical CPU, addressed the way the OS wants it for pinning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalCpu {
    pub group: u16,
    /// Bit index inside the group (0..63).
    pub bit: u32,
}

impl LogicalCpu {
    /// A flat id: group * 64 + bit. Matches Task Manager numbering on single-group machines.
    pub fn flat_id(self) -> u32 {
        self.group as u32 * 64 + self.bit
    }
}

/// One physical core as the OS reports it.
#[derive(Debug, Clone)]
pub struct RawCore {
    /// Higher = faster core on hybrid parts (Windows EfficiencyClass). 0 on uniform parts.
    pub efficiency_class: u8,
    pub logical: Vec<LogicalCpu>,
}

#[derive(Debug, Clone)]
pub struct RawCache {
    pub level: u8,
    /// "data", "instruction", "unified", "trace"
    pub kind: &'static str,
    pub size_bytes: u64,
    pub line_bytes: u32,
}

#[derive(Debug, Clone, Default)]
pub struct RawModule {
    pub capacity_bytes: Option<u64>,
    pub speed_mts: Option<u32>,
    pub configured_speed_mts: Option<u32>,
    pub bank_label: Option<String>,
    pub device_locator: Option<String>,
    pub smbios_type: Option<u32>,
    pub part_number: Option<String>,
    pub manufacturer: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct RawDrive {
    pub root: String,
    pub filesystem: Option<String>,
    pub bus_type: Option<u32>,
    pub vendor: Option<String>,
    pub product: Option<String>,
    pub disk_numbers: Vec<u32>,
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub removable: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RawPnp {
    pub name: String,
    pub class: Option<String>,
    pub device_id: Option<String>,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RawPowerStatus {
    /// 0 = battery, 1 = AC, 255 = unknown.
    pub ac_line: u8,
    pub battery_flag: u8,
    pub battery_percent: u8,
    /// Bit 0 = battery saver / Energy Saver active (as reported to apps).
    pub system_status_flag: u8,
}

/// A heap buffer with a chosen power-of-two alignment (page alignment is what unbuffered file I/O and wide SIMD loads
/// want).
pub struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
}

// The buffer is plain bytes owned by this struct; sharing &AlignedBuf across threads only hands out shared slices.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize, align: usize) -> Option<Self> {
        let layout = Layout::from_size_align(len.max(1), align).ok()?;
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() { None } else { Some(Self { ptr, layout }) }
    }

    pub fn len(&self) -> usize {
        self.layout.size()
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid, zero-initialised and owned for layout.size() bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and &mut self guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in new() with this exact layout.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}
