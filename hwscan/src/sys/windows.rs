//! windows.rs - Windows implementation of the platform layer (x86_64 and aarch64).

use super::{LogicalCpu, RawCache, RawCore, RawDrive, RawModule, RawPnp, RawPowerStatus};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

use windows::Win32::Foundation::{ERROR_SUCCESS, HANDLE};
use windows::Win32::Storage::FileSystem::{
    FILE_FLAG_NO_BUFFERING, FILE_FLAG_WRITE_THROUGH, GetDiskFreeSpaceExW, GetDriveTypeW, GetLogicalDrives,
    GetVolumeInformationW, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
};
use windows::Win32::System::IO::DeviceIoControl;
use windows::Win32::System::Ioctl::{
    DISK_EXTENT, IOCTL_STORAGE_QUERY_PROPERTY, PropertyStandardQuery, STORAGE_DEVICE_DESCRIPTOR,
    STORAGE_PROPERTY_QUERY, StorageDeviceProperty,
};
use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
use windows::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD, RRF_RT_REG_SZ, RegGetValueW};
use windows::Win32::System::SystemInformation::{
    CacheData, CacheInstruction, CacheTrace, CacheUnified, GROUP_AFFINITY, GetLogicalProcessorInformationEx,
    GlobalMemoryStatusEx, LOGICAL_PROCESSOR_RELATIONSHIP, MEMORYSTATUSEX, RelationCache, RelationProcessorCore,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};
use windows::Win32::System::Threading::{GetCurrentThread, SetThreadGroupAffinity};
use windows::core::{HSTRING, PCWSTR};

const DRIVE_REMOVABLE: u32 = 2;
const DRIVE_FIXED: u32 = 3;

// Registry

pub fn reg_dword_hklm(subkey: &str, value: &str) -> Option<u32> {
    let (k, v) = (HSTRING::from(subkey), HSTRING::from(value));
    let mut data: u32 = 0;
    let mut size = std::mem::size_of::<u32>() as u32;
    // SAFETY: data/size point at a valid u32 and its size.
    let rc = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            &k,
            &v,
            RRF_RT_REG_DWORD,
            None,
            Some(&mut data as *mut u32 as *mut _),
            Some(&mut size),
        )
    };
    (rc == ERROR_SUCCESS).then_some(data)
}

pub fn reg_string_hklm(subkey: &str, value: &str) -> Option<String> {
    let (k, v) = (HSTRING::from(subkey), HSTRING::from(value));
    let mut buf = vec![0u16; 512];
    let mut size = (buf.len() * 2) as u32;
    // SAFETY: buf is valid for `size` bytes.
    let rc = unsafe {
        RegGetValueW(HKEY_LOCAL_MACHINE, &k, &v, RRF_RT_REG_SZ, None, Some(buf.as_mut_ptr() as *mut _), Some(&mut size))
    };
    if rc != ERROR_SUCCESS {
        return None;
    }
    let n = (size as usize / 2).min(buf.len());
    let s = String::from_utf16_lossy(&buf[..n]);
    Some(s.trim_end_matches('\0').trim().to_string())
}

// CPU

const CPU0_KEY: &str = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0";

pub fn cpu_brand() -> Option<String> {
    reg_string_hklm(CPU0_KEY, "ProcessorNameString")
}

/// The firmware's nominal clock (the "~MHz" value: base clock, not boost).
pub fn cpu_nominal_mhz() -> Option<u32> {
    reg_dword_hklm(CPU0_KEY, "~MHz")
}

/// Walk GetLogicalProcessorInformationEx for one relationship, calling `f` on each record.
fn for_each_lpi(rel: LOGICAL_PROCESSOR_RELATIONSHIP, mut f: impl FnMut(&SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)) {
    let mut len: u32 = 0;
    // SAFETY: size query with a null buffer; failure (insufficient buffer) is expected.
    let _ = unsafe { GetLogicalProcessorInformationEx(rel, None, &mut len) };
    if len == 0 {
        return;
    }
    // u64 backing keeps the records 8-byte aligned.
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    let ptr = buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
    // SAFETY: buf holds at least `len` bytes.
    if unsafe { GetLogicalProcessorInformationEx(rel, Some(ptr), &mut len) }.is_err() {
        return;
    }
    let base = buf.as_ptr() as *const u8;
    let mut off = 0usize;
    while off + 8 <= len as usize {
        // SAFETY: the OS packed variable-size records into the buffer; each starts at `off` and carries its own Size.
        let rec = unsafe { &*(base.add(off) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) };
        if rec.Size == 0 {
            break;
        }
        f(rec);
        off += rec.Size as usize;
    }
}

fn mask_bits(group: u16, mask: usize) -> impl Iterator<Item = LogicalCpu> {
    (0..usize::BITS).filter(move |b| mask >> b & 1 == 1).map(move |bit| LogicalCpu { group, bit })
}

pub fn cpu_cores() -> Vec<RawCore> {
    let mut cores = Vec::new();
    for_each_lpi(RelationProcessorCore, |rec| {
        // SAFETY: RelationProcessorCore records hold the Processor variant.
        let p = unsafe { &rec.Anonymous.Processor };
        let groups = p.GroupCount as usize;
        let first = p.GroupMask.as_ptr();
        let mut logical = Vec::new();
        for g in 0..groups {
            // SAFETY: GroupMask is a variable-length array of GroupCount entries inside this record.
            let ga: &GROUP_AFFINITY = unsafe { &*first.add(g) };
            logical.extend(mask_bits(ga.Group, ga.Mask));
        }
        cores.push(RawCore { efficiency_class: p.EfficiencyClass, logical });
    });
    cores
}

pub fn cpu_caches() -> Vec<RawCache> {
    let mut out = Vec::new();
    for_each_lpi(RelationCache, |rec| {
        // SAFETY: RelationCache records hold the Cache variant.
        let c = unsafe { &rec.Anonymous.Cache };
        let kind = match c.Type {
            t if t == CacheData => "data",
            t if t == CacheInstruction => "instruction",
            t if t == CacheUnified => "unified",
            t if t == CacheTrace => "trace",
            _ => "unknown",
        };
        out.push(RawCache { level: c.Level, kind, size_bytes: c.CacheSize as u64, line_bytes: c.LineSize as u32 });
    });
    out
}

/// Pin the calling thread to one logical CPU. Returns false if the OS refused.
pub fn pin_current_thread(cpu: LogicalCpu) -> bool {
    let ga = GROUP_AFFINITY { Mask: 1usize << cpu.bit, Group: cpu.group, Reserved: [0; 3] };
    // SAFETY: GetCurrentThread is a pseudo-handle; ga is a valid struct.
    unsafe { SetThreadGroupAffinity(GetCurrentThread(), &ga, None).as_bool() }
}

// Memory

/// Physical memory visible to the OS (bytes).
pub fn mem_total_visible() -> Option<u64> {
    let mut ms = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    // SAFETY: ms is a valid MEMORYSTATUSEX with dwLength set.
    unsafe { GlobalMemoryStatusEx(&mut ms) }.ok()?;
    Some(ms.ullTotalPhys)
}

fn v_u64(v: Option<&wmi::Variant>) -> Option<u64> {
    use wmi::Variant as V;
    match v? {
        V::String(s) => s.trim().parse().ok(),
        V::UI1(x) => Some(*x as u64),
        V::UI2(x) => Some(*x as u64),
        V::UI4(x) => Some(*x as u64),
        V::UI8(x) => Some(*x),
        V::I1(x) => u64::try_from(*x).ok(),
        V::I2(x) => u64::try_from(*x).ok(),
        V::I4(x) => u64::try_from(*x).ok(),
        V::I8(x) => u64::try_from(*x).ok(),
        _ => None,
    }
}

fn v_str(v: Option<&wmi::Variant>) -> Option<String> {
    match v? {
        wmi::Variant::String(s) => {
            let t = s.trim();
            (!t.is_empty()).then(|| t.to_string())
        }
        _ => None,
    }
}

type Row = HashMap<String, wmi::Variant>;

fn wmi_query(namespace: &str, query: &str) -> Result<Vec<Row>, String> {
    let con = wmi::WMIConnection::with_namespace_path(namespace).map_err(|e| format!("WMI connect: {e}"))?;
    con.raw_query::<Row>(query).map_err(|e| format!("WMI query: {e}"))
}

pub fn memory_modules() -> Result<Vec<RawModule>, String> {
    let rows = wmi_query(
        r"ROOT\CIMV2",
        "SELECT Capacity, Speed, ConfiguredClockSpeed, BankLabel, DeviceLocator, SMBIOSMemoryType, \
         PartNumber, Manufacturer FROM Win32_PhysicalMemory",
    )?;
    Ok(rows
        .iter()
        .map(|r| RawModule {
            capacity_bytes: v_u64(r.get("Capacity")),
            speed_mts: v_u64(r.get("Speed")).map(|x| x as u32),
            configured_speed_mts: v_u64(r.get("ConfiguredClockSpeed")).map(|x| x as u32),
            bank_label: v_str(r.get("BankLabel")),
            device_locator: v_str(r.get("DeviceLocator")),
            smbios_type: v_u64(r.get("SMBIOSMemoryType")).map(|x| x as u32),
            part_number: v_str(r.get("PartNumber")),
            manufacturer: v_str(r.get("Manufacturer")),
        })
        .collect())
}

pub fn pnp_devices() -> Result<Vec<RawPnp>, String> {
    let rows = wmi_query(r"ROOT\CIMV2", "SELECT Name, PNPClass, DeviceID, Status FROM Win32_PnPEntity")?;
    Ok(rows
        .iter()
        .map(|r| RawPnp {
            name: v_str(r.get("Name")).unwrap_or_default(),
            class: v_str(r.get("PNPClass")),
            device_id: v_str(r.get("DeviceID")),
            status: v_str(r.get("Status")),
        })
        .collect())
}

// Drives

fn c_str_at(buf: &[u8], off: u32) -> Option<String> {
    let off = off as usize;
    if off == 0 || off >= buf.len() {
        return None;
    }
    let end = buf[off..].iter().position(|&b| b == 0).map_or(buf.len(), |p| off + p);
    let s = String::from_utf8_lossy(&buf[off..end]).trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Query bus type / ids and the backing disk numbers of a volume (e.g. "D:").
fn volume_device_info(letter: char, d: &mut RawDrive) {
    // access 0 = a query-only open, which works without elevation
    let Ok(f) = OpenOptions::new().access_mode(0).share_mode(3).open(format!(r"\\.\{letter}:")) else {
        return;
    };
    let h = HANDLE(f.as_raw_handle());

    let query = STORAGE_PROPERTY_QUERY {
        PropertyId: StorageDeviceProperty,
        QueryType: PropertyStandardQuery,
        AdditionalParameters: [0],
    };
    let mut out = vec![0u64; 128]; // 1 KiB, 8-byte aligned
    let mut got: u32 = 0;
    // SAFETY: in/out buffers are valid for the sizes passed.
    let ok = unsafe {
        DeviceIoControl(
            h,
            IOCTL_STORAGE_QUERY_PROPERTY,
            Some(&query as *const _ as *const _),
            std::mem::size_of::<STORAGE_PROPERTY_QUERY>() as u32,
            Some(out.as_mut_ptr() as *mut _),
            (out.len() * 8) as u32,
            Some(&mut got),
            None,
        )
    };
    if ok.is_ok() && got as usize >= std::mem::size_of::<STORAGE_DEVICE_DESCRIPTOR>() {
        // SAFETY: the driver wrote a STORAGE_DEVICE_DESCRIPTOR at the start of `out`.
        let desc = unsafe { &*(out.as_ptr() as *const STORAGE_DEVICE_DESCRIPTOR) };
        let bytes: Vec<u8> = out.iter().flat_map(|w| w.to_le_bytes()).take(got as usize).collect();
        d.bus_type = Some(desc.BusType.0 as u32);
        d.vendor = c_str_at(&bytes, desc.VendorIdOffset);
        d.product = c_str_at(&bytes, desc.ProductIdOffset);
    }

    // VOLUME_DISK_EXTENTS: u32 count (+pad to 8) then DISK_EXTENT[count] (24 bytes each).
    let mut ext = vec![0u64; 1 + 3 * 8];
    // SAFETY: out buffer valid for its size; no input buffer.
    let ok = unsafe {
        DeviceIoControl(
            h,
            IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
            None,
            0,
            Some(ext.as_mut_ptr() as *mut _),
            (ext.len() * 8) as u32,
            Some(&mut got),
            None,
        )
    };
    if ok.is_ok() {
        let n = (ext[0] as u32 as usize).min(8);
        // SAFETY: extents start at byte 8 and `n` of them fit in `ext`.
        let first = unsafe { (ext.as_ptr() as *const u8).add(8) as *const DISK_EXTENT };
        for i in 0..n {
            // SAFETY: bounded by n <= 8 entries of the buffer.
            d.disk_numbers.push(unsafe { (*first.add(i)).DiskNumber });
        }
    }
}

pub fn drives() -> Vec<RawDrive> {
    // SAFETY: no arguments.
    let mask = unsafe { GetLogicalDrives() };
    let mut out = Vec::new();
    for i in 0..26u32 {
        if mask >> i & 1 == 0 {
            continue;
        }
        let letter = (b'A' + i as u8) as char;
        let root = format!("{letter}:\\");
        let hroot = HSTRING::from(root.as_str());
        // SAFETY: valid null-terminated path.
        let dtype = unsafe { GetDriveTypeW(&hroot) };
        if dtype != DRIVE_FIXED && dtype != DRIVE_REMOVABLE {
            continue;
        }
        let mut d = RawDrive { root: root.clone(), removable: dtype == DRIVE_REMOVABLE, ..Default::default() };
        let (mut avail, mut total) = (0u64, 0u64);
        // SAFETY: out pointers are valid u64s.
        if unsafe { GetDiskFreeSpaceExW(&hroot, Some(&mut avail), Some(&mut total), None) }.is_ok() {
            d.free_bytes = Some(avail);
            d.total_bytes = Some(total);
        }
        let mut fs = [0u16; 64];
        // SAFETY: fs buffer is valid; other outputs not requested.
        if unsafe { GetVolumeInformationW(PCWSTR(hroot.as_ptr()), None, None, None, None, Some(&mut fs)) }.is_ok() {
            let n = fs.iter().position(|&c| c == 0).unwrap_or(fs.len());
            d.filesystem = Some(String::from_utf16_lossy(&fs[..n]));
        }
        volume_device_info(letter, &mut d);
        out.push(d);
    }
    out
}

/// Open a file for unbuffered (cache-bypassing) I/O. Reads/writes must use sector-aligned offsets, lengths and buffers.
pub fn open_unbuffered(path: &Path, write: bool) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    if write {
        o.write(true).create_new(true).custom_flags(FILE_FLAG_NO_BUFFERING.0 | FILE_FLAG_WRITE_THROUGH.0);
    } else {
        o.read(true).custom_flags(FILE_FLAG_NO_BUFFERING.0);
    }
    o.open(path)
}

/// Positional read (does not move a shared file pointer).
pub fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, off)
}

// Power

pub fn power_status() -> Option<RawPowerStatus> {
    let mut s = SYSTEM_POWER_STATUS::default();
    // SAFETY: s is a valid out struct.
    unsafe { GetSystemPowerStatus(&mut s) }.ok()?;
    Some(RawPowerStatus {
        ac_line: s.ACLineStatus,
        battery_flag: s.BatteryFlag,
        battery_percent: s.BatteryLifePercent,
        system_status_flag: s.SystemStatusFlag,
    })
}
