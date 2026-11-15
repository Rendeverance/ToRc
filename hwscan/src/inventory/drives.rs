//! drives.rs - drives: bus type (NVMe/USB/SATA...), free space, and any MODELS directory on them.

use crate::sys;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct Drive {
    pub root: String,
    pub bus: String,
    pub vendor: Option<String>,
    pub product: Option<String>,
    pub disk_numbers: Vec<u32>,
    pub filesystem: Option<String>,
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub removable: bool,
    /// `<root>MODELS` if it exists.
    pub models_dir: Option<String>,
    /// Largest file >= 4 GiB under the models dir (the drive test reads it).
    pub largest_model_file: Option<String>,
    pub largest_model_bytes: Option<u64>,
}

impl Drive {
    pub fn is_usb(&self) -> bool {
        self.bus == "USB"
    }
}

pub const LARGE_FILE: u64 = 4 << 30;

pub fn bus_name(t: Option<u32>) -> String {
    match t {
        Some(1) => "SCSI",
        Some(2) => "ATAPI",
        Some(3) => "ATA",
        Some(4) => "1394",
        Some(5) => "SSA",
        Some(6) => "FibreChannel",
        Some(7) => "USB",
        Some(8) => "RAID",
        Some(9) => "iSCSI",
        Some(10) => "SAS",
        Some(11) => "SATA",
        Some(12) => "SD",
        Some(13) => "MMC",
        Some(14) => "Virtual",
        Some(15) => "FileBackedVirtual",
        Some(16) => "StorageSpaces",
        Some(17) => "NVMe",
        Some(18) => "SCM",
        Some(19) => "UFS",
        _ => "unknown",
    }
    .to_string()
}

/// Bounded walk: depth <= 6, at most 20 000 entries. Returns the largest file >= LARGE_FILE.
fn largest_file(dir: &Path) -> Option<(PathBuf, u64)> {
    let mut best: Option<(PathBuf, u64)> = None;
    let mut stack = vec![(dir.to_path_buf(), 0u32)];
    let mut seen = 0usize;
    while let Some((d, depth)) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            seen += 1;
            if seen > 20_000 {
                return best;
            }
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() && depth < 6 {
                stack.push((e.path(), depth + 1));
            } else if ft.is_file() {
                let len = e.metadata().map(|m| m.len()).unwrap_or(0);
                if len >= LARGE_FILE && best.as_ref().is_none_or(|b| len > b.1) {
                    best = Some((e.path(), len));
                }
            }
        }
    }
    best
}

pub fn collect() -> Vec<Drive> {
    sys::drives()
        .into_iter()
        .map(|d| {
            let models = Path::new(&d.root).join("MODELS");
            let models_dir = models.is_dir().then(|| models.display().to_string());
            let big = models_dir.as_ref().and_then(|_| largest_file(&models));
            Drive {
                bus: bus_name(d.bus_type),
                root: d.root,
                vendor: d.vendor,
                product: d.product,
                disk_numbers: d.disk_numbers,
                filesystem: d.filesystem,
                total_bytes: d.total_bytes,
                free_bytes: d.free_bytes,
                removable: d.removable,
                models_dir,
                largest_model_file: big.as_ref().map(|b| b.0.display().to_string()),
                largest_model_bytes: big.map(|b| b.1),
            }
        })
        .collect()
}
