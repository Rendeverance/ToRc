//! memory.rs - RAM inventory: total, and (Windows) SMBIOS modules, speed and channels.

use crate::sys;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Module {
    pub capacity_bytes: Option<u64>,
    pub speed_mts: Option<u32>,
    pub configured_speed_mts: Option<u32>,
    pub bank_label: Option<String>,
    pub device_locator: Option<String>,
    pub memory_type: String,
    pub part_number: Option<String>,
    pub manufacturer: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryInfo {
    /// What the OS can use.
    pub visible_bytes: Option<u64>,
    /// Sum of SMBIOS module capacities.
    pub installed_bytes: Option<u64>,
    pub modules: Vec<Module>,
    /// Distinct channel letters parsed from SMBIOS bank labels ("P0 CHANNEL A").
    pub channels: Vec<String>,
    /// Configured speed (MT/s), lowest across modules.
    pub speed_mts: Option<u32>,
    /// speed x 8 bytes x channels: only meaningful if each label is a 64-bit channel.
    pub theoretical_gbs_if_64bit_channels: Option<f64>,
    pub error: Option<String>,
}

fn smbios_type_name(t: Option<u32>) -> String {
    match t {
        Some(24) => "DDR3".into(),
        Some(26) => "DDR4".into(),
        Some(27) => "LPDDR".into(),
        Some(28) => "LPDDR2".into(),
        Some(29) => "LPDDR3".into(),
        Some(30) => "LPDDR4".into(),
        Some(34) => "DDR5".into(),
        Some(35) => "LPDDR5".into(),
        Some(0) | None => "unknown".into(),
        Some(x) => format!("SMBIOS type {x}"),
    }
}

/// "P0 CHANNEL A" -> "A"; "ChannelB-DIMM0" -> "B".
fn channel_of(label: &str) -> Option<String> {
    let up = label.to_ascii_uppercase();
    let i = up.find("CHANNEL")?;
    let rest = up[i + "CHANNEL".len()..].trim_start_matches([' ', '_', '-']);
    let tok: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
    (!tok.is_empty()).then_some(tok)
}

pub fn collect() -> MemoryInfo {
    let visible_bytes = sys::mem_total_visible();
    let (raw, error) = match sys::memory_modules() {
        Ok(v) => (v, None),
        Err(e) => (Vec::new(), Some(e)),
    };
    let modules: Vec<Module> = raw
        .iter()
        .map(|m| Module {
            capacity_bytes: m.capacity_bytes,
            speed_mts: m.speed_mts,
            configured_speed_mts: m.configured_speed_mts,
            bank_label: m.bank_label.clone(),
            device_locator: m.device_locator.clone(),
            memory_type: smbios_type_name(m.smbios_type),
            part_number: m.part_number.clone(),
            manufacturer: m.manufacturer.clone(),
        })
        .collect();
    let installed_bytes = (!modules.is_empty()).then(|| modules.iter().filter_map(|m| m.capacity_bytes).sum());
    let mut channels: Vec<String> = modules
        .iter()
        .filter_map(|m| {
            m.bank_label.as_deref().and_then(channel_of).or_else(|| m.device_locator.as_deref().and_then(channel_of))
        })
        .collect();
    channels.sort();
    channels.dedup();
    let speed_mts = modules.iter().filter_map(|m| m.configured_speed_mts.filter(|&s| s > 0).or(m.speed_mts)).min();
    let theoretical = match (speed_mts, channels.len()) {
        (Some(s), n) if n > 0 => Some(s as f64 * 8.0 * n as f64 / 1000.0),
        _ => None,
    };
    MemoryInfo {
        visible_bytes,
        installed_bytes,
        modules,
        channels,
        speed_mts,
        theoretical_gbs_if_64bit_channels: theoretical,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::channel_of;

    #[test]
    fn channel_labels() {
        assert_eq!(channel_of("P0 CHANNEL A").as_deref(), Some("A"));
        assert_eq!(channel_of("ChannelB-DIMM0").as_deref(), Some("B"));
        assert_eq!(channel_of("BANK 0"), None);
    }
}
