//! power.rs - section 2: power state, the things that silently cap the machine.

use crate::sys;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default)]
pub struct PowerInfo {
    /// Some(true) = on AC.
    pub on_ac: Option<bool>,
    pub battery_percent: Option<u8>,
    /// GetSystemPowerStatus SystemStatusFlag bit 0 (battery/Energy Saver as apps see it).
    pub saver_flag_from_power_status: Option<bool>,
    /// Raw HKLM\SYSTEM\CurrentControlSet\Control\Power EnergySaverState.
    pub energy_saver_state_raw: Option<u32>,
    /// Interpretation of the raw value (1 = on, 2 = off).
    pub energy_saver: String,
    pub active_scheme_guid: Option<String>,
    pub active_scheme_name: Option<String>,
    pub overlay_ac_guid: Option<String>,
    pub overlay_ac_name: Option<String>,
    pub overlay_dc_guid: Option<String>,
    pub overlay_dc_name: Option<String>,
    /// Processor boost mode of the active scheme (AC), as powercfg reports it; overlays may override.
    pub boost_mode_ac: Option<String>,
    /// Energy-performance preference % of the active scheme (AC); 0 = max performance.
    pub epp_ac: Option<u32>,
}

impl PowerInfo {
    pub fn energy_saver_on(&self) -> bool {
        self.energy_saver_state_raw == Some(1)
    }
}

const POWER_KEY: &str = r"SYSTEM\CurrentControlSet\Control\Power";
const SCHEMES_KEY: &str = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes";

fn overlay_name(guid: &str) -> String {
    match guid.to_ascii_lowercase().as_str() {
        "ded574b5-45a0-4f42-8737-46345c09c238" => "Best performance",
        "3af9b8d9-7c97-431d-ad78-34a8bfea439f" => "Better performance",
        "961cc777-2547-4f9d-8174-7d86181b8a7a" => "Best power efficiency",
        "00000000-0000-0000-0000-000000000000" => "Balanced (no overlay)",
        _ => "unknown overlay",
    }
    .to_string()
}

fn scheme_name(guid: &str) -> Option<String> {
    let n = match guid.to_ascii_lowercase().as_str() {
        "381b4222-f694-41f0-9685-ff5bb260df2e" => "Balanced",
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c" => "High performance",
        "a1841308-3541-4fab-bc81-f71556f20b4a" => "Power saver",
        "e9a42b02-d5df-448d-aa00-03f14749eb61" => "Ultimate Performance",
        _ => return None,
    };
    Some(n.to_string())
}

#[cfg_attr(not(windows), allow(dead_code))]
fn boost_name(v: u32) -> String {
    match v {
        0 => "Disabled",
        1 => "Enabled",
        2 => "Aggressive",
        3 => "Efficient Enabled",
        4 => "Efficient Aggressive",
        5 => "Aggressive At Guaranteed",
        6 => "Efficient Aggressive At Guaranteed",
        _ => return format!("unknown ({v})"),
    }
    .to_string()
}

#[cfg(windows)]
fn powercfg(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("powercfg").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// "Power Scheme GUID: 381b...  (Balanced)" -> (guid, name)
#[cfg(windows)]
fn parse_active_scheme(s: &str) -> Option<(String, Option<String>)> {
    let after = s.split_once(':')?.1.trim();
    let guid: String = after.chars().take_while(|c| c.is_ascii_hexdigit() || *c == '-').collect();
    let name = after.split_once('(').and_then(|(_, r)| r.rsplit_once(')')).map(|(n, _)| n.trim().to_string());
    (guid.len() == 36).then_some((guid, name))
}

/// The "Current AC Power Setting Index: 0x..." value from `powercfg /qh`. Falls back to the penultimate hex line (AC
/// precedes DC) on a localised Windows.
#[cfg(windows)]
fn parse_ac_index(s: &str) -> Option<u32> {
    let hex = |l: &str| l.rsplit_once("0x").and_then(|(_, h)| u32::from_str_radix(h.trim(), 16).ok());
    let lines: Vec<&str> = s.lines().filter(|l| l.contains("0x")).collect();
    if let Some(l) = lines.iter().find(|l| l.contains("Current AC")) {
        return hex(l);
    }
    lines.len().checked_sub(2).and_then(|i| hex(lines[i]))
}

pub fn collect() -> PowerInfo {
    let mut p = PowerInfo::default();
    if let Some(s) = sys::power_status() {
        p.on_ac = match s.ac_line {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        };
        // 128 = no battery, 255 = unknown.
        if s.battery_flag != 128 && s.battery_percent <= 100 {
            p.battery_percent = Some(s.battery_percent);
        }
        p.saver_flag_from_power_status = Some(s.system_status_flag & 1 == 1);
    }
    p.energy_saver_state_raw = sys::reg_dword_hklm(POWER_KEY, "EnergySaverState");
    p.energy_saver = match p.energy_saver_state_raw {
        Some(1) => "ON (caps the CPU)".into(),
        Some(2) => "off".into(),
        Some(v) => format!("unknown value {v}"),
        None => "not present".into(),
    };
    p.overlay_ac_guid = sys::reg_string_hklm(SCHEMES_KEY, "ActiveOverlayAcPowerScheme");
    p.overlay_dc_guid = sys::reg_string_hklm(SCHEMES_KEY, "ActiveOverlayDcPowerScheme");
    p.overlay_ac_name = p.overlay_ac_guid.as_deref().map(overlay_name);
    p.overlay_dc_name = p.overlay_dc_guid.as_deref().map(overlay_name);
    p.active_scheme_guid = sys::reg_string_hklm(SCHEMES_KEY, "ActivePowerScheme");

    #[cfg(windows)]
    {
        if let Some((g, n)) = powercfg(&["/getactivescheme"]).as_deref().and_then(parse_active_scheme) {
            p.active_scheme_guid = Some(g);
            p.active_scheme_name = n;
        }
        p.boost_mode_ac = powercfg(&["/qh", "SCHEME_CURRENT", "SUB_PROCESSOR", "PERFBOOSTMODE"])
            .as_deref()
            .and_then(parse_ac_index)
            .map(boost_name);
        p.epp_ac = powercfg(&["/qh", "SCHEME_CURRENT", "SUB_PROCESSOR", "PERFEPP"]).as_deref().and_then(parse_ac_index);
    }
    if p.active_scheme_name.is_none() {
        p.active_scheme_name = p.active_scheme_guid.as_deref().and_then(scheme_name);
    }
    p
}
