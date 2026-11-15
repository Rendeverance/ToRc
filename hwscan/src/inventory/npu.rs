//! npu.rs - NPUs, best effort: PnP device name / class / PCI id matching. A disabled or driverless NPU is invisible
//! here; a positive match says nothing about speed.

use crate::sys;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Npu {
    pub name: String,
    pub class: Option<String>,
    pub device_id: Option<String>,
    pub status: Option<String>,
    /// Which rule matched.
    pub matched: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NpuInfo {
    pub method: &'static str,
    pub npus: Vec<Npu>,
    pub error: Option<String>,
}

/// Whole-token, case-sensitive: avoids "Input" matching "NPU".
const NAME_TOKENS: &[&str] = &["NPU", "IPU", "XDNA", "QNN", "HTP"];
const NAME_PHRASES: &[&str] = &["hexagon", "neural processing", "ai boost", "ryzen ai"];
/// PCI ids of known NPUs (AMD Phoenix/Hawk Point IPU, AMD Strix XDNA2, Intel Meteor/Lunar Lake).
const PCI_IDS: &[&str] = &["VEN_1022&DEV_1502", "VEN_1022&DEV_17F0", "VEN_8086&DEV_7D1D", "VEN_8086&DEV_643E"];

fn match_rule(name: &str, class: Option<&str>, id: Option<&str>) -> Option<String> {
    if class.is_some_and(|c| c.eq_ignore_ascii_case("ComputeAccelerator")) {
        return Some("PnP class ComputeAccelerator".into());
    }
    if let Some(t) = name.split(|c: char| !c.is_ascii_alphanumeric()).find(|t| NAME_TOKENS.contains(t)) {
        return Some(format!("name token {t}"));
    }
    let lower = name.to_ascii_lowercase();
    if let Some(p) = NAME_PHRASES.iter().find(|p| lower.contains(*p)) {
        return Some(format!("name phrase \"{p}\""));
    }
    let up = id.unwrap_or("").to_ascii_uppercase();
    PCI_IDS.iter().find(|p| up.contains(*p)).map(|p| format!("PCI id {p}"))
}

pub fn collect() -> NpuInfo {
    let method = "best effort: PnP name/class/PCI-id match";
    match sys::pnp_devices() {
        Ok(devs) => NpuInfo {
            method,
            npus: devs
                .into_iter()
                .filter_map(|d| {
                    match_rule(&d.name, d.class.as_deref(), d.device_id.as_deref()).map(|matched| Npu {
                        name: d.name,
                        class: d.class,
                        device_id: d.device_id,
                        status: d.status,
                        matched,
                    })
                })
                .collect(),
            error: None,
        },
        Err(e) => NpuInfo { method, npus: Vec::new(), error: Some(e) },
    }
}

#[cfg(test)]
mod tests {
    use super::match_rule;

    #[test]
    fn rules() {
        assert!(match_rule("USB Input Device", Some("HIDClass"), None).is_none());
        assert!(match_rule("Qualcomm(R) Hexagon(TM) NPU", None, None).is_some());
        assert!(match_rule("AMD IPU Device", None, None).is_some());
        assert!(match_rule("Intel(R) AI Boost", None, None).is_some());
        assert!(match_rule("Computer", Some("Computer"), None).is_none());
        assert!(match_rule("x", None, Some(r"PCI\VEN_1022&DEV_1502&SUBSYS")).is_some());
    }
}
