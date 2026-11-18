//! profile.rs - the settings a server runs with, as one file to keep beside a benchmark, pass to someone else, and
//! read back in on another machine (AUTOTUNE.md, Sharing settings).
//!
//! A profile holds the values in use and what chose each (a switch, the tune file or the engine's default), the
//! TORC_* switches the server was started with, the model file it was measured on and the hardware. Reading one in
//! writes the values a switch or the tune file chose into this machine's tune file for the same model, under the
//! layout they were measured in, so the server uses them from its next start until torc-tune measures this machine's
//! own. A default is not read in: this machine has its own, which can differ. Switches are not read in here: the
//! desktop app takes those into its own settings.

use serde_json::{Map, Value, json};

/// The settings a profile may set, with the values each may take: the prompt block, guided splits, pinned threads,
/// the threads a reply uses (0 is all of them) and the NPU's hot experts a layer.
const NAMES: [(&str, u64, u64); 5] =
    [("block", 32, 16384), ("guided", 0, 1), ("pin", 0, 1), ("decode_threads", 0, 1024), ("npu_hot", 0, 64)];

/// The layouts a tune file keeps apart (tune::layout).
const LAYOUTS: [&str; 4] = ["one-cpu", "one-npu", "two-cpu", "two-npu"];

/// The values this server runs with, wherever each came from. The block is given for the models that read it from the
/// tune file (the Qwen family's: Gemma 4 and gpt-oss read TORC_BLOCK alone and keep no tune file).
pub fn in_use() -> Value {
    let mut m = Map::new();
    if crate::pipe::model_id() != 0 {
        m.insert("block".into(), json!(crate::tune::block()));
    }
    m.insert("guided".into(), json!(crate::moe::guided() as u8));
    m.insert("pin".into(), json!(crate::pool::pinned() as u8));
    m.insert("decode_threads".into(), json!(crate::pool::decode_width()));
    m.insert("threads".into(), json!(crate::pool::get().threads));
    if crate::npu::up() {
        m.insert("npu_hot".into(), json!(crate::npu::hot_asked()));
    }
    Value::Object(m)
}

/// What chose each value in_use gives: "switch", "tuned" (the tune file: torc-tune or a profile read in) or "default".
pub fn chosen() -> Value {
    // read: whether a tuned value is used here at all (Linux doesn't apply a tuned "pin": pool::apply_tuned_pin)
    let by = |given: bool, tuned: &str, read: bool| {
        if given {
            "switch"
        } else if read && crate::tune::get(tuned).is_some() {
            "tuned"
        } else {
            "default"
        }
    };
    let s = crate::switches::given;
    json!({
        // TORC_BLOCK=0 counts as not set (tune::block)
        "block": by(crate::switches::num::<usize>("TORC_BLOCK").is_some_and(|b| b > 0), "block", true),
        "guided": by(s("TORC_NPU_GUIDED"), "guided", true),
        "pin": by(s("TORC_PIN"), "pin", !cfg!(target_os = "linux")),
        "decode_threads": by(s("TORC_DECODE_THREADS"), "decode_threads", true),
        "npu_hot": by(s("TORC_NPU_HOT"), "npu_hot", true),
    })
}

/// The TORC_* switches this process was started with. A switch holding a path or a network address is given by name
/// only, so a profile never carries a folder name or an address.
pub fn switches() -> Value {
    let mut m = Map::new();
    for (k, v) in std::env::vars().filter(|(k, _)| k.starts_with("TORC_")) {
        let hidden = if ["TORC_HOST", "TORC_PIPE", "TORC_PIPE_BIND"].contains(&k.as_str()) {
            "<address>"
        } else if v.contains(['/', '\\', ':']) {
            "<path>"
        } else {
            v.as_str()
        };
        m.insert(k, json!(hidden));
    }
    Value::Object(m)
}

/// The hardware in a line: the CPU's name where the system gives it, its CPU count and the memory installed.
pub fn hardware() -> String {
    let name = std::env::var("PROCESSOR_IDENTIFIER").ok().or_else(|| {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()?
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
    });
    let cpus = std::thread::available_parallelism().map_or(0, |n| n.get());
    let gb = crate::weights::total_phys().unwrap_or(0) >> 30;
    let os = std::env::consts::OS;
    match name {
        Some(n) => format!("{n}, {cpus} CPUs, {gb} GB, {os}"),
        None => format!("{} {cpus} CPUs, {gb} GB, {os}", std::env::consts::ARCH),
    }
}

/// This server's profile: what the server's /torc/settings answers, and what a benchmark keeps with its results.
pub fn export(model: &str) -> Value {
    json!({
        "torc_settings": 1,
        "model": model,
        "model_key": format!("{:016x}", crate::pipe::model_id()),
        "arch": crate::tune::arch(),
        "hardware": hardware(),
        "layout": crate::tune::layout(),
        "in_use": in_use(),
        "chosen": chosen(),
        "switches": switches(),
    })
}

/// A profile's layout and the values to write, checked: the layout one a tune file keeps, each value in its range,
/// and the model the same file as this one when the profile names it (`key`, this model's; 0 when unknown). A value
/// the profile says was the engine's default is left out.
fn checked(p: &Value, key: u64) -> Result<(String, Vec<(&'static str, u64)>), String> {
    // a profile names the model file it was measured on by a key from the file's size and shape
    match &p["model_key"] {
        Value::Null => {}
        Value::String(k) if u64::from_str_radix(k, 16).ok() == Some(key) => {}
        Value::String(_) => {
            return Err("these settings were measured on another model file: load that model to read them in".into());
        }
        _ => return Err("the model key in these settings is not one ToRc writes".into()),
    }
    let lay = p["layout"].as_str().unwrap_or("");
    if !LAYOUTS.contains(&lay) {
        return Err(format!("no layout I know in these settings ({lay:?})"));
    }
    let mut out = Vec::new();
    for (name, lo, hi) in NAMES {
        let v = &p["in_use"][name];
        if v.is_null() || p["chosen"][name] == "default" {
            continue;
        }
        match v.as_u64() {
            Some(x) if (lo..=hi).contains(&x) => out.push((name, x)),
            _ => return Err(format!("{name}: {v} is not a value it can take ({lo} to {hi})")),
        }
    }
    if out.is_empty() {
        return Err("these settings are all the engine's defaults, so there is nothing to read in".into());
    }
    Ok((lay.to_string(), out))
}

/// Reads a profile in: its values written into this machine's tune file for the loaded model, under the profile's
/// layout. Returns that layout and the settings written. The server reads the tune file at its start, so it takes them
/// after a restart, and only when it runs in that layout.
pub fn import(p: &Value) -> Result<(String, Vec<String>), String> {
    let key = crate::pipe::model_id();
    if key == 0 || crate::tune::file().is_none() {
        return Err("this model keeps no tuned settings, so there is nothing to read them into".into());
    }
    let (lay, vals) = checked(p, key)?;
    let from = p["hardware"].as_str().unwrap_or("").to_string();
    let when = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let mut done = Vec::new();
    for (name, v) in vals {
        let entry = json!({"value": v, "imported": true, "measured_on": from, "when": when});
        crate::tune::put_in(&lay, name, entry).map_err(|e| e.to_string())?;
        done.push(format!("{name} {v}"));
    }
    Ok((lay, done))
}

#[cfg(test)]
mod tests {
    use super::checked;
    use serde_json::json;

    #[test]
    fn a_profile_is_checked_before_it_is_read_in() {
        let p = json!({"model_key": "00000000000000ab", "layout": "one-npu",
            "in_use": {"block": 1024, "guided": 1, "npu_hot": 32, "threads": 8}});
        let (lay, v) = checked(&p, 0xab).unwrap();
        assert_eq!(lay, "one-npu");
        // threads is shown, not read in: the app sets the thread count
        assert_eq!(v, vec![("block", 1024), ("guided", 1), ("npu_hot", 32)]);
        // another model file
        assert!(checked(&p, 0xac).is_err());
        // a value out of range, a layout it does not keep, nothing to read in
        assert!(checked(&json!({"layout": "one-npu", "in_use": {"npu_hot": 65}}), 1).is_err());
        assert!(checked(&json!({"layout": "three-npu", "in_use": {"block": 512}}), 1).is_err());
        assert!(checked(&json!({"layout": "one-cpu", "in_use": {}}), 1).is_err());
        // a value that was the engine's default stays out; a profile of nothing but defaults is refused
        let d = json!({"layout": "one-npu", "in_use": {"block": 1024, "guided": 0},
            "chosen": {"block": "default", "guided": "tuned"}});
        assert_eq!(checked(&d, 1).unwrap().1, vec![("guided", 0)]);
        assert!(
            checked(&json!({"layout": "one-cpu", "in_use": {"block": 512}, "chosen": {"block": "default"}}), 1)
                .is_err()
        );
        // a key that is not text is refused rather than skipped
        assert!(checked(&json!({"model_key": 171, "layout": "one-cpu", "in_use": {"block": 512}}), 0xab).is_err());
        // no model named: read into the loaded one
        assert_eq!(
            checked(&json!({"layout": "one-cpu", "in_use": {"block": 512}}), 1).unwrap().1,
            vec![("block", 512)]
        );
    }
}
