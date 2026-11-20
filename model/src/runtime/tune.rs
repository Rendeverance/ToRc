//! tune.rs - the settings measured best on this machine for this model (AUTOTUNE.md, torc-tune): where they live and
//! how a setting reads them. An environment variable always wins over a tuned value, and no file (or no entry) means
//! the built-in rules.
//!
//! There is one JSON file per machine and model: tune\<machine>\<model>.json in ToRc's cache folder (cache_dir). Inside
//! it there is one object per layout (one machine or two, NPU or CPU, since the best block changes with both), each
//! holding every setting's chosen value with the scores it was chosen on.

use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::OnceLock;

/// This machine: FNV-1a of the CPU's identifier, the CPU count and the memory installed, as a hex string. A new CPU or
/// more memory means re-tuning; a new engine build does not (the model key covers the file, not yet the kernels).
pub fn machine() -> String {
    let cpu = std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default();
    let n = std::thread::available_parallelism().map_or(0, |n| n.get());
    let mem = crate::weights::total_phys().unwrap_or(0) >> 30;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in format!("{cpu}|{n}|{mem}").bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The layout a tuned value is for: "one-npu", "one-cpu", "two-npu" or "two-cpu" (from TORC_PIPE and TORC_NPU, as the
/// server runs).
pub fn layout() -> String {
    let two = crate::switches::given("TORC_PIPE");
    let npu = crate::switches::on("TORC_NPU");
    format!("{}-{}", if two { "two" } else { "one" }, if npu { "npu" } else { "cpu" })
}

/// The tune file for the loaded model on this machine (None before Model::load, or when there is no cache directory).
/// Keyed by the model alone, not the context: torc-tune loads at 8192 and a server at its own context, and when the
/// key held the context (pipe::ident) a server never found what torc-tune chose.
pub fn file() -> Option<PathBuf> {
    let id = crate::pipe::model_id();
    if id == 0 {
        return None;
    }
    Some(cache_dir()?.join("tune").join(machine()).join(format!("{id:016x}.json")))
}

/// The folder where ToRc keeps what it learns on this machine (the tune files, the expert usage, the two-machine
/// split): %LOCALAPPDATA%\torc, else $XDG_CACHE_HOME/torc, else ~/.cache/torc. None when none of those is set.
pub fn cache_dir() -> Option<PathBuf> {
    // an empty or relative value is not a place: it would put the files wherever the program happened to start
    let var = |k: &str| std::env::var_os(k).map(PathBuf::from).filter(|p| p.is_absolute());
    var("LOCALAPPDATA")
        .or_else(|| var("XDG_CACHE_HOME"))
        .or_else(|| var("HOME").map(|h| h.join(".cache")))
        .map(|b| b.join("torc"))
}

fn read() -> Value {
    file()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}))
}

/// The tuned value of `name` for this layout, if torc-tune chose one. It is read once per process, so after tuning
/// again the server must be restarted.
pub fn get(name: &str) -> Option<u64> {
    static T: OnceLock<Value> = OnceLock::new();
    T.get_or_init(read).get(layout())?.get(name)?.get("value")?.as_u64()
}

/// Stores `entry` (the value and its evidence) as `name`'s value for this layout, keeping the rest of the file.
pub fn put(name: &str, entry: Value) -> std::io::Result<PathBuf> {
    put_in(&layout(), name, entry)
}

/// put, for a given layout ("one-npu" and so on) rather than the one this process runs.
pub fn put_in(lay: &str, name: &str, entry: Value) -> std::io::Result<PathBuf> {
    let p = file().ok_or_else(|| std::io::Error::other("no model loaded or no cache directory"))?;
    let mut v = read();
    if !v.is_object() {
        v = json!({});
    }
    let l = v.as_object_mut().unwrap().entry(lay).or_insert_with(|| json!({}));
    if !l.is_object() {
        *l = json!({});
    }
    l.as_object_mut().unwrap().insert(name.to_string(), entry);
    std::fs::create_dir_all(p.parent().unwrap())?;
    // written beside the file and renamed over it, so a crash half way never leaves it unreadable
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&v).unwrap())?;
    std::fs::rename(&tmp, &p)?;
    Ok(p)
}

/// The loaded model's architecture (general.architecture), for the defaults that differ by model.
static ARCH: OnceLock<String> = OnceLock::new();

/// Records the loaded model's architecture (Model::load).
pub fn set_arch(arch: &str) {
    let _ = ARCH.set(arch.to_string());
}

/// The loaded model's architecture, or "" before a load.
pub fn arch() -> &'static str {
    ARCH.get().map_or("", String::as_str)
}

/// Whether the loaded model sits whole in memory (no expert cache), set by Model::load.
static RESIDENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Records whether the loaded model sits whole in memory (Model::load).
pub fn set_resident(on: bool) {
    RESIDENT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the loaded model reads its prompt block from the tune file (the Qwen family's Model::load says so). Gemma 4
/// and gpt-oss keep a tune file too, for what the GPU tier learns, but read the block from TORC_BLOCK alone.
static READS_BLOCK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Records that the loaded model reads its prompt block from the tune file (Model::load).
pub fn set_reads_block() {
    READS_BLOCK.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the loaded model reads its prompt block from the tune file.
pub fn reads_block() -> bool {
    READS_BLOCK.load(std::sync::atomic::Ordering::Relaxed)
}

/// The NPU's prompt block and hot experts a layer on one machine, by architecture, with the whole model in memory.
/// The Qwen3.6-35B was measured reading faster with these than with the general 256 and 16, writing as fast and giving
/// the same replies, and the other Qwen3.5 / 3.6 MoE models share its build. Every other model keeps the general
/// values, since neither is better on every model, and so does a model read from disk through the expert cache.
fn npu_defaults() -> Option<(usize, usize)> {
    if !RESIDENT.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    match arch() {
        "qwen35moe" => Some((1024, 32)),
        _ => None,
    }
}

/// The hot experts a layer the NPU tier takes when TORC_NPU_HOT and the tune file say nothing: the model's own
/// default (npu_defaults), or 16.
pub fn npu_hot_default() -> usize {
    npu_defaults().map_or(16, |d| d.1)
}

/// The prompt block by rule: 128 on the two-machine pipeline, with the NPU or not; with the NPU tier on one machine
/// the model's own default (npu_defaults), else 256; and 512 on one machine's CPU.
pub fn block_rule() -> usize {
    if crate::switches::given("TORC_PIPE") {
        128
    } else if crate::switches::on("TORC_NPU") {
        npu_defaults().map_or(256, |d| d.0)
    } else {
        512
    }
}

/// The prompt block a server reads with: TORC_BLOCK, else the tuned value, else the rule.
pub fn block() -> usize {
    crate::switches::num("TORC_BLOCK")
        .filter(|&b: &usize| b > 0)
        .or_else(|| get("block").map(|b| b as usize).filter(|&b| b > 0))
        .unwrap_or_else(block_rule)
}
