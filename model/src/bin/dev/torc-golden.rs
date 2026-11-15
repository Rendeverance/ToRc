//! torc-golden <record|check> <models folder> [threads] [only=name]: checks that a change left every model's output
//! alone, bit for bit.
//!
//! Each entry of tools/golden.json names a model file, a prompt, how many positions to run and the prompt block size
//! (0: token by token). The prompt goes through the model's own tokenizer (torc-tok), the tokens through torc-run (or
//! torc-gemma for Gemma 4) with every position's logits dumped, and the dump is hashed. `record` stores the hash for
//! this CPU family (aarch64 or x86_64: their kernels round differently, so each keeps its own); `check` runs everything
//! again and names any model whose logits moved, exiting 1 (or 3 when it compared nothing, so a run with no model files
//! or no recorded hashes can't pass for a check). A new model is added by adding an entry, and every other
//! model is checked in minutes. torc-tok, torc-run and torc-gemma are taken from this executable's own folder, and the
//! NPU and GPU tiers are left off: they are not bit-exact by design.

use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

const ARCH: &str = if cfg!(target_arch = "aarch64") { "aarch64" } else { "x86_64" };

/// tools/golden.json, found from the current folder upwards (the workspace root or anywhere inside it).
fn golden_path() -> PathBuf {
    let mut d = std::env::current_dir().unwrap_or_default();
    loop {
        let p = d.join("tools").join("golden.json");
        if p.exists() {
            return p;
        }
        if !d.pop() {
            eprintln!("torc-golden: tools/golden.json not found here or above (run it inside the workspace)");
            std::process::exit(2);
        }
    }
}

/// A sibling executable of this one.
fn tool(name: &str) -> PathBuf {
    let me = std::env::current_exe().unwrap_or_default();
    me.with_file_name(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// The BLAKE3 of entry e's logits, or None when its model file is not in `models`.
fn run(e: &Value, root: &Path, models: &Path, threads: &str) -> Option<String> {
    let model = models.join(e["file"].as_str().unwrap_or_default());
    if !model.exists() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("torc-golden-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let (toks, dump) = (dir.join("p.int32"), dir.join("logits.f32"));
    let prompt = root.join("tools").join(e["prompt"].as_str().unwrap_or_default());
    let quiet = |c: &mut Command| -> std::process::Output {
        c.env("TORC_NPU", "0").env_remove("TORC_GPU_ATTN").output().unwrap_or_else(|x| {
            eprintln!("torc-golden: {x}");
            std::process::exit(2)
        })
    };
    let t = quiet(Command::new(tool("torc-tok")).arg(&model).arg(&prompt).arg(&toks));
    if !t.status.success() {
        eprintln!("torc-golden: {}: torc-tok failed\n{}", e["name"], String::from_utf8_lossy(&t.stderr));
        std::process::exit(2)
    }
    let runner = if e["gemma"].as_bool().unwrap_or(false) { "torc-gemma" } else { "torc-run" };
    let mut c = Command::new(tool(runner));
    c.arg(&model).arg(&toks).arg(e["n"].as_u64().unwrap_or(32).to_string()).arg(threads);
    c.arg(format!("dump={}", dump.display()));
    if let Some(b) = e["block"].as_u64().filter(|&b| b > 0) {
        c.arg(format!("block={b}"));
    }
    let r = quiet(&mut c);
    let bytes = std::fs::read(&dump).ok().filter(|_| r.status.success());
    let _ = std::fs::remove_dir_all(&dir);
    match bytes {
        Some(b) => Some(blake3::hash(&b).to_hex().to_string()),
        None => {
            // what it printed too: torc-gemma's block check says there which positions differ
            let (out, err) = (String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr));
            eprintln!("torc-golden: {}: {runner} failed\n{out}{err}", e["name"]);
            std::process::exit(2)
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let record = match a.first().map(String::as_str) {
        Some("record") => true,
        Some("check") => false,
        _ => {
            eprintln!("usage: torc-golden <record|check> <models folder> [threads] [only=name]");
            std::process::exit(2)
        }
    };
    let models = PathBuf::from(a.get(1).cloned().unwrap_or_else(|| ".".into()));
    let threads = a.get(2).filter(|v| !v.starts_with("only=")).cloned().unwrap_or_else(|| "8".into());
    let only = a.iter().find_map(|v| v.strip_prefix("only="));
    let path = golden_path();
    let root = path.parent().and_then(Path::parent).unwrap_or(Path::new(".")).to_path_buf();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let mut g: Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("torc-golden: {}: {e}", path.display());
        std::process::exit(2)
    });
    let (mut ran, mut compared, mut moved) = (0, 0, Vec::new());
    for e in g["entries"].as_array_mut().into_iter().flatten() {
        let name = e["name"].as_str().unwrap_or_default().to_string();
        if only.is_some_and(|o| o != name) {
            continue;
        }
        let Some(h) = run(e, &root, &models, &threads) else {
            println!("{name:28} skipped (no {})", e["file"].as_str().unwrap_or_default());
            continue;
        };
        ran += 1;
        let want = e["hash"][ARCH].as_str().map(str::to_string);
        if record {
            if !e["hash"].is_object() {
                e["hash"] = json!({});
            }
            e["hash"][ARCH] = json!(h);
            println!("{name:28} recorded {}", &h[..16]);
        } else if let Some(w) = want {
            compared += 1;
            if w == h {
                println!("{name:28} IDENTICAL");
            } else {
                println!("{name:28} MOVED (was {}, now {})", &w[..16], &h[..16]);
                moved.push(name);
            }
        } else {
            println!("{name:28} no {ARCH} hash recorded yet ({})", &h[..16]);
        }
    }
    if record {
        let out = serde_json::to_string_pretty(&g).unwrap_or_default() + "\n";
        if let Err(e) = std::fs::write(&path, out) {
            eprintln!("torc-golden: {}: {e}", path.display());
            std::process::exit(2);
        }
    }
    println!(
        "{ran} run, {compared} compared, {} moved{}",
        moved.len(),
        if moved.is_empty() { String::new() } else { format!(": {}", moved.join(", ")) }
    );
    // a check that compared nothing proves nothing: no model files here, or no hashes for this CPU family yet
    // (`record` on the main branch first, then `check` on the change)
    if !record && compared == 0 {
        println!("NOTHING COMPARED: record the hashes on the main branch first, then check the change");
        std::process::exit(3);
    }
    std::process::exit(if moved.is_empty() { 0 } else { 1 });
}
