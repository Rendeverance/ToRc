//! node.rs - what needs Node: ToolFunnel (an npm package) and the benchmark script. The app has no Node of its own, so
//! it uses the one on the machine, or the first time either is wanted it fetches the official portable build from
//! nodejs.org, checks it against Node's published SHA-256 list and keeps it in the app's folder. ToolFunnel itself is
//! installed from npm: its sha512 checked against npm's, and only plain files with safe names unpacked.

use crate::Shell;
use base64::Engine;
use serde_json::{Value, json};
use sha2::{Digest, Sha256, Sha512};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use tauri::{AppHandle, Emitter};

const REGISTRY: &str = "https://registry.npmjs.org/toolfunnel/latest";
const NODE_DIST: &str = "https://nodejs.org/dist/latest-v22.x/";

fn version_ok(v: &str) -> bool {
    let core = v.split('-').next().unwrap_or("");
    let parts: Vec<&str> = core.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
        && v.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
}

fn say(app: &AppHandle, line: &str) {
    let _ = app.emit("torc:joblog", format!("{line}\n"));
}

fn get(url: &str) -> Result<Vec<u8>, String> {
    let res = ureq::get(url).call().map_err(|e| e.to_string())?;
    let mut b = Vec::new();
    res.into_reader().take(512 << 20).read_to_end(&mut b).map_err(|e| e.to_string())?;
    Ok(b)
}

impl Shell {
    /// ToolFunnel is an MCP server like the rest: it lives in mcps, with the servers added to it inside its own folder
    pub(crate) fn tf_root(&self) -> PathBuf {
        self.mcps_dir().join("toolfunnel")
    }

    fn tf_program(&self, v: &str) -> PathBuf {
        self.tf_root().join(format!("pkg-{v}")).join("bin").join("toolfunnel.js")
    }

    pub fn tf_installed(&self) -> String {
        let v = self.settings()["toolfunnel"].as_str().unwrap_or("").to_string();
        if version_ok(&v) && self.tf_program(&v).exists() { v } else { String::new() }
    }

    /// Node: the app's own copy if it fetched one, else the one on the PATH.
    pub fn node(&self) -> Option<PathBuf> {
        let own = self.tools_dir().join("node").join(if cfg!(windows) { "node.exe" } else { "bin/node" });
        if own.exists() {
            return Some(own);
        }
        let finder = if cfg!(windows) { "where" } else { "which" };
        let out = crate::jobs::quiet(Command::new(finder).arg("node")).output().ok()?;
        let first = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
        (!first.is_empty() && Path::new(&first).exists()).then(|| PathBuf::from(first))
    }

    /// Node, fetched from nodejs.org when the machine has none: the archive for this system, checked against
    /// SHASUMS256.txt, unpacked into the app's folder.
    fn ensure_node(&self, app: &AppHandle) -> Result<PathBuf, String> {
        if let Some(n) = self.node() {
            return Ok(n);
        }
        let arch = if cfg!(target_arch = "aarch64") { "arm64" } else { "x64" };
        let tail = if cfg!(windows) { format!("-win-{arch}.zip") } else { format!("-linux-{arch}.tar.xz") };
        say(app, "No Node on this machine: fetching the portable build from nodejs.org");
        let sums = String::from_utf8(get(&format!("{NODE_DIST}SHASUMS256.txt"))?).map_err(|e| e.to_string())?;
        let (want, file) = sums
            .lines()
            .filter_map(|l| l.split_once("  "))
            .find(|(_, f)| f.ends_with(&tail))
            .ok_or("nodejs.org lists no build for this system")?;
        let bytes = get(&format!("{NODE_DIST}{file}"))?;
        let got = format!("{:x}", Sha256::digest(&bytes));
        if got != want {
            return Err("the Node download does not match the hash nodejs.org publishes".into());
        }
        say(app, &format!("{file}: {} MB, hash checked. Unpacking", bytes.len() >> 20));
        let tmp = self.tools_dir().join("node-download");
        let _ = std::fs::create_dir_all(&tmp);
        let archive = tmp.join(file);
        std::fs::write(&archive, &bytes).map_err(|e| e.to_string())?;
        // the system's own unpacker: Expand-Archive on Windows, tar elsewhere
        #[cfg(windows)]
        let ok = crate::jobs::quiet(Command::new("powershell").args([
            "-NoProfile",
            "-Command",
            &format!(
                "Expand-Archive -LiteralPath '{}' -DestinationPath '{}' -Force",
                crate::jobs::ps_quoted(&archive),
                crate::jobs::ps_quoted(&tmp)
            ),
        ]))
        .status();
        #[cfg(not(windows))]
        let ok = Command::new("tar").arg("-xJf").arg(&archive).arg("-C").arg(&tmp).status();
        if !ok.map(|s| s.success()).unwrap_or(false) {
            return Err("the Node archive would not unpack".into());
        }
        let inner = tmp.join(file.trim_end_matches(".zip").trim_end_matches(".tar.xz"));
        let dest = self.tools_dir().join("node");
        // a Node folder left half-made by an earlier try (this is only reached when no working Node is there)
        if dest.exists() {
            std::fs::remove_dir_all(&dest).map_err(|e| e.to_string())?;
        }
        std::fs::rename(&inner, &dest).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&archive);
        let _ = std::fs::remove_dir(&tmp); // empty by now; left alone if anything else is in it
        self.node().ok_or_else(|| "Node did not unpack as expected".into())
    }

    /// What npm says ToolFunnel's current release is: (version, tarball, integrity).
    fn tf_release(&self) -> Result<(String, String, String), String> {
        let meta: Value = serde_json::from_slice(&get(REGISTRY)?).map_err(|e| e.to_string())?;
        let v = meta["version"].as_str().unwrap_or("").to_string();
        if !version_ok(&v) {
            return Err("npm gave a version that does not read as one".into());
        }
        let t = meta["dist"]["tarball"].as_str().unwrap_or("").to_string();
        let i = meta["dist"]["integrity"].as_str().unwrap_or("").to_string();
        Ok((v, t, i))
    }

    pub fn tf_latest(&self) -> String {
        self.tf_release().map(|r| r.0).unwrap_or_default()
    }

    pub fn tf_install(&self, app: &AppHandle) -> Value {
        let r = (|| -> Result<String, String> {
            self.ensure_node(app)?;
            say(app, "Asking npm for the current ToolFunnel release");
            let (v, tarball, integrity) = self.tf_release()?;
            say(app, &format!("ToolFunnel {v}: downloading"));
            let archive = get(&tarball)?;
            let want = integrity.strip_prefix("sha512-").ok_or("npm gave no sha512 for it")?;
            let got = base64::engine::general_purpose::STANDARD.encode(Sha512::digest(&archive));
            if got != want {
                return Err("the download does not match the hash npm publishes for it".into());
            }
            say(app, &format!("{} bytes, hash checked. Unpacking", archive.len()));
            let mut tar = Vec::new();
            flate2::read::GzDecoder::new(&archive[..]).read_to_end(&mut tar).map_err(|e| e.to_string())?;
            let dir = self.tf_root().join(format!("pkg-{v}"));
            let mut files = 0;
            for (name, body) in untar(&tar) {
                let rel = name.strip_prefix("package/").unwrap_or(&name);
                let file = dir.join(rel);
                if let Some(d) = file.parent() {
                    std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                }
                std::fs::write(&file, body).map_err(|e| e.to_string())?;
                files += 1;
            }
            if !self.tf_program(&v).exists() {
                return Err("the release did not unpack as expected".into());
            }
            self.settings_set(&json!({"toolfunnel": v}));
            say(app, &format!("{files} files. ToolFunnel {v} is installed."));
            Ok(v)
        })();
        match r {
            Ok(v) => json!({"ok": true, "version": v}),
            Err(e) => {
                say(app, &e);
                json!({"ok": false, "error": e})
            }
        }
    }

    pub fn tf_uninstall(&self, app: &AppHandle) -> Value {
        let s = self.settings();
        if s["tools"] == "toolfunnel" {
            self.settings_set(&json!({"tools": "basic"}));
        }
        self.sync(app);
        for e in std::fs::read_dir(self.tf_root()).into_iter().flatten().flatten() {
            if e.file_name().to_string_lossy().starts_with("pkg-") {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
        self.settings_set(&json!({"toolfunnel": ""}));
        json!({"ok": true})
    }

    /// The benchmark: tools/bench/torc-bench.js on Node, against the running server, with the settings that server
    /// was started with.
    pub fn bench(&self, app: &AppHandle) -> Value {
        let (port, cfg, model) = match self.server.lock().unwrap().as_ref() {
            Some(sv) if sv.ready && !sv.worker => (sv.port, sv.cfg.clone(), sv.model.clone()),
            _ => return json!({"ok": false, "error": "load a model first: the benchmark measures the running server"}),
        };
        let node = match self.ensure_node(app) {
            Ok(n) => n,
            Err(e) => return json!({"ok": false, "error": e}),
        };
        // the script and its texts are inside the app, so a copied exe can run it: written out beside the settings
        let dir = self.dir.join("bench");
        if let Err(e) = std::fs::create_dir_all(&dir)
            .and_then(|_| BENCH.iter().try_for_each(|(f, s)| std::fs::write(dir.join(f), s)))
        {
            return json!({"ok": false, "error": format!("could not write the benchmark to {}: {e}", dir.display())});
        }
        let script = dir.join("torc-bench.js");
        let ctx = cfg["ctx"].as_u64().unwrap_or(0);
        let mut sizes = vec!["alice", "1k", "4k", "8k"];
        if ctx >= 18000 {
            sizes.push("16k");
        }
        if ctx >= 34816 {
            sizes.push("32k");
        }
        let results = dirs_documents().join("torc-bench.csv");
        if let Some(d) = results.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        say(app, &format!("The results go to {}", results.display()));
        let on = |k: &str| if cfg[k] == true { "on" } else { "off" };
        let settings = format!(
            "threads {}, ctx {ctx}, NPU {}, GPU {}, {}",
            cfg["threads"].as_str().unwrap_or("auto"),
            on("npu"),
            on("gpu"),
            if cfg["worker"].as_str().is_some_and(|w| !w.is_empty()) { "two machines" } else { "one machine" }
        );
        let name = Path::new(&model).file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let mut cmd = Command::new(node);
        crate::without_app_vars(&mut cmd);
        cmd.arg(script)
            .arg(port.to_string())
            .args(["--model", &name, "--sizes", &sizes.join(","), "--settings", &settings])
            .arg("--out")
            .arg(&results)
            .envs(crate::switches_for(&cfg));
        // hwscan from the engine's folder names the hardware in the results
        let hwscan = PathBuf::from(self.settings()["binDir"].as_str().unwrap_or("")).join(if cfg!(windows) {
            "hwscan.exe"
        } else {
            "hwscan"
        });
        if hwscan.is_file() {
            cmd.arg("--hwscan").arg(hwscan);
        }
        self.follow(app, cmd)
    }
}

/// tools/bench: the benchmark script and the texts it sends
const BENCH: [(&str, &str); 7] = [
    ("torc-bench.js", include_str!("../../tools/bench/torc-bench.js")),
    ("lb_alice.txt", include_str!("../../tools/bench/lb_alice.txt")),
    ("lb_p1k.txt", include_str!("../../tools/bench/lb_p1k.txt")),
    ("lb_p4k.txt", include_str!("../../tools/bench/lb_p4k.txt")),
    ("lb_p8192.txt", include_str!("../../tools/bench/lb_p8192.txt")),
    ("lb_p16k.txt", include_str!("../../tools/bench/lb_p16k.txt")),
    ("lb_p32k.txt", include_str!("../../tools/bench/lb_p32k.txt")),
];

fn dirs_documents() -> PathBuf {
    // a portable app keeps its results in its own data folder
    if let Some(r) = crate::portable_root() {
        return r.join("data").join("results");
    }
    let home =
        std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from).unwrap_or_default();
    home.join("Documents")
}

/// ToolFunnel as an MCP server: Node running its program, with its home in the app's folder. None when it is not
/// installed or there is no Node.
pub fn toolfunnel_server(shell: &Shell) -> Option<(String, Command)> {
    let v = shell.tf_installed();
    if v.is_empty() {
        return None;
    }
    let node = shell.node()?;
    let mut cmd = Command::new(node);
    cmd.arg(shell.tf_program(&v)).env("TOOLFUNNEL_HOME", shell.tf_root().join("home"));
    Some((format!("toolfunnel {v}"), cmd))
}

pub fn run_bench(shell: &Shell, app: &AppHandle, _s: &Value) -> Value {
    shell.bench(app)
}

/// The plain files of a tar archive, by name: a long name (GNU L, or a pax path) is followed, and a name that could
/// land outside the folder (rooted with either slash, with a drive or "..") is left out, as is everything that is not a
/// plain file (links included).
fn untar(buf: &[u8]) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut long = String::new();
    let mut at = 0usize;
    while at + 512 <= buf.len() {
        let head = &buf[at..at + 512];
        if head.iter().all(|&b| b == 0) {
            break;
        }
        let text = |from: usize, len: usize| {
            let s = &head[from..from + len];
            let end = s.iter().position(|&b| b == 0).unwrap_or(len);
            String::from_utf8_lossy(&s[..end]).to_string()
        };
        let size = usize::from_str_radix(text(124, 12).trim(), 8).unwrap_or(0);
        let kind = head[156];
        let body = buf.get(at + 512..at + 512 + size).unwrap_or(&[]);
        let prefix = text(345, 155);
        let name = if long.is_empty() {
            if prefix.is_empty() { text(0, 100) } else { format!("{prefix}/{}", text(0, 100)) }
        } else {
            std::mem::take(&mut long)
        };
        match kind {
            b'L' => long = String::from_utf8_lossy(body).trim_end_matches('\0').to_string(),
            b'x' => {
                let r = String::from_utf8_lossy(body);
                long = r.lines().find_map(|l| l.split_once(" path=").map(|(_, p)| p.to_string())).unwrap_or_default();
            }
            b'0' | 0 => {
                // a rooted name (/x, or \x on Windows), a drive (c:), ".." or an empty step lands outside the folder
                let bad = name.starts_with(['/', '\\'])
                    || name.contains(':')
                    || name.split(['/', '\\']).any(|p| p == ".." || p.is_empty());
                if !bad {
                    out.push((name, body.to_vec()));
                }
            }
            _ => {}
        }
        at += 512 + size.div_ceil(512) * 512;
    }
    out
}
