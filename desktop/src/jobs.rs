//! jobs.rs - the engine's other ways of running: this machine as the second of two (torc-run serving the later
//! layers to the first), and the tuner. Each is a child process whose output goes to the
//! page as it comes.

use crate::{EXE, Server, Shell, switches_for};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

/// The port the second machine of two listens on for the first.
pub const WORKER_PORT: u16 = 50052;

/// No console window for a child started by the app.
pub fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd
}

/// How likely an address is to be on the network the first machine reaches this one by: a home or office network
/// first, then a direct link (169.254, a USB4 or Ethernet cable between two machines with no router), then the rest,
/// and virtual adapters (WSL, Hyper-V, VPNs, Bluetooth) after all of those.
fn rank(name: &str, address: &str) -> u32 {
    let n = name.to_lowercase();
    let virt = [
        "vethernet",
        "wsl",
        "hyper-v",
        "vpn",
        "nord",
        "tailscale",
        "zerotier",
        "virtualbox",
        "vmware",
        "bluetooth",
        "loopback",
    ]
    .iter()
    .any(|v| n.contains(v));
    let o: Vec<u32> = address.split('.').filter_map(|p| p.parse().ok()).collect();
    let lan = o.len() == 4 && (o[0] == 10 || (o[0] == 192 && o[1] == 168) || (o[0] == 172 && (16..32).contains(&o[1])));
    let link = o.len() == 4 && o[0] == 169 && o[1] == 254;
    (if virt { 10 } else { 0 })
        + if lan {
            0
        } else if link {
            1
        } else {
            2
        }
}

/// This machine's IPv4 addresses on the networks it is connected to, each with its adapter's name, likeliest first.
pub fn interfaces() -> Vec<(String, String)> {
    #[cfg(windows)]
    let raw = {
        let ps = "Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred | Where-Object { $_.IPAddress -ne \
                  '127.0.0.1' } | ForEach-Object { $_.InterfaceAlias + \"`t\" + $_.IPAddress }";
        let out = quiet(Command::new("powershell").args(["-NoProfile", "-Command", ps])).output();
        out.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default()
    };
    #[cfg(not(windows))]
    let raw = {
        // "2: eth0    inet 192.168.1.5/24 ..." from ip -4 -o addr
        let out = Command::new("ip").args(["-4", "-o", "addr"]).output();
        let text = out.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
        text.lines()
            .filter_map(|l| {
                let w: Vec<&str> = l.split_whitespace().collect();
                let a = w.get(3)?.split('/').next()?;
                let name = w.get(1)?;
                (a != "127.0.0.1").then(|| format!("{name}\t{a}"))
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut v: Vec<(String, String)> = raw
        .lines()
        .filter_map(|l| l.trim().split_once('\t').map(|(n, a)| (n.trim().to_string(), a.trim().to_string())))
        .collect();
    v.sort_by_key(|(n, a)| rank(n, a));
    v
}

/// The address the second machine listens on: the one picked on the Load screen if this machine still has it, every
/// network for "any", else the likeliest, or this machine alone when no network can be found. A worker has no
/// password, so it is not on every network by default.
fn bind_for(choice: &str) -> String {
    if choice == "any" {
        return "0.0.0.0".into();
    }
    let mine: Vec<String> = interfaces().into_iter().map(|(_, a)| a).collect();
    if mine.iter().any(|a| a == choice) {
        return choice.into();
    }
    mine.into_iter().next().unwrap_or_else(|| "127.0.0.1".into())
}

impl Shell {
    /// This machine as the second of two: torc-run holding the later layers until the first machine connects. Ready
    /// once it says it is serving.
    pub fn start_worker(&self, app: &AppHandle, s: &Value) -> Value {
        let bin = PathBuf::from(s["binDir"].as_str().unwrap_or(""));
        let exe = bin.join(format!("torc-run{EXE}"));
        let model = s["lastModel"].as_str().unwrap_or("").to_string();
        if !exe.exists() {
            return json!({"ok": false, "kept": true, "error": format!("torc-run is not in {}", bin.display())});
        }
        if !Path::new(&model).exists() {
            return json!({"ok": false, "kept": true, "error": "that model file is not there"});
        }
        self.stop_server();
        let mut env = switches_for(s);
        env.retain(|(k, _)| k != "TORC_PIPE");
        // an address set in Advanced wins
        let bind = match env.iter().find(|(k, _)| k == "TORC_PIPE_BIND") {
            Some((_, v)) => v.clone(),
            None => {
                let b = bind_for(s["workerBind"].as_str().unwrap_or(""));
                env.push(("TORC_PIPE_BIND".into(), b.clone()));
                b
            }
        };
        let ctx = s["ctx"].as_u64().unwrap_or(4096);
        let mut cmd = Command::new(&exe);
        crate::without_app_vars(&mut cmd);
        cmd.args([model.as_str(), &format!("serve-pipe={WORKER_PORT}"), &format!("ctx={ctx}")])
            .envs(env)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = match quiet(&mut cmd).spawn() {
            Ok(c) => c,
            Err(e) => return json!({"ok": false, "kept": true, "error": format!("torc-run would not start: {e}")}),
        };
        // its log goes to the page; the line that says it is listening is watched for here
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        if let Some(err) = child.stderr.take() {
            let app = app.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    if line.contains("serving layers on") {
                        let _ = tx.send(());
                    }
                    let _ = app.emit("torc:log", format!("{line}\n"));
                }
            });
        }
        let child = Arc::new(Mutex::new(child));
        *self.server.lock().unwrap() = Some(Server {
            child: child.clone(),
            port: 0,
            ready: false,
            worker: true,
            cfg: s.clone(),
            model: model.clone(),
        });
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(crate::LOAD_SECONDS) {
            if rx.recv_timeout(Duration::from_millis(500)).is_ok() {
                if let Some(sv) = self.server.lock().unwrap().as_mut() {
                    sv.ready = true;
                }
                crate::watch_exit(app, child);
                let name = Path::new(&model).file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let shown: Vec<String> =
                    if bind == "0.0.0.0" { interfaces().into_iter().map(|(_, a)| a).collect() } else { vec![bind] };
                let addresses: Vec<String> = shown.iter().map(|a| format!("{a}:{WORKER_PORT}")).collect();
                return json!({"ok": true, "worker": true, "name": name, "addresses": addresses});
            }
            if let Ok(Some(_)) = child.lock().unwrap().try_wait() {
                *self.server.lock().unwrap() = None;
                return json!({"ok": false, "error": "the engine did not start as the second machine"});
            }
        }
        self.stop_server();
        json!({"ok": false, "error": "the engine did not start as the second machine"})
    }

    /// The tuner (the benchmark needs Node, see node.rs): the engine is stopped for it, as it needs the model to
    /// itself. What it prints goes to the page as it comes; the answer comes when it ends.
    pub fn run_job(&self, app: &AppHandle, kind: &str, args: &Value) -> Value {
        let s = self.settings();
        if kind == "bench" {
            return crate::node::run_bench(self, app, &s);
        }
        if kind == "convert" {
            return self.convert(app, &s, args);
        }
        let bin = PathBuf::from(s["binDir"].as_str().unwrap_or(""));
        let exe = bin.join(format!("torc-tune{EXE}"));
        if !exe.exists() {
            return json!({"ok": false, "kept": true, "error": format!("torc-tune is not in {}", bin.display())});
        }
        let model = s["lastModel"].as_str().unwrap_or("").to_string();
        self.stop_server();
        let threads = s["threads"].as_str().map_or_else(|| s["threads"].to_string(), str::to_string);
        let ctx = s["ctx"].as_u64().unwrap_or(16384).to_string();
        let mut cmd = Command::new(&exe);
        crate::without_app_vars(&mut cmd);
        cmd.args([model.as_str(), &threads, &ctx]).envs(switches_for(&s));
        self.follow(app, cmd)
    }

    /// A .torc model's architecture (from `torc info`) and its file name, from which the page picks the model maker's
    /// recommended sampling.
    pub fn model_arch(&self, path: &str) -> Value {
        let s = self.settings();
        let bin = PathBuf::from(s["binDir"].as_str().unwrap_or(""));
        let mut c = Command::new(bin.join(format!("torc{EXE}")));
        crate::without_app_vars(&mut c);
        let first = quiet(c.args(["info", path]))
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or("").to_string())
            .unwrap_or_default();
        let arch = first.split(", arch ").nth(1).and_then(|r| r.split(',').next()).unwrap_or("").trim().to_string();
        let name = PathBuf::from(path).file_stem().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        json!({"arch": arch, "name": name})
    }

    /// A GGUF the user picked, before it is converted: its architecture, whether this engine runs it, its size, and
    /// where the converted file goes by default (models beside a portable app, else beside the GGUF).
    pub fn gguf_check(&self, src: &str) -> Value {
        let s = self.settings();
        let bin = PathBuf::from(s["binDir"].as_str().unwrap_or(""));
        let run = |exe: &str, args: &[&str]| {
            let mut c = Command::new(bin.join(format!("{exe}{EXE}")));
            crate::without_app_vars(&mut c);
            quiet(c.args(args)).output().ok().filter(|o| o.status.success())
        };
        let Some(info) = run("torc", &["gguf-info", src]) else {
            let error = format!("torc could not read {src} as a GGUF (is torc in {}?)", bin.display());
            return json!({"ok": false, "error": error});
        };
        let first = String::from_utf8_lossy(&info.stdout).lines().next().unwrap_or("").to_string();
        let arch = first.split(", arch ").nth(1).and_then(|r| r.split(',').next()).unwrap_or("").trim().to_string();
        let archs = run("torc-server", &["--archs"]).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
        // torc plan makes the load's own checks (architecture, weight types, the model's settings, each layer's
        // experts), so a model that couldn't run is refused here, with the reason, before it is converted
        let mut plan = Command::new(bin.join(format!("torc{EXE}")));
        crate::without_app_vars(&mut plan);
        let plan = quiet(plan.args(["plan", src])).output().ok();
        let supported = plan.as_ref().is_some_and(|o| o.status.success());
        let why = plan
            .filter(|o| !o.status.success())
            .map(|o| {
                let e = String::from_utf8_lossy(&o.stderr).trim().to_string();
                e.split_once("won't run in this build of ToRc: ").map_or(e.clone(), |(_, w)| w.to_string())
            })
            .unwrap_or_default();
        let src_path = PathBuf::from(src);
        let stem = src_path.file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let folder =
            crate::portable_dir("models").filter(|d| d.is_dir()).or_else(|| src_path.parent().map(PathBuf::from));
        let out = folder.unwrap_or_default().join(format!("{stem}.torc"));
        let bytes = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
        let free = out.parent().and_then(free_bytes);
        json!({"ok": true, "arch": arch, "supported": supported, "why": why, "archs": archs.unwrap_or_default(),
            "bytes": bytes, "free": free, "out": out.display().to_string()})
    }

    /// Converts a GGUF into a ToRc file with torc convert-gguf, its output to the page as it comes
    fn convert(&self, app: &AppHandle, s: &Value, args: &Value) -> Value {
        let (src, dst) = (args["src"].as_str().unwrap_or(""), args["dst"].as_str().unwrap_or(""));
        if src.is_empty() || dst.is_empty() {
            return json!({"ok": false, "error": "give the GGUF and where the ToRc file goes"});
        }
        if std::path::Path::new(dst).exists() {
            return json!({"ok": false, "error": format!("{dst} is there already: choose another name")});
        }
        let need = std::fs::metadata(src).map(|m| m.len()).unwrap_or(0);
        if let Some(free) = Path::new(dst).parent().and_then(free_bytes).filter(|&f| f < need) {
            let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
            let error = format!("there is not enough space: it needs {:.1} GB, {:.1} GB is free", gb(need), gb(free));
            return json!({"ok": false, "error": error});
        }
        let bin = PathBuf::from(s["binDir"].as_str().unwrap_or(""));
        let mut cmd = Command::new(bin.join(format!("torc{EXE}")));
        crate::without_app_vars(&mut cmd);
        cmd.args(["convert-gguf", src, dst]);
        self.follow(app, cmd)
    }

    /// Runs a job to its end, its output to the page as "joblog", Stop able to end it.
    pub fn follow(&self, app: &AppHandle, mut cmd: Command) -> Value {
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child: Child = match quiet(&mut cmd).spawn() {
            Ok(c) => c,
            Err(e) => return json!({"ok": false, "error": e.to_string()}),
        };
        let pipes = [
            child.stdout.take().map(|o| Box::new(o) as Box<dyn std::io::Read + Send>),
            child.stderr.take().map(|e| Box::new(e) as Box<dyn std::io::Read + Send>),
        ];
        let mut readers = Vec::new();
        for p in pipes.into_iter().flatten() {
            let app = app.clone();
            readers.push(std::thread::spawn(move || {
                for line in BufReader::new(p).lines().map_while(Result::ok) {
                    let _ = app.emit("torc:joblog", format!("{line}\n"));
                }
            }));
        }
        let child = Arc::new(Mutex::new(child));
        *self.job.lock().unwrap() = Some(child.clone());
        let code = loop {
            if let Ok(Some(st)) = child.lock().unwrap().try_wait() {
                break st.code();
            }
            std::thread::sleep(Duration::from_millis(200));
        };
        for r in readers {
            let _ = r.join();
        }
        *self.job.lock().unwrap() = None;
        json!({"ok": code == Some(0), "code": code})
    }

    pub fn stop_job(&self) {
        if let Some(j) = self.job.lock().unwrap().as_ref() {
            let _ = j.lock().unwrap().kill();
        }
    }
}

/// A path for the inside of a single-quoted PowerShell string, where a quote is written twice
#[cfg(windows)]
pub fn ps_quoted(p: &Path) -> String {
    p.to_string_lossy().replace('\'', "''")
}

/// The free space on the drive holding `dir`, as the system reports it; None when it can't be read
fn free_bytes(dir: &Path) -> Option<u64> {
    #[cfg(windows)]
    let out = quiet(Command::new("powershell").args([
        "-NoProfile",
        "-Command",
        &format!("(Get-Item -LiteralPath '{}').PSDrive.Free", ps_quoted(dir)),
    ]))
    .output();
    #[cfg(not(windows))]
    let out = Command::new("df").args(["-B1", "--output=avail"]).arg(dir).output();
    let text = String::from_utf8_lossy(&out.ok()?.stdout).to_string();
    text.lines().rev().find_map(|l| l.trim().parse().ok())
}
