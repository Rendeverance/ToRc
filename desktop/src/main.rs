//! torc-desktop - the ToRc desktop app. The page is ui/renderer; this is the shell behind it: the settings, saved chats
//! and primers, the file dialogs, torc-server as a child process, and a chat turn (the reply streamed to the page,
//! tools run between rounds). jobs.rs has the second machine and the tuner, tools.rs the MCP servers, node.rs Node,
//! ToolFunnel and the benchmark, basic.rs the two built-in tools.

// no console window behind the app in a release build
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::webview::PageLoadEvent;
use tauri::{AppHandle, Emitter, Manager, State, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;

mod basic;
use basic::{BASIC, basic_specs, run_basic};
mod jobs;
mod node;
mod tools;

const BRIDGE: &str = include_str!("../bridge.js");
const SWITCHES_MD: &str = include_str!("../../SWITCHES.md");
pub(crate) const LOAD_SECONDS: u64 = 600;
const MAX_ROUNDS: usize = 8;
const MAX_ATTACH_BYTES: u64 = 4 << 20;
const CHAT_FORMAT: &str = "torc-chat-1";
const SPLASH: Duration = Duration::from_millis(1500);

#[cfg(windows)]
pub(crate) const EXE: &str = ".exe";
#[cfg(not(windows))]
pub(crate) const EXE: &str = "";

/// torc-server while it runs (or torc-run, as the second machine of two).
pub(crate) struct Server {
    pub child: Arc<Mutex<Child>>,
    pub port: u16,
    pub ready: bool,
    pub worker: bool,
    /// the settings it was started with, for the benchmark
    pub cfg: Value,
    pub model: String,
}

pub(crate) struct Shell {
    /// the app's own data folder: settings.json, chats/, primers/, toolfunnel/, node/
    pub dir: PathBuf,
    pub server: Mutex<Option<Server>>,
    /// the request in flight, so Stop can cut it
    turn: Mutex<Option<TcpStream>>,
    stopped: AtomicBool,
    /// the tuner or the benchmark, while one runs
    pub job: Mutex<Option<Arc<Mutex<Child>>>>,
    /// the MCP servers, and one sync of them at a time
    pub servers: Mutex<Vec<tools::Server>>,
    pub syncing: Mutex<()>,
    /// tool calls waiting for the user's yes or no
    pub waiting: Mutex<std::collections::HashMap<u64, std::sync::mpsc::Sender<bool>>>,
    pub asked: std::sync::atomic::AtomicU64,
}

// ---- settings: one JSON file, over the defaults

/// The app's own folder when it runs portable, which a folder named data beside it marks (the zip ships one,
/// with chats and models beside it)
pub fn portable_root() -> Option<PathBuf> {
    std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)).filter(|r| r.join("data").is_dir())
}

/// A folder beside the app when it runs portable (tools: Node and ToolFunnel; mcps: the user's MCP servers), else
/// None
pub fn portable_dir(name: &str) -> Option<PathBuf> {
    portable_root().map(|r| r.join(name))
}

fn defaults() -> Value {
    // the engine folder beside the app (a portable zip), else the one this checkout builds
    let server = if cfg!(windows) { "torc-server.exe" } else { "torc-server" };
    let here = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
    let beside = here.map(|d| d.join("engine")).filter(|d| d.join(server).is_file());
    let bin = beside.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("target").join("release"));
    json!({
        "saveChats": true, "chatDir": "", "autoCompact": true, "compactShare": 15, "compactMaxWords": 5000,
        "primer": "", "firstView": "chat",
        "binDir": bin.to_string_lossy(), "port": 8080, "lastModel": "", "ctx": 16384, "replyMax": 16384, "threads": "auto",
        "npu": false, "gpu": false, "worker": "", "asWorker": false, "workerBind": "", "env": {},
        "tools": "basic", "basic": {"now": true, "calc": true, "ctx": true, "compact": false, "wake": false},
        "mcp": [], "toolfunnel": "", "agentSetup": false,
    })
}

impl Shell {
    fn settings_file(&self) -> PathBuf {
        self.dir.join("settings.json")
    }

    pub(crate) fn settings(&self) -> Value {
        let mut s = defaults();
        let Ok(text) = std::fs::read_to_string(self.settings_file()) else { return s };
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(saved)) => {
                for (k, v) in saved {
                    s[k] = v;
                }
                s
            }
            // a file that isn't JSON is kept aside, never thrown away
            _ => {
                let _ = std::fs::rename(self.settings_file(), self.dir.join("settings.json.unreadable"));
                s
            }
        }
    }

    /// Writes beside the file and renames, so a crash half way leaves the old settings whole.
    fn save(&self, file: &Path, text: &str) -> std::io::Result<()> {
        if let Some(d) = file.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = file.with_extension("tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, file)
    }

    /// A change the page asks for. Advanced's switches are kept to TORC_* names with text values: the engine gets them
    /// as its environment, so nothing else (PATH, a library path) can be slipped in through them.
    pub(crate) fn settings_set(&self, patch: &Value) -> Value {
        // one change at a time: two at once would each read the old file, and the second write would undo the first
        static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = self.settings();
        if let Value::Object(p) = patch {
            for (k, v) in p {
                s[k] = if k == "env" { switches_only(v) } else { v.clone() };
            }
        }
        let _ = self.save(&self.settings_file(), &serde_json::to_string_pretty(&s).unwrap_or_default());
        s
    }
}

/// The TORC_* names with text values out of an env object; anything else is dropped
fn switches_only(env: &Value) -> Value {
    let named = |k: &str| {
        k.strip_prefix("TORC_").is_some_and(|r| {
            !r.is_empty() && r.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        })
    };
    let kept = env.as_object().into_iter().flatten().filter(|(k, v)| named(k) && v.is_string());
    Value::Object(kept.map(|(k, v)| (k.clone(), v.clone())).collect())
}

// ---- chats and primers, as plain files

fn valid_id(id: &Value) -> Option<String> {
    let id = id.as_str()?;
    (!id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
        .then(|| id.to_string())
}

impl Shell {
    /// `name` beside the app when it runs portable, else in the app's data folder (also when the portable folder
    /// could not be written and the app fell back to the system's folder)
    pub(crate) fn beside(&self, name: &str) -> PathBuf {
        match portable_root() {
            Some(r) if self.dir == r.join("data") => r.join(name),
            _ => self.dir.join(name),
        }
    }

    /// The MCP servers, ToolFunnel among them: mcps beside a portable app, else in the data folder
    pub(crate) fn mcps_dir(&self) -> PathBuf {
        self.beside("mcps")
    }

    /// Node: tools beside a portable app, else the data folder itself
    pub(crate) fn tools_dir(&self) -> PathBuf {
        match portable_root() {
            Some(r) if self.dir == r.join("data") => r.join("tools"),
            _ => self.dir.clone(),
        }
    }

    fn chat_dir(&self) -> PathBuf {
        match self.settings()["chatDir"].as_str() {
            Some(d) if !d.is_empty() => PathBuf::from(d),
            _ => self.beside("chats"),
        }
    }

    fn chat_read(&self, id: &str) -> Option<Value> {
        let text = std::fs::read_to_string(self.chat_dir().join(format!("{id}.json"))).ok()?;
        let mut c: Value = serde_json::from_str(&text).ok()?;
        (c["format"] == CHAT_FORMAT && c["turns"].is_array()).then(|| {
            c["id"] = json!(id);
            c
        })
    }

    fn chats(&self) -> Value {
        let mut rows = Vec::new();
        for e in std::fs::read_dir(self.chat_dir()).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(id) = name.strip_suffix(".json") else { continue };
            if let Some(c) = self.chat_read(id) {
                rows.push(json!({"id": id, "title": c["title"], "updated": c["updated"], "model": c["model"]}));
            }
        }
        rows.sort_by(|a, b| b["updated"].as_f64().unwrap_or(0.0).total_cmp(&a["updated"].as_f64().unwrap_or(0.0)));
        Value::Array(rows)
    }

    fn chat_put(&self, chat: &Value) -> Value {
        let Some(id) = valid_id(&chat["id"]) else { return json!({"ok": false}) };
        if self.settings()["saveChats"] == false {
            return json!({"ok": false});
        }
        let mut c = chat.clone();
        c["format"] = json!(CHAT_FORMAT);
        let file = self.chat_dir().join(format!("{id}.json"));
        json!({"ok": self.save(&file, &serde_json::to_string(&c).unwrap_or_default()).is_ok()})
    }

    /// Only a file that reads as a ToRc chat is ever deleted.
    fn chat_del(&self, id: &Value) -> Value {
        let Some(id) = valid_id(id) else { return json!({"ok": false}) };
        if self.chat_read(&id).is_none() {
            return json!({"ok": false});
        }
        json!({"ok": std::fs::remove_file(self.chat_dir().join(format!("{id}.json"))).is_ok()})
    }

    fn primers(&self) -> Value {
        let mut out = Vec::new();
        for e in std::fs::read_dir(self.beside("primers")).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(id) = name.strip_suffix(".md") else { continue };
            let Ok(raw) = std::fs::read_to_string(e.path()) else { continue };
            let (first, rest) = raw.split_once('\n').unwrap_or((&raw, ""));
            match first.trim().strip_prefix("# ") {
                Some(n) => out.push(json!({"id": id, "name": n, "text": rest.trim_start_matches(['\r', '\n'])})),
                None => out.push(json!({"id": id, "name": id, "text": raw})),
            }
        }
        out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        Value::Array(out)
    }

    fn primer_put(&self, p: &Value) -> Value {
        let name = p["name"].as_str().unwrap_or("Primer").replace(['\r', '\n'], " ");
        let id = match valid_id(&p["id"]) {
            Some(id) => id,
            None => {
                let slug: String = name
                    .to_lowercase()
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .filter(|w| !w.is_empty())
                    .collect::<Vec<_>>()
                    .join("-")
                    .chars()
                    .take(40)
                    .collect();
                let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
                format!("{}-{t:x}", if slug.is_empty() { "primer" } else { &slug })
            }
        };
        let file = self.beside("primers").join(format!("{id}.md"));
        let text = format!("# {name}\n\n{}", p["text"].as_str().unwrap_or(""));
        json!({"ok": self.save(&file, &text).is_ok(), "id": id})
    }

    /// The primers folder is the app's own, so every .md in it is a primer (unlike the chats folder, which can be one
    /// the user picked: hence chat_del's check).
    fn primer_del(&self, id: &Value) -> Value {
        let Some(id) = valid_id(id) else { return json!({"ok": false}) };
        json!({"ok": std::fs::remove_file(self.beside("primers").join(format!("{id}.md"))).is_ok()})
    }
}

// ---- the switches, read from SWITCHES.md

fn switches() -> Value {
    let mut out = Vec::new();
    let mut group = String::new();
    for line in SWITCHES_MD.lines() {
        if let Some(g) = line.strip_prefix("## ") {
            group = g.to_string();
        }
        if !line.starts_with("| `") {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 4 {
            continue;
        }
        let names: Vec<String> = cells[1]
            .split('`')
            .filter(|w| {
                w.starts_with("TORC_") && w.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            })
            .map(str::to_string)
            .collect();
        for n in &names {
            let kind = if cells[2].contains("the rest") {
                match cells[2].split(';').find(|p| p.contains(&format!("`{n}`"))) {
                    Some(own) => own.split(':').next().unwrap_or("").trim().to_string(),
                    None => cells[2].split("the rest").nth(1).unwrap_or("").trim().to_string(),
                }
            } else {
                cells[2].to_string()
            };
            out.push(json!({"name": n, "group": group, "kind": kind, "text": cells[3], "row": names}));
        }
    }
    Value::Array(out)
}

// ---- torc-server as a child process

fn answers(port: u16) -> bool {
    let Ok(mut s) = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(2)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = s.write_all(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    let mut head = [0u8; 12];
    s.read_exact(&mut head).is_ok() && head.starts_with(b"HTTP/1.1 200")
}

/// The loaded engine's settings: its profile (GET /torc/settings), or with `profile` that profile read in (POST), for
/// the engine's next start. The engine answers in JSON either way.
fn engine_settings(shell: &Shell, profile: Option<&Value>) -> Value {
    let port = match shell.server.lock().unwrap().as_ref() {
        Some(sv) if sv.ready && !sv.worker => sv.port,
        _ => return json!({"ok": false, "error": "load a model first: these are the running engine's settings"}),
    };
    let Ok(mut s) = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(2)) else {
        return json!({"ok": false, "error": "the engine did not answer"});
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(30)));
    let body = profile.map(Value::to_string).unwrap_or_default();
    let head = match profile {
        Some(_) => format!(
            "POST /torc/settings HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        ),
        None => "GET /torc/settings HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".to_string(),
    };
    let mut got = Vec::new();
    if s.write_all(head.as_bytes())
        .and_then(|_| s.write_all(body.as_bytes()))
        .and_then(|_| s.read_to_end(&mut got))
        .is_err()
    {
        return json!({"ok": false, "error": "the engine did not answer"});
    }
    let text = String::from_utf8_lossy(&got);
    let (status, reply) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let reply: Value = serde_json::from_str(reply).unwrap_or(Value::Null);
    match (status.starts_with("HTTP/1.1 200"), profile) {
        (true, None) => json!({"ok": true, "profile": reply}),
        (true, Some(_)) => {
            json!({"ok": true, "written": reply["written"], "layout": reply["layout"], "running": reply["running"]})
        }
        // an engine from before 0.1.4 has no settings to give
        (false, _) if status.starts_with("HTTP/1.1 404") => {
            json!({"ok": false, "error": "this engine is older than the settings files (ToRc 0.1.4)"})
        }
        (false, _) => json!({"ok": false, "error": reply["error"].as_str().unwrap_or("the engine refused them")}),
    }
}

/// The app's own switch (TORC_DEVTOOLS_PORT) is not an engine switch: kept from the engine and the benchmark, which
/// record every TORC_ variable they see
pub(crate) fn without_app_vars(cmd: &mut Command) -> &mut Command {
    cmd.env_remove("TORC_DEVTOOLS_PORT")
}

/// What the Load screen asks for, read from the saved settings (never from the page).
pub(crate) fn switches_for(s: &Value) -> Vec<(String, String)> {
    let mut own = s["env"].as_object().cloned().unwrap_or_default();
    // the app talks to its server on 127.0.0.1 only, so the server must listen there
    own.remove("TORC_HOST");
    let mut env: Vec<(String, String)> =
        own.iter().map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
    let set = |env: &mut Vec<(String, String)>, k: &str, v: &str| {
        if !own.contains_key(k) {
            env.push((k.into(), v.into()));
        }
    };
    if s["npu"] == true && npu_kind() == "snapdragon" {
        set(&mut env, "TORC_NPU", "1");
    }
    if s["gpu"] == true {
        set(&mut env, "TORC_GPU_ATTN", "1");
    }
    if let Some(w) = s["worker"].as_str().filter(|w| !w.is_empty()) {
        set(&mut env, "TORC_PIPE", w);
    }
    if s["saveChats"] == false {
        set(&mut env, "TORC_MARKS", "0");
    }
    env
}

fn npu_kind() -> &'static str {
    let cpu = cpu_name().to_lowercase();
    if cpu.contains("snapdragon") || cpu.contains("oryon") || cpu.contains("qualcomm") {
        "snapdragon"
    } else if cpu.contains("core(tm) ultra") || cpu.contains("core ultra") {
        "intel"
    } else if cpu.contains("ryzen ai") {
        "amd"
    } else {
        "none"
    }
}

#[cfg(windows)]
fn cpu_name() -> String {
    let out = jobs::quiet(&mut Command::new("reg"))
        .args(["query", r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "/v", "ProcessorNameString"])
        .output();
    out.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default()
}

#[cfg(not(windows))]
fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default()
}

fn lines_of(r: impl Read + Send + 'static, app: AppHandle) {
    std::thread::spawn(move || {
        for line in BufReader::new(r).lines().map_while(Result::ok) {
            let _ = app.emit("torc:log", format!("{line}\n"));
        }
    });
}

/// Tells the page when the engine stops by itself after it was ready. An engine the app stopped itself (to load
/// another model or context, or to unload) is no longer the current one and is not reported: reporting it marked the
/// engine that replaced it as stopped.
pub(crate) fn watch_exit(app: &AppHandle, child: Arc<Mutex<Child>>) {
    let app = app.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let exited = child.lock().unwrap().try_wait();
            match exited {
                Ok(None) => continue,
                Ok(Some(code)) => {
                    let shell = app.state::<Arc<Shell>>();
                    let current = shell.server.lock().unwrap().as_ref().is_some_and(|s| Arc::ptr_eq(&s.child, &child));
                    if current {
                        let _ = app.emit("torc:state", json!({"state": "stopped", "code": code.code()}));
                    }
                }
                Err(_) => {}
            }
            break;
        }
    });
}

impl Shell {
    pub(crate) fn stop_server(&self) {
        if let Some(s) = self.server.lock().unwrap().take() {
            let mut child = s.child.lock().unwrap();
            let _ = child.kill();
            // gone before this returns, so its port is free for the next engine and nothing mistakes it for another
            // program's server
            let _ = child.wait();
        }
    }

    fn load(&self, app: &AppHandle) -> Value {
        let s = self.settings();
        if s["asWorker"] == true {
            return self.start_worker(app, &s);
        }
        let port = s["port"].as_u64().filter(|p| (1..=65535).contains(p)).unwrap_or(0) as u16;
        let ctx = s["ctx"].as_u64().unwrap_or(0);
        if port == 0 {
            return json!({"ok": false, "kept": true, "error": "the port has to be a number from 1 to 65535"});
        }
        if ctx < 1024 {
            return json!({"ok": false, "kept": true, "error": "the context has to be 1024 tokens or more"});
        }
        let bin = PathBuf::from(s["binDir"].as_str().unwrap_or(""));
        let exe = bin.join(format!("torc-server{EXE}"));
        let model = s["lastModel"].as_str().unwrap_or("").to_string();
        if !exe.exists() {
            return json!({"ok": false, "kept": true, "error": format!("torc-server is not in {}", bin.display())});
        }
        if !Path::new(&model).exists() {
            return json!({"ok": false, "kept": true, "error": "that model file is not there"});
        }
        self.stop_server();
        // a server that answers before this one has started is somebody else's
        if answers(port) {
            let error =
                format!("something else is already serving on port {port}. Stop it, or change the port in Settings");
            return json!({"ok": false, "error": error});
        }
        let threads = s["threads"].as_str().map_or_else(|| s["threads"].to_string(), str::to_string);
        let mut cmd = Command::new(&exe);
        without_app_vars(&mut cmd);
        cmd.args([model.as_str(), &port.to_string(), &threads, &ctx.to_string()])
            .envs(switches_for(&s))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return json!({"ok": false, "kept": true, "error": format!("torc-server would not start: {e}")}),
        };
        if let Some(err) = child.stderr.take() {
            lines_of(err, app.clone());
        }
        let child = Arc::new(Mutex::new(child));
        *self.server.lock().unwrap() = Some(Server {
            child: child.clone(),
            port,
            ready: false,
            worker: false,
            cfg: s.clone(),
            model: model.clone(),
        });
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(LOAD_SECONDS) {
            if let Ok(Some(_)) = child.lock().unwrap().try_wait() {
                *self.server.lock().unwrap() = None;
                return json!({"ok": false, "error": "the engine stopped while it was loading the model"});
            }
            if answers(port) {
                if let Some(sv) = self.server.lock().unwrap().as_mut() {
                    sv.ready = true;
                }
                watch_exit(app, child);
                let name = Path::new(&model).file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                return json!({"ok": true, "name": name});
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        self.stop_server();
        json!({"ok": false, "error": "the engine did not come up"})
    }
}

// ---- a chat turn

/// One request to the server, streamed: each piece of text goes to the page as it comes. What comes back: the text,
/// the tool calls, the timings and usage, and why it ended; or the error.
fn request(shell: &Shell, app: &AppHandle, port: u16, messages: &[Value], opts: &Value, tools: &[Value]) -> Value {
    // Sampled, never greedy: a thinking model decoded greedily can repeat itself without end. The values are Qwen's
    // recommended ones, with and without thinking; the page can send its own
    let (temp, top_p) = if opts["think"] == false { (0.7, 0.8) } else { (0.6, 0.95) };
    let mut body = json!({
        "model": "local", "stream": true, "stream_options": {"include_usage": true}, "messages": messages,
        "max_tokens": opts["max"].as_u64().unwrap_or(1024),
        "temperature": opts["temperature"].as_f64().unwrap_or(temp),
        "top_p": opts["top_p"].as_f64().unwrap_or(top_p),
        "top_k": opts["top_k"].as_u64().unwrap_or(20),
        "presence_penalty": opts["presence_penalty"].as_f64().unwrap_or(0.0),
        "frequency_penalty": opts["frequency_penalty"].as_f64().unwrap_or(0.0),
    });
    if opts["think"] == false {
        body["chat_template_kwargs"] = json!({"enable_thinking": false});
    }
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    let body = body.to_string();
    let mut s = match TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(e) => return json!({"ok": false, "error": e.to_string()}),
    };
    *shell.turn.lock().unwrap() = s.try_clone().ok();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    if s.write_all(head.as_bytes()).and_then(|_| s.write_all(body.as_bytes())).is_err() {
        return json!({"ok": false, "error": "the engine did not take the request"});
    }
    let quiet = opts["quiet"] == true;
    let mut r = BufReader::new(s);
    let mut status = String::new();
    let _ = r.read_line(&mut status);
    let ok_status = status.starts_with("HTTP/1.1 200");
    // past the head
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
            break;
        }
    }
    let (mut text, mut calls, mut timings, mut usage, mut finish, mut error) =
        (String::new(), Vec::new(), Value::Null, Value::Null, String::new(), String::new());
    if !ok_status {
        let mut rest = String::new();
        let _ = r.read_to_string(&mut rest);
        let j: Value = serde_json::from_str(&rest).unwrap_or(Value::Null);
        let e = j["error"]["message"].as_str().or(j["error"].as_str()).unwrap_or(rest.trim());
        return json!({"ok": false, "error": e});
    }
    let mut ended = false;
    for line in r.lines() {
        let Ok(line) = line else { break };
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data.contains("[DONE]") {
            ended = true;
            continue;
        }
        let Ok(j) = serde_json::from_str::<Value>(data) else { continue };
        if !j["error"].is_null() {
            error = j["error"]["message"].as_str().or(j["error"].as_str()).unwrap_or("the engine failed").to_string();
        }
        let choice = &j["choices"][0];
        let delta = &choice["delta"];
        let (c, th) = (delta["content"].as_str().unwrap_or(""), delta["reasoning_content"].as_str().unwrap_or(""));
        if !c.is_empty() || !th.is_empty() {
            text.push_str(c);
            if !quiet {
                let _ = app.emit("torc:delta", json!({"text": c, "thinking": th}));
            }
        }
        if let Some(tc) = delta["tool_calls"].as_array() {
            calls.extend(tc.iter().cloned());
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            finish = f.to_string();
        }
        if !j["timings"].is_null() {
            timings = j["timings"].clone();
        }
        if !j["usage"].is_null() {
            usage = j["usage"].clone();
        }
    }
    *shell.turn.lock().unwrap() = None;
    if shell.stopped.load(Ordering::SeqCst) {
        return json!({"ok": true, "stopped": true, "text": text, "calls": [], "timings": timings, "usage": usage});
    }
    if !error.is_empty() {
        return json!({"ok": false, "error": error, "text": text});
    }
    if !ended && finish.is_empty() {
        let error = "the engine closed the connection before the reply was finished";
        return json!({"ok": false, "error": error, "text": text});
    }
    json!({"ok": true, "text": text, "calls": calls, "timings": timings, "usage": usage, "finish": finish})
}

impl Shell {
    fn turn(&self, app: &AppHandle, messages: Value, opts: Value) -> Value {
        self.stopped.store(false, Ordering::SeqCst);
        let port = match self.server.lock().unwrap().as_ref() {
            Some(s) if s.ready && !s.worker => s.port,
            _ => return json!({"ok": false, "error": "no model is loaded"}),
        };
        let s = self.settings();
        // the built-in tools and the MCP servers', synced first: the toolbox is built afresh for every turn
        let specs = if opts["plain"] == true {
            Vec::new()
        } else {
            let mut v = basic_specs(&s);
            v.extend(self.mcp_specs());
            v
        };
        let mut messages = messages.as_array().cloned().unwrap_or_default();
        let mut trail: Vec<Value> = Vec::new();
        let mut reply = Value::Null;
        // the basic tools the model was offered (only those run), and what ctx reports: the context the server was
        // started with and the tokens the last round used
        let offered = |name: &str| specs.iter().any(|t| t["function"]["name"] == name);
        let size = self.server.lock().unwrap().as_ref().and_then(|sv| sv.cfg["ctx"].as_u64()).unwrap_or(0);
        let mut compact = false;
        let mut wakes: Vec<Value> = Vec::new();
        for _ in 0..MAX_ROUNDS {
            reply = request(self, app, port, &messages, &opts, &specs);
            reply["trail"] = json!(trail);
            reply["compact"] = json!(compact);
            reply["wake"] = json!(wakes);
            let calls = reply["calls"].as_array().cloned().unwrap_or_default();
            if reply["ok"] != true || reply["stopped"] == true || calls.is_empty() {
                return reply;
            }
            let u = &reply["usage"];
            let used = u["prompt_tokens"].as_u64().unwrap_or(0) + u["completion_tokens"].as_u64().unwrap_or(0);
            let asked = json!({"role": "assistant", "content": reply["text"], "tool_calls": calls});
            messages.push(asked.clone());
            let mut round = vec![asked];
            for call in &calls {
                let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                let args: Value =
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
                let result = if BASIC.iter().any(|(n, ..)| *n == name) && offered(&name) {
                    compact |= name == "compact";
                    if let (true, Ok((minutes, message))) = (name == "wake", basic::wake(&args)) {
                        wakes.push(json!({"minutes": minutes, "message": message}));
                    }
                    run_basic(&name, &args, &basic::Context { used, size }).unwrap_or_else(|e| format!("error: {e}"))
                } else {
                    self.run_mcp(app, &name, &args).unwrap_or_else(|| format!("error: there is no tool called {name}"))
                };
                let answer = json!({"role": "tool", "tool_call_id": call["id"], "content": result});
                messages.push(answer.clone());
                round.push(answer);
                let _ = app.emit(
                    "torc:tooluse",
                    json!({"name": name, "args": call["function"]["arguments"], "result": result}),
                );
            }
            if self.stopped.load(Ordering::SeqCst) {
                return json!({"ok": true, "stopped": true, "text": reply["text"], "calls": [], "trail": trail,
                    "compact": compact});
            }
            trail.extend(round);
        }
        reply["calls"] = json!([]);
        reply["trail"] = json!(trail);
        reply["compact"] = json!(compact);
        reply["wake"] = json!(wakes);
        reply["note"] = json!(format!("stopped after {MAX_ROUNDS} rounds of tool calls without an answer"));
        reply
    }

    fn stop_turn(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Some(s) = self.turn.lock().unwrap().as_ref() {
            let _ = s.shutdown(Shutdown::Both);
        }
    }
}

// ---- the page's calls

/// A file or folder dialog; a folder dialog opens at `start`.
fn pick(app: &AppHandle, kind: &str, start: PathBuf) -> Value {
    const TEXT: [&str; 17] = [
        "txt", "md", "json", "csv", "log", "rs", "py", "js", "ts", "c", "h", "toml", "yaml", "yml", "xml", "html",
        "css",
    ];
    let d = app.dialog().file();
    let paths: Vec<String> = match kind {
        "folder" => d
            .set_directory(start)
            .blocking_pick_folder()
            .into_iter()
            .filter_map(|p| p.into_path().ok())
            .map(|p| p.display().to_string())
            .collect(),
        "model" => match portable_root().map(|r| r.join("models")).filter(|m| m.is_dir()) {
            Some(m) => d.set_directory(m),
            None => d,
        }
        .add_filter("ToRc or GGUF model", &["torc", "gguf"])
        .blocking_pick_file()
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .map(|p| p.display().to_string())
        .collect(),
        _ => d
            .add_filter("Text", &TEXT)
            .blocking_pick_files()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|p| p.into_path().ok())
            .map(|p| p.display().to_string())
            .collect(),
    };
    json!(paths)
}

/// A Save As dialog, then `text` written to the file chosen (nothing when the dialog is closed).
fn save_text(app: &AppHandle, text: &str, name: &str) -> Value {
    let Some(path) = app.dialog().file().set_file_name(name).blocking_save_file().and_then(|p| p.into_path().ok())
    else {
        return json!({"ok": false, "cancelled": true});
    };
    match std::fs::write(&path, text) {
        Ok(()) => json!({"ok": true, "path": path.display().to_string()}),
        Err(e) => json!({"ok": false, "error": e.to_string()}),
    }
}

fn read_text(file: &str) -> Value {
    match std::fs::metadata(file) {
        Ok(m) if m.len() > MAX_ATTACH_BYTES => json!({"ok": false, "error": "over 4 MB"}),
        Ok(_) => match std::fs::read_to_string(file) {
            Ok(text) => {
                json!({"ok": true, "name": Path::new(file).file_name().map(|n| n.to_string_lossy()), "text": text})
            }
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        },
        Err(e) => json!({"ok": false, "error": e.to_string()}),
    }
}

#[tauri::command]
async fn call(app: AppHandle, shell: State<'_, Arc<Shell>>, name: String, args: Vec<Value>) -> Result<Value, String> {
    let shell = shell.inner().clone();
    let a = move |i: usize| args.get(i).cloned().unwrap_or(Value::Null);
    // the slow ones (a load, a reply, a dialog) run off the main thread
    let out = tauri::async_runtime::spawn_blocking(move || match name.as_str() {
        "info" => {
            let dark = app.get_webview_window("main").and_then(|w| w.theme().ok()) == Some(tauri::Theme::Dark);
            let chats = shell.chat_dir().display().to_string();
            json!({"version": env!("CARGO_PKG_VERSION"), "npu": npu_kind(), "dark": dark, "chatsDir": chats})
        }
        "settings" => shell.settings(),
        "settingsSet" => shell.settings_set(&a(0)),
        "switches" => switches(),
        "pick" => {
            let start = a(1).as_str().filter(|d| !d.is_empty()).map_or_else(|| shell.chat_dir(), PathBuf::from);
            pick(&app, a(0).as_str().unwrap_or("text"), start)
        }
        "readText" => read_text(a(0).as_str().unwrap_or("")),
        "saveText" => save_text(&app, a(0).as_str().unwrap_or(""), a(1).as_str().unwrap_or("code.txt")),
        "load" => shell.load(&app),
        "unload" => {
            shell.stop_server();
            json!(true)
        }
        "chat" => {
            let r = shell.turn(&app, a(0), a(1));
            // after the last piece, a mark the same way as the pieces: the page then knows it has them all
            let _ = app.emit("torc:delta", json!({"end": true}));
            r
        }
        "answer" => {
            shell.answer(&a(0), &a(1));
            json!(true)
        }
        "job" => shell.run_job(&app, a(0).as_str().unwrap_or(""), &a(1)),
        "engineSettings" => engine_settings(&shell, None),
        "engineSettingsSet" => engine_settings(&shell, Some(&a(0))),
        "ggufCheck" => shell.gguf_check(a(0).as_str().unwrap_or("")),
        "modelArch" => shell.model_arch(a(0).as_str().unwrap_or("")),
        "chats" => shell.chats(),
        "chatGet" => valid_id(&a(0)).and_then(|id| shell.chat_read(&id)).unwrap_or(Value::Null),
        "chatPut" => shell.chat_put(&a(0)),
        "chatDel" => shell.chat_del(&a(0)),
        "primers" => shell.primers(),
        "primerPut" => shell.primer_put(&a(0)),
        "primerDel" => shell.primer_del(&a(0)),
        "tools" => json!(BASIC.iter().map(|(n, t, on)| json!({"name": n, "text": t, "on": on})).collect::<Vec<_>>()),
        "mcp" => shell.sync(&app),
        "mcpDiscover" => tools::discover(&app, a(0).as_str().unwrap_or("")),
        "tfPlace" => shell.tf_place(a(0).as_str().unwrap_or(""), a(1).as_str().unwrap_or("")),
        "tfHave" => json!(shell.tf_installed()),
        "tfLatest" => json!(shell.tf_latest()),
        "tfInstall" => shell.tf_install(&app),
        "tfUninstall" => shell.tf_uninstall(&app),
        "tfManage" => shell.manage(&a(0), &a(1)),
        "addresses" => {
            json!(jobs::interfaces().into_iter().map(|(n, a)| json!({"name": n, "address": a})).collect::<Vec<_>>())
        }
        other => json!({"ok": false, "error": format!("{other}: not a call this shell knows")}),
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok(out)
}

#[tauri::command]
fn stop(app: AppHandle, shell: State<'_, Arc<Shell>>) {
    shell.stop_turn();
    shell.refuse_waiting(&app);
}

#[tauri::command]
fn job_stop(shell: State<'_, Arc<Shell>>) {
    shell.stop_job();
}

/// What both windows share, being one browser profile: its folder (in data, when portable) and its arguments.
fn browser<'a, R: tauri::Runtime, M: Manager<R>>(
    w: WebviewWindowBuilder<'a, R, M>,
    cache: Option<PathBuf>,
) -> WebviewWindowBuilder<'a, R, M> {
    let w = match cache {
        Some(d) => w.data_directory(d),
        None => w,
    };
    // TORC_DEVTOOLS_PORT=n: the page on WebView2's DevTools port, for the test that drives it. Tauri sets its own
    // browser arguments, so WebView2's environment variable for them is not read
    #[cfg(windows)]
    let w = match std::env::var("TORC_DEVTOOLS_PORT") {
        Ok(p) => w.additional_browser_args(&format!(
            "--remote-debugging-port={p} --disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection"
        )),
        Err(_) => w,
    };
    w
}

/// Shows the app's window once the splash has had its time (or at once, when there is no splash), and fades the splash
/// out over it.
fn reveal(main: tauri::WebviewWindow, started: Instant) {
    let Some(splash) = main.get_webview_window("splash") else {
        REVEALED.store(true, Ordering::SeqCst);
        let _ = main.show();
        return;
    };
    std::thread::spawn(move || {
        std::thread::sleep(SPLASH.saturating_sub(started.elapsed()));
        let _ = splash.eval("document.body.classList.add('leaving')");
        REVEALED.store(true, Ordering::SeqCst);
        let _ = main.show();
        let _ = main.set_focus();
        std::thread::sleep(Duration::from_millis(380));
        let _ = splash.close();
    });
}

/// Set once the app's window is shown: a splash that draws later than that stays hidden.
static REVEALED: AtomicBool = AtomicBool::new(false);

/// Says why the app cannot start, where someone will see it: a message box on Windows (a windowed program has no
/// console), the terminal elsewhere.
fn fatal(text: &str) {
    #[cfg(windows)]
    {
        #[link(name = "user32")]
        unsafe extern "system" {
            fn MessageBoxW(owner: *mut std::ffi::c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
        }
        let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
        let (t, c) = (wide(text), wide("ToRc"));
        const MB_ICONERROR: u32 = 0x10;
        // SAFETY: both strings are NUL-terminated UTF-16 that outlive the call; no owner window
        unsafe { MessageBoxW(std::ptr::null_mut(), t.as_ptr(), c.as_ptr(), MB_ICONERROR) };
    }
    #[cfg(not(windows))]
    eprintln!("{text}");
}

fn main() {
    let run = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // portable: settings, Node, ToolFunnel, the benchmark and the window's own cache in data, chats in
            // chats, beside the app. Otherwise the system's folder for app data.
            // a portable folder that cannot be written to (unpacked into Program Files, say) falls back, and says so
            let writable = |d: &Path| {
                let probe = d.join(".write-test");
                let ok = std::fs::write(&probe, b"").is_ok();
                let _ = std::fs::remove_file(&probe);
                ok
            };
            let root = portable_root();
            let portable = root.as_ref().is_some_and(|r| writable(&r.join("data")));
            let dir =
                if portable { root.clone().unwrap_or_default().join("data") } else { app.path().app_data_dir()? };
            if let (Some(r), false) = (&root, portable) {
                app.dialog()
                    .message(format!(
                        "ToRc cannot write to {}, so your settings and chats are kept in {} instead. To keep \
                         everything in one folder, move ToRc to a folder of your own, such as one in Documents.",
                        r.display(),
                        dir.display()
                    ))
                    .title("ToRc")
                    .show(|_| {});
            }
            app.manage(Arc::new(Shell {
                dir: dir.clone(),
                server: Mutex::new(None),
                turn: Mutex::new(None),
                stopped: AtomicBool::new(false),
                job: Mutex::new(None),
                servers: Mutex::new(Vec::new()),
                syncing: Mutex::new(()),
                waiting: Mutex::new(std::collections::HashMap::new()),
                asked: std::sync::atomic::AtomicU64::new(0),
            }));
            let cache = portable.then(|| dir.join("webview"));
            let webview = |w| browser(w, cache.clone());
            // the splash, while the app's page loads behind it. Each window shows once its page has drawn, so
            // neither flashes white in dark mode. A splash that cannot open costs nothing: the app shows without it
            let splash = WebviewWindowBuilder::new(app, "splash", WebviewUrl::App("renderer/splash.html".into()))
                .title("ToRc")
                .inner_size(620.0, 300.0)
                .resizable(false)
                .decorations(false)
                .center()
                .visible(false)
                .on_page_load(|w, p| {
                    if p.event() == PageLoadEvent::Finished && !REVEALED.load(Ordering::SeqCst) {
                        let _ = w.show();
                    }
                });
            let _ = webview(splash).build();
            // a page that never finishes loading would leave nothing on screen: after a while the window shows anyway
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(15));
                if !REVEALED.swap(true, Ordering::SeqCst) {
                    if let Some(w) = handle.get_webview_window("main") {
                        let _ = w.show();
                    }
                    if let Some(s) = handle.get_webview_window("splash") {
                        let _ = s.close();
                    }
                }
            });
            let started = Instant::now();
            let main = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("renderer/index.html".into()))
                .title("ToRc")
                .inner_size(1280.0, 840.0)
                .min_inner_size(900.0, 600.0)
                .visible(false)
                .initialization_script(BRIDGE)
                .on_page_load(move |w, p| {
                    if p.event() == PageLoadEvent::Finished {
                        reveal(w, started);
                    }
                });
            webview(main).build()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![call, stop, job_stop])
        .on_window_event(|w, e| {
            match e {
                // a file dropped on the window: its path goes to the page, which attaches it
                tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) => {
                    let p: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
                    let _ = w.emit("torc:drop", p);
                }
                // everything started from here stops with the app: the engine, a job and the tool servers
                tauri::WindowEvent::Destroyed if w.label() == "main" => {
                    if let Some(shell) = w.try_state::<Arc<Shell>>() {
                        shell.stop_job();
                        shell.stop_server();
                        shell.close_servers();
                    }
                }
                _ => {}
            }
        })
        .run(tauri::generate_context!());
    if let Err(e) = run {
        fatal(&format!(
            "ToRc could not open its window ({e}).\n\nOn Windows the window is Microsoft's WebView2 Runtime, which Windows \
             11 has built in and most Windows 10 machines have from Edge's updates; where it is missing (Windows 10 \
             LTSC, Windows Server), install it from https://developer.microsoft.com/microsoft-edge/webview2/. On Linux \
             it is WebKitGTK 4.1 (on Ubuntu or Debian: sudo apt install libwebkit2gtk-4.1-0)."
        ));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn env_keeps_only_switches() {
        let env = serde_json::json!({"TORC_NPU": "1", "PATH": "x", "TORC_X": 1, "torc_lower": "1", "TORC_": "1"});
        assert_eq!(super::switches_only(&env), serde_json::json!({"TORC_NPU": "1"}));
    }
}
