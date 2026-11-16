//! tools.rs - the MCP servers the model may call: the user's own
//! servers (a command line run over stdio, or a URL spoken to over Streamable HTTP), or ToolFunnel when that is chosen.
//! A call to a server the user added is put to the user first; ToolFunnel, which this app installed, has its own policy
//! gate, and its management functions reach the model only while the user allows it.

use crate::Shell;
use crate::jobs::quiet;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};

const PROTOCOL: &str = "2025-06-18";
pub const CALL_SECONDS: u64 = 120;

type Pending = Arc<Mutex<HashMap<u64, Sender<Result<Value, String>>>>>;

enum Link {
    Stdio { child: Arc<Mutex<Child>>, stdin: Arc<Mutex<ChildStdin>>, pending: Pending },
    Http { url: String, session: Option<String>, protocol: Option<String> },
}

/// One MCP server: how it is reached, and what it said about itself.
pub struct Server {
    pub name: String,
    pub target: String,
    pub ours: bool,
    link: Option<Link>,
    pub state: String,
    pub error: String,
    pub tools: Vec<Value>,
    next: u64,
}

/// What a server's settings entry asks for: a command line, or a URL. A portable app starts it in its mcps folder,
/// so a server kept there can be named without its path.
fn command_line(line: &str) -> Command {
    // as Node's spawn with shell: true runs it, cmd /d /s /c "line", the line passed as it is: cmd does not read the
    // backslash-escaped quotes Rust would otherwise put round it
    #[cfg(windows)]
    let mut c = {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        c.args(["/d", "/s", "/c"]).raw_arg(format!("\"{line}\""));
        c
    };
    #[cfg(not(windows))]
    let mut c = {
        let mut c = Command::new("sh");
        c.args(["-c", line]);
        c
    };
    if let Some(d) = crate::portable_dir("mcps").filter(|d| d.is_dir()) {
        c.current_dir(d);
    }
    c
}

impl Server {
    pub fn new(name: &str, target: &str, ours: bool) -> Server {
        Server {
            name: name.into(),
            target: target.into(),
            ours,
            link: None,
            state: "disconnected".into(),
            error: String::new(),
            tools: Vec::new(),
            next: 1,
        }
    }

    /// Starts or reaches the server, then the MCP handshake and its tool list. A failure leaves it in "error" with
    /// the reason, for the Tools screen.
    pub fn connect(&mut self, app: &AppHandle, cmd: Option<Command>) {
        if self.state == "connected" {
            return;
        }
        self.state = "connecting".into();
        self.error.clear();
        let r = self.open(app, cmd).and_then(|_| self.initialize());
        match r {
            Ok(()) => {
                self.state = "connected".into();
                let _ = app.emit("torc:log", format!("{}: {} tools\n", self.name, self.tools.len()));
            }
            Err(e) => {
                let _ = app.emit("torc:log", format!("{}: {e}\n", self.name));
                self.close();
                self.state = "error".into();
                self.error = e;
            }
        }
    }

    fn open(&mut self, app: &AppHandle, cmd: Option<Command>) -> Result<(), String> {
        if self.target.starts_with("http://") || self.target.starts_with("https://") {
            self.link = Some(Link::Http { url: self.target.clone(), session: None, protocol: None });
            return Ok(());
        }
        let mut c = cmd.unwrap_or_else(|| command_line(&self.target));
        c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = quiet(&mut c).spawn().map_err(|e| e.to_string())?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let stdin = Arc::new(Mutex::new(stdin));
        // its answers, matched to their requests by id; a request of its own is refused
        {
            let (pending, stdin, name, app) = (pending.clone(), stdin.clone(), self.name.clone(), app.clone());
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    let Ok(msg) = serde_json::from_str::<Value>(line.trim()) else { continue };
                    let id = msg["id"].as_u64();
                    if let Some(tx) = id.and_then(|i| pending.lock().unwrap().remove(&i)) {
                        let r = if msg["error"].is_null() {
                            Ok(msg["result"].clone())
                        } else {
                            Err(msg["error"]["message"].as_str().unwrap_or("MCP error").to_string())
                        };
                        let _ = tx.send(r);
                    } else if !msg["id"].is_null() && msg["method"].is_string() {
                        let no = json!({"jsonrpc": "2.0", "id": msg["id"],
                            "error": {"code": -32601, "message": "not supported by this client"}});
                        let _ = writeln!(stdin.lock().unwrap(), "{no}");
                    }
                }
                // the server has gone: whatever waits on it is answered now
                for (_, tx) in pending.lock().unwrap().drain() {
                    let _ = tx.send(Err("the server exited".into()));
                }
                let _ = app.emit("torc:log", format!("{name}: stopped\n"));
            });
        }
        if let Some(err) = child.stderr.take() {
            let (name, app) = (self.name.clone(), app.clone());
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    if !line.trim().is_empty() {
                        let _ = app.emit("torc:log", format!("{name}: {line}\n"));
                    }
                }
            });
        }
        self.link = Some(Link::Stdio { child: Arc::new(Mutex::new(child)), stdin, pending });
        Ok(())
    }

    fn initialize(&mut self) -> Result<(), String> {
        let r = self.request(
            "initialize",
            json!({"protocolVersion": PROTOCOL, "capabilities": {},
                "clientInfo": {"name": "torc", "version": env!("CARGO_PKG_VERSION")}}),
            30,
        )?;
        if let Some(Link::Http { protocol, .. }) = &mut self.link {
            *protocol = Some(r["protocolVersion"].as_str().unwrap_or(PROTOCOL).to_string());
        }
        self.notify("notifications/initialized", json!({}))?;
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..20 {
            let p = cursor.as_ref().map_or(json!({}), |c| json!({"cursor": c}));
            let t = self.request("tools/list", p, 30)?;
            tools.extend(t["tools"].as_array().cloned().unwrap_or_default());
            cursor = t["nextCursor"].as_str().map(str::to_string);
            if cursor.is_none() {
                break;
            }
        }
        self.tools = tools;
        Ok(())
    }

    pub fn call(&mut self, tool: &str, args: &Value, secs: u64) -> Result<Value, String> {
        if self.state != "connected" {
            return Err(format!("{} is not connected", self.name));
        }
        self.request("tools/call", json!({"name": tool, "arguments": args}), secs)
    }

    fn request(&mut self, method: &str, params: Value, secs: u64) -> Result<Value, String> {
        let id = self.next;
        self.next += 1;
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        match self.link.as_mut().ok_or("not open")? {
            Link::Stdio { stdin, pending, .. } => {
                let (tx, rx) = channel();
                pending.lock().unwrap().insert(id, tx);
                if writeln!(stdin.lock().unwrap(), "{body}").is_err() {
                    pending.lock().unwrap().remove(&id);
                    return Err("the server is not running".into());
                }
                match rx.recv_timeout(Duration::from_secs(secs)) {
                    Ok(r) => r,
                    Err(_) => {
                        pending.lock().unwrap().remove(&id);
                        Err(format!("{method}: timed out"))
                    }
                }
            }
            Link::Http { url, session, protocol } => {
                let (r, sid) = http_post(url, session.as_deref(), protocol.as_deref(), &body, secs)?;
                if sid.is_some() {
                    *session = sid;
                }
                let r = r.ok_or("no answer in the response")?;
                if !r["error"].is_null() {
                    return Err(r["error"]["message"].as_str().unwrap_or("MCP error").to_string());
                }
                Ok(r["result"].clone())
            }
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        let body = json!({"jsonrpc": "2.0", "method": method, "params": params});
        match self.link.as_mut().ok_or("not open")? {
            Link::Stdio { stdin, .. } => writeln!(stdin.lock().unwrap(), "{body}").map_err(|e| e.to_string()),
            Link::Http { url, session, protocol } => {
                let (_, sid) = http_post(url, session.as_deref(), protocol.as_deref(), &body, 30)?;
                if sid.is_some() {
                    *session = sid;
                }
                Ok(())
            }
        }
    }

    /// Stops the server: stdin closed, half a second to leave, then the whole process tree ended. An HTTP session
    /// is ended with a DELETE that gets two seconds.
    pub fn close(&mut self) {
        match self.link.take() {
            Some(Link::Stdio { child, stdin, .. }) => {
                drop(stdin);
                std::thread::sleep(Duration::from_millis(500));
                let mut c = child.lock().unwrap();
                if let Ok(None) = c.try_wait() {
                    #[cfg(windows)]
                    {
                        let pid = c.id().to_string();
                        let _ = quiet(Command::new("taskkill").args(["/T", "/F", "/PID", &pid])).status();
                    }
                    let _ = c.kill();
                }
            }
            Some(Link::Http { url, session: Some(sid), .. }) => {
                let _ = ureq::delete(&url).set("Mcp-Session-Id", &sid).timeout(Duration::from_secs(2)).call();
            }
            _ => {}
        }
        self.state = "disconnected".into();
        self.tools.clear();
    }
}

/// One POST of the Streamable HTTP transport: the answer to a request (from JSON or from server-sent events) and the
/// session id the server gave, if any.
fn http_post(
    url: &str,
    session: Option<&str>,
    protocol: Option<&str>,
    body: &Value,
    secs: u64,
) -> Result<(Option<Value>, Option<String>), String> {
    let mut req = ureq::post(url)
        .timeout(Duration::from_secs(secs))
        .set("Content-Type", "application/json")
        .set("Accept", "application/json, text/event-stream");
    if let Some(s) = session {
        req = req.set("Mcp-Session-Id", s);
    }
    if let Some(p) = protocol {
        req = req.set("MCP-Protocol-Version", p);
    }
    let res = req.send_string(&body.to_string()).map_err(|e| match e {
        ureq::Error::Status(code, _) => format!("HTTP {code}"),
        other => other.to_string(),
    })?;
    let sid = res.header("mcp-session-id").map(str::to_string);
    if body["id"].is_null() {
        return Ok((None, sid));
    }
    let sse = res.header("content-type").unwrap_or("").contains("text/event-stream");
    let text = res.into_string().map_err(|e| e.to_string())?;
    let msgs: Vec<Value> = if sse {
        text.split("\n\n")
            .filter_map(|ev| {
                let data: Vec<&str> = ev.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim_start).collect();
                serde_json::from_str(&data.join("\n")).ok()
            })
            .collect()
    } else {
        match serde_json::from_str::<Value>(&text).map_err(|e| e.to_string())? {
            Value::Array(a) => a,
            v => vec![v],
        }
    };
    Ok((msgs.into_iter().find(|m| m["id"] == body["id"]), sid))
}

fn result_text(r: &Value) -> String {
    r["content"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .map(|p| {
                    if p["type"] == "text" {
                        p["text"].as_str().unwrap_or("").to_string()
                    } else {
                        format!("[{}]", p["type"].as_str().unwrap_or(""))
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn safe(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

impl Shell {
    /// Brings the running servers in line with the settings: the user's own direct servers with Basic tools,
    /// ToolFunnel alone when it is chosen. A server that failed is tried again. Returns each
    /// server's state for the page.
    pub fn sync(&self, app: &AppHandle) -> Value {
        let _one = self.syncing.lock().unwrap();
        let s = self.settings();
        let mut want: Vec<(String, String, bool, Option<Command>)> = Vec::new();
        if s["tools"] == "toolfunnel" {
            if let Some((target, cmd)) = crate::node::toolfunnel_server(self) {
                want.push(("toolfunnel".into(), target, true, Some(cmd)));
            }
        } else {
            for m in s["mcp"].as_array().into_iter().flatten().filter(|m| m["via"] == "direct") {
                let (n, t) = (m["name"].as_str().unwrap_or(""), m["target"].as_str().unwrap_or(""));
                want.push((n.into(), t.into(), false, None));
            }
        }
        let mut servers = self.servers.lock().unwrap();
        servers.retain_mut(|sv| {
            let keep = want.iter().any(|(n, t, ..)| *n == sv.name && *t == sv.target);
            if !keep {
                sv.close();
            }
            keep
        });
        for (name, target, ours, cmd) in want {
            if !servers.iter().any(|sv| sv.name == name) {
                servers.push(Server::new(&name, &target, ours));
            }
            let sv = servers.iter_mut().find(|sv| sv.name == name).unwrap();
            if sv.state != "connected" {
                sv.connect(app, cmd);
            }
        }
        states(&servers)
    }

    pub fn close_servers(&self) {
        for sv in self.servers.lock().unwrap().iter_mut() {
            sv.close();
        }
        self.servers.lock().unwrap().clear();
    }

    /// A ToolFunnel management function (tf_mcp_add and the like), run by ToRc for the user.
    pub fn manage(&self, name: &Value, args: &Value) -> Value {
        let mut servers = self.servers.lock().unwrap();
        let Some(tf) = servers.iter_mut().find(|sv| sv.name == "toolfunnel" && sv.state == "connected") else {
            return json!({"ok": false, "error": "ToolFunnel is not running"});
        };
        match tf.call("toolfunnel_run_tool", &json!({"name": name, "args": args}), 60) {
            Ok(r) => {
                let text = result_text(&r);
                let bad = r["isError"] == true;
                json!({"ok": !bad, "text": text, "error": if bad { text.clone() } else { String::new() }})
            }
            Err(e) => json!({"ok": false, "error": e}),
        }
    }

    /// The MCP tools the model may call now, as chat completions tool specs, beside the basic ones.
    pub fn mcp_specs(&self) -> Vec<Value> {
        let servers = self.servers.lock().unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for sv in servers.iter().filter(|sv| sv.state == "connected") {
            for t in &sv.tools {
                let name: String = format!("{}__{}", safe(&sv.name), safe(t["name"].as_str().unwrap_or("")))
                    .chars()
                    .take(64)
                    .collect();
                if !seen.insert(name.clone()) {
                    continue;
                }
                let schema = if t["inputSchema"].is_object() {
                    t["inputSchema"].clone()
                } else {
                    json!({"type": "object", "properties": {}})
                };
                let desc: String = t["description"].as_str().unwrap_or("").chars().take(1024).collect();
                out.push(
                    json!({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}),
                );
            }
        }
        out
    }

    /// Runs the MCP tool the model asked for by its spec name, asking the user first when the server is one they
    /// added. None when no server has a tool of that name.
    pub fn run_mcp(&self, app: &AppHandle, spec: &str, args: &Value) -> Option<String> {
        let (server, tool, ours) = {
            let servers = self.servers.lock().unwrap();
            let sv = servers.iter().filter(|sv| sv.state == "connected").find(|sv| {
                sv.tools.iter().any(|t| {
                    let n: String = format!("{}__{}", safe(&sv.name), safe(t["name"].as_str().unwrap_or("")))
                        .chars()
                        .take(64)
                        .collect();
                    n == spec
                })
            })?;
            let t = sv.tools.iter().find(|t| {
                let n: String = format!("{}__{}", safe(&sv.name), safe(t["name"].as_str().unwrap_or("")))
                    .chars()
                    .take(64)
                    .collect();
                n == spec
            })?;
            (
                sv.name.clone(),
                t["name"].as_str().unwrap_or("").to_string(),
                sv.ours && self.settings()["tools"] == "toolfunnel",
            )
        };
        if ours && blocked_setup(&server, &tool, args, &self.settings()) {
            return Some(
                "refused: the user has not allowed the agent to configure ToolFunnel. They can turn it on under Tools."
                    .into(),
            );
        }
        if !ours && !self.ask_user(app, &server, &tool, args) {
            return Some("refused: the user did not allow this call.".into());
        }
        let mut servers = self.servers.lock().unwrap();
        let sv = servers.iter_mut().find(|sv| sv.name == server)?;
        Some(match sv.call(&tool, args, CALL_SECONDS) {
            Ok(r) => format!("{}{}", if r["isError"] == true { "error: " } else { "" }, result_text(&r)),
            Err(e) => format!("error: {e}"),
        })
    }

    /// Puts a call in front of the user and waits for their yes or no (or Stop, which is a no).
    fn ask_user(&self, app: &AppHandle, server: &str, tool: &str, args: &Value) -> bool {
        let (tx, rx) = channel();
        let id = {
            let mut w = self.waiting.lock().unwrap();
            let id = w.len() as u64 + 1 + self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            w.insert(id, tx);
            id
        };
        let _ = app.emit("torc:ask", json!({"id": id, "server": server, "tool": tool, "args": args}));
        rx.recv().unwrap_or(false)
    }

    pub fn answer(&self, id: &Value, allowed: &Value) {
        if let Some(tx) = id.as_u64().and_then(|i| self.waiting.lock().unwrap().remove(&i)) {
            let _ = tx.send(allowed == &json!(true));
        }
    }

    /// Stop refuses every call still waiting for the user, and takes its question down.
    pub fn refuse_waiting(&self, app: &AppHandle) {
        for (id, tx) in self.waiting.lock().unwrap().drain() {
            let _ = tx.send(false);
            let _ = app.emit("torc:ask", json!({"id": id, "done": true}));
        }
    }
}

impl Shell {
    /// A server added through ToolFunnel has to sit inside ToolFunnel's own folder, which its guard keeps every path
    /// to. The script it runs is copied there (its folder with it, when the script has a folder of its own in mcps),
    /// and the command comes back with that argument pointing at the copy. Arguments that are not files stay as they
    /// are: a server started by a package name (npx, uvx) has nothing to copy.
    pub fn tf_place(&self, name: &str, target: &str) -> Value {
        let parts: Vec<String> = split_line(target);
        let Some((command, args)) = parts.split_first() else {
            return json!({"ok": false, "error": "give the server's command"});
        };
        // the name becomes a folder name
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            let error = "a server added to ToolFunnel needs a name of letters, digits, - and _";
            return json!({"ok": false, "error": error});
        }
        let mcps = self.mcps_dir();
        let dest = self.tf_root().join("home").join("mcp").join("servers").join(name);
        let mut out = Vec::new();
        for a in args {
            let p = std::path::Path::new(a);
            let file = if p.is_absolute() { p.to_path_buf() } else { mcps.join(p) };
            if !file.is_file() {
                out.push(a.clone());
                continue;
            }
            // a script in a folder of its own in mcps (a package with its node_modules) takes that whole folder
            // along; a script lying loose in mcps, or anywhere else, goes alone
            let top = file
                .strip_prefix(&mcps)
                .ok()
                .filter(|rel| rel.components().count() > 1)
                .and_then(|rel| rel.components().next())
                .map(|c| mcps.join(c));
            let (copied, inside) = match &top {
                Some(t) => (copy_dir(t, &dest), file.strip_prefix(t).unwrap_or(&file).to_path_buf()),
                None => (copy_file(&file, &dest), std::path::PathBuf::from(file.file_name().unwrap_or_default())),
            };
            if let Err(e) = copied {
                let error = format!("could not copy {} into ToolFunnel's folder: {e}", file.display());
                return json!({"ok": false, "error": error});
            }
            out.push(format!("mcp/servers/{name}/{}", inside.to_string_lossy().replace('\\', "/")));
        }
        json!({"ok": true, "command": command, "args": out})
    }
}

/// A command line split as the page splits it: on spaces, a "quoted part" kept whole without its quotes
fn split_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut any = false;
    for c in line.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                any = true;
            }
            c if c.is_whitespace() && !quoted => {
                if any {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            c => {
                cur.push(c);
                any = true;
            }
        }
    }
    if any {
        out.push(cur);
    }
    out
}

fn copy_file(file: &std::path::Path, dest: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    std::fs::copy(file, dest.join(file.file_name().unwrap_or_default())).map(|_| ())
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let kind = e.file_type()?;
        if kind.is_dir() {
            copy_dir(&e.path(), &to.join(e.file_name()))?;
        } else if kind.is_file() {
            std::fs::copy(e.path(), to.join(e.file_name()))?;
        }
    }
    Ok(())
}

/// The tools a server offers, by starting it once and asking: what ToolFunnel is told to expose when a server is
/// added through it.
pub fn discover(app: &AppHandle, target: &str) -> Value {
    let mut sv = Server::new("discover", target, false);
    sv.connect(app, None);
    let names: Vec<Value> = sv.tools.iter().map(|t| t["name"].clone()).collect();
    let r = if sv.state == "connected" {
        json!({"ok": true, "tools": names})
    } else {
        json!({"ok": false, "error": format!("the server did not start: {}", sv.error)})
    };
    sv.close();
    r
}

fn states(servers: &[Server]) -> Value {
    json!(
        servers
            .iter()
            .map(|sv| json!({"name": sv.name, "state": sv.state, "error": sv.error,
            "tools": sv.tools.iter().map(|t| t["name"].clone()).collect::<Vec<_>>()}))
            .collect::<Vec<_>>()
    )
}

/// ToolFunnel's management functions change what the model may do: they reach it only while the user allows it.
fn blocked_setup(server: &str, tool: &str, args: &Value, s: &Value) -> bool {
    if server != "toolfunnel" || tool != "toolfunnel_run_tool" {
        return false;
    }
    let name = args["name"].as_str().unwrap_or("").trim().to_lowercase();
    name.starts_with("tf_") && name != "tf_list" && s["agentSetup"] != true
}
