/*
 * ToRc - torc-server, the chat server
 *
 *                                                                                    ,---,_          ,
 *     Author: Rendeverance                                                           _>   `'-.  .--'/
 *                                                                               .--'` ._      `/   <_
 *                                                                                 >,-' ._'.. ..__ . ' '-.
 *                                                                             .-'   .'`         `'.     '.
 *                                                                              >   / >`-.     .-'< \ , '._\
 *                                                                             /    ; '-._>   <_.-' ;  '._>
 *                                                                            `>  ,/  /___\ /___\  \_  /
 *                                                                             `.-|(|  \o_/  \o_/   |)|`
 *                                                                                 \;        \      ;/
 *                                                                                   \  .-,   )-.  /
 *                                                                                    /`  .'-'.  `\
 *                                                                                   ;_.-`.___.'-.;
 *
 *      torc-server.rs
 *      Big models, small machines.
 */
//! torc-server <model> [port] [threads] [ctx] [name]: a chat server for every model ToRc runs.
//!
//! One conversation at a time, over HTTP: GET /v1/models, and POST /v1/chat/completions (OpenAI), /v1/messages
//! (Anthropic) and /v1/responses (OpenAI Responses), each as one JSON reply or as that API's stream of events. This
//! file is the HTTP side. The model answers through chat/serve.rs, each family from its own folder under models/,
//! picked by the file's architecture.

use model::api;
use model::chat::serve::{Reply, Serve, tool_calls};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// OpenAI's usage block. cached_tokens is how many prompt tokens were not read again (from the state kept from the last
/// request, its mark or the system prompt's).
fn usage(np: usize, nr: usize, nt: usize, kept: usize) -> Value {
    json!({"prompt_tokens": np, "completion_tokens": nr, "total_tokens": np + nr,
        "prompt_tokens_details": {"cached_tokens": kept}, "completion_tokens_details": {"reasoning_tokens": nt}})
}

/// The engine's own timing of a reply, in llama-server's form: the prompt tokens read and the time reading them, the
/// reply tokens and the time from the first token going out to the end, and the time to the first token (ttft_ms).
/// No HTTP or client time is in it.
fn timings((pn, ps, ft, rn, rs): (usize, f64, f64, usize, f64)) -> Value {
    let per = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
    json!({"prompt_n": pn, "prompt_ms": ps * 1e3, "prompt_per_second": per(pn, ps), "ttft_ms": ft * 1e3,
        "predicted_n": rn, "predicted_ms": rs * 1e3, "predicted_per_second": per(rn, rs)})
}

fn respond(s: &mut TcpStream, status: &str, ctype: &str, body: &str) {
    // the charset named: a client that guesses (Windows PowerShell 5.1 reads ISO-8859-1) shows UTF-8 text wrongly
    let _ = write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

/// The largest request body read (a chat with long attachments runs to a few MB).
const MAX_BODY: usize = 16 << 20;

/// A request's first line and headers together may take this many bytes.
const MAX_HEAD: usize = 64 << 10;

/// A whole request must arrive within this long.
const REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);

/// The connection while a request is read. Each read waits at most until the request's deadline (and never more than
/// 10 s), so a client that sends its head a byte at a time can't hold the server past REQUEST_DEADLINE: read_line
/// would otherwise keep waiting as long as each byte comes inside the socket's timeout.
struct Deadline {
    s: TcpStream,
    end: std::time::Instant,
}

impl Read for Deadline {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.end.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return Err(std::io::ErrorKind::TimedOut.into());
        }
        self.s.set_read_timeout(Some(left.min(std::time::Duration::from_secs(10))))?;
        self.s.read(buf)
    }
}

/// One line of a request's head, within what is left of MAX_HEAD. None when the head is too long, the line is cut
/// short or the read fails.
fn head_line(r: &mut BufReader<Deadline>, left: &mut usize) -> Option<String> {
    // nothing left: a head that has used all of MAX_HEAD without ending is too long, not finished
    if *left == 0 {
        return None;
    }
    let mut h = String::new();
    let n = r.by_ref().take(*left as u64).read_line(&mut h).ok()?;
    *left -= n;
    if n > 0 && !h.ends_with('\n') {
        return None;
    }
    Some(h)
}

/// Whether an Origin header's value names this machine (a page served from it), as against another site's page.
fn local_origin(v: &str) -> bool {
    let host = v.trim().split("://").nth(1).unwrap_or("");
    let host = host.split('/').next().unwrap_or("");
    let name =
        if host.starts_with('[') { host.split(']').next().unwrap_or("") } else { host.split(':').next().unwrap_or("") };
    matches!(name, "localhost" | "127.0.0.1" | "[::1")
}

fn handle(e: &mut dyn Serve, mut s: TcpStream) {
    // one connection at a time, so a client that stalls before finishing its request must not hold the server (the
    // whole request has REQUEST_DEADLINE, see Deadline), and nor must one that stops reading the reply without closing
    // (a write that can't go out in a minute ends it)
    let _ = s.set_write_timeout(Some(std::time::Duration::from_secs(60)));
    let Ok(rs) = s.try_clone() else { return };
    let mut r = BufReader::new(Deadline { s: rs, end: std::time::Instant::now() + REQUEST_DEADLINE });
    let mut left = MAX_HEAD;
    let Some(line) = head_line(&mut r, &mut left) else { return };
    let (method, path) = {
        let mut it = line.split_whitespace();
        (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
    };
    let (mut len, mut foreign) = (0usize, false);
    loop {
        let Some(h) = head_line(&mut r, &mut left) else { return };
        if h == "\r\n" || h.is_empty() {
            break;
        }
        let low = h.to_ascii_lowercase();
        if let Some(v) = low.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        if let Some(v) = low.strip_prefix("origin:") {
            foreign = !local_origin(v);
        }
    }
    // a browser names the page a request comes from, and another site's page is refused, whatever address the server
    // listens on: it could otherwise run the model from any open tab. Clients that aren't web pages send no Origin
    if foreign {
        respond(
            &mut s,
            "403 Forbidden",
            "application/json",
            &json!({"error": "requests from another site's web page are refused"}).to_string(),
        );
        return;
    }
    if len > MAX_BODY {
        respond(
            &mut s,
            "413 Payload Too Large",
            "application/json",
            &json!({"error": format!("request body over {} MB", MAX_BODY >> 20)}).to_string(),
        );
        return;
    }
    let mut body = vec![0u8; len];
    let mut got = 0;
    while got < len {
        match r.read(&mut body[got..]) {
            Ok(n) if n > 0 => got += n,
            _ => return,
        }
    }
    // the path without its query (some clients ask for /v1/messages?beta=true)
    let path = path.split('?').next().unwrap_or("");
    match (method.as_str(), path) {
        ("GET", "/v1/models") => {
            let j = json!({"object": "list", "data": [{"id": e.name(), "object": "model", "owned_by": "torc"}]});
            respond(&mut s, "200 OK", "application/json", &j.to_string());
        }
        ("POST", "/v1/chat/completions" | "/v1/messages" | "/v1/responses" | "/v1/messages/count_tokens") => {
            let Ok(req) = serde_json::from_slice::<Value>(&body) else {
                return respond(&mut s, "400 Bad Request", "application/json", r#"{"error":"bad json"}"#);
            };
            let (a, c) = match path {
                "/v1/chat/completions" => (Api::Chat, api::from_chat_completions(&req)),
                "/v1/responses" => (Api::Responses, api::from_responses(&req)),
                _ => (Api::Messages, api::from_messages(&req)),
            };
            if path == "/v1/messages/count_tokens" {
                let n = e.count(&c);
                return respond(&mut s, "200 OK", "application/json", &json!({"input_tokens": n}).to_string());
            }
            complete(e, &mut s, a, c);
        }
        _ => respond(&mut s, "404 Not Found", "application/json", r#"{"error":"not found"}"#),
    }
}

/// The API a request came in by, which decides the shape of its reply.
#[derive(Clone, Copy, PartialEq)]
enum Api {
    /// OpenAI chat completions
    Chat,
    /// Anthropic Messages
    Messages,
    /// OpenAI Responses
    Responses,
}

/// The HTTP status for a reply that failed: a prompt that does not fit the context is the request's fault, anything
/// else the server's.
fn status(e: &format::Error) -> &'static str {
    if model::chat::serve::is_too_long(e) { "400 Bad Request" } else { "500 Internal Server Error" }
}

/// Runs a request and writes its reply in the shape of the API it came in by: one JSON body, or that API's stream of
/// events.
fn complete(e: &mut dyn Serve, s: &mut TcpStream, a: Api, c: api::Chat) {
    let name = e.name().to_string();
    if c.stream {
        let _ = write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        );
    }
    let Ok(mut out) = s.try_clone() else { return };
    // a failed write means the client closed the stream, and run then stops the reply
    let mut send = |t: &str| t.is_empty() || write!(out, "{t}").is_ok() && out.flush().is_ok();
    match a {
        Api::Chat => {
            let id = format!(
                "chatcmpl-{}",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis())
            );
            let chunk = |delta: Value, fin: Value| {
                json!({"id": id, "object": "chat.completion.chunk", "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
            };
            if c.stream {
                send(&format!("data: {}\n\n", chunk(json!({"role": "assistant"}), Value::Null)));
                let r = e.reply(&c, &mut |rs, t| {
                    let d = if rs { json!({"reasoning_content": t}) } else { json!({"content": t}) };
                    send(&format!("data: {}\n\n", chunk(d, Value::Null)))
                });
                match r {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        if !calls.is_empty() {
                            send(&format!(
                                "data: {}\n\n",
                                chunk(json!({"tool_calls": tool_calls(&calls)}), Value::Null)
                            ));
                        }
                        send(&format!("data: {}\n\n", chunk(json!({}), json!(fin))));
                        // the usage chunk (stream_options.include_usage) has no choices, as OpenAI sends it
                        let u = json!({"id": id, "object": "chat.completion.chunk", "model": name, "choices": [],
                            "usage": usage(np, nr, nt, e.kept()), "timings": timings(e.timed())});
                        send(&format!("data: {u}\n\n"));
                    }
                    Err(err) => {
                        send(&format!(
                            "data: {}\n\n",
                            json!({"error": {"message": err.to_string(), "type": "server_error"}})
                        ));
                    }
                }
                send("data: [DONE]\n\n");
            } else {
                let (mut reply, mut reasoning) = (String::new(), String::new());
                match e.reply(&c, &mut |rs, t| {
                    if rs {
                        reasoning.push_str(t)
                    } else {
                        reply.push_str(t)
                    }
                    true
                }) {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        let mut msg = json!({"role": "assistant", "content": reply});
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        if !calls.is_empty() {
                            msg["tool_calls"] = tool_calls(&calls);
                        }
                        let j = json!({"id": id, "object": "chat.completion", "model": name,
                            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                            "usage": usage(np, nr, nt, e.kept()), "timings": timings(e.timed())});
                        respond(s, "200 OK", "application/json", &j.to_string());
                    }
                    Err(err) => {
                        respond(s, status(&err), "application/json", &json!({"error": err.to_string()}).to_string())
                    }
                }
            }
        }
        Api::Messages => {
            let u = |np, nr, nt, kept| api::Usage { prompt: np, cached: kept, output: nr, reasoning: nt };
            if c.stream {
                let mut ms = api::MessagesStream::new(c.show_reasoning);
                let np0 = e.count(&c);
                send(&ms.start(&name, np0));
                let r = e.reply(&c, &mut |rs, t| send(&ms.delta(rs, t)));
                match r {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        send(&ms.finish(&tool_calls(&calls), fin, &u(np, nr, nt, e.kept())));
                    }
                    Err(err) => {
                        send(&api::MessagesStream::error(&err.to_string()));
                    }
                }
            } else {
                let (mut reply, mut reasoning) = (String::new(), String::new());
                match e.reply(&c, &mut |rs, t| {
                    if rs {
                        reasoning.push_str(t)
                    } else {
                        reply.push_str(t)
                    }
                    true
                }) {
                    Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                        let j = api::messages_reply(
                            &name,
                            &reasoning,
                            &reply,
                            &tool_calls(&calls),
                            fin,
                            &u(np, nr, nt, e.kept()),
                            c.show_reasoning,
                        );
                        respond(s, "200 OK", "application/json", &j.to_string());
                    }
                    Err(err) => respond(
                        s,
                        status(&err),
                        "application/json",
                        &api::messages_error("api_error", &err.to_string()).to_string(),
                    ),
                }
            }
        }
        Api::Responses => {
            let mut rs = api::ResponsesStream::new(&name, c.stream, c.show_reasoning);
            send(&rs.start());
            let r = e.reply(&c, &mut |reasoning, t| send(&rs.delta(reasoning, t)));
            match r {
                Ok(Reply { prompt: np, output: nr, reasoning: nt, finish: fin, calls }) => {
                    let u = api::Usage { prompt: np, cached: e.kept(), output: nr, reasoning: nt };
                    let body = rs.finish(&tool_calls(&calls), fin, &u);
                    if c.stream {
                        send(&body);
                    } else {
                        respond(s, "200 OK", "application/json", &body);
                    }
                }
                Err(err) if c.stream => {
                    send(&rs.error(&err.to_string()));
                }
                Err(err) => respond(
                    s,
                    status(&err),
                    "application/json",
                    &json!({"error": {"message": err.to_string(), "type": "server_error"}}).to_string(),
                ),
            }
        }
    }
}

fn main() {
    // every panic's message and backtrace is appended to panic.log next to the executable, and a panic on the main
    // thread (a request's) ends the server there and then, rather than leaving the process alive with its listener
    // closed and nothing in its log. A helper thread's panic carries on as before (the NPU helper's ends only its
    // request)
    let dflt = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        dflt(info);
        let bt = std::backtrace::Backtrace::force_capture();
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let name = std::thread::current().name().unwrap_or("?").to_string();
        if let Some(p) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("panic.log")))
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p)
        {
            let _ = writeln!(f, "{t} {name}: {info}\n{bt}");
        }
        if name == "main" {
            std::process::exit(101);
        }
    }));
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: torc-server <model.torc> [port] [threads|auto] [ctx] [name]");
        eprintln!(
            "       torc-server <model.torc> --tokenize <text>   (the prompt's ids as the server makes them, BOS first)"
        );
        eprintln!("       torc-server --archs                          (the architectures it runs, one line)");
        std::process::exit(2);
    }
    if a[1] == "--archs" {
        println!("{}", model::models::ARCHS.join(" "));
        return;
    }
    // --tokenize: the ids of a text, special tokens matched, with the model's BOS when it has one (for checking the
    // tokenizer against llama-tokenize)
    if a.get(2).map(String::as_str) == Some("--tokenize") {
        let tok = model::tok::Tokenizer::open(&a[1]).unwrap_or_else(|e| {
            eprintln!("torc-server: {e}");
            std::process::exit(1)
        });
        let text = a.get(3).cloned().unwrap_or_default().replace("\\n", "\n");
        match tok.encode(&text, true) {
            Ok(ids) => println!("{:?}", tok.bos().into_iter().chain(ids).collect::<Vec<u32>>()),
            Err(e) => {
                eprintln!("torc-server: {e}");
                std::process::exit(1)
            }
        }
        return;
    }
    let port: u16 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(8080);
    if !cfg!(feature = "gpu") && model::switches::on("TORC_GPU_ATTN") {
        eprintln!("torc-server: TORC_GPU_ATTN is set, but this build has no GPU tier (--features torc-model/gpu)");
    }
    // threads: a number, or auto (or nothing) for every CPU of the fastest core class (pool.rs), never more than there
    // are CPUs
    let asked: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let threads = model::pool::init(asked).threads as i32;
    eprintln!("torc-server: threads {threads}{}", if asked == 0 { " (auto: the fastest cores)" } else { "" });
    let ctx: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(8192);
    // each family answers through its own module (serve.rs), chosen by the file's architecture
    let arch = format::read::Model::open(&a[1]).ok().and_then(|f| {
        format::meta::Meta::parse(&f.metadata).ok().and_then(|mm| mm.str("general.architecture").map(str::to_string))
    });
    let (path, name) = (a[1].as_str(), a.get(5).cloned());
    let mut e = match arch.as_deref() {
        Some("gemma4") => model::models::gemma::serve::open(path, ctx, name),
        Some("gpt-oss") => model::models::gptoss::serve::open(path, ctx, name),
        // every other architecture is one of Model's (models/qwen), which refuses a file it can't run
        _ => model::models::qwen::serve::open(path, ctx, name),
    };
    listen(&mut *e, port);
}

/// Serves requests on the port, one at a time, until the process ends.
fn listen(e: &mut dyn Serve, port: u16) {
    // this machine only, unless TORC_HOST says otherwise (the server has no authentication of its own; a front end in
    // front of it does)
    let host = model::switches::get("TORC_HOST").unwrap_or_else(|| "127.0.0.1".into());
    let l = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|err| {
        eprintln!("torc-server: cannot listen on {host}:{port}: {err}");
        std::process::exit(1)
    });
    eprintln!("torc-server: {} on {host}:{port}", e.name());
    // between requests: after TORC_IDLE_SAVE seconds without one (default 60, 0 = never) the current chat goes to disk
    // as well (save_leaving), so a reload or a restart comes back to it warm
    let idle = model::switches::num::<u64>("TORC_IDLE_SAVE").unwrap_or(60);
    let _ = l.set_nonblocking(true);
    let mut last = std::time::Instant::now();
    loop {
        match l.accept() {
            Ok((c, _)) => {
                // on Windows an accepted socket inherits the listener's mode
                let _ = c.set_nonblocking(false);
                // only a reply counts: a client's model polls between chats do not keep the chat off the disk
                let n = e.replies();
                handle(e, c);
                if e.replies() != n {
                    last = std::time::Instant::now();
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if idle > 0 && last.elapsed().as_secs() >= idle {
                    e.idle();
                    last = std::time::Instant::now() + std::time::Duration::from_secs(86400 * 365);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(err) => {
                eprintln!("torc-server: accept: {err}");
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
}

// The server's guards that need no socket: which pages count as this machine's, and which errors are the caller's.
#[cfg(test)]
mod guards {
    use super::*;

    #[test]
    fn only_this_machine_counts_as_local() {
        for o in [
            "http://localhost",
            "http://localhost:8080",
            "http://127.0.0.1:8080/chat",
            "http://[::1]:8080",
            " http://localhost ",
        ] {
            assert!(local_origin(o), "{o}");
        }
        for o in [
            "https://example.com",
            "http://localhost.example.com",
            "http://127.0.0.1.example.com:8080",
            "null",
            "",
            "file://",
            "http://[::2]:80",
        ] {
            assert!(!local_origin(o), "{o}");
        }
    }

    #[test]
    fn a_prompt_over_the_context_is_the_callers_error() {
        assert_eq!(status(&model::chat::serve::too_long(9000, 8192)), "400 Bad Request");
        assert_eq!(status(&format::Error("the NPU went away".into())), "500 Internal Server Error");
        // an engine fault that happens to mention the context is still the server's
        assert_eq!(status(&format::Error("position past the context".into())), "500 Internal Server Error");
    }
}
