//! pipe.rs - two machines as a layer pipeline (qwen4exp, and the block-path architectures).
//!
//! The main machine runs layers 0..S of a block and sends the streams (or residual rows) to the worker - `torc-run
//! <model> serve-pipe=<port>` on the other machine, with the same file - which runs S..L and the head and sends back
//! the logits asked for. A prompt goes block by block, with the main machine already on block k while the worker does
//! block k-1 (Model::prefill_blocks); a token or a verify pass waits for its reply. Every layer runs the same code on
//! the same inputs on whichever machine holds it, so the bits are the same as on one machine. Each machine's expert
//! cache only sees its own layers' experts. The worker keeps the KV and DeltaNet states of its layers, so every pass of
//! a run goes through it - decode too, at one hop each way per token, against a round trip per layer for the expert
//! split (ep.rs).
//!
//! Set TORC_PIPE=host:port on the main machine, and TORC_PIPE_SPLIT=S (by default half the layers). A request is
//! "TPP1", S, pos0, n and want (0 for no logits, 1 for the last row's, 2 for every row's; with n = 0 it is a control
//! request, OP_BEGIN / OP_COMMIT), then n tokens (u32), then n x hc x dm f32 streams. The reply is the worker's compute
//! time for it (microseconds, u32), the count of f32s, then the logits.
//!
//! The hello (protocol 2, first on every connection) is "TPH1", VERSION, the model's identity (a u64 and two u32s:
//! FNV-1a of the file's length, layers, width, streams and tensor count) and the main machine's context. The worker
//! answers 0, or 1 if it has another model or protocol (and then hangs up), followed by its own context. The hello
//! catches a worker that was left running with a different file of the same shape, which would otherwise compute half
//! of every token with the wrong weights without any error. A worker too old to know the hello hangs up on it.
//!
//! Protocol 5: OP_SAVE and OP_LOAD name a mark slot (0 for the server's last prompt, 1 for its system prompt), and
//! "TPK1" asks the worker to save or load its layers' half of a state on its own disk (markfile.rs,
//! Model::worker_mark).

use format::{Result, err};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Mutex, OnceLock};

const MAGIC: u32 = 0x3150_5054;
const HELLO: u32 = 0x3148_5054;
/// A saved-state request ("TPK1", protocol 5): l0 | op << 16 (KEY_SAVE or KEY_LOAD), the key's two halves, then the
/// positions. The worker saves or loads its layers l0.. at those positions (Model::worker_mark) and answers one f32: 1
/// if done (for a load, the file was there and matched), 0 if not.
const KEYOP: u32 = 0x314B_5054;
/// A head request ("THD1", protocol 8): r0, r1, the vector's block count, then its Q8 bytes (Q8s::bytes, k then g). The
/// worker answers with the output head's rows r0..r1 over that vector (Model::head_rows). This is for decoding with the
/// worker's layers pulled back (TORC_HEAD_SPLIT): the worker is idle then, and each row is the same dot product on
/// either machine.
const HEADOP: u32 = 0x3144_4854;
/// The most blocks a head request may carry (a model width of 64k).
const HEAD_MAX_BLOCKS: usize = 256;
pub const KEY_SAVE: u32 = 1;
pub const KEY_LOAD: u32 = 2;
/// The pipe protocol version. Both machines must run the same one: any change to what goes over the wire (a new op,
/// field or frame, or a different layout of the bytes) moves it on, so a mismatched pair refuses at the hello instead
/// of misreading each other.
const VERSION: u32 = 10;
/// In the hello's version word: the main machine is coming back after a broken connection, and the worker keeps its
/// layers' state and marks (so a cut prompt resumes from a checkpoint, see reconnect_resume). Without it, a connection
/// starts clean.
const RESUME: u32 = 1 << 31;

/// This process's model (from Model::load): its identity for the hello, its context, and its shape's hash
static IDENT: OnceLock<(u64, u32, u64)> = OnceLock::new();

/// This process's model identity (0 before Model::load), used as the key of its saved states (markfile).
pub fn ident() -> u64 {
    // mixed with the context, since a state saved under another context has a different indexer layout
    IDENT.get().map_or(0, |&(id, ctx, _)| id ^ (ctx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// The loaded model's shape hash (0 before Model::load): the tune file's key, since a value tuned for a shape holds
/// for every file of that shape, at any context
pub fn model_id() -> u64 {
    IDENT.get().map_or(0, |&(_, _, shape)| shape)
}

/// Records the loaded model's identity (set by Model::load; the first model loaded is the one the pipe carries).
/// The identity is the file's model id (the hash of its contents), so two files of one shape and size are told
/// apart; the shape hash is kept beside it for the tune file.
pub fn set_ident(model: &[u8; 32], file_bytes: u64, n_layer: u32, d_model: u32, hc: u32, n_tensors: usize, n_ctx: u32) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in [file_bytes, n_layer as u64, d_model as u64, hc as u64, n_tensors as u64] {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    let id = u64::from_le_bytes(model[..8].try_into().unwrap());
    let _ = IDENT.set((id, n_ctx, h));
}

/// The worker's address and the connection to it (TORC_PIPE): made on first use, and made again by reconnect().
struct Link {
    addr: String,
    s: Mutex<TcpStream>,
}

static LINK: OnceLock<Option<Link>> = OnceLock::new();

/// TCP keepalive on a pipe connection: probes after 20 s of silence, every 5 s, giving up after 6. A long prompt over a
/// USB4 link can lose the connection on one side only: the worker gets a reset while the main machine's socket stays
/// ESTABLISHED, and its server would wait on it forever. With the probes, that side gets an error within about 50 s
/// (the usual "starting the next request from zero" and a reconnect) instead of hanging.
#[cfg(windows)]
fn keepalive(s: &TcpStream) {
    use std::os::windows::io::AsRawSocket;
    unsafe extern "system" {
        fn setsockopt(s: usize, level: i32, name: i32, val: *const i32, len: i32) -> i32;
    }
    const SOL_SOCKET: i32 = 0xffff;
    const SO_KEEPALIVE: i32 = 0x0008;
    const IPPROTO_TCP: i32 = 6;
    const TCP_KEEPIDLE: i32 = 3;
    const TCP_KEEPCNT: i32 = 16;
    const TCP_KEEPINTVL: i32 = 17;
    let h = s.as_raw_socket() as usize;
    for (level, name, v) in [
        (SOL_SOCKET, SO_KEEPALIVE, 1),
        (IPPROTO_TCP, TCP_KEEPIDLE, 20),
        (IPPROTO_TCP, TCP_KEEPINTVL, 5),
        (IPPROTO_TCP, TCP_KEEPCNT, 6),
    ] {
        // SAFETY: a live socket handle and an i32 option value
        if unsafe { setsockopt(h, level, name, &v, 4) } != 0 {
            eprintln!("two boxes (pipeline): keepalive option {name} not set");
        }
    }
}

/// The same on Linux, with Linux's numbers for the four options.
#[cfg(target_os = "linux")]
fn keepalive(s: &TcpStream) {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn setsockopt(fd: i32, level: i32, name: i32, val: *const i32, len: u32) -> i32;
    }
    const SOL_SOCKET: i32 = 1;
    const SO_KEEPALIVE: i32 = 9;
    const IPPROTO_TCP: i32 = 6;
    const TCP_KEEPIDLE: i32 = 4;
    const TCP_KEEPINTVL: i32 = 5;
    const TCP_KEEPCNT: i32 = 6;
    let fd = s.as_raw_fd();
    for (level, name, v) in [
        (SOL_SOCKET, SO_KEEPALIVE, 1),
        (IPPROTO_TCP, TCP_KEEPIDLE, 20),
        (IPPROTO_TCP, TCP_KEEPINTVL, 5),
        (IPPROTO_TCP, TCP_KEEPCNT, 6),
    ] {
        // SAFETY: a live socket and an i32 option value
        if unsafe { setsockopt(fd, level, name, &v, 4) } != 0 {
            eprintln!("two boxes (pipeline): keepalive option {name} not set");
        }
    }
}

// Roadmap: keepalive on other systems (macOS numbers its options differently)
#[cfg(not(any(windows, target_os = "linux")))]
fn keepalive(_s: &TcpStream) {}

/// A connection to the worker at `a`, with its hello answered. Err((true, ..)) means the worker refused it (another
/// model, or an older torc-run); Err((false, ..)) means there is no connection.
fn connect(a: &str, resume: bool) -> std::result::Result<TcpStream, (bool, String)> {
    let mut s = TcpStream::connect(a).map_err(|e| (false, format!("cannot reach {a}: {e}")))?;
    let _ = s.set_nodelay(true);
    keepalive(&s);
    let Some(&(id, ctx, _)) = IDENT.get() else { return Ok(s) };
    let mut b = Vec::with_capacity(20);
    for v in [HELLO, VERSION | if resume { RESUME } else { 0 }, id as u32, (id >> 32) as u32, ctx] {
        b.extend(v.to_le_bytes());
    }
    let refused = |e: String| {
        (
            true,
            format!(
                "the worker at {a} refused this model: {e} (it serves another file, or is an older \
            torc-run: restart it with this model)"
            ),
        )
    };
    s.write_all(&b).map_err(|e| refused(e.to_string()))?;
    // the answer must come within 30 s, else there is no connection (a worker can still be busy with an old one)
    let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let ok = read_u32(&mut s).map_err(|e| (false, format!("no answer from the worker at {a}: {e}")))?;
    let wctx = read_u32(&mut s).map_err(|e| (false, format!("no answer from the worker at {a}: {e}")))?;
    // every later read waits at most TORC_PIPE_TIMEOUT seconds (default 90; 0 = forever). Keepalive catches a dead
    // link in about 50 s, and this catches a link that is alive but stuck (USB4 with receive coalescing on: frames
    // still cross, a reply never arrives), so waits are not bounded only by TCP; the prompt and the reply then resume
    // from a mark (torc-server). It is generous: a worker's longest legitimate silence is one prompt block or a large
    // state saved to disk, a matter of seconds
    let t = crate::switches::num::<u64>("TORC_PIPE_TIMEOUT").unwrap_or(90);
    let _ = s.set_read_timeout((t > 0).then(|| std::time::Duration::from_secs(t)));
    if ok != 0 {
        return Err(refused("a different model or protocol".into()));
    }
    if wctx < ctx {
        eprintln!("two boxes (pipeline): the worker's context is {wctx}, ours {ctx}: longer chats will fail there");
    }
    Ok(s)
}

/// The connection to the worker (TORC_PIPE), made on first use. A worker that refuses the hello stops the process,
/// because half a model run against another file's other half gives garbage, and one machine may not hold the whole
/// model.
pub fn client() -> Option<&'static Mutex<TcpStream>> {
    LINK.get_or_init(|| {
        let a = crate::switches::get("TORC_PIPE")?;
        match connect(&a, false) {
            Ok(s) => {
                eprintln!("two boxes (pipeline): worker at {a}");
                Some(Link { addr: a, s: Mutex::new(s) })
            }
            Err((true, e)) => {
                eprintln!("two boxes (pipeline): {e}");
                std::process::exit(3)
            }
            // an unreachable worker stops the process too: with a per-machine file this machine holds zeros where the
            // worker's layers are, so carrying on alone would produce garbage. TORC_PIPE_FALLBACK=1 runs on one machine
            // instead, for a file that holds every layer
            Err((false, e)) if crate::switches::on("TORC_PIPE_FALLBACK") => {
                eprintln!("two boxes (pipeline): {e}; one box (TORC_PIPE_FALLBACK=1)");
                None
            }
            Err((false, e)) => {
                eprintln!("two boxes (pipeline): {e}; stopping (TORC_PIPE_FALLBACK=1 runs one box)");
                std::process::exit(3)
            }
        }
    })
    .as_ref()
    .map(|l| &l.s)
}

/// A new connection to the worker after an error on the old one: no replies from an abandoned request are left to read,
/// and the worker's states go back to zero (a new connection is a new run there). The caller starts its sequence again
/// from zero too. With one machine there is nothing to do.
pub fn reconnect() -> Result<()> {
    reconnect_with(false)
}

/// reconnect, with the worker keeping its layers' state and marks (RESUME): the caller goes back to a mark of its own
/// sequence (Model::load_mark) and carries on.
pub fn reconnect_resume() -> Result<()> {
    reconnect_with(true)
}

fn reconnect_with(resume: bool) -> Result<()> {
    let Some(l) = LINK.get().and_then(|o| o.as_ref()) else { return Ok(()) };
    // the old connection is shut first, because the worker serves one connection at a time and only answers the new
    // hello once the old one has ended
    let _ = l.s.lock().unwrap_or_else(|p| p.into_inner()).shutdown(std::net::Shutdown::Both);
    // a link that drops usually comes back within seconds to a minute, so keep trying for about a minute; a refusal
    // (another model or protocol) is final
    let mut tries = 0;
    let s = loop {
        match connect(&l.addr, resume) {
            Ok(s) => break s,
            Err((false, e)) if tries < 12 => {
                tries += 1;
                eprintln!("two boxes (pipeline): {e}; again in 5 s ({tries} of 12)");
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
            Err((_, e)) => return err(format!("two boxes (pipeline): {e}")),
        }
    };
    *l.s.lock().unwrap_or_else(|p| p.into_inner()) = s;
    eprintln!("two boxes (pipeline): connected to the worker again");
    Ok(())
}

/// A shard's layers (format's shard.rs): `<model>.layers` holds "a-b", written by `torc check --layers` once it has
/// verified them. Every other layer's bytes are holes (zeros), so the shards fix the split: the main machine (a = 0)
/// runs 0..b and the worker runs a..; any other split, a shift or a pull would run on zeros.
static SHARD: OnceLock<(usize, usize)> = OnceLock::new();

/// Reads `<model>.layers` if there is one (called from Model::load).
pub fn set_shard(model: &str) {
    let r = std::fs::read_to_string(format!("{model}.layers")).ok().and_then(|v| {
        let (a, b) = v.trim().split_once('-')?;
        Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))
    });
    if let Some((a, b)) = r {
        eprintln!("shard: this file holds layers {a}..{b} only");
        let _ = SHARD.set((a, b));
    }
}

/// The shard's layers (a, b), if the model file is a shard.
pub fn shard() -> Option<(usize, usize)> {
    SHARD.get().copied()
}

/// Set by torc-run before it loads the model to serve a pipeline's later layers (serve-pipe=).
static WORKER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_worker() {
    WORKER.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Set by Model::load before the weights load, for a model that runs over the pipeline (other architectures ignore
/// TORC_PIPE and run every layer here).
static PIPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_piped() {
    PIPED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The layers this machine runs on a pipeline, for what it loads ahead of use: the expert cache's warm set and budget,
/// and the NPU's hot experts. TORC_CACHE_LAYERS=a-b when it is given. Otherwise a shard's layers, or, on the main
/// machine of a model that pipelines (TORC_PIPE, without TORC_PIPE_PULL or TORC_PIPE_FALLBACK, which can leave it
/// running every layer alone), 0..S + 2, and on the worker S - 2..L, S being TORC_PIPE_SPLIT or half the layers. The
/// two layers either side cover a split learned a layer or two off S; for two machines that differ, set
/// TORC_PIPE_SPLIT (or TORC_CACHE_LAYERS on each). None otherwise: the machine loads every layer, as one machine does.
pub fn own_layers(n_layer: usize) -> Option<(usize, usize)> {
    if let Some(v) = crate::switches::get("TORC_CACHE_LAYERS") {
        let (a, b) = v.split_once('-')?;
        let (a, b) = (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?.min(n_layer));
        return (a < b).then_some((a, b));
    }
    if let Some((a, b)) = shard() {
        return Some((a.min(n_layer), b.min(n_layer)));
    }
    let worker = WORKER.load(std::sync::atomic::Ordering::Relaxed);
    let main = PIPED.load(std::sync::atomic::Ordering::Relaxed)
        && crate::switches::given("TORC_PIPE")
        && !crate::switches::on("TORC_PIPE_PULL")
        && !crate::switches::on("TORC_PIPE_FALLBACK");
    if !worker && !main {
        return None;
    }
    let s = crate::switches::num::<usize>("TORC_PIPE_SPLIT").unwrap_or(n_layer / 2).min(n_layer);
    Some(if worker { (s.saturating_sub(2), n_layer) } else { (0, (s + 2).min(n_layer)) })
}

/// The first layer the worker runs: a shard's end on the main machine (TORC_PIPE_SPLIT must agree), else
/// TORC_PIPE_SPLIT, else the balance measured on an earlier run of a model of this shape (save_split), else half the
/// layers.
pub fn split(n_layer: usize) -> usize {
    if let Some((a, b)) = shard() {
        let env = crate::switches::num::<usize>("TORC_PIPE_SPLIT");
        if a != 0 || env.is_some_and(|s| s != b) {
            eprintln!(
                "shard: this file holds layers {a}..{b}, so the split must be {b} (TORC_PIPE_SPLIT {env:?}); stopping"
            );
            std::process::exit(2);
        }
        return b;
    }
    if let Some(s) = crate::switches::num::<usize>("TORC_PIPE_SPLIT") {
        return s.min(n_layer);
    }
    split_file(n_layer)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|v| v.round() as usize)
        .filter(|&s| s > 0 && s < n_layer)
        .unwrap_or(n_layer / 2)
}

/// The loaded model's shape (layers, width, streams), set at load. The saved split is kept per model shape, not per
/// depth alone, since the 80B and Flash Next both have 48 layers.
static SHAPE: OnceLock<String> = OnceLock::new();

pub fn set_shape(n_layer: u32, d_model: u32, hc: u32) {
    let _ = SHAPE.set(format!("{n_layer}-{d_model}-{hc}"));
}

fn split_file(n_layer: usize) -> Option<std::path::PathBuf> {
    let d = crate::tune::cache_dir()?;
    std::fs::create_dir_all(&d).ok()?;
    let key = SHAPE.get().cloned().unwrap_or_else(|| n_layer.to_string());
    Some(d.join(format!("pipe-split-{key}.txt")))
}

/// The saved split's text, if any (torc-tune puts it back after its runs, so its candidates cannot move it).
pub fn saved_split(n_layer: usize) -> Option<String> {
    split_file(n_layer).and_then(|p| std::fs::read_to_string(p).ok())
}

/// Puts the saved split back to `v` (None: no file, as before).
pub fn restore_split(n_layer: usize, v: Option<String>) {
    if let Some(p) = split_file(n_layer) {
        let _ = match v {
            Some(v) => std::fs::write(p, v),
            None => std::fs::remove_file(p),
        };
    }
}

/// After a prompt: this machine's seconds over layers 0..s and the worker's over s..n_layer (its compute time, from the
/// replies) give each machine's time per layer, and the split that balances them is kept for the next run (and
/// printed).
pub fn save_split(n_layer: usize, s: usize, ours: f64) {
    let theirs = WORKER_US.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    if s == 0 || s >= n_layer || ours <= 0.0 || theirs <= 0.0 {
        return;
    }
    let (m, w) = (ours / s as f64, theirs / (n_layer - s) as f64);
    let best = n_layer as f64 * w / (m + w);
    // kept as a running mean with the saved value: one run's balance depends on its block size and its cache, so a
    // single run's answer jumps around
    let p = split_file(n_layer);
    let old = p.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|v| v.trim().parse::<f64>().ok());
    let keep = old.map_or(best, |o| (o + best) / 2.0).clamp(1.0, (n_layer - 1) as f64);
    eprintln!(
        "two boxes (pipeline): split {s}: ours {ours:.2} s, worker {theirs:.2} s -> balanced {best:.1}, kept {keep:.1}"
    );
    if let Some(p) = p {
        let _ = std::fs::write(p, format!("{keep:.2}\n"));
    }
}

/// The worker's compute time, summed from its replies (microseconds).
static WORKER_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Starts the worker's compute time from zero at a prompt's first block, so save_split weighs this prompt alone and not
/// the last reply's decode passes.
pub fn worker_time_reset() {
    WORKER_US.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Profile counters (TORC_PROF): a pass's round trip as the main machine sees it (from writing the request to reading
/// the reply), the worker's compute time for it, and the count, in microseconds; reset with the other profile tables.
pub static TRIP: [std::sync::atomic::AtomicU64; 3] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

pub fn trip_add(wall_us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    TRIP[0].fetch_add(wall_us, Relaxed);
    TRIP[2].fetch_add(1, Relaxed);
}

/// Control requests (n = 0, in `want`): the worker's DeltaNet layers begin a verify block, or keep the first `want >>
/// KEPT_SHIFT` tokens of it.
pub const OP_BEGIN: u32 = 16;
/// Where OP_COMMIT carries the count kept: from bit 16 up, clear of every op bit (OP_LOAD is 256, so a field starting
/// at bit 8 would turn odd counts into loads). It shares bits 16.. with TOPK_SHIFT: KEPT only rides on OP_COMMIT (which
/// has no tokens) and TOPK only on a one-token logits request (mode 1); the worker refuses a TOPK on anything else
/// (Model::hc_back).
pub const KEPT_SHIFT: u32 = 16;
pub const OP_COMMIT: u32 = 8;
/// Control request: send back the states of the worker's layers (KV and DeltaNet), packed in f32 words.
pub const OP_PULL: u32 = 32;
/// Control request: the worker's layers go back to a zero state (the main machine is starting a new sequence).
pub const OP_RESET: u32 = 64;
/// Control requests: the worker keeps its DeltaNet states as its mark (at the end of a server's prompt), or goes back
/// to it (the chat's next turn resumes there; attention needs nothing, since past the mark every position is rewritten
/// before it is read).
pub const OP_SAVE: u32 = 128;
pub const OP_LOAD: u32 = 256;
/// The worker's mark slots: 0 for the server's last prompt, 1 for its system prompt, and 2 for a long prompt's
/// checkpoint. The slot rides in bits 16.. of an OP_SAVE / OP_LOAD, which have no tokens, so KEPT and TOPK never ride
/// on them.
pub const MARK_SLOTS: usize = 3;

pub fn slot_bits(slot: usize) -> u32 {
    (slot.min(MARK_SLOTS - 1) as u32) << 16
}

pub fn slot_of(want: u32) -> usize {
    ((want >> 16) as usize).min(MARK_SLOTS - 1)
}

/// Sends a control request to the worker (for its layers from `split`), if there is one, and reads its empty reply.
pub fn control(split: usize, op: u32) -> Result<()> {
    let Some(cl) = client() else { return Ok(()) };
    let mut s = cl.lock().unwrap_or_else(|p| p.into_inner());
    s.write_all(&request(split, 0, &[], op, &[])).map_err(io)?;
    recv(&mut s, &mut [])
}

/// Sends a saved-state request to the worker (KEYOP): its layers from `split` at `npos` positions, saved (KEY_SAVE) or
/// loaded (KEY_LOAD) under `key`. Ok(false) means it was not done (no worker, or no such file there).
pub fn mark_op(split: usize, op: u32, key: u64, npos: usize) -> Result<bool> {
    let Some(cl) = client() else { return Ok(false) };
    let mut s = cl.lock().unwrap_or_else(|p| p.into_inner());
    let mut b = Vec::with_capacity(20);
    for v in [KEYOP, split as u32 | op << 16, key as u32, (key >> 32) as u32, npos as u32] {
        b.extend(v.to_le_bytes());
    }
    s.write_all(&b).map_err(io)?;
    let mut r = [0.0f32];
    recv(&mut s, &mut r)?;
    Ok(r[0] == 1.0)
}

/// HEADOP from the main machine: asks the worker for rows r0..r1 of the head over xq, runs `local` (this machine's
/// rows) in the meantime, then reads the worker's rows into out (r1 - r0 of them).
pub fn head_split(xq: kernels::Q8s, r0: usize, r1: usize, out: &mut [f32], local: &mut dyn FnMut()) -> Result<()> {
    let Some(cl) = client() else { return err("two boxes (pipeline): no worker for the head") };
    let mut s = cl.lock().unwrap_or_else(|p| p.into_inner());
    let (kb, gb) = xq.bytes();
    let mut b = Vec::with_capacity(20 + kb.len() + gb.len());
    for v in [HEADOP, r0 as u32, r1 as u32, xq.len() as u32, 0] {
        b.extend(v.to_le_bytes());
    }
    b.extend_from_slice(kb);
    b.extend_from_slice(gb);
    s.write_all(&b).map_err(io)?;
    local();
    recv(&mut s, &mut out[..r1 - r0])
}

/// A request's bytes.
pub fn request(s: usize, pos0: u32, tokens: &[u32], want: u32, res: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(20 + tokens.len() * 4 + res.len() * 4);
    for v in [MAGIC, s as u32, pos0, tokens.len() as u32, want] {
        b.extend(v.to_le_bytes());
    }
    for t in tokens {
        b.extend(t.to_le_bytes());
    }
    // SAFETY: the f32s viewed as bytes
    b.extend_from_slice(unsafe { std::slice::from_raw_parts(res.as_ptr() as *const u8, res.len() * 4) });
    b
}

fn io(e: std::io::Error) -> format::Error {
    format::Error(format!("two boxes (pipeline): {e}"))
}

fn read_u32(s: &mut TcpStream) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    s.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Reads a reply's logits into `out` (which must be exactly its length).
pub fn recv(s: &mut TcpStream, out: &mut [f32]) -> Result<()> {
    let us = read_u32(s).map_err(io)?;
    WORKER_US.fetch_add(us as u64, std::sync::atomic::Ordering::Relaxed);
    TRIP[1].fetch_add(us as u64, std::sync::atomic::Ordering::Relaxed);
    let k = read_u32(s).map_err(io)? as usize;
    if k != out.len() {
        return err(format!("two boxes (pipeline): {k} logits back, {} asked", out.len()));
    }
    // SAFETY: f32 has no invalid bit patterns
    let b = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, k * 4) };
    s.read_exact(b).map_err(io)
}

/// The most f32s a reply that gives its own length (recv_any) may claim. A whole-context state pull is well under it
/// (Flash Next's worker half at 139k positions is about 1 GB), and a misaligned stream's garbage word is not allocated
/// (one bad u32 could otherwise ask for up to 16 GB).
const ANY_MAX: usize = 1 << 30;

/// A reply of whatever length it says (a state pull, see Model::pull_range), up to ANY_MAX values.
pub fn recv_any(s: &mut TcpStream) -> Result<Vec<f32>> {
    let us = read_u32(s).map_err(io)?;
    WORKER_US.fetch_add(us as u64, std::sync::atomic::Ordering::Relaxed);
    let k = read_u32(s).map_err(io)? as usize;
    if k > ANY_MAX {
        return err(format!("two boxes (pipeline): a reply of {k} values (out of step with the worker?)"));
    }
    let mut out = vec![0.0f32; k];
    // SAFETY: f32 has no invalid bit patterns
    let b = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, k * 4) };
    s.read_exact(b).map_err(io)?;
    Ok(out)
}

/// A reply that also carries the out-normed hidden rows after the logits (want | WANT_HIDDEN): the logits go into `out`
/// and the rest into `hidden` (whose length the caller sets).
pub fn recv_with_hidden(s: &mut TcpStream, out: &mut [f32], hidden: &mut [f32]) -> Result<()> {
    let us = read_u32(s).map_err(io)?;
    WORKER_US.fetch_add(us as u64, std::sync::atomic::Ordering::Relaxed);
    TRIP[1].fetch_add(us as u64, std::sync::atomic::Ordering::Relaxed);
    let k = read_u32(s).map_err(io)? as usize;
    if k != out.len() + hidden.len() {
        return err(format!("two boxes (pipeline): {k} floats back, {} asked", out.len() + hidden.len()));
    }
    for v in [out, hidden] {
        // SAFETY: f32 has no invalid bit patterns
        let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) };
        s.read_exact(b).map_err(io)?;
    }
    Ok(())
}

/// In `want`: also send the out-normed hidden rows after the logits (the main machine's MTP block drafts from them).
pub const WANT_HIDDEN: u32 = 4;

/// In `want` (a prompt block's request, the prompt's last before a pull): as each of the worker's layers finishes this
/// block, its state at positions 0..pos0 + n goes out as an early frame (a u32 byte count, then that layer's
/// layer_states bytes), one frame a layer in order, before the reply. The transfer then runs while the worker's later
/// layers compute, instead of after the prompt (Model::pull_states).
pub const WANT_STATES: u32 = 512;

/// In `want` (a token's request, mode 1): bits 16.. = K > 0 means only the top K logits come back, as K values (highest
/// first, equal values by the lower id) followed by their K ids (u32 bits in f32 slots). Greedy needs K = 1 and top-k
/// sampling needs K >= top_k: the main machine puts them in a row of -inf (half_recv) and picks exactly as it would
/// from the full row. It shares bits 16.. with KEPT_SHIFT (see there).
pub const TOPK_SHIFT: u32 = 16;

/// In want (a top-K token request, mode 1), bits 10..15 = v > 0 split the head's rows between the machines. The worker
/// sends an early frame (a u32 byte count, then the head's input: the last row's Q8 blocks) as soon as its layers and
/// the head's mixer are done, and then the top K of rows split_rows(V, v).. only (the ids are the row numbers). The
/// main machine computes rows 0..split_rows(V, v) itself in the meantime, by the same per-row dot product, so the bits
/// are the same, and the whole row's top K lies within its rows plus the worker's K. Opt-in.
pub const HEAD_SPLIT_SHIFT: u32 = 10;
pub const HEAD_SPLIT_MASK: u32 = 63;

/// The first row the worker computes for split v (in 64ths of the vocabulary, rounded down to 64 rows).
pub fn split_rows(rows: usize, v: u32) -> usize {
    (rows * v as usize / 64) / 64 * 64
}

/// The top k of a row (value descending, then id ascending, so argmax's first maximum comes first; see ops::top_k): the
/// values, then the ids.
pub fn topk(l: &[f32], k: usize) -> Vec<f32> {
    let c = crate::ops::top_k(l, k);
    let mut out: Vec<f32> = c.iter().map(|x| x.1).collect();
    out.extend(c.iter().map(|x| f32::from_bits(x.0)));
    out
}

/// The address a worker listens on: TORC_PIPE_BIND, else every interface. A worker has no authentication, so on a
/// machine that other networks can reach, name the link's own address.
pub fn bind_host() -> String {
    crate::switches::get("TORC_PIPE_BIND").unwrap_or_else(|| "0.0.0.0".into())
}

/// The worker: serves requests in turn until the main machine hangs up, then takes the next connection (a new run,
/// whose states start again from zero).
pub fn serve(m: &mut crate::forward::Model, port: u16) -> Result<()> {
    let host = bind_host();
    let l = TcpListener::bind((host.as_str(), port)).map_err(io)?;
    eprintln!("two boxes (pipeline): serving layers on {host}:{port} (no authentication: a trusted link only)");
    let d = m.stream_width();
    for conn in l.incoming() {
        let Ok(mut s) = conn else { continue };
        let _ = s.set_nodelay(true);
        keepalive(&s);
        eprintln!("two boxes (pipeline): main box connected");
        // a new run starts clean, unless the hello says the main machine is resuming (RESUME); this is cleared at the
        // first message
        let mut fresh = true;
        let (mut nreq, mut comp) = (0usize, 0f64);
        // TORC_PROF: the worker's own profile of the decode passes since the last reset (a warm-up's replay starts
        // again from a zero state), printed when the main machine hangs up
        let prof = crate::switches::given("TORC_PROF");
        let (mut armed, mut ndec, mut tdec, mut c0) = (true, 0usize, 0f64, None);
        // the prompt rows' compute time since the last decode pass (a server run prints its prompt's profile there)
        let mut tprompt = 0f64;
        let mut pf0 = page_faults();
        // the prompt rows this connection carried (a prompt-only run's profile is per prompt token)
        let mut nrows = 0usize;
        if prof {
            // this connection's own profile (the decode one resets again at its first pass)
            m.prof = Default::default();
            for (_, _, table) in crate::phase_tables() {
                for a in table {
                    a.store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }
            c0 = m.w.cache.as_ref().map(|c| c.stats());
        }
        // the worker serves one connection at a time: one that says nothing in its first 30 s is dropped, so it can't
        // hold the worker while the real main machine waits. After that a quiet link is a main machine between chats
        let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(30)));
        let mut first = true;
        loop {
            let mut h = [0u8; 20];
            match s.read_exact(&mut h) {
                Ok(()) if first => {
                    first = false;
                    let _ = s.set_read_timeout(None);
                }
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("two boxes (pipeline): {e}");
                    break;
                }
            }
            let g = |i: usize| u32::from_le_bytes(h[4 * i..4 * i + 4].try_into().unwrap());
            if fresh {
                // a connection starts with the hello, which checks the model and the protocol
                if g(0) != HELLO {
                    eprintln!("two boxes (pipeline): no hello: refused");
                    break;
                }
                fresh = false;
                if g(1) & RESUME != 0 {
                    eprintln!("two boxes (pipeline): main box resumes (state kept)");
                } else {
                    m.clear_states();
                    m.clear_marks();
                }
            }
            if g(0) == HELLO {
                let (id, ctx) = (g(2) as u64 | (g(3) as u64) << 32, g(4));
                let ok = g(1) & !RESUME == VERSION && IDENT.get().is_none_or(|&(me, _, _)| me == id);
                let mut b = Vec::with_capacity(8);
                b.extend((!ok as u32).to_le_bytes());
                b.extend((m.n_ctx() as u32).to_le_bytes());
                let _ = s.write_all(&b).and_then(|_| s.flush());
                if !ok {
                    eprintln!("two boxes (pipeline): the main box runs another model (or protocol {}): refused", g(1));
                    break;
                }
                eprintln!("two boxes (pipeline): main box runs this model (context {ctx}, ours {})", m.n_ctx());
                continue;
            }
            if g(0) == KEYOP {
                let (l0, op, key, npos) =
                    ((g(1) & 0xffff) as usize, g(1) >> 16, g(2) as u64 | (g(3) as u64) << 32, g(4));
                let t = std::time::Instant::now();
                let ok = match m.worker_mark(op, key, l0, npos as usize) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("two boxes (pipeline): {e}");
                        false
                    }
                };
                let mut b = Vec::with_capacity(12);
                b.extend(((t.elapsed().as_secs_f64() * 1e6) as u32).to_le_bytes());
                b.extend(1u32.to_le_bytes());
                b.extend((ok as u32 as f32).to_le_bytes());
                if let Err(e) = s.write_all(&b).and_then(|_| s.flush()) {
                    eprintln!("two boxes (pipeline): {e}");
                    break;
                }
                continue;
            }
            if g(0) == HEADOP {
                let (r0, r1, nb) = (g(1) as usize, g(2) as usize, g(3) as usize);
                if nb == 0 || nb > HEAD_MAX_BLOCKS {
                    eprintln!("two boxes (pipeline): bad head request ({nb} blocks)");
                    break;
                }
                let t = std::time::Instant::now();
                let mut xq = kernels::Q8::new(nb);
                let mut xm = xq.view_mut();
                let (kb, gb) = xm.bytes_mut();
                if let Err(e) = s.read_exact(kb).and_then(|_| s.read_exact(gb)) {
                    eprintln!("two boxes (pipeline): {e}");
                    break;
                }
                let y = match m.head_rows(xq.view(), r0, r1) {
                    Ok(y) => y,
                    Err(e) => {
                        eprintln!("two boxes (pipeline): {e}");
                        break;
                    }
                };
                let mut b = Vec::with_capacity(8 + y.len() * 4);
                b.extend(((t.elapsed().as_secs_f64() * 1e6) as u32).to_le_bytes());
                b.extend((y.len() as u32).to_le_bytes());
                // SAFETY: the f32s viewed as bytes
                b.extend_from_slice(unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len() * 4) });
                if let Err(e) = s.write_all(&b).and_then(|_| s.flush()) {
                    eprintln!("two boxes (pipeline): {e}");
                    break;
                }
                continue;
            }
            if g(0) != MAGIC {
                eprintln!("two boxes (pipeline): bad request");
                break;
            }
            let (l0, pos0, n, want) = (g(1) as usize, g(2), g(3) as usize, g(4));
            if let Some((a, _)) = shard().filter(|&(a, _)| a != l0) {
                eprintln!(
                    "two boxes (pipeline): the main box splits at {l0}, but this shard starts at layer {a}: refused"
                );
                break;
            }
            if n > m.n_ctx() {
                // before allocating n rows: a mismatched main machine (or a corrupt header) is refused
                eprintln!("two boxes (pipeline): bad request ({n} rows, ctx {})", m.n_ctx());
                break;
            }
            let mut tb = vec![0u8; n * 4];
            let mut res = vec![0.0f32; n * d];
            // SAFETY: f32 has no invalid bit patterns
            let rb = unsafe { std::slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u8, n * d * 4) };
            if let Err(e) = s.read_exact(&mut tb).and_then(|_| s.read_exact(rb)) {
                eprintln!("two boxes (pipeline): {e}");
                break;
            }
            let tokens: Vec<u32> = tb.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            if prof && n == 0 && want & OP_RESET != 0 {
                armed = true;
                // a new prompt starts its profile from zero (not from the last reply's decode passes)
                m.prof = Default::default();
                for (_, _, table) in crate::phase_tables() {
                    for a in table {
                        a.store(0, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                (nrows, tprompt) = (0, 0.0);
                pf0 = page_faults();
            }
            if prof && n == 1 && armed {
                armed = false;
                // the prompt just served: where this machine's share of it went (a server prints one per request)
                if nrows > 0 {
                    let (p, k) = (m.prof, 1e3 / nrows as f64);
                    eprintln!(
                        "worker, {nrows} prompt rows: {tprompt:.2} s computing, {} page faults; ms a row: deltanet {:.3} attention {:.3} moe {:.3} output {:.3} norms {:.3}",
                        page_faults() - pf0,
                        p.rec * k,
                        p.attn * k,
                        p.moe * k,
                        p.out * k,
                        p.norm * k
                    );
                    for (name, phases, table) in crate::phase_tables() {
                        let parts: Vec<String> = phases
                            .iter()
                            .zip(table.iter())
                            .filter(|(_, v)| v.load(std::sync::atomic::Ordering::Relaxed) > 0)
                            .map(|(ph, v)| {
                                format!(
                                    "{ph} {:.3}",
                                    v.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / nrows as f64
                                )
                            })
                            .collect();
                        if !parts.is_empty() {
                            eprintln!("worker {name} ms/row: {}", parts.join(", "));
                        }
                    }
                    (nrows, tprompt) = (0, 0.0);
                }
                (ndec, tdec) = (0, 0.0);
                m.prof = Default::default();
                for (_, _, table) in crate::phase_tables() {
                    for a in table {
                        a.store(0, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                c0 = m.w.cache.as_ref().map(|c| c.stats());
            }
            let t = std::time::Instant::now();
            let r = if n > 0 && want & WANT_STATES != 0 {
                // the layers' state frames go out through a thread of their own, so the next layer computes while the
                // last one's state is on the wire; the reply follows once they are all written
                let ws = match s.try_clone() {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("two boxes (pipeline): {e}");
                        break;
                    }
                };
                std::thread::scope(|sc| {
                    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
                    let wr = sc.spawn(move || -> std::io::Result<()> {
                        let mut ws = ws;
                        for f in rx {
                            ws.write_all(&f)?;
                        }
                        ws.flush()
                    });
                    let r = {
                        let mut early = |b: &[u8]| -> std::io::Result<()> {
                            let mut f = Vec::with_capacity(4 + b.len());
                            f.extend((b.len() as u32).to_le_bytes());
                            f.extend_from_slice(b);
                            tx.send(f).map_err(|_| std::io::Error::other("the state frames' writer stopped"))
                        };
                        m.hc_back(&tokens, pos0, l0, res, want, &mut early)
                    };
                    drop(tx);
                    let w =
                        wr.join().unwrap_or_else(|_| Err(std::io::Error::other("the state frames' writer panicked")));
                    r.and_then(|o| w.map(|_| o).map_err(io))
                })
            } else {
                // the head split's early frame goes out from inside hc_back, before the worker's share of the head
                let mut early = |b: &[u8]| -> std::io::Result<()> {
                    s.write_all(&(b.len() as u32).to_le_bytes())?;
                    s.write_all(b)?;
                    s.flush()
                };
                m.hc_back(&tokens, pos0, l0, res, want, &mut early)
            };
            let out = match r {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("two boxes (pipeline): {e}");
                    break;
                }
            };
            let dt = t.elapsed().as_secs_f64();
            comp += dt;
            nreq += 1;
            if n > 1 {
                nrows += n;
                tprompt += dt;
            }
            if n == 1 {
                ndec += 1;
                tdec += dt;
            }
            let mut b = Vec::with_capacity(8 + out.len() * 4);
            b.extend(((dt * 1e6) as u32).to_le_bytes());
            b.extend((out.len() as u32).to_le_bytes());
            // SAFETY: the f32s viewed as bytes
            b.extend_from_slice(unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4) });
            if let Err(e) = s.write_all(&b).and_then(|_| s.flush()) {
                eprintln!("two boxes (pipeline): {e}");
                break;
            }
        }
        eprintln!("two boxes (pipeline): main box gone after {nreq} requests, {comp:.2} s computing");
        // the experts this run used go into the machine's usage file (a worker is stopped rather than exiting)
        if let Some(c) = &m.w.cache {
            c.usage_flush();
        }
        if prof && ndec == 0 && nrows > 0 {
            // a prompt-only run (torc-run nologits, prefill tests): where this machine's share of the prompt went
            let (p, k) = (m.prof, 1e3 / nrows as f64);
            eprintln!(
                "worker, {nrows} prompt rows: {:.2} s computing; ms a row: deltanet {:.3} attention {:.3} moe {:.3} output {:.3} norms {:.3}",
                comp,
                p.rec * k,
                p.attn * k,
                p.moe * k,
                p.out * k,
                p.norm * k
            );
            for (name, phases, table) in crate::phase_tables() {
                let ns: Vec<u64> = table.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect();
                if ns.iter().any(|&v| v > 0) {
                    let parts: Vec<String> = phases
                        .iter()
                        .zip(&ns)
                        .filter(|(n, _)| **n != "-")
                        .map(|(n, &v)| format!("{n} {:.3}", v as f64 / 1e6 / nrows as f64))
                        .collect();
                    eprintln!("worker {name} ms/row: {}", parts.join(", "));
                }
            }
            if let (Some(c), Some(a)) = (&m.w.cache, c0) {
                let s = c.stats();
                let (rq, h) = (s.asked - a.asked, s.resident - a.resident);
                eprintln!(
                    "worker expert cache over the prompt: {rq} requests, {:.1}% hits, {:.0} MB read, waited {:.2} s",
                    100.0 * h as f64 / rq.max(1) as f64,
                    (s.loaded_bytes - a.loaded_bytes) as f64 / 1e6,
                    s.blocked - a.blocked
                );
            }
        }
        if prof && ndec > 0 {
            let (p, k) = (m.prof, 1e3 / ndec as f64);
            eprintln!(
                "worker, {ndec} decode passes: {:.2} ms a pass; deltanet {:.2} attention {:.2} moe {:.2} output {:.2} norms {:.2}",
                tdec * k,
                p.rec * k,
                p.attn * k,
                p.moe * k,
                p.out * k,
                p.norm * k
            );
            for (name, phases, table) in crate::phase_tables() {
                let ns: Vec<u64> = table.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect();
                if ns.iter().any(|&v| v > 0) {
                    let parts: Vec<String> = phases
                        .iter()
                        .zip(&ns)
                        .filter(|(n, _)| **n != "-")
                        .map(|(n, &v)| format!("{n} {:.2}", v as f64 / 1e6 / ndec as f64))
                        .collect();
                    eprintln!("worker {name} ms/pass: {}", parts.join(", "));
                }
            }
            if let (Some(c), Some(a)) = (&m.w.cache, c0) {
                let s = c.stats();
                let (rq, h) = (s.asked - a.asked, s.resident - a.resident);
                eprintln!(
                    "worker expert cache over them: {rq} requests, {:.1}% hits, {:.0} MB read, waited {:.2} s",
                    100.0 * h as f64 / rq.max(1) as f64,
                    (s.loaded_bytes - a.loaded_bytes) as f64 / 1e6,
                    s.blocked - a.blocked
                );
            }
        }
    }
    Ok(())
}

/// The process's page faults so far (Windows; 0 elsewhere). The worker prints them per prompt (TORC_PROF).
pub fn page_faults() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Counters {
            cb: u32,
            page_fault_count: u32,
            rest: [usize; 8],
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(p: *mut std::ffi::c_void, c: *mut Counters, cb: u32) -> i32;
        }
        let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
        // SAFETY: a plain Win32 call on this process, with a correctly sized struct
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0 {
            return c.page_fault_count as u64;
        }
    }
    0
}
