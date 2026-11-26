//! status.rs - what the server is doing right now, kept where a page can read it.
//!
//! `/health`, `/slots` and `/metrics` answer from counters that commit when a reply ends. That is what a monitor of a
//! finished run wants and what a person watching a thirty second prefill cannot see: the run gives them nothing until
//! it is over. Two things move while a reply is being written, and both are kept here: how far the prompt of the reply
//! in progress has been read, which the model already measures for [TORC_METRICS](SWITCHES.md) and only needs
//! somewhere to put it, and the replies that went through the server, which the server itself sees. `torc-server`
//! answers `GET /torc/status` from them, and the page it serves at `/` draws that.
//!
//! Nothing here changes what a model produces. The counters describe a run in progress, and the text a server was
//! asked and answered is kept only when `TORC_STATUS_TEXT` asks for it, as the address a request came from is kept
//! only when `TORC_STATUS_CLIENT` does, because a request is someone's business.
//!
//! Inside one block of a prompt there is nothing further to count: a block is one pass over the model's layers, and a
//! long one can take minutes. So the layers of that pass are counted too, and the page shows them as what they are -
//! progress inside a block, never a speed worked out from it.

use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

/// How many finished replies the page can look back at. It is a window onto the server, not a history: the history is
/// what TORC_METRICS writes.
const WINDOW: usize = 40;

/// The most characters of a request and of its reply that are kept. Long enough to tell one chat from another at a
/// glance, short enough that a window of them cannot grow the server's memory.
pub const SNIPPET: usize = 220;

/// Where the monotonic clock starts. The seconds a prompt has been reading are measured from the process, not from the
/// wall clock, so a step across a daylight-saving change cannot make one negative.
static CLOCK: LazyLock<std::time::Instant> = LazyLock::new(std::time::Instant::now);

/// Nanoseconds since that, which is what the atomics below hold. Never zero: every holder below reads 0 as "unset",
/// and on a Windows clock tick the first reading after the clock starts can be zero nanoseconds old.
fn nanos() -> u64 {
    CLOCK.elapsed().as_nanos().max(1) as u64
}

// A prompt being read: its tokens, how many of them are read, and the nanoseconds the reading has taken. Zero in all
// three when no prompt is being read, so a monitor sees the end of a reply the same way it sees a server that has no
// progress to report at all.
static READ_TOKENS: AtomicUsize = AtomicUsize::new(0);
static READ_DONE: AtomicUsize = AtomicUsize::new(0);
static READ_NS: AtomicU64 = AtomicU64::new(0);
// the rate of the block the engine last reported, held as a float's bits: a rate is a float, and an atomic that has
// to carry one carries its bits
static READ_RATE: AtomicU64 = AtomicU64::new(0);
// when the engine last reported a block, so that the rate above can be shown with its age. A block of `TORC_BLOCK`
// tokens at a long context takes minutes, and a number that has not moved for five of them is not a live speed
// unless the page says how old it is
static READ_RATE_NS: AtomicU64 = AtomicU64::new(0);
// Where this prompt's first block ended: the tokens read at that point and the nanosecond it was reported. The block
// that starts a prompt is also the one that pays for the machine coming back to the model - the pages of a mapped
// file touched for the first time, a worker's graphs set up - and it yields nothing until it is over. An average
// measured from before it says as much about that setup as about reading, and on a prompt nobody meant to be long
// (an agent's overhead around two typed letters) that is most of the run. From the first block's end to here, both
// the tokens and the seconds are measured.
static READ_FROM_DONE: AtomicUsize = AtomicUsize::new(0);
static READ_FROM_NS: AtomicU64 = AtomicU64::new(0);
// the moment the engine last reported `done`. A finished prompt's span has to end on a moment the engine reported,
// because its reply is written long after the reading stopped and the counters only go back to zero when the reply
// ends - a span ending on `read_end` itself would quietly include the writing of the answer
static READ_DONE_NS: AtomicU64 = AtomicU64::new(0);
// What the reading of the last prompt cost after its first block ended: the tokens and the nanoseconds between the
// first block's report and the engine's last one. A finished reply's record asks for the pair once, after the
// counters below have gone back to zero; a prompt read in one block has no span, and 0 tokens says so.
static LAST_READ_TOKENS: AtomicUsize = AtomicUsize::new(0);
static LAST_READ_NS: AtomicU64 = AtomicU64::new(0);
// How far inside the block being read the model has got: the layers of this forward pass that are done, and how many
// it will run. A block is one pass over the layers, so this is the only progress inside a block that exists at all;
// it says how far a block still has to go, which a token count cannot, and it is not a rate and never becomes one.
static READ_LAYER: AtomicUsize = AtomicUsize::new(0);
static READ_LAYERS: AtomicUsize = AtomicUsize::new(0);

/// A prompt of `total` tokens has started being read. Call it from the point where the tokens are known, which is
/// where a format's own progress reporting starts.
pub fn read_begin(total: usize) {
    READ_TOKENS.store(total, Ordering::Relaxed);
    READ_DONE.store(0, Ordering::Relaxed);
    READ_NS.store(nanos(), Ordering::Relaxed);
    READ_RATE.store(0, Ordering::Relaxed);
    READ_RATE_NS.store(0, Ordering::Relaxed);
    READ_FROM_DONE.store(0, Ordering::Relaxed);
    READ_FROM_NS.store(0, Ordering::Relaxed);
    READ_LAYER.store(0, Ordering::Relaxed);
    READ_LAYERS.store(0, Ordering::Relaxed);
    READ_DONE_NS.store(0, Ordering::Relaxed);
    // the last prompt's span belongs to the last prompt: the record this prompt will leave has its own span or
    // none, and never the previous prompt's
    LAST_READ_TOKENS.store(0, Ordering::Relaxed);
    LAST_READ_NS.store(0, Ordering::Relaxed);
}

/// The rate the engine measured for the block it has just finished reading, in tokens per second. This is not derived
/// here: models/qwen/serve.rs already works it out of its own block timing to write its TORC_METRICS progress line,
/// and the page is shown that number rather than one invented beside it. It is what a display of live prefill speed
/// wants, as against the average over the whole prompt, which `reading()` gives and which says more about the blocks
/// that are already behind us than about the one being read now.
/// `full_pass` says whether the finished pass read a whole block's rows. A prompt is read in segments (the base, its
/// opener, the new stretch), and a segment's last pass is usually a partial one; only a full pass ended is a block
/// ended, so only one of those may mark where the average after the first block starts. A short prompt whose every
/// pass is partial then has no such point at all, which is right: there was never a block to read past.
pub fn read_rate(tokens_per_second: f64, full_pass: bool) {
    let at = nanos();
    READ_RATE.store(tokens_per_second.to_bits(), Ordering::Relaxed);
    READ_RATE_NS.store(at, Ordering::Relaxed);
    // the first full-pass rate marks where a measurable span can start: `read_done` has just been called for the
    // block that ended, so its tokens and its instant are both the engine's, and nothing before them has to be
    if full_pass && READ_FROM_NS.load(Ordering::Relaxed) == 0 {
        READ_FROM_NS.store(at, Ordering::Relaxed);
        READ_FROM_DONE.store(READ_DONE.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

/// `done` of that prompt's tokens are read. Held as the engine reports it, in both places it is written: a prompt
/// that is read through a pipeline worker can lose the worker mid-stretch and be read again from the last checkpoint
/// (models/qwen/serve.rs), and then the honest number goes back. A page that divides it by the seconds the reading
/// has taken needs no promise that it only rises, and a number here that disagreed with the same moment's
/// TORC_METRICS line would be a number nobody could check.
pub fn read_done(done: usize) {
    READ_DONE.store(done, Ordering::Relaxed);
    READ_DONE_NS.store(nanos(), Ordering::Relaxed);
}

/// The prompt is read, or the reply stopped trying. The counters go back to zero, which is also how a monitor that
/// watches for a counter dropping learns that a task ended.
pub fn read_end() {
    // once, while the prompt is still standing: the reply's end calls this a second time, and a second time would
    // store the zeroed counters over the span the record is about to ask for
    let at = READ_FROM_NS.load(Ordering::Relaxed);
    if at != 0 {
        let (reported, done, from) = (
            READ_DONE_NS.load(Ordering::Relaxed),
            READ_DONE.load(Ordering::Relaxed),
            READ_FROM_DONE.load(Ordering::Relaxed),
        );
        // the span ends at the engine's last report, not at this call: between the last block and here the reply
        // was already being written, and its seconds are not the prompt's reading
        let (tokens, ns) = if reported > at && done > from { (done - from, reported - at) } else { (0, 0) };
        LAST_READ_TOKENS.store(tokens, Ordering::Relaxed);
        LAST_READ_NS.store(ns, Ordering::Relaxed);
    }
    READ_TOKENS.store(0, Ordering::Relaxed);
    READ_DONE.store(0, Ordering::Relaxed);
    READ_NS.store(0, Ordering::Relaxed);
    READ_RATE.store(0, Ordering::Relaxed);
    READ_RATE_NS.store(0, Ordering::Relaxed);
    READ_FROM_DONE.store(0, Ordering::Relaxed);
    READ_FROM_NS.store(0, Ordering::Relaxed);
    READ_LAYER.store(0, Ordering::Relaxed);
    READ_LAYERS.store(0, Ordering::Relaxed);
}

/// A forward pass over one block is starting, and this machine will run `total` of its layers. Called where that pass
/// is entered, not estimated from the model: a pass over part of the layers (a pipeline worker) reports its own
/// share, and a reader should not be told the whole model is being read here when it is not.
pub fn pass_begin(total: usize) {
    READ_LAYER.store(0, Ordering::Relaxed);
    READ_LAYERS.store(total, Ordering::Relaxed);
}

/// One more layer of the pass has run. This is called from the layer loop, so it is one relaxed store per layer of a
/// pass that takes seconds; nothing in the pass reads it back, which is what lets it be relaxed at all.
pub fn pass_layer() {
    READ_LAYER.fetch_add(1, Ordering::Relaxed);
}

/// How far inside the pass the model is: `(layers done, layers in the pass)`, `(0, 0)` between prompts and before the
/// first pass of one starts, so a page can tell "not started yet" from "at the first layer".
pub fn pass_progress() -> (usize, usize) {
    let total = READ_LAYERS.load(Ordering::Relaxed);
    if total == 0 || READ_TOKENS.load(Ordering::Relaxed) == 0 {
        return (0, 0);
    }
    (READ_LAYER.load(Ordering::Relaxed).min(total), total)
}

/// The rate of the last-read block, or 0.0 when no prompt is being read.
pub fn reading_rate() -> f64 {
    f64::from_bits(READ_RATE.load(Ordering::Relaxed))
}

/// The seconds since the engine last reported a block, or `None` between prompts. The engine only measures a block
/// when it has finished one, so at a long context this is the difference between a page showing a rate that stopped
/// moving minutes ago and one that says the block being read has been going for that long.
pub fn reading_rate_age() -> Option<f64> {
    let at = READ_RATE_NS.load(Ordering::Relaxed);
    if at == 0 || READ_TOKENS.load(Ordering::Relaxed) == 0 {
        return None;
    }
    Some((nanos().saturating_sub(at)) as f64 / 1e9)
}

/// The rate of the reading since this prompt's first block ended: the tokens read after that point over the seconds
/// between that point and the engine's last report of the reading. Both ends are reports the engine made, which is
/// what makes the number move when a block has been read and hold still otherwise: a denominator taken from the
/// clock here would fall on every poll of a page, including the polls between two blocks, and a number that drifts
/// down while nothing new was measured is the fade the reply's rate was kept off for the same reason.
///
/// Zero when a second block has not been reported yet, since until then there is no span between two measurements,
/// and zero when a prompt read again from a checkpoint has fallen back behind that point.
pub fn reading_sustained() -> f64 {
    let (from_ns, end_ns) = (READ_FROM_NS.load(Ordering::Relaxed), READ_DONE_NS.load(Ordering::Relaxed));
    let (tokens, done, from_done) = (
        READ_TOKENS.load(Ordering::Relaxed),
        READ_DONE.load(Ordering::Relaxed),
        READ_FROM_DONE.load(Ordering::Relaxed),
    );
    if tokens == 0 || from_ns == 0 || end_ns <= from_ns || done <= from_done {
        return 0.0;
    }
    (done - from_done) as f64 / ((end_ns - from_ns) as f64 / 1e9)
}

/// What the reading of the last prompt cost after its first block ended: the tokens and the seconds between the
/// first block's report and the engine's last one, or `None` when the prompt was a single block or stopped before a
/// second report. The record of a finished reply divides these two rather than the prompt's whole seconds, because
/// the seconds before the first block ends are the machine coming back to the model, not reading.
pub fn last_reading_span() -> Option<(usize, f64)> {
    let (tokens, ns) = (LAST_READ_TOKENS.load(Ordering::Relaxed), LAST_READ_NS.load(Ordering::Relaxed));
    if tokens == 0 || ns == 0 {
        return None;
    }
    Some((tokens, ns as f64 / 1e9))
}

/// The prompt being read: its tokens, the tokens read, and the seconds the reading has taken. `(0, 0, 0.0)` between
/// replies. The seconds come with it so that a page can show a rate from one reading rather than having to compare
/// two of its own polls, which is the difference between the number moving at once and moving after the first block.
pub fn reading() -> (usize, usize, f64) {
    let (tokens, done, ns) =
        (READ_TOKENS.load(Ordering::Relaxed), READ_DONE.load(Ordering::Relaxed), READ_NS.load(Ordering::Relaxed));
    if tokens == 0 || ns == 0 {
        return (0, 0, 0.0);
    }
    (tokens, done, (nanos().saturating_sub(ns)) as f64 / 1e9)
}

// A reply being written: the nanoseconds since its first token, which with the tokens written since is the same
// average the engine prints for itself. Held here rather than in torc-server's own state because the page divides
// by it, and a division by a time measured somewhere else in the process is how a rate ends up wrong.
static WRITE_NS: AtomicU64 = AtomicU64::new(0);

// When a token of the reply in progress was written, the last few seconds of them only. The average above settles onto
// one number and then barely moves, which is the wrong thing to show a person watching a machine: a reply that slows
// to a halt looks like one still running at the speed it started at, and a speculative-decoding burst looks like
// nothing changed. A rate over the last tokens moves with the engine, which is what a live display is for.
const WRITE_WINDOW_NS: u64 = 3_000_000_000;

// The number of last tokens a rate is taken from, rather than every stamp in the window: an average over three seconds
// is still an average, and what a person watching wants is the rate of what the engine has just written. The time
// window above is the cap, so that a reply which stopped a minute ago is not measured from stamps that old.
const RECENT_TOKENS: usize = 16;

fn written() -> &'static Mutex<VecDeque<u64>> {
    static STAMPS: OnceLock<Mutex<VecDeque<u64>>> = OnceLock::new();
    STAMPS.get_or_init(|| Mutex::new(VecDeque::with_capacity(64)))
}

/// The rate the tokens in `stamps` were written at, per second: their count minus one over the span between the first
/// and the last, which is the interval they were actually written in. `None` below two stamps, where there is no span
/// to divide by and any rate would be invented. Split out from the live queue so that the arithmetic can be asked
/// about a known sequence instead of a clock.
fn rate_over(stamps: &[u64]) -> Option<f64> {
    let (first, last) = (*stamps.first()?, *stamps.last()?);
    let span = last.checked_sub(first)?;
    // one token every 200 microseconds would be 5 000 t/s, past any engine this page describes: what is really
    // happening is several stamps in the same instant, and a rate from them is a division by a number too small to mean
    if span < 200_000 {
        return None;
    }
    Some(((stamps.len() - 1) as f64 * 1e9) / span as f64)
}

/// A token of a reply has been written. The first one of a reply starts the interval the writing is measured over and
/// answers true; the ones after it answer false. This runs once per token, next to the work of writing it: one
/// compare-and-exchange on an atomic already in this thread's cache, and one push onto a queue that holds three
/// seconds of tokens - against a forward pass per token, neither is a thing the reply notices.
pub fn note_token() -> bool {
    let now = nanos();
    let first = WRITE_NS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed).is_ok();
    let mut q = written().lock().unwrap();
    if first {
        q.clear();
    }
    q.push_back(now);
    while let Some(&oldest) = q.front() {
        if now.saturating_sub(oldest) > WRITE_WINDOW_NS {
            q.pop_front();
        } else {
            break;
        }
    }
    first
}

/// The rate of the reply in progress over the last `WRITE_WINDOW_NS`, or `None` when the tokens written in it are too
/// few to measure a span from. A caller that wants a number then falls back to the whole-reply average, which is the
/// number the engine itself prints: both are honest, one is current and one is not, and the page says which it shows.
pub fn writing_rate() -> Option<f64> {
    let now = nanos();
    let mut q = written().lock().unwrap();
    while let Some(&oldest) = q.front() {
        if now.saturating_sub(oldest) > WRITE_WINDOW_NS {
            q.pop_front();
        } else {
            break;
        }
    }
    rate_over(&q.iter().skip(q.len().saturating_sub(RECENT_TOKENS)).copied().collect::<Vec<_>>())
}

/// The seconds the reply in progress has been writing. `0.0` between replies, so that a rate asked of it is not a
/// division by a stale interval.
pub fn writing() -> f64 {
    let ns = WRITE_NS.load(Ordering::Relaxed);
    if ns == 0 {
        return 0.0;
    }
    (nanos().saturating_sub(ns)) as f64 / 1e9
}

/// The reply is over. As with a prompt, the counter goes back to zero, which is what tells a monitor watching for a
/// counter to drop that the run ended. The stamps go with it: a rate of three seconds of a reply that has finished
/// would be attributed to the next one.
pub fn write_end() {
    WRITE_NS.store(0, Ordering::Relaxed);
    written().lock().unwrap().clear();
}

/// One reply that went through the server, as the page shows it. The seconds are the model's own measurements, from
/// Serve::timed: what reading the prompt took, what the wait for the first token was, and what writing the reply
/// took. A rate is those two numbers divided, which the page does rather than the server, so that a page can show a
/// rate over its own window of them.
pub struct Record {
    /// the server's request number, which is what /slots calls `id_task`
    pub task: u64,
    /// when the reply started, as seconds since the Unix epoch
    pub started: u64,
    /// what the engine read for this request, and how many tokens of the same prompt it did not read again because
    /// the chat's cache still held them. Their sum, `prompt_total`, is the prompt the request arrived with, and that
    /// is what the page's prompt column shows; `prompt` itself keeps the meaning `usage.prompt_tokens` and
    /// `/metrics`'s `prompt_tokens_total` carry here, which is the part that was read.
    pub prompt: usize,
    pub cached: usize,
    /// the tokens written, reasoning included
    pub output: usize,
    pub prompt_s: f64,
    pub first_token_s: f64,
    pub output_s: f64,
    /// the tokens read after this prompt's first block ended, and the seconds the engine spent on them: the pair
    /// the page divides to get the prompt's reading speed. `(0, 0.0)` when the prompt was a single block - then the
    /// whole prompt is that block, and the page reads the pair above
    pub reading: (usize, f64),
    /// why it ended: "stop", "length" or "tool_calls"
    pub finish: &'static str,
    /// what was asked, and the start of the answer: empty unless the text switch is on
    pub asked: String,
    pub reply: String,
    /// whether `reply` is the start of a thinking stretch rather than of the answer. A thinking model stopped by a
    /// small token cap has said nothing but thinking, and a page that shows an empty column next to a screenful of
    /// text the person beside it can read is worse than one that says what it is showing.
    pub thinking: bool,
    /// the address the request came from, empty unless `TORC_STATUS_CLIENT` asks for it: a prompt tells you what
    /// was asked, an address tells you who asked it, and that is a second person's business
    pub client: String,
}

impl Record {
    fn value(&self) -> Value {
        json!({
            "task": self.task,
            "started": self.started,
            "prompt": self.prompt,
            "cached": self.cached,
            // the prompt the request arrived with: the two counts above added, not a third measurement. The rate
            // columns stay over `prompt`, because reading 1,543 of 78,110 tokens is what took the seconds.
            "prompt_total": self.prompt + self.cached,
            "output": self.output,
            "prompt_seconds": self.prompt_s,
            "first_token_seconds": self.first_token_s,
            "output_seconds": self.output_s,
            "reading_tokens": self.reading.0,
            "reading_seconds": self.reading.1,
            "finish": self.finish,
            // what `keep` decided about the text, as it was then: the page is told separately whether this run keeps
            // text, so that it can say why the column is empty rather than showing an empty one
            "asked": self.asked,
            "reply": self.reply,
            "thinking": self.thinking,
            "client": self.client,
            // a name is the operator's word for an address, never the server's guess at one, so an address that is
            // not mapped is reported as the address it is
            "client_name": client_name(&self.client),
        })
    }
}

/// What a reply's row shows: what the model answered, or its thinking when thinking is all there was. Here rather
/// than at the call site, because which of two texts a page puts in front of someone is a rule worth a test.
pub fn snippet(answer: &str, thinking: &str) -> (String, bool) {
    match (answer.is_empty(), thinking.is_empty()) {
        (false, _) => (cut(answer), false),
        (true, false) => (cut(thinking), true),
        _ => (String::new(), false),
    }
}

/// Whether a reply's text may be kept and shown. Read each time rather than once: it changes nothing about what the
/// model does, only about what the server holds for a page, and a server should not be holding someone's text
/// because of a decision made at startup that nobody can undo.
fn text_kept() -> bool {
    crate::switches::on("TORC_STATUS_TEXT")
}

/// Whether the address a request came from may be kept. A separate switch from the text one because they name
/// different people: the text is the conversation, the address is who had it, and someone serving a page on the LAN
/// may want one and not the other. Off by default, like the text: a server should not be holding a client's address
/// for a table nobody asked to see.
fn client_kept() -> bool {
    crate::switches::on("TORC_STATUS_CLIENT")
}

/// A request's text, with the control tokens a chat template leaves in it taken out, and cut to the window. The last
/// user message is what a person typed; a system prompt is the same on every request and says nothing about this one.
pub fn asked(msgs: &[Value]) -> String {
    let text = |m: &Value| match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        // a multimodal message keeps its text in parts
        Some(Value::Array(parts)) => {
            parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(" ")
        }
        _ => String::new(),
    };
    let last = msgs
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .map(text)
        .next_back()
        .unwrap_or_default();
    cut(&last)
}

/// The first characters of a reply, on a character boundary.
pub fn cut(text: &str) -> String {
    match text.chars().take(SNIPPET).collect::<String>() {
        // a cut in the middle of a word is fine for a snippet; a cut in the middle of a character is not
        t if text.chars().count() > SNIPPET => t + "…",
        t => t,
    }
}

/// The replies kept, oldest first in the queue. This lock is never taken before the server's own state lock: the two
/// places that touch the window (the reply wrapper and `/torc/status`) both hold the state first, so there is one
/// order to find and no cycle to wait in.
fn window() -> &'static Mutex<VecDeque<Record>> {
    static KEPT: OnceLock<Mutex<VecDeque<Record>>> = OnceLock::new();
    KEPT.get_or_init(|| Mutex::new(VecDeque::with_capacity(WINDOW)))
}

/// How many replies this run has finished. The window holds the last `WINDOW` of them and no more, so the total a
/// page names cannot be counted from it. A request the server refused before generating - a prompt longer than the
/// context, say - is not in this count either, because it never finished a reply: a heading that says what this
/// server has done should not be answerable by a number that also counts what it turned down.
static FINISHED: AtomicUsize = AtomicUsize::new(0);

/// A reply is finished: keep it for the page, dropping the oldest when the window is full. This is the one place the
/// text and address switches are decided, so that a reply's text and its client's address are either held or not held
/// rather than kept and then hidden.
pub fn keep(mut r: Record) {
    FINISHED.fetch_add(1, Ordering::Relaxed);
    if !text_kept() {
        r.asked.clear();
        r.reply.clear();
        // a record with no text does not have a kind of text either
        r.thinking = false;
    }
    if !client_kept() {
        r.client.clear();
    }
    let mut q = window().lock().unwrap();
    q.push_back(r);
    while q.len() > WINDOW {
        q.pop_front();
    }
}

/// The replies this run finished, more than the window can hold.
pub fn finished() -> usize {
    FINISHED.load(Ordering::Relaxed)
}

/// `TORC_STATUS_NAMES`, read once: `192.168.1.196=desk, 192.168.1.184=laptop`. Entries are separated by commas or
/// semicolons and a pair by its first `=`, so spaces around the sign are allowed; `:` is accepted as the sign when an
/// entry has no `=`, because a list of pairs reads either way and nobody should have to remember which one the code
/// picked. An address with colons in it (IPv6) has to be written with `=` for that reason, which the example in
/// SWITCHES.md shows in the form that cannot go wrong.
fn parse_names(spec: &str) -> HashMap<String, String> {
    spec.split([',', ';'])
        .filter_map(|e| match e.split_once('=') {
            Some(p) => Some(p),
            None => e.split_once(':'),
        })
        .map(|(addr, name)| (addr.trim().to_string(), name.trim().to_string()))
        .filter(|(addr, name)| !addr.is_empty() && !name.is_empty())
        .collect()
}

/// What a kept address is called. `TORC_STATUS_CLIENT` decides whether an address is kept at all; this only says what
/// to call one that is kept, because a name is easier to read than a number and the name of a machine on the LAN is
/// not something this crate can ask for without a resolver, which the tree has no dependency for. An address that is
/// not mapped stays an address: a name guessed at would be worse than an address that is certainly right.
pub fn client_name(addr: &str) -> Option<&'static str> {
    if addr.is_empty() {
        return None;
    }
    static NAMES: LazyLock<HashMap<String, String>> =
        LazyLock::new(|| parse_names(&std::env::var("TORC_STATUS_NAMES").unwrap_or_default()));
    NAMES.get(addr).map(String::as_str)
}

/// The replies the page may show, newest first.
pub fn recent() -> Vec<Value> {
    let q = window().lock().unwrap();
    q.iter().rev().map(Record::value).collect()
}

/// Whether a request for the page or its data may be answered. It is off unless `TORC_STATUS` asks for it, and a
/// machine that is not this one may see it only when `TORC_STATUS_NET` says the network may: the counters name the
/// model and what it is doing, which is worth keeping to the machine it runs on unless someone opens it.
pub fn allowed(local: bool) -> bool {
    crate::switches::on("TORC_STATUS") && (local || crate::switches::on("TORC_STATUS_NET"))
}

/// Whether another machine must carry the API key to see the page (`TORC_STATUS_KEY`), as it must for the API. Off by
/// default, so a page opened to the network with `TORC_STATUS_NET` behaves as it always has. This machine never needs
/// the key, as it never does for the API; and with no `TORC_API_KEY` set there is no key to carry, so the server
/// refuses the page to every other machine rather than show it to all of them.
pub fn keyed() -> bool {
    crate::switches::on("TORC_STATUS_KEY")
}

/// How far the page can be seen, which is what the start-up warning has to say.
#[derive(Clone, Copy)]
pub struct Reach {
    /// `TORC_STATUS`: the page is served at all
    pub page: bool,
    /// `TORC_STATUS_NET`: other machines may ask for it
    pub net: bool,
    /// the server listens on more than this machine (`TORC_HOST`)
    pub open: bool,
    /// `TORC_STATUS_KEY`: other machines must carry the API key
    pub keyed: bool,
    /// `TORC_API_KEY` is set
    pub key: bool,
}

impl Reach {
    pub fn now(open: bool, key: bool) -> Reach {
        let on = crate::switches::on;
        Reach { page: on("TORC_STATUS"), net: on("TORC_STATUS_NET"), open, keyed: keyed(), key }
    }
}

/// The line the server prints at start when its page shows what someone asked or where they asked it from, so whoever
/// turned those on hears who else can read them. Nothing when neither is on, or when the page is off: the warning is
/// about the two switches, and the counters alone are not someone's business.
pub fn warning(text: bool, client: bool, r: Reach) -> Option<String> {
    let what = match (text, client) {
        (true, true) => "the text of each request and its reply, and the address it came from,",
        (true, false) => "the text of each request and its reply",
        (false, true) => "the address each request came from",
        (false, false) => return None,
    };
    if !r.page {
        return None;
    }
    let who = if !r.net || !r.open {
        "anyone who can use this machine"
    } else if r.keyed && r.key {
        "this machine and any machine that has the API key"
    } else if r.keyed {
        "this machine only, since TORC_STATUS_KEY asks other machines for a key and TORC_API_KEY is not set"
    } else {
        "any machine that can reach this server, with no key asked for: set TORC_STATUS_KEY=1 and TORC_API_KEY to \
         keep it to machines that have the key"
    };
    Some(format!("torc-server: warning: the status page shows {what} to {who}"))
}

/// The page itself, kept in the binary so that serving it needs no folder beside the executable: it reads two URLs,
/// both below, and has no other files to load.
pub const PAGE: &str = include_str!("status.html");

/// The mark the desktop app puts in the corner of its own windows. It is taken from that app's own file rather than
/// copied into a second one, so that a page and the app beside it cannot drift into two different logos, and it is
/// compiled in for the same reason the page is: a server binary copied onto another machine has no folder to carry.
pub const MARK: &[u8] = include_bytes!("../../../ui/assets/mark-color.png");

/// The name a model is called by on any screen a person reads: the file it was loaded from, with the folder it lives
/// in and the extension it has both left off. This is the rule the desktop app uses when it says what it has loaded.
/// Both separators are looked for because the path arrives from whichever machine this server happens to run on, and
/// which one that machine uses is none of this function's business. A file with no extension answers its own name.
pub fn model_name(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or("");
    match file.rfind('.') {
        // a name that starts with a dot has no extension to take off: that dot is part of the name
        Some(i) if i > 0 => file[..i].to_string(),
        _ => file.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::switches;

    // A Windows checkout renders its line endings as CRLF, and the page is compiled from the file as it stands,
    // so what the assertions read has the stray carriage returns taken out. The server itself serves PAGE whole;
    // nothing a browser draws turns on an end of line.
    static PAGE_LF: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| PAGE.replace("\r\n", "\n"));

    // the read and write counters these tests move are the process's own, so the tests that move them take turns
    static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// An address is called by what the operator named it and by nothing else: an unmapped address, an empty one and
    /// a malformed entry all have to leave the row showing the address rather than a blank or a guess.
    #[test]
    fn an_address_is_named_only_when_someone_named_it() {
        let m = parse_names("192.168.1.196=desk, 192.168.1.184:laptop;10.0.0.1 = box-two ;=nope =x;fe80::1=router");
        assert_eq!(m.get("192.168.1.196").map(String::as_str), Some("desk"));
        assert_eq!(m.get("192.168.1.184").map(String::as_str), Some("laptop"));
        assert_eq!(m.get("10.0.0.1").map(String::as_str), Some("box-two"), "spaces around the sign are allowed");
        assert_eq!(m.get("fe80::1").map(String::as_str), Some("router"), "an address with colons needs the = sign");
        assert_eq!(m.len(), 4, "an entry without one side of the pair is not a name");
        assert_eq!(client_name(""), None, "a record that keeps no address gets no name either");
    }

    /// The name on the page is the one a person recognises, not the path a switch was given. Neither separator nor
    /// either kind of name is allowed to produce a path or an empty heading.
    #[test]
    fn a_model_is_called_by_its_file_and_not_by_the_folder_it_lives_in() {
        assert_eq!(model_name(r"C:\llm\models\Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"), "Qwen_Qwen3.6-35B-A3B-Q4_0");
        assert_eq!(model_name("/home/rainer/llm-models/qwen36-35b.torc"), "qwen36-35b");
        assert_eq!(model_name("model"), "model");
        assert_eq!(model_name(".cache"), ".cache");
        assert_eq!(model_name("/srv/models/"), "");
        assert_eq!(model_name(""), "");
    }

    fn record(task: u64) -> Record {
        Record {
            task,
            started: 1_700_000_000 + task,
            prompt: 100 + task as usize,
            cached: 40,
            output: 12,
            prompt_s: 1.5,
            first_token_s: 0.2,
            output_s: 3.0,
            reading: (90, 0.5),
            finish: "stop",
            asked: format!("question {task}"),
            reply: format!("answer {task}"),
            thinking: false,
            // TEST-NET, so that a fixture address cannot be mistaken for a machine somewhere
            client: format!("198.51.100.{task}"),
        }
    }

    /// The prompt column is the prompt that was sent, which is the number a conversation's rows grow against, and it
    /// is the sum of two counts the record already holds. The rate over it is not: that stays on the read part.
    #[test]
    fn the_prompt_column_of_a_record_is_the_prompt_that_was_sent() {
        let r = record(7).value();
        assert_eq!(
            r["prompt"].as_u64().unwrap() + r["cached"].as_u64().unwrap(),
            r["prompt_total"].as_u64().unwrap(),
            "the total is the part read and the part reused, nothing else"
        );
        assert!(
            PAGE_LF.contains("data-l=\"prompt tok\">${num(r.prompt_total)}"),
            "the table's prompt column shows the whole prompt"
        );
        assert!(
            PAGE_LF.contains("rate(r.prompt, r.prompt_seconds)"),
            "while the reading rate stays over the tokens that were read"
        );
    }

    /// One test for the window and for the two switches that decide what a row may hold: all three are the process's,
    /// so two tests touching them would change each other's results.
    #[test]
    fn the_window_holds_the_newest_replies_first_and_their_text_only_when_it_is_asked_for() {
        let before = finished();
        for t in 0..(WINDOW as u64 + 5) {
            keep(record(t));
        }
        let shown = recent();
        assert_eq!(shown.len(), WINDOW, "the window is bounded");
        let first = shown[0]["task"].as_u64().unwrap();
        let last = shown[shown.len() - 1]["task"].as_u64().unwrap();
        assert_eq!(first, WINDOW as u64 + 4, "newest first");
        assert_eq!(last, 5, "the oldest five fell out");
        assert_eq!(shown[0]["asked"].as_str().unwrap(), "", "no text unless it is asked for");
        assert_eq!(shown[0]["client"].as_str().unwrap(), "", "no address either, until it is asked for");

        switches::set("TORC_STATUS_TEXT", "1");
        keep(record(900));
        assert_eq!(recent()[0]["asked"].as_str().unwrap(), "question 900", "asked for, and kept");
        // the address has its own switch: a server whose operator will show the conversation need not show who had it
        assert_eq!(recent()[0]["client"].as_str().unwrap(), "", "keeping the text does not keep the address");
        switches::set("TORC_STATUS_TEXT", "0");
        keep(record(901));
        assert_eq!(recent()[1]["asked"].as_str().unwrap(), "question 900", "what was kept stays kept");
        assert_eq!(recent()[0]["asked"].as_str().unwrap(), "", "and what arrives now is not");
        switches::set("TORC_STATUS_CLIENT", "1");
        keep(record(902));
        assert_eq!(recent()[0]["client"].as_str().unwrap(), "198.51.100.902", "asked for, and kept");
        assert_eq!(recent()[0]["asked"].as_str().unwrap(), "", "and the text stays where its switch left it");
        switches::set("TORC_STATUS_CLIENT", "0");
        keep(record(903));
        assert_eq!(recent()[1]["client"].as_str().unwrap(), "198.51.100.902", "what was kept stays kept");
        assert_eq!(recent()[0]["client"].as_str().unwrap(), "", "and what arrives now is not");
        // the total the page names is the replies that finished, not the ones still in the window: nine of these
        // fell out of the table and are still counted, which is the difference between a heading and a row count
        assert_eq!(finished(), before + WINDOW + 9, "every reply kept is counted, kept or dropped");
    }

    #[test]
    fn the_prompt_counters_move_and_then_clear() {
        let _turn = COUNTERS.lock().unwrap();
        // the baseline is a state between two prompts, whatever test ran before this one
        read_end();
        // a forward pass outside a prompt is not reported: there is no block being read, and a page shown a layer
        // count then would read a decode step as prefill progress
        pass_begin(40);
        assert_eq!(pass_progress(), (0, 0), "no pass of a prompt that is not being read");
        read_begin(4096);
        for done in [256usize, 512, 2048] {
            read_done(done);
        }
        let (tokens, done, seconds) = reading();
        assert_eq!((tokens, done), (4096, 2048), "as far as the engine has read");
        assert!(seconds > 0.0, "the reading has taken time, and says so");
        // a prompt read again from a checkpoint goes back; the page is told so, rather than being shown the highest
        // number seen so far, which is a number the engine's own log would contradict
        read_done(1024);
        assert_eq!(reading().1, 1024, "a step back is kept as a step back");
        // the block rate is the engine's own number and reaches the page as it was measured
        read_rate(50.9, true);
        assert!((reading_rate() - 50.9).abs() < 1e-9, "the rate the engine measured, unchanged");
        // the rate comes with how long ago it was measured, because a block of this prompt's block size takes
        // minutes at a long context and the page has to be able to say so
        assert!(reading_rate_age().is_some_and(|age| age < 1.0), "measured just now");
        // inside a block, the layers of the pass are what move, and they are cleared with the rest of the prompt's
        // counters so that a page can never be shown a layer count left over from a block already finished
        pass_begin(40);
        assert_eq!(pass_progress(), (0, 40), "at the first layer, with a total so that zero is not mistaken");
        for _ in 0..17 {
            pass_layer();
        }
        assert_eq!(pass_progress(), (17, 40), "how far the pass has travelled");
        pass_begin(8);
        assert_eq!(pass_progress(), (0, 8), "a pass over part of the layers starts from nothing again");
        read_end();
        assert_eq!(reading(), (0, 0, 0.0), "nothing between replies");
        assert_eq!(reading_rate(), 0.0, "and no rate of a block that is not being read");
        assert_eq!(reading_rate_age(), None, "nor an age for one");
        assert_eq!(pass_progress(), (0, 0), "nor a pass still running after the prompt was read");

        assert!(note_token(), "the first token of a reply starts its interval");
        assert!(!note_token(), "the tokens after it do not restart it");
        assert!(writing() > 0.0, "the writing is measured from that first token");
        // two tokens written in the same instant are not a rate over recent seconds, and saying so is what lets the
        // page fall back to the average and name it, instead of holding its last number up
        assert!(writing_rate().is_none(), "no recent rate from two tokens in one instant");
        write_end();
        assert_eq!(writing(), 0.0, "nothing between replies");
        assert!(writing_rate().is_none(), "and a finished reply leaves no stamps for the next one");
        assert!(note_token(), "and the next reply starts an interval of its own");
        write_end();
    }

    /// A prompt's own average starts where the measuring starts: the block that begins a prompt also pays for the
    /// machine coming back to the model, and an average taken from before it describes that setup as reading speed.
    #[test]
    fn the_average_of_a_prompt_starts_after_its_first_block() {
        let _turn = COUNTERS.lock().unwrap();
        read_begin(4096);
        assert_eq!(reading_sustained(), 0.0, "nothing read yet, so nothing to average");
        // the block that begins the prompt: the slow one, and the reason the average below must not start before it
        std::thread::sleep(std::time::Duration::from_millis(150));
        read_done(1024);
        read_rate(20.0, true);
        assert_eq!(reading_sustained(), 0.0, "one block is a point, not a span between two");
        std::thread::sleep(std::time::Duration::from_millis(50));
        read_done(2048);
        read_rate(30.0, true);
        let (_, done, seconds) = reading();
        let whole = done as f64 / seconds;
        let since = reading_sustained();
        assert!(since > whole, "the cold first block is out of {since:.0}, and in the prompt's {whole:.0}");
        // and it holds between two reports. The span ends at a report, so a page polling twice inside one block gets
        // the same number twice, which is the difference between a measurement and a clock running down
        let held = since;
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert_eq!(reading_sustained(), held, "the average moves with the engine's reports, not with the polls");
        // a prompt read again from a checkpoint below that point has no tokens since it, and no rate is shown for it
        read_done(512);
        assert_eq!(reading_sustained(), 0.0, "a step back behind the first point is not an average");
        read_end();
        assert_eq!(reading_sustained(), 0.0, "and nothing between prompts");
    }

    /// A finished prompt's record keeps the span the live tile shows. The engine's last report of the reading ends
    /// it - not the end of the reply, which comes after the answer has been written.
    #[test]
    fn the_record_of_a_prompt_keeps_the_span_after_its_first_block() {
        let _turn = COUNTERS.lock().unwrap();
        // one block is a point: there is no span between two readings, and the record falls back to the prompt
        read_begin(500);
        std::thread::sleep(std::time::Duration::from_millis(20));
        read_done(500);
        read_rate(25.0, false);
        read_end();
        assert!(last_reading_span().is_none(), "one block stays a point");
        // a short prompt read in segment-edge passes (base, opener) reports twice without ever ending a block:
        // partial passes do not mark a block ended, so there is no span to average over
        read_begin(28);
        read_done(25);
        read_rate(20.0, false);
        read_done(28);
        read_rate(12.0, false);
        read_end();
        assert!(last_reading_span().is_none(), "two partial passes are still no block");
        // two blocks: the span lies between their reports, and the writing of the answer after it cannot move the end
        read_begin(2048);
        std::thread::sleep(std::time::Duration::from_millis(80));
        read_done(1024);
        read_rate(20.0, true);
        std::thread::sleep(std::time::Duration::from_millis(50));
        read_done(2048);
        read_rate(40.0, true);
        read_end();
        let (tokens, seconds) = last_reading_span().expect("two reports span a reading");
        assert_eq!(tokens, 1024, "the tokens are the ones read after the first block ended");
        assert!(seconds > 0.04 && seconds < 0.2, "the seconds are between the two reports, not {seconds:.3}");
        // the reply's end calls read_end a second time, and that second call must not write the zeroed counters
        // over the span the record asks for after it
        read_end();
        let (again, _) = last_reading_span().expect("the span survives the reply's end");
        assert_eq!(again, 1024, "the record asks for the span after the reply ended");
        // and a new prompt owns the number from its first instant
        read_begin(1024);
        assert!(last_reading_span().is_none(), "the next prompt starts with a span of its own, which is none yet");
        // and hand the counters back the way they were found: another test asserts what a state between two prompts
        // looks like, and it should not have to know which test ran before it
        read_end();
    }

    #[test]
    fn a_recent_rate_is_the_span_the_last_tokens_were_written_in() {
        // the arithmetic on a spacing we choose rather than on a clock, because a test that sleeps is a test that
        // sometimes fails on a busy machine
        let steady: Vec<u64> = (0..21).map(|i| 1_000_000_000 + i * 50_000_000).collect();
        let rate = rate_over(&steady).expect("twenty one stamps span a second");
        assert!((rate - 20.0).abs() < 0.01, "20 t/s is what those stamps were written at");
        // a reply that slows to a quarter of its starting speed. The last tokens give the new rate; dividing tokens by
        // the reply's whole seconds - what the old display did - still carries the fast start and reads nearly twice
        // the speed the engine is at. That drift is the fade the live number must not have.
        let fast: Vec<u64> = (0..40).map(|i| 1_000_000_000 + i * 50_000_000).collect();
        let slow: Vec<u64> = (0..8).map(|i| 1_000_000_000 + 40 * 50_000_000 + i * 250_000_000).collect();
        let all: Vec<u64> = fast.iter().chain(slow.iter()).copied().collect();
        let whole = (all.len() - 1) as f64 * 1e9 / (all[all.len() - 1] - all[0]) as f64;
        let recent = rate_over(&all[all.len() - RECENT_TOKENS..]).expect("the last tokens span more than a moment");
        assert!(whole > 12.0 && whole < 13.0, "the average over the whole reply says {whole:.1} t/s");
        assert!(recent > 6.0 && recent < 7.5, "the last tokens say {recent:.1}, which is the speed it is at now");
        assert!(rate_over(&[]).is_none(), "no tokens, no rate");
        assert!(rate_over(&[5_000_000_000]).is_none(), "one token, no span");
        // stamps in one instant divide by a span too small to mean anything: a hundredth of a millisecond between two
        // tokens is a speculative burst, not 20 000 t/s
        assert!(rate_over(&[5_000_000_000, 5_000_000_050]).is_none(), "two stamps 50 microseconds apart are no rate");
    }

    #[test]
    fn the_page_is_off_until_it_is_asked_for_and_then_only_as_far_as_it_was_opened() {
        // again one test for the matrix: the switches are the process's
        switches::set("TORC_STATUS", "0");
        switches::set("TORC_STATUS_NET", "0");
        assert!(!allowed(true) && !allowed(false), "off is off on both sides");
        switches::set("TORC_STATUS", "1");
        assert!(allowed(true), "this machine sees it");
        assert!(!allowed(false), "another machine does not, until it is opened");
        switches::set("TORC_STATUS_NET", "1");
        assert!(allowed(false), "opened, it does");
        switches::set("TORC_STATUS", "0");
        switches::set("TORC_STATUS_NET", "1");
        assert!(!allowed(false), "opening the network does not turn the page on");
    }

    #[test]
    fn the_warning_comes_only_with_the_two_switches_and_says_who_can_read_them() {
        // the reach is passed in rather than read from the switches, which other tests here change as they run
        let r = |page, net, open, keyed, key| Reach { page, net, open, keyed, key };
        let all = r(true, true, true, false, false);
        assert!(warning(false, false, all).is_none(), "the counters alone warn nobody");
        assert!(warning(true, false, r(false, true, true, false, false)).is_none(), "an off page shows nothing");
        let w = warning(true, false, all).unwrap();
        assert!(w.contains("text") && !w.contains("address"), "{w}");
        assert!(w.contains("any machine") && w.contains("TORC_STATUS_KEY=1"), "an open page says how to close it: {w}");
        let w = warning(false, true, all).unwrap();
        assert!(w.contains("address") && !w.contains("text"), "{w}");
        let w = warning(true, true, all).unwrap();
        assert!(w.contains("text") && w.contains("address"), "{w}");
        // not opened to the network, either by the page's switch or by where the server listens
        for local in [r(true, false, true, false, false), r(true, true, false, false, false)] {
            let w = warning(true, false, local).unwrap();
            assert!(w.contains("this machine") && !w.contains("any machine"), "{w}");
        }
        let w = warning(true, false, r(true, true, true, true, true)).unwrap();
        assert!(w.contains("has the API key") && !w.contains("TORC_STATUS_KEY=1"), "{w}");
        let w = warning(true, false, r(true, true, true, true, false)).unwrap();
        assert!(w.contains("this machine only") && w.contains("TORC_API_KEY"), "{w}");
    }

    #[test]
    fn the_page_passes_on_the_key_it_was_opened_with() {
        // TORC_STATUS_KEY refuses another machine's request without the key, and the page reads two addresses after
        // it loads: a key left off either is a page that opens and then goes blank
        assert!(PAGE_LF.contains("get(\"key\")"), "the page must read ?key= from its own address");
        assert!(PAGE_LF.contains("withKey(\"/torc/status\")"), "the data must be asked for with the key");
        assert!(PAGE_LF.contains("withKey(\"/torc/mark.png\")"), "so must the mark");
    }

    #[test]
    fn the_page_reads_the_address_the_data_is_answered_at() {
        // the route is written in two places, and a rename that misses one shows as a blank page
        assert!(PAGE_LF.contains("/torc/status"), "the page must poll what the server answers");
        assert!(!PAGE_LF.contains("http://") && !PAGE_LF.contains("https://"), "it loads nothing from anywhere else");
    }

    #[test]
    fn the_page_asks_for_the_fields_that_answer_tuning_and_current_speed() {
        // these keys are written on one side in Rust and on the other in JavaScript: a rename on the server side is a
        // line that goes blank on the page, and nobody notices until they are looking for that line
        for field in [
            "prompt_rate",
            "prompt_rate_sustained",
            "output_rate",
            "prompt_done",
            "prompt_tokens",
            "chosen",
            "in_use",
            "prompt_rate_age_s",
            "prompt_layer",
            "prompt_layers",
            "client",
            "client_name",
            "model_name",
            "reading_tokens",
            "reading_seconds",
            "prompt_total",
        ] {
            assert!(PAGE_LF.contains(field), "the page reads {field}, which status_json answers");
        }
        // the header is the desktop app's: its mark served by this server (not a copy of it in another folder, and not
        // a fetch to somewhere else), and the product named before the model, so a page on a LAN is recognisably ours
        // (the address is set from the script, so that it can carry the key the page was opened with)
        assert!(PAGE_LF.contains(".src = withKey(\"/torc/mark.png\")"), "the app's mark is asked for from this server");
        // the app's own file and crop, at the size a heading across a room is read at rather than the 28 px a
        // window draws it at - the file is not resized, only the box it is drawn into
        assert!(PAGE_LF.contains(".brand .mark { width: 50px"), "the mark is sized for the page, not for the app");
        // CSS has no line comment. A `//` in the stylesheet parses as the beginning of a selector and quietly takes the
        // rule after it with it: no error anywhere, and a mark that was meant to be 50 px is drawn at its file's 96
        let style = PAGE.split("<style>").nth(1).unwrap().split("</style>").next().unwrap();
        assert!(!style.lines().any(|l| l.trim_start().starts_with("//")), "no line comments in the stylesheet");
        assert!(PAGE_LF.contains("ToRc ${d.engine"), "the page leads with the product and its version");
        // the tuning answer belongs to the layout word as a hover, not to the page as a line above the rates: it is
        // looked up rather than watched, and it is the word it describes that should show when something is wrong
        assert!(PAGE_LF.contains("class=\"lay"), "the layout word is what carries the tuning answer");
        assert!(PAGE_LF.contains("title=\"${esc(note.text)}\""), "the answer is in a hover on that word");
        assert!(!PAGE_LF.contains("id=\"tuned\""), "and it is not a paragraph of its own over the numbers");
        // the layers of a pass are the only thing that moves between two blocks, so the page shows that as a second
        // segment of the bar - and it must stay drawn apart from the measured part. It rests on the layers costing the
        // same each, which wide MoE layers do not, so a reader has to be able to see which part is which
        assert!(PAGE_LF.contains(".bar i.part { opacity"), "the measured and the in-flight segment are drawn apart");
        // the two graphs are compared with each other by an eye reading across, so they are drawn at one height down
        // the page; that is a rule about the layout, so the assertion is about the layout too
        assert!(PAGE_LF.contains("#readspark, #writespark { margin-top: auto"), "both graphs sit at the same height");
        // and what is drawn inside them: a rate of nothing keeps its line, at a zero that sits inside the frame and
        // not on its edge, because a plot that stopped and a machine that is idle must not look the same
        assert!(PAGE_LF.contains("y1=\"${y(0)}\""), "the graphs always draw where their zero is, idle included");
        // dashed, so that a baseline cannot be read as a rate that held, and painted before the first answer so
        // that an open page is never an empty box
        // the plot keeps its own air: a CSS padding on the svg would push the drawing down inside its own box and the
        // box would clip the bottom of it, which is the row the line at 0 lives on
        assert!(
            !PAGE_LF.contains("padding-top: 6px"),
            "the gap above a graph is drawn, not padded - padding clips the line at 0 away"
        );
        assert!(
            PAGE_LF.contains("const stem = begun || pts.length < 2"),
            "the stem at the left end goes away once the window has scrolled past the point it belongs to"
        );
        assert!(
            PAGE_LF.contains("if (!H[which].length) BEGUN[which] = now;"),
            "the page remembers where each series began, past the point where its samples are trimmed"
        );
        assert!(
            // the frame's height is approached a step per poll: a ruler that snaps to what the window holds moves the
            // whole line sideways in scale while the machine holds one speed
            PAGE_LF.contains("cur + Math.sign(want - cur) * step")
                && PAGE_LF.contains("TOP[which] = Math.max(hi, eased);"),
            "the top of the scale is approached, never jumped to, and never left below the window's own high",
        );
        assert!(
            // staleness in the engine's own cadence: a block measured only when it ends is not a stalled number
            PAGE_LF.contains("return Math.max(8, 2 * Math.max(0, ...c.gaps));"),
            "a rate dims against the gap between the engine's reports, not against a fixed age"
        );
        assert!(
            // the bar over a graph is drawn at the graph's width, from one shared value
            PAGE_LF.contains(":root { --plot: 220px; }")
                && PAGE_LF.contains(".bar, #readspark, #writespark { width: var(--plot); max-width: 100%; }"),
            "the progress bar and the plot under it share one width"
        );
        assert!(
            PAGE_LF.contains("const y = (v) => 6 + (1 - v / top) * (h - 8);"),
            "zero is drawn inside the box, with the air above it part of the drawing"
        );
        assert!(
            PAGE_LF.contains("stroke-opacity=\".6\"") && PAGE_LF.contains("stroke-dasharray=\"3 3\""),
            "the line at 0 is dashed, in the series colour, under any measured rate"
        );
        assert!(
            PAGE_LF.contains("spark(\"write\", \"writespark\", \"var(--write)\");\npoll();"),
            "both graphs are drawn before the first answer arrives"
        );
        // the tint is the frame's, not the drawn history's: the history starts when the page was opened, and an empty
        // rest of the width reads as a plot that stopped rather than as a page that has not seen anything yet
        assert!(
            PAGE_LF.contains("width=\"${w}\" height=\"${h}\" fill=\""),
            "the plot area is tinted across its whole width, not only where history was drawn"
        );
        assert!(PAGE_LF.contains("fill-opacity=\".14\""), "and the line keeps a stronger tint down to its zero");
        // the history has a start, so it gets an edge: a stem from zero to the first value, and one sample alone is
        // still a measurement even though it cannot be a line. Once the window has scrolled past that first value the
        // stem is dropped, because the point it closes is no longer on the screen
        assert!(
            PAGE_LF.contains("y2=\"${y(pts[0][1])}\""),
            "the drawn series is closed at its left end by a stem, only while its first value is on screen"
        );
        // the state line is a line of its own above the tiles. As a column of the row it took width from the two
        // tiles that are compared side by side, which is what separated them once a narrow window wrapped the row
        assert!(
            PAGE_LF.contains("<div class=\"state\"><div id=\"state\" class=\"badge\">"),
            "the state badge is not a column of the tile row"
        );
        assert!(!PAGE_LF.contains("<div class=\"k\">state</div>"), "and it needs no label to say what it is");
        // one clock, 24-hour, whatever the browser's locale: the times are read against durations, and an hour that
        // might mean twelve hours either way is the ambiguity worth removing rather than leaving to the machine
        assert!(PAGE_LF.contains("{hour12: false}"), "the page keeps one clock, not the browser's");
        // the cache column names the thing it counts, the way the phrase a reader searches for is "cache hit"
        assert!(PAGE_LF.contains("Cache Hit"), "the cache column is called what people call it");
        // one header wraps, so all of them keep to the top of the column rather than to its middle
        assert!(
            PAGE_LF.contains("th { color: var(--ink3); vertical-align: top"),
            "a wrapped header does not pull the others down"
        );
        // a named address is shown as the name with the address behind it on hover, and an unnamed one as its address
        assert!(PAGE_LF.contains("r.client_name || r.client"), "an unmapped address is still shown as itself");
        // the element, not just the script that fills it: a page that draws into an id the markup never created
        // throws in the browser and stops drawing everything after that line
        assert!(PAGE_LF.contains("id=\"readpart\""), "the measured and the in-flight segment both exist in the markup");
        // the percent under the bar and the address column both exist as elements, and the column is hidden rather than
        // left empty when the switch is off, so that a table never shows a heading over nothing
        assert!(PAGE_LF.contains("id=\"readpct\""), "the bar's own label exists in the markup");
        assert!(
            PAGE_LF.contains("id=\"thclient\" hidden"),
            "the address column starts hidden and is asked for by name"
        );
        // what a running prompt says in words grows and shrinks. The numbers and graphs are laid out above those words
        // and their height is held, so that the page does not breathe under a reader's eye while a prompt is read
        assert!(PAGE_LF.contains("#readsub, #writesub { min-height"), "the text lines hold their height");
        // that a model was never measured on this machine is the one thing a reader must not have to infer from a
        // slow number, so the page has words for it
        assert!(PAGE_LF.contains("not measured on this machine"), "it says when a model runs on built-in rules");
        // a value somebody set with a switch is not the same fact as a value nobody chose: the page counts them
        // apart, because "not tuned" would make a deliberate choice look like an accident
        assert!(PAGE_LF.contains("set by switches"), "it names a switch-set value as such");
        assert!(PAGE_LF.contains("built-in rule"), "and names an unset one as that");
        // the count is over the settings this engine reports running, not over every name in the tuning file: an x86
        // build has no NPU tier, and a fifth item in a sentence about four values would be a wrong sentence
        assert!(PAGE_LF.contains("k in u"), "it counts only the settings the engine reports running");
        assert!(
            PAGE_LF.contains("rate over the last tokens"),
            "and it says which interval a live rate is measured over"
        );
        // a finished reply leaves no counters behind, so the tile would show a dash. The last reply's speed is a real
        // number about this server, so it is said in words instead
        assert!(
            PAGE_LF.contains("const lastWrite = last ? rate(last.output, last.output_seconds)"),
            "the past write rate comes from the record of replies, not from a counter"
        );
        assert!(PAGE_LF.contains("the last reply:"), "and the page names where that speed came from");
        assert!(PAGE_LF.contains("last prompt read:"), "the same for the last prompt's read speed");
        // a past speed stays words under the tile. The tile's own number means "this second", and a finished reply
        // has no this second, so it must not fill it in
        assert!(
            PAGE_LF.contains(".textContent = write ?"),
            "the write tile's number comes only from the reply being written"
        );
        // a request and its text are one entry: the line the shared cell border draws between them reads as another
        // row of the table, so the pair is marked and keeps a single line under it
        assert!(PAGE_LF.contains("tr.pair td { border-bottom: 0"), "a request and its text take one row's height");
        assert!(PAGE_LF.contains("class=\"pair\""), "and the row is marked when its text follows");
    }

    #[test]
    fn a_narrow_screen_keeps_naming_every_number() {
        // on a phone the header row alone is about twice the screen's width, so the columns get squeezed, headers
        // wrap over several lines and the last three leave the screen. Below that width the header goes away and a
        // reply becomes a block where each value carries the header's own words, so the two lists of names are one
        // list on purpose - a column renamed in the header but not in its label would print an unnumbered number on
        // exactly the one screen where nothing else names it.
        const HEADS: [&str; 9] = [
            "time",
            "from",
            "prompt tok",
            "Cache Hit tok",
            "reply tok",
            "read t/s",
            "write t/s",
            "model time s",
            "ended",
        ];
        for head in HEADS {
            assert!(PAGE_LF.contains(&format!("data-l=\"{head}\"")), "a narrow screen still says what {head} is");
        }
        assert!(
            PAGE_LF.contains("@media (max-width: 640px)"),
            "the change is width-based, not device-based: a narrow window gets the same block"
        );
        assert!(PAGE_LF.contains("#replies thead { display: none; }"), "the header that does not fit goes away");
        assert!(
            PAGE_LF.contains("#replies td[data-l]::before { content: attr(data-l); display: block;"),
            "a caption sharing a line with its value splits a long number over two lines instead"
        );
        // a reply's end sits beside its start rather than at the end of the block, which is what makes the three
        // lines read as when, what and how fast
        assert!(
            PAGE_LF.contains("#replies td[data-l=\"ended\"] { grid-column: 3; grid-row: 1; }"),
            "and it is placed there, not left to fall out of the column order"
        );
    }

    #[test]
    fn a_row_shows_the_answer_and_says_when_it_is_thinking_instead() {
        let (text, thinking) = snippet("The third planet is Earth.", "Let me consider the planets.");
        assert_eq!((text.as_str(), thinking), ("The third planet is Earth.", false), "the answer wins");
        let (text, thinking) = snippet("", "Let me consider the planets.");
        assert_eq!((text.as_str(), thinking), ("Let me consider the planets.", true), "thinking, named as such");
        let (text, thinking) = snippet("", "");
        assert_eq!((text.as_str(), thinking), ("", false), "nothing said, nothing to name");
    }

    #[test]
    fn a_snippet_is_cut_on_a_character_not_a_byte() {
        let long = "über".repeat(200);
        let c = cut(&long);
        assert!(c.chars().count() <= SNIPPET + 1, "cut to the window");
        assert!(c.is_char_boundary(c.len()), "and on a boundary");
        assert!(c.ends_with("…"), "and says so");
    }
}
