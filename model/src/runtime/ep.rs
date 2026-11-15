//! ep.rs - two machines for a prompt, expert parallel.
//!
//! The main machine sends a block's odd-numbered routed experts to a worker (`torc-run <model> serve-ep=<port>` on the
//! other machine, with the same file), computes the even ones and the shared one in the meantime, and receives each of
//! the worker's pairs' output rows. The rows land in their own places, and every token sums its experts in increasing
//! id as on one machine, so the bits are the same. Each machine's expert cache then holds only its half of the experts.
//! This is for prompt blocks only (TORC_EP_MIN_N tokens, 32 by default): a single token's pass pays the round trip at
//! every layer.
//!
//! Set TORC_EP=host:port on the main machine. A request is "TEP2", layer, group, hc, n, nb and dm, then every token's
//! input as Q8 (n x nb blocks of the K-quants' array, then n x nb of the 32-groups), then the jobs: a count, and for
//! each job its expert, token count and tokens. The reply is the pair count, then each pair's row (dm f32), with the
//! jobs in order and the tokens in order.

use crate::moe::{EJob, mats_of, run_ejobs};
use crate::weights::Weights;
use format::{Result, err};
use kernels::{Q8, Q8s};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Mutex, OnceLock};

const MAGIC: u32 = 0x3250_4554;

/// On the main machine, the seconds summed for sending requests and for waiting for replies after the local work (the
/// TORC_PROF tables print them via stats()).
static SEND_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn add(a: &std::sync::atomic::AtomicU64, t: std::time::Instant) {
    a.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// (seconds sending, seconds waiting for the worker) so far.
pub fn stats() -> (f64, f64) {
    let r = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
    (r(&SEND_NS), r(&WAIT_NS))
}

/// The connection to the worker (TORC_EP), made on first use.
pub fn client() -> Option<&'static Mutex<TcpStream>> {
    static C: OnceLock<Option<Mutex<TcpStream>>> = OnceLock::new();
    C.get_or_init(|| {
        let a = crate::switches::get("TORC_EP")?;
        match TcpStream::connect(&a) {
            Ok(s) => {
                let _ = s.set_nodelay(true);
                eprintln!("two boxes: worker at {a}");
                Some(Mutex::new(s))
            }
            Err(e) => {
                eprintln!("two boxes: cannot reach {a}: {e}; one box");
                None
            }
        }
    })
    .as_ref()
}

/// The fewest tokens a block must have to send experts away (TORC_EP_MIN_N).
pub fn min_tokens() -> usize {
    static N: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_EP_MIN_N").unwrap_or(32));
    *N
}

fn u32s(b: &mut Vec<u8>, v: &[u32]) {
    for x in v {
        b.extend(x.to_le_bytes());
    }
}

/// Sends a block's jobs (expert, tokens) for a layer, with every token's Q8 input going along.
#[allow(clippy::too_many_arguments)]
pub fn send(
    s: &mut TcpStream,
    layer: u32,
    group: u32,
    hc: bool,
    gelu: bool,
    n: usize,
    nb: usize,
    dm: usize,
    xq: Q8s,
    jobs: &[(usize, &[usize])],
) -> std::io::Result<()> {
    let t = std::time::Instant::now();
    let r = send_inner(s, layer, group, hc, gelu, n, nb, dm, xq, jobs);
    add(&SEND_NS, t);
    r
}

#[allow(clippy::too_many_arguments)]
fn send_inner(
    s: &mut TcpStream,
    layer: u32,
    group: u32,
    hc: bool,
    gelu: bool,
    n: usize,
    nb: usize,
    dm: usize,
    xq: Q8s,
    jobs: &[(usize, &[usize])],
) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(64 + jobs.len() * 16);
    // the flags word: bit 0 is hc (qwen4exp's ggml quantisation) and bit 1 GELU experts (gemma4)
    u32s(&mut b, &[MAGIC, layer, group, hc as u32 | (gelu as u32) << 1, n as u32, nb as u32, dm as u32]);
    s.write_all(&b)?;
    let (kb, gb) = xq.sub(..n * nb).bytes();
    s.write_all(kb)?;
    s.write_all(gb)?;
    b.clear();
    u32s(&mut b, &[jobs.len() as u32]);
    for (e, toks) in jobs {
        u32s(&mut b, &[*e as u32, toks.len() as u32]);
        for &t in toks.iter() {
            b.extend((t as u32).to_le_bytes());
        }
    }
    s.write_all(&b)?;
    s.flush()
}

fn read_u32(s: &mut TcpStream) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    s.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Reads the reply's rows (pairs x dm f32, with the jobs in order).
pub fn recv(s: &mut TcpStream, pairs: usize, dm: usize, out: &mut [f32]) -> Result<()> {
    let t = std::time::Instant::now();
    let r = recv_inner(s, pairs, dm, out);
    add(&WAIT_NS, t);
    r
}

fn recv_inner(s: &mut TcpStream, pairs: usize, dm: usize, out: &mut [f32]) -> Result<()> {
    let got = read_u32(s).map_err(|e| format::Error(format!("two boxes: {e}")))? as usize;
    if got != pairs {
        return err(format!("two boxes: {got} rows back, {pairs} asked"));
    }
    // SAFETY: f32 has no invalid bit patterns, and out holds pairs x dm
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, pairs * dm * 4) };
    s.read_exact(bytes).map_err(|e| format::Error(format!("two boxes: {e}")))?;
    Ok(())
}

/// The worker: serves requests in turn until the main machine hangs up, then takes the next connection.
pub fn serve(w: &Weights, port: u16) -> Result<()> {
    let host = crate::pipe::bind_host();
    let l = TcpListener::bind((host.as_str(), port)).map_err(|e| format::Error(format!("serve-ep: {e}")))?;
    eprintln!("two boxes: serving experts on {host}:{port} (no authentication: a trusted link only)");
    for conn in l.incoming() {
        let Ok(mut s) = conn else { continue };
        let _ = s.set_nodelay(true);
        eprintln!("two boxes: main box connected");
        loop {
            match serve_one(w, &mut s) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    eprintln!("two boxes: {e}");
                    break;
                }
            }
        }
    }
    Ok(())
}

/// The most a request may ask for: tokens in a block, quantised blocks a token, and the output width.
const MAX_N: usize = 1 << 14;
const MAX_NB: usize = 1 << 12;
const MAX_DM: usize = 1 << 16;

/// One request; returns false when the connection closed cleanly.
fn serve_one(w: &Weights, s: &mut TcpStream) -> Result<bool> {
    let io = |e: std::io::Error| format::Error(format!("two boxes: {e}"));
    let mut h = [0u8; 28];
    match s.read_exact(&mut h) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(e) => return Err(io(e)),
    }
    let g = |i: usize| u32::from_le_bytes(h[4 * i..4 * i + 4].try_into().unwrap());
    if g(0) != MAGIC {
        return err("two boxes: bad request");
    }
    let (layer, group, hc, gelu, n, nb, dm) =
        (g(1), g(2), g(3) & 1 != 0, g(3) & 2 != 0, g(4) as usize, g(5) as usize, g(6) as usize);
    // sizes off the wire are bounded before anything is allocated or indexed with them
    if n == 0 || n > MAX_N || nb == 0 || nb > MAX_NB || dm == 0 || dm > MAX_DM || group as usize >= w.file.experts.len()
    {
        return err("two boxes: bad request");
    }
    let n_expert = w.layer_experts(group).n_expert;
    let mut xq = Q8::new(n * nb);
    let mut xm = xq.view_mut();
    let (kb, gb) = xm.bytes_mut();
    s.read_exact(kb).map_err(io)?;
    s.read_exact(gb).map_err(io)?;
    let nj = read_u32(s).map_err(io)? as usize;
    if nj > n_expert as usize {
        return err("two boxes: bad request");
    }
    let mut jobs: Vec<(u32, Vec<usize>)> = Vec::with_capacity(nj);
    for _ in 0..nj {
        let e = read_u32(s).map_err(io)?;
        let k = read_u32(s).map_err(io)? as usize;
        if e >= n_expert || k > n {
            return err("two boxes: bad request");
        }
        let mut toks = Vec::with_capacity(k);
        for _ in 0..k {
            let t = read_u32(s).map_err(io)? as usize;
            if t >= n {
                return err("two boxes: bad request");
            }
            toks.push(t);
        }
        jobs.push((e, toks));
    }
    let pairs: usize = jobs.iter().map(|j| j.1.len()).sum();
    let t_comp = std::time::Instant::now();
    let mut out = vec![0.0f32; pairs * dm];
    let per = w.cache.as_ref().map_or(128, |c| c.layer_slots(layer).clamp(1, 128));
    let mut g0 = 0;
    for chunk in jobs.chunks(per) {
        let ids: Vec<u32> = chunk.iter().map(|j| j.0).collect();
        let slots = w.experts_of(layer, group, &ids)?;
        let mut ej: Vec<EJob> = Vec::with_capacity(chunk.len());
        for (j, (slot, lay)) in chunk.iter().zip(slots) {
            ej.push(EJob { mats: mats_of(slot, lay), toks: j.1.clone(), g0 });
            g0 += j.1.len();
        }
        run_ejobs(hc, gelu, &ej, xq.view(), nb, dm, &mut out);
    }
    let comp = t_comp.elapsed().as_secs_f64();
    let t_w = std::time::Instant::now();
    s.write_all(&(pairs as u32).to_le_bytes()).map_err(io)?;
    // SAFETY: the f32 rows viewed as bytes
    let ob = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4) };
    s.write_all(ob).map_err(io)?;
    s.flush().map_err(io)?;
    // every 48 requests (a block of the 48-layer models): the compute and reply-writing seconds, and the pairs
    static W: std::sync::Mutex<(u32, f64, f64, usize)> = std::sync::Mutex::new((0, 0.0, 0.0, 0));
    let mut g = W.lock().unwrap();
    *g = (g.0 + 1, g.1 + comp, g.2 + t_w.elapsed().as_secs_f64(), g.3 + pairs);
    if g.0.is_multiple_of(48) {
        eprintln!("two boxes worker: {} requests, compute {:.2} s, reply {:.2} s, {} pairs", g.0, g.1, g.2, g.3);
    }
    Ok(true)
}
