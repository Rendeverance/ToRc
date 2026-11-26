//! inkling.rs - Inkling 2 Mini (general.architecture inkling2_loop).
//!
//! How it differs from the other models here:
//! - 24 layers of weights make 48 layer passes a token: layers 6 to 17 run three times. Every pass has its own keys,
//!   values and convolution state. From the second pass on, each looped layer's inputs take a message from the pass
//!   before (a mix of that pass's block outputs) through a low-rank control: (1 - forget) x + gain message.
//! - The layers come in blocks of six. A block's first layer starts a new residual stream, and the stream it leaves
//!   joins a list that later layers mix back in (a softmax over the list and the current stream, by learned queries)
//!   before attention, before the experts and before the head.
//! - Attention has no rotary positions. Each query makes a small vector (16 a head) that, against a learned table by
//!   distance, gives a bias for every key up to 1024 positions back, and a sink logit of its own that takes a share of
//!   the weight and points at nothing. Queries and keys are clipped to a norm, values are normed, and the output is
//!   gated. Five layers of six look back over a window of 1024 positions; the sixth sees everything, with its queries
//!   scaled up slowly with the position.
//! - Short causal convolutions (4 taps, the last 3 inputs kept) follow the keys, the values, the attention output and
//!   the experts' output; on the full-attention layers the keys' and values' taps also depend on the input.
//! - The routed experts (128, 8 a token) are picked by sigmoid scores plus a bias, weighted by the scores alone
//!   (normed to sum 1, times 8), and use SITU: beta tanh(gate / beta) sigmoid(gate) * lb tanh(up / lb).
//!
//! A block of n tokens runs every matrix product for all n at once; each token's numbers are the same bits as a block
//! of one.

use crate::moe::{Mat, apply_group_multi};
use crate::ops::rms_norm;
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::meta::{Meta, Value};
use format::{BlockKind, Result, err};
use kernels::{Q8, QK, ops as kops};

/// The most tokens one call of forward takes.
pub const BLOCK_MAX: usize = 1024;

#[derive(Clone, Debug)]
pub struct InkHp {
    /// the layers of weights, and the layer passes a token makes
    pub n_layer: usize,
    pub n_pass: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    /// the vocabulary (the embedding and the head are padded past it)
    pub n_vocab: usize,
    pub eps: f32,
    /// the sliding window, and which layers have it
    pub window: usize,
    pub swa: Vec<bool>,
    pub n_expert: usize,
    pub n_used: usize,
    /// the experts' width, and that padded to whole 256-number blocks (as the converter pads gate's and up's rows)
    pub d_ff: usize,
    pub d_ff_pad: usize,
    /// the convolutions' taps
    pub n_conv: usize,
    /// layers a block
    pub block: usize,
    /// the relative-position vector's length a head, and how far back the bias table goes
    pub n_rel: usize,
    pub extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub w_scale: f32,
    pub softcap: f32,
    /// the looped layers [start, end), how many times they run, and the control's rank
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    pub rank: usize,
}

fn up256(n: usize) -> usize {
    n.div_ceil(QK) * QK
}

impl InkHp {
    pub fn from_meta(metadata: &[u8]) -> Result<InkHp> {
        let m = Meta::parse(metadata)?;
        let arch = m.str("general.architecture").unwrap_or("");
        if arch != "inkling2_loop" {
            return err(format!("not an Inkling 2 file ({arch})"));
        }
        let missing = |k: &str| format::Error(format!("missing {arch}.{k}"));
        let need = |k: &str| m.arch_u32(k).map(|v| v as usize).ok_or_else(|| missing(k));
        let needf = |k: &str| m.arch_f32(k).ok_or_else(|| missing(k));
        let n_layer = need("block_count")?;
        // a count that is the same for every layer, stored as one number or as a list
        let same = |k: &str| -> Result<usize> {
            if let Some(v) = m.arch_u32(k) {
                return Ok(v as usize);
            }
            let v = m.ints(&m.arch_key(k)).ok_or_else(|| missing(k))?;
            match v.first() {
                Some(&a) if v.len() == n_layer && v.iter().all(|&b| b == a) && a > 0 => Ok(a as usize),
                _ => err(format!("{arch}.{k}: layers that differ are not built in ToRc yet")),
            }
        };
        let swa = match m.get(&m.arch_key("attention.sliding_window_pattern")) {
            Some(Value::Array(_, v)) if v.len() == n_layer => {
                v.iter().map(|x| matches!(x, Value::Bool(true))).collect::<Vec<bool>>()
            }
            _ => return Err(missing("attention.sliding_window_pattern")),
        };
        let d_ff = same("expert_feed_forward_length")?;
        let (loop_start, loop_end, loop_count) = (need("loop.start")?, need("loop.end")?, need("loop.count")?);
        let block = need("attn_res.block_size")?;
        let h = InkHp {
            n_layer,
            n_pass: n_layer + (loop_count.max(1) - 1) * loop_end.saturating_sub(loop_start),
            d_model: need("embedding_length")?,
            n_head: same("attention.head_count")?,
            n_head_kv: same("attention.head_count_kv")?,
            head_dim: need("attention.key_length")?,
            n_vocab: need("vocab_size_unpadded")?,
            eps: needf("attention.layer_norm_rms_epsilon")?,
            window: need("attention.sliding_window")?,
            swa,
            n_expert: need("expert_count")?,
            n_used: same("expert_used_count")?,
            d_ff,
            d_ff_pad: up256(d_ff),
            n_conv: need("shortconv.l_cache")?,
            block,
            n_rel: need("attention.relative_length")?,
            extent: need("attention.relative_extent")?,
            qk_clip: needf("attention.qk_clip")?,
            rel_clip: needf("attention.relative_clip")?,
            log_floor: need("attention.log_scale_floor")? as f32,
            log_alpha: needf("attention.log_scale_alpha")?,
            situ_beta: needf("activation.situ_beta")?,
            situ_lb: needf("activation.situ_linear_beta")?,
            w_scale: m.arch_f32("expert_weights_scale").unwrap_or(1.0),
            softcap: m.arch_f32("final_logit_softcapping").unwrap_or(0.0),
            loop_start,
            loop_end,
            loop_count,
            rank: need("loop.gate_rank")?,
        };
        if h.n_head_kv == 0
            || !h.n_head.is_multiple_of(h.n_head_kv)
            || !h.d_model.is_multiple_of(QK)
            || !(h.n_head * h.head_dim).is_multiple_of(QK)
            || h.n_conv != 4
            || block == 0
            || loop_count < 2
            || loop_start == 0
            || loop_start >= loop_end
            || loop_end > n_layer
            || !loop_start.is_multiple_of(block)
            || !loop_end.is_multiple_of(block)
            || h.extent != h.window
            || !h.swa[0]
            || m.arch_u32("loop.varying").unwrap_or(0) != 0
        {
            return err("Inkling 2: a shape this build cannot run");
        }
        Ok(h)
    }
}

/// A looped layer's control for one input (the query's, the keys', the values' or the experts'): down (rank x 2
/// d_model), gain and forget (d_model x rank each).
struct Ctrl {
    down: Vec<f32>,
    gain: Vec<f32>,
    forget: Vec<f32>,
}

/// What a looped layer has besides its weights: its four message queries (4 x d_model) and controls.
struct Loop {
    msg_query: Vec<f32>,
    ctrl: [Ctrl; 4],
}

/// A layer's own numbers (its matrices stay in the file).
struct ILayer {
    /// the sliding-window layers' one input norm, or the full-attention layers' three (query, keys, values)
    in_norm: Vec<Vec<f32>>,
    q_clip: Vec<f32>,
    rel_bias: Vec<f32>,
    rel_clip: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    attn_conv: Vec<f32>,
    ffn_conv: Vec<f32>,
    /// the queries that mix the earlier blocks into the attention's inputs (one, or three on a full-attention layer;
    /// none on the first layer) and into the experts' input
    attn_query: Vec<f32>,
    ffn_query: Vec<f32>,
    ffn_norm: Vec<f32>,
    router: Vec<f32>,
    pick_bias: Vec<f32>,
    group: u32,
    looped: Option<Loop>,
}

/// One layer pass's state: its keys and values at every position (so a prompt that extends an earlier one finds
/// them), and the last three inputs of its four convolutions (keys, values, attention output, the experts' output),
/// oldest first.
struct Pass {
    k: Vec<f32>,
    v: Vec<f32>,
    cap: usize,
    hist: Vec<f32>,
}

/// The time spent in each phase of forward, summed over calls (TORC_PROF=1).
pub static PROF: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
pub const PHASES: [&str; 9] = [
    "embed, mixes and head",
    "inputs",
    "q, k, v products",
    "convolutions and clips",
    "attention",
    "output",
    "router",
    "experts",
    "expert sums",
];

pub struct Ink {
    pub h: InkHp,
    pub w: Weights,
    layers: Vec<ILayer>,
    pass: Vec<Pass>,
    tok_norm: Vec<f32>,
    out_norm: Vec<f32>,
    out_query: Vec<f32>,
    pub n_ctx: u32,
}

/// Runs f(i) for i in 0..n: on this thread for a single item, else over the pool (each item writes only its own
/// rows, so the bits are the same either way).
fn each(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n < 2 { (0..n).for_each(f) } else { crate::pool::par_for(n, f) }
}

/// The rows of x (n x d) quantised for the matrix products.
fn quant(x: &[f32], n: usize, d: usize) -> Q8 {
    let nb = d / QK;
    let mut q = Q8::new(n * nb);
    let qp = crate::pool::Q8Ptr::new(q.view_mut());
    // SAFETY: item t owns token t's blocks of q
    each(n, &|t| kops::q8k_quantize(&x[t * d..(t + 1) * d], unsafe { qp.slice(t * nb, nb) }));
    q
}

/// x over its root mean square (no weights).
fn rms(out: &mut [f32], x: &[f32], eps: f32) {
    let mut ss = 0.0f64;
    for &v in x {
        ss += v as f64 * v as f64;
    }
    let scale = 1.0f32 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v * scale;
    }
}

/// The product of two rows, summed in 16 lanes in a fixed order (so the bits do not depend on the machine's vector
/// width): lane i takes every 16th pair from i, then the lanes are added in pairs.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] = x[i].mul_add(y[i], acc[i]);
        }
    }
    for (i, (x, y)) in ra.iter().zip(rb).enumerate() {
        acc[i] = x.mul_add(*y, acc[i]);
    }
    let mut w = 16;
    while w > 1 {
        w /= 2;
        for i in 0..w {
            acc[i] += acc[i + w];
        }
    }
    acc[0]
}

/// x scaled down to a length of at most `limit`, after an optional scale a number.
fn clip(x: &mut [f32], scale: Option<&[f32]>, limit: f32) {
    if let Some(s) = scale {
        for (v, k) in x.iter_mut().zip(s) {
            *v *= k;
        }
    }
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    let f = (limit / norm).clamp(0.0, 1.0);
    for v in x.iter_mut() {
        *v *= f;
    }
}

/// The mix over the depth: each token's earlier block outputs and its current stream, each over its root mean
/// square, weighted by a softmax of their products with each of the `nq` queries (over the square root of the width).
/// n x nq x dm.
fn mix(blocks: &[Vec<f32>], cur: &[f32], query: &[f32], n: usize, dm: usize) -> Vec<f32> {
    let nq = query.len() / dm;
    let k = blocks.len() + 1;
    let mut out = vec![0.0f32; n * nq * dm];
    let op = SyncPtr::new(&mut out);
    each(n, &|t| {
        let mut st = vec![0.0f32; k * dm];
        for (i, b) in blocks.iter().enumerate() {
            rms(&mut st[i * dm..(i + 1) * dm], &b[t * dm..(t + 1) * dm], 1e-6);
        }
        rms(&mut st[(k - 1) * dm..], &cur[t * dm..(t + 1) * dm], 1e-6);
        // SAFETY: item t owns token t's rows of out
        let o = unsafe { op.slice(t * nq * dm, nq * dm) };
        let scale = 1.0 / (dm as f32).sqrt();
        for j in 0..nq {
            let q = &query[j * dm..(j + 1) * dm];
            let s: Vec<f32> = (0..k).map(|i| dot(&st[i * dm..(i + 1) * dm], q) * scale).collect();
            let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = s.iter().map(|v| (v - mx).exp()).sum();
            let oj = &mut o[j * dm..(j + 1) * dm];
            for (i, v) in s.iter().enumerate() {
                let p = (v - mx).exp() / sum;
                for (a, b) in oj.iter_mut().zip(&st[i * dm..(i + 1) * dm]) {
                    *a = p.mul_add(*b, *a);
                }
            }
        }
    });
    out
}

/// A looped layer's input with the message of the pass before: (1 - forget) x + gain message, gain and forget each
/// tanh of a rank-wide product of x and the message over their root mean squares. In place, n rows of dm.
fn control(x: &mut [f32], msg: &[f32], c: &Ctrl, n: usize, dm: usize, rank: usize) {
    // the rank-wide products first (each token's inputs over their root mean squares, then CTRL_SPLIT runs of the
    // rank a token), then the gains and forgets in CTRL_SPLIT runs of the width a token: every number is the same dot
    // as a token at a time, so the bits do not depend on n or on the split
    let mut g = vec![0.0f32; n * 2 * dm];
    {
        let gp = SyncPtr::new(&mut g);
        each(n, &|t| {
            // SAFETY: item t owns token t's row of g
            let gt = unsafe { gp.slice(t * 2 * dm, 2 * dm) };
            rms(&mut gt[..dm], &x[t * dm..(t + 1) * dm], 1e-6);
            rms(&mut gt[dm..], &msg[t * dm..(t + 1) * dm], 1e-6);
        });
    }
    let mut low = vec![0.0f32; n * rank];
    {
        let lp = SyncPtr::new(&mut low);
        let per = rank.div_ceil(CTRL_SPLIT);
        each(n * CTRL_SPLIT, &|i| {
            let (t, s) = (i / CTRL_SPLIT, i % CTRL_SPLIT);
            let (r0, r1) = ((s * per).min(rank), ((s + 1) * per).min(rank));
            let gt = &g[t * 2 * dm..(t + 1) * 2 * dm];
            // SAFETY: item i owns ranks r0..r1 of token t
            let lt = unsafe { lp.slice(t * rank + r0, r1 - r0) };
            for (o, r) in lt.iter_mut().zip(r0..r1) {
                *o = dot(&c.down[r * 2 * dm..(r + 1) * 2 * dm], gt);
            }
        });
    }
    let xp = SyncPtr::new(x);
    let per = dm.div_ceil(CTRL_SPLIT);
    each(n * CTRL_SPLIT, &|i| {
        let (t, s) = (i / CTRL_SPLIT, i % CTRL_SPLIT);
        let (i0, i1) = ((s * per).min(dm), ((s + 1) * per).min(dm));
        // SAFETY: item i owns numbers i0..i1 of token t's row of x
        let xt = unsafe { xp.slice(t * dm + i0, i1 - i0) };
        let mt = &msg[t * dm + i0..t * dm + i1];
        let lt = &low[t * rank..(t + 1) * rank];
        for (j, (xv, mv)) in xt.iter_mut().zip(mt).enumerate() {
            let i = i0 + j;
            let gain = dot(&c.gain[i * rank..(i + 1) * rank], lt).tanh();
            let forget = dot(&c.forget[i * rank..(i + 1) * rank], lt).tanh();
            *xv = (*xv - forget * *xv) + gain * mv;
        }
    });
}

/// How many runs a token's control is split into, so a single token's work spreads over the pool.
const CTRL_SPLIT: usize = 16;

/// A causal convolution of 4 taps with its residual, in place over n rows of c numbers, in token order: each
/// number's last three inputs come from `hist` (3 x c, oldest first), which moves on. `taps` is c x 4, the last tap
/// on the current input. `dynamic` adds taps that depend on the input (n rows of heads x 4, a head's first tap on
/// the current input). The sum with the input is scaled by 1 / sqrt(taps + 1), the dynamic taps counted too.
fn convolve(x: &mut [f32], n: usize, c: usize, taps: &[f32], dynamic: Option<(&[f32], usize)>, hist: &mut [f32]) {
    let scale = 1.0 / ((if dynamic.is_some() { 8.0f32 } else { 4.0 }) + 1.0).sqrt();
    for t in 0..n {
        let xt = &mut x[t * c..(t + 1) * c];
        for i in 0..c {
            let w = &taps[i * 4..i * 4 + 4];
            let past = [hist[i], hist[c + i], hist[2 * c + i], xt[i]];
            let mut out = 0.0f32;
            for k in 0..4 {
                out += past[k] * w[k];
            }
            if let Some((d, heads)) = dynamic {
                let hh = i / (c / heads);
                let tap = &d[(t * heads + hh) * 4..(t * heads + hh) * 4 + 4];
                for lag in 0..4 {
                    out += past[3 - lag] * tap[lag];
                }
            }
            hist[i] = past[1];
            hist[c + i] = past[2];
            hist[2 * c + i] = past[3];
            xt[i] = (past[3] + out) * scale;
        }
    }
}

impl Ink {
    pub fn load(path: &str, n_ctx: u32) -> Result<Ink> {
        // ARM: the experts on large pages by default (TORC_EXPERTS_LARGE=0 turns it off). Box 1, 26 Nov: the cold first
        // pass 5.4 -> 18.8 tok/s and the experts' products 30 -> 100 GB/s when each token asks for other experts; the
        // mapped pages are handed back after the copy there (weights.rs), so only one copy is held
        if cfg!(target_arch = "aarch64") && !crate::switches::given("TORC_EXPERTS_LARGE") {
            crate::switches::set("TORC_EXPERTS_LARGE", "1");
        }
        let w = Weights::load(path)?;
        let h = InkHp::from_meta(&w.file.metadata)?;
        crate::pipe::set_ident(
            &w.file.header.model_id,
            w.file.header.file_bytes,
            h.n_layer as u32,
            h.d_model as u32,
            0,
            w.file.tensors.len(),
            n_ctx,
        );
        let (dm, kvd) = (h.d_model, h.n_head_kv * h.head_dim);
        let mut layers = Vec::with_capacity(h.n_layer);
        for l in 0..h.n_layer {
            let lw = l as u32;
            let group = w
                .file
                .blocks
                .iter()
                .find(|b| b.layer == lw && b.kind == BlockKind::Moe && b.experts.is_some())
                .and_then(|b| b.experts)
                .ok_or_else(|| format::Error(format!("layer {l}: no routed experts")))?;
            let x = w.layer_experts(group);
            if x.n_expert as usize != h.n_expert || x.d_in as usize != dm || x.width as usize != h.d_ff_pad {
                return err(format!(
                    "layer {l}: experts of {} x {} x {}, expected {} x {} x {} (convert the GGUF again)",
                    x.n_expert, x.d_in, x.width, h.n_expert, dm, h.d_ff_pad
                ));
            }
            let f = |s: &str| -> Result<Vec<f32>> { w.layer(lw, s)?.f32s() };
            let in_norm = if h.swa[l] {
                vec![f("attn_norm.weight")?]
            } else {
                vec![f("attn_q_inp_norm.weight")?, f("attn_k_inp_norm.weight")?, f("attn_v_inp_norm.weight")?]
            };
            let looped = if (h.loop_start..h.loop_end).contains(&l) {
                let ctrl = |p: &str| -> Result<Ctrl> {
                    Ok(Ctrl {
                        down: f(&format!("loop_{p}_ctrl_down.weight"))?,
                        gain: f(&format!("loop_{p}_ctrl_gain.weight"))?,
                        forget: f(&format!("loop_{p}_ctrl_forget.weight"))?,
                    })
                };
                Some(Loop {
                    msg_query: f("loop_msg_query.weight")?,
                    ctrl: [ctrl("q")?, ctrl("k")?, ctrl("v")?, ctrl("ffn")?],
                })
            } else {
                None
            };
            let ly = ILayer {
                in_norm,
                q_clip: f("attn_q_clip.weight")?,
                rel_bias: f("attn_rel_bias.weight")?,
                rel_clip: f("attn_rel_clip.weight")?,
                k_conv: f("attn_k_conv.weight")?,
                v_conv: f("attn_v_conv.weight")?,
                attn_conv: f("attn_post_conv.weight")?,
                ffn_conv: f("ffn_post_conv.weight")?,
                attn_query: if l > 0 { f("attn_res_query.weight")? } else { Vec::new() },
                ffn_query: f("ffn_res_query.weight")?,
                ffn_norm: f("ffn_norm.weight")?,
                router: f("ffn_gate_inp.weight")?,
                pick_bias: f("exp_probs_b.bias")?,
                group,
                looped,
            };
            if ly.rel_bias.len() != h.n_rel * h.extent
                || ly.q_clip.len() != h.n_head * h.head_dim
                || ly.k_conv.len() != kvd * 4
                || ly.attn_query.len()
                    != if l == 0 {
                        0
                    } else if h.swa[l] {
                        dm
                    } else {
                        3 * dm
                    }
            {
                return err(format!("layer {l}: a tensor of an unexpected shape"));
            }
            layers.push(ly);
        }
        // a layer pass's state: the passes after the first come after the layers, a looped stretch at a time
        let pass = (0..h.n_pass)
            .map(|_| {
                let cap = n_ctx as usize;
                Pass { k: vec![0.0; cap * kvd], v: vec![0.0; cap * kvd], cap, hist: vec![0.0; 3 * (2 * kvd + 2 * dm)] }
            })
            .collect();
        let tok_norm = w.need("token_embd_norm.weight")?.f32s()?;
        let out_norm = w.need("output_norm.weight")?.f32s()?;
        let out_query = w.need("output_res_query.weight")?.f32s()?;
        Ok(Ink { h, w, layers, pass, tok_norm, out_norm, out_query, n_ctx })
    }

    pub fn n_vocab(&self) -> usize {
        self.h.n_vocab
    }

    /// Every pass's convolution state, end to end: with the keys and values (kept by position), the model's whole
    /// state after the tokens read so far. A prompt can go on from position p only with the state as it was at p.
    pub fn conv_state(&self) -> Vec<f32> {
        self.pass.iter().flat_map(|p| p.hist.iter().copied()).collect()
    }

    /// Puts conv_state back.
    pub fn set_conv_state(&mut self, state: &[f32]) -> Result<()> {
        if state.len() != self.pass.iter().map(|p| p.hist.len()).sum::<usize>() {
            return err("a convolution state of another shape");
        }
        let mut at = 0;
        for p in &mut self.pass {
            let n = p.hist.len();
            p.hist.copy_from_slice(&state[at..at + n]);
            at += n;
        }
        Ok(())
    }

    /// Forgets the sequence: every pass's convolution state back to zeros (keys and values are overwritten as the
    /// positions are read again).
    pub fn reset(&mut self) {
        for p in &mut self.pass {
            p.hist.fill(0.0);
        }
    }

    /// n tokens at positions pos0.. (1 to BLOCK_MAX of them; the convolution state must be the one after position
    /// pos0 - 1, as it is when the sequence is read in order, or put back by set_conv_state): writes the
    /// logits of the last one (or of all n when `all` is set, n x n_vocab) into logits.
    pub fn forward(&mut self, tokens: &[u32], pos0: u32, logits: &mut [f32], all: bool) -> Result<()> {
        let h = self.h.clone();
        let n = tokens.len();
        let _w = (n <= 16).then(crate::pool::Width::decode);
        let dm = h.d_model;
        if n == 0 || n > BLOCK_MAX || pos0 as usize + n > self.n_ctx as usize {
            return err("a block of 1 to BLOCK_MAX tokens inside the context");
        }
        if pos0 == 0 {
            self.reset();
        }
        let mut lap = crate::Laps::start(&PROF);
        // the embedding (padded rows in the file), normed
        let emb = self.w.need("token_embd.weight")?;
        let (rb, nv) = (emb.row_bytes(), emb.shape().1);
        let mut x = vec![0.0f32; n * dm];
        let mut row = vec![0.0f32; emb.shape().0];
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= h.n_vocab {
                return err("token out of range");
            }
            kops::dequantize(emb.t.ty, &emb.data[tok as usize * rb..(tok as usize + 1) * rb], &mut row);
            rms_norm(&mut x[t * dm..(t + 1) * dm], &row[..dm], &self.tok_norm, h.eps);
        }
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        lap.lap(0);
        for l in 0..h.loop_start {
            self.layer(l, l, &mut x, &mut blocks, None, pos0, n)?;
        }
        // every pass over the looped layers starts from here; from the second on, with the messages of the one before
        let (prelude, prelude_blocks) = (x.clone(), blocks.clone());
        let span = h.loop_end - h.loop_start;
        for pass in 0..h.loop_count {
            let mut bank: Option<(Vec<Vec<f32>>, Vec<f32>)> = None;
            if pass > 0 {
                // the pass before: its blocks after the first looped one, and its output
                bank = Some((blocks[prelude_blocks.len() + 1..].to_vec(), std::mem::replace(&mut x, prelude.clone())));
                blocks = prelude_blocks.clone();
            }
            for l in h.loop_start..h.loop_end {
                let s = if pass == 0 { l } else { h.n_layer + (pass - 1) * span + (l - h.loop_start) };
                let msg = match (&bank, &self.layers[l].looped) {
                    (Some((b, out)), Some(lp)) => Some(mix(b, out, &lp.msg_query, n, dm)),
                    _ => None,
                };
                self.layer(s, l, &mut x, &mut blocks, msg.as_deref(), pos0, n)?;
            }
        }
        for l in h.loop_end..h.n_layer {
            self.layer(l, l, &mut x, &mut blocks, None, pos0, n)?;
        }
        // the output: the mix, the norm, the head, the soft cap
        let mut lap = crate::Laps::start(&PROF);
        let first = if all { 0 } else { n - 1 };
        let m = n - first;
        let mixed = mix(&blocks, &x, &self.out_query, n, dm);
        let mut xo = vec![0.0f32; m * dm];
        for t in first..n {
            rms_norm(
                &mut xo[(t - first) * dm..(t - first + 1) * dm],
                &mixed[t * dm..(t + 1) * dm],
                &self.out_norm,
                h.eps,
            );
        }
        if logits.len() < m * h.n_vocab {
            return err("logits buffer too small");
        }
        let head = self.w.need("output.weight")?;
        let mut full = vec![0.0f32; m * nv];
        apply_group_multi(&mut [(Mat::of(&head), &mut full[..])], quant(&xo, m, dm).view(), m);
        for t in 0..m {
            for (o, &v) in logits[t * h.n_vocab..(t + 1) * h.n_vocab].iter_mut().zip(&full[t * nv..]) {
                *o = if h.softcap != 0.0 { (v / h.softcap).tanh() * h.softcap } else { v };
            }
        }
        lap.lap(0);
        Ok(())
    }

    /// Layer pass `s` (on layer `l`'s weights) over the block's streams `x`, with the pass before's messages (n x 4
    /// x d_model: for the query, the keys, the values and the experts) where it is a later pass of a looped layer.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        s: usize,
        l: usize,
        x: &mut Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        msg: Option<&[f32]>,
        pos0: u32,
        n: usize,
    ) -> Result<()> {
        let h = &self.h;
        let lw = l as u32;
        let (dm, hd, nh, nkv) = (h.d_model, h.head_dim, h.n_head, h.n_head_kv);
        let (qd, kvd, nr) = (nh * hd, nkv * hd, h.n_rel);
        let swa = h.swa[l];
        let ly = &self.layers[l];
        let st = &mut self.pass[s];
        let mut lap = crate::Laps::start(&PROF);
        // message j of token t
        let part = |j: usize| -> Vec<f32> {
            let m = msg.unwrap();
            let mut v = vec![0.0f32; n * dm];
            for t in 0..n {
                v[t * dm..(t + 1) * dm].copy_from_slice(&m[(t * 4 + j) * dm..(t * 4 + j + 1) * dm]);
            }
            v
        };
        // 1. the attention's inputs: the mix of the earlier blocks (one, or one each for the query, the keys and the
        // values), the messages, the norms
        let mut ins: Vec<Vec<f32>> = if ly.attn_query.is_empty() {
            vec![x.clone()]
        } else {
            let mixed = mix(blocks, x, &ly.attn_query, n, dm);
            let nq = ly.attn_query.len() / dm;
            (0..nq)
                .map(|j| {
                    let mut v = vec![0.0f32; n * dm];
                    for t in 0..n {
                        v[t * dm..(t + 1) * dm].copy_from_slice(&mixed[(t * nq + j) * dm..(t * nq + j + 1) * dm]);
                    }
                    v
                })
                .collect()
        };
        if let (Some(_), Some(lp)) = (msg, &ly.looped) {
            // the messages make the three inputs differ
            while ins.len() < 3 {
                ins.push(ins[0].clone());
            }
            for (j, v) in ins.iter_mut().enumerate() {
                control(v, &part(j), &lp.ctrl[j], n, dm, h.rank);
            }
        }
        for (j, v) in ins.iter_mut().enumerate() {
            let wn = &ly.in_norm[if ly.in_norm.len() == 3 { j } else { 0 }];
            let vp = SyncPtr::new(v);
            each(n, &|t| {
                // SAFETY: item t owns token t's row
                let r = unsafe { vp.slice(t * dm, dm) };
                let c = r.to_vec();
                rms_norm(r, &c, wn, h.eps);
            });
        }
        let qs: Vec<Q8> = ins.iter().map(|v| quant(v, n, dm)).collect();
        lap.lap(1);
        let (qi, ki, vi) = (0, (ins.len() - 1).min(1), ins.len() - 1);
        // 2. the products: queries, relative vectors, sinks and the output's gate from the query's input; keys and
        // values (and their taps on a full-attention layer) from their own
        let t_ = |s: &str| self.w.layer(lw, s);
        let (mut q, mut rel, mut sink, mut gate) =
            (vec![0.0f32; n * qd], vec![0.0f32; n * nh * nr], vec![0.0f32; n * nh], vec![0.0f32; n * qd]);
        let (mut kk, mut vv) = (vec![0.0f32; n * kvd], vec![0.0f32; n * kvd]);
        let (mut ktap, mut vtap) = (vec![0.0f32; n * nkv * 4], vec![0.0f32; n * nkv * 4]);
        {
            let (tq, tr, ts, tg) =
                (t_("attn_q.weight")?, t_("attn_rel.weight")?, t_("attn_sink_proj.weight")?, t_("attn_gate.weight")?);
            apply_group_multi(
                &mut [
                    (Mat::of(&tq), &mut q[..]),
                    (Mat::of(&tr), &mut rel[..]),
                    (Mat::of(&ts), &mut sink[..]),
                    (Mat::of(&tg), &mut gate[..]),
                ],
                qs[qi].view(),
                n,
            );
            let (tk, tv) = (t_("attn_k.weight")?, t_("attn_v.weight")?);
            if swa {
                apply_group_multi(&mut [(Mat::of(&tk), &mut kk[..])], qs[ki].view(), n);
                apply_group_multi(&mut [(Mat::of(&tv), &mut vv[..])], qs[vi].view(), n);
            } else {
                let (pk, pv) = (t_("attn_k_conv_proj.weight")?, t_("attn_v_conv_proj.weight")?);
                apply_group_multi(&mut [(Mat::of(&tk), &mut kk[..]), (Mat::of(&pk), &mut ktap[..])], qs[ki].view(), n);
                apply_group_multi(&mut [(Mat::of(&tv), &mut vv[..]), (Mat::of(&pv), &mut vtap[..])], qs[vi].view(), n);
            }
        }
        lap.lap(2);
        let (hk, rest) = st.hist.split_at_mut(3 * kvd);
        let (hv, rest) = rest.split_at_mut(3 * kvd);
        let (ha, hf) = rest.split_at_mut(3 * dm);
        convolve(&mut kk, n, kvd, &ly.k_conv, (!swa).then_some((&ktap[..], nkv)), hk);
        convolve(&mut vv, n, kvd, &ly.v_conv, (!swa).then_some((&vtap[..], nkv)), hv);
        // the clips and norms, the keys and values into the cache
        let cap = st.cap;
        {
            let (qp, rp) = (SyncPtr::new(&mut q), SyncPtr::new(&mut rel));
            let (kp, vp) = (SyncPtr::new(&mut kk), SyncPtr::new(&mut vv));
            let (ck, cv) = (SyncPtr::new(&mut st.k), SyncPtr::new(&mut st.v));
            each(n, &|t| {
                let pos = pos0 as usize + t;
                // the full-attention layers' queries grow slowly with the position
                let tau = if swa { 1.0 } else { 1.0 + h.log_alpha * ((pos + 1) as f32 / h.log_floor).max(1.0).ln() };
                // SAFETY: item t owns token t's rows and its position's rows of the cache
                let (qt, rt) = unsafe { (qp.slice(t * qd, qd), rp.slice(t * nh * nr, nh * nr)) };
                let (kt, vt) = unsafe { (kp.slice(t * kvd, kvd), vp.slice(t * kvd, kvd)) };
                for hh in 0..nh {
                    let qh = &mut qt[hh * hd..(hh + 1) * hd];
                    clip(qh, Some(&ly.q_clip[hh * hd..(hh + 1) * hd]), h.qk_clip);
                    let rh = &mut rt[hh * nr..(hh + 1) * nr];
                    clip(rh, Some(&ly.rel_clip[hh * nr..(hh + 1) * nr]), h.rel_clip);
                    for v in qh.iter_mut().chain(rh.iter_mut()) {
                        *v *= tau;
                    }
                }
                for g in 0..nkv {
                    clip(&mut kt[g * hd..(g + 1) * hd], None, h.qk_clip);
                    let vh = &mut vt[g * hd..(g + 1) * hd];
                    let c = vh.to_vec();
                    rms(vh, &c, h.eps);
                }
                let slot = (pos % cap) * kvd;
                unsafe {
                    ck.slice(slot, kvd).copy_from_slice(kt);
                    cv.slice(slot, kvd).copy_from_slice(vt);
                }
            });
        }
        lap.lap(3);
        // 3. attention: each head's scores over the keys it sees, plus the bias by distance less the query's sink
        // logit, softmaxed with the sink's zero beside them
        let mut att = vec![0.0f32; n * qd];
        {
            let ap = SyncPtr::new(&mut att);
            let (ck, cv) = (&st.k, &st.v);
            let scale = 1.0 / (hd as f32).sqrt();
            let per = nh / nkv;
            // an item is one head over a tile of up to ATT_QT tokens (so a single token's heads spread over the pool,
            // and in a prompt block each key and value row is read once for the tile's queries while it is in the
            // core's cache). Each query's numbers are the same steps in the same order as one query alone (its scores
            // over its own keys in position order, the same maximum and sums, its value sums in position order), so
            // the bits do not depend on n, the tile or how the items are dealt
            const ATT_QT: usize = 8;
            let tiles = n.div_ceil(ATT_QT);
            // in a prompt block an item also takes every query head of one key head (GQA), so each key and value row
            // is read once for all of them; a single token keeps a head an item, to spread over the pool
            let hpi = if n > 1 { per } else { 1 };
            let groups = nh / hpi;
            if n == 1 {
                // one token: nothing to tile (the tiled code below cost writing 3-5% on box1,
                // 26 Nov). The same steps as the tiled path for one query, so the same bits
                let pos = pos0 as usize;
                let lo = if swa { (pos + 1).saturating_sub(h.window) } else { 0 };
                // an item is one key head with its `per` query heads, so each key and value row is read from memory
                // once for all of them (the cache is f32 by position: at depth it is most of a token's bytes). Each
                // head's numbers are the same steps in the same order, so the same bits
                let span = pos + 1 - lo;
                each(nkv, &|g| {
                    let h0 = g * per;
                    let mut bias = vec![0.0f32; per * h.extent];
                    let mut sc = vec![0.0f32; per * span];
                    for j in 0..per {
                        let rh = &rel[(h0 + j) * nr..(h0 + j + 1) * nr];
                        for (d, b) in bias[j * h.extent..(j + 1) * h.extent].iter_mut().enumerate() {
                            *b = dot(&ly.rel_bias[d * nr..(d + 1) * nr], rh);
                        }
                    }
                    for (i, p) in (lo..=pos).enumerate() {
                        let d = pos - p;
                        let k = &ck[(p % cap) * kvd + g * hd..(p % cap) * kvd + (g + 1) * hd];
                        for j in 0..per {
                            let hh = h0 + j;
                            let b = if d < h.extent { bias[j * h.extent + d] } else { 0.0 };
                            sc[j * span + i] = dot(&q[hh * hd..(hh + 1) * hd], k) * scale + (b - sink[hh]);
                        }
                    }
                    let mut sums = vec![0.0f32; per];
                    for j in 0..per {
                        let row = &mut sc[j * span..(j + 1) * span];
                        let mut mx = 0.0f32;
                        for &v in row.iter() {
                            mx = mx.max(v);
                        }
                        let mut sum = (-mx).exp();
                        for v in row.iter_mut() {
                            *v = (*v - mx).exp();
                            sum += *v;
                        }
                        sums[j] = sum;
                    }
                    // SAFETY: item g owns heads h0..h0 + per of the token's row of att
                    let og = unsafe { ap.slice(h0 * hd, per * hd) };
                    for (i, p) in (lo..=pos).enumerate() {
                        let v = &cv[(p % cap) * kvd + g * hd..(p % cap) * kvd + (g + 1) * hd];
                        for j in 0..per {
                            let wgt = sc[j * span + i] / sums[j];
                            for (a, b) in og[j * hd..(j + 1) * hd].iter_mut().zip(v) {
                                *a = wgt.mul_add(*b, *a);
                            }
                        }
                    }
                    for (a, gt) in og.iter_mut().zip(&gate[h0 * hd..(h0 + per) * hd]) {
                        *a *= 1.0 / (1.0 + (-gt).exp());
                    }
                });
            } else {
                each(tiles * groups, &|it| {
                    let (tile, h0) = (it / groups, (it % groups) * hpi);
                    let (t0, t1) = (tile * ATT_QT, ((tile + 1) * ATT_QT).min(n));
                    let g = h0 / per;
                    // the item's queries: (token, head) for t0..t1 and h0..h0 + hpi, query j = (t - t0) * hpi + (hh - h0)
                    let nq = (t1 - t0) * hpi;
                    let tok = |j: usize| t0 + j / hpi;
                    let head = |j: usize| h0 + j % hpi;
                    // each query's first key, and the item's span of keys
                    let lo_of = |t: usize| {
                        let pos = pos0 as usize + t;
                        if swa { (pos + 1).saturating_sub(h.window) } else { 0 }
                    };
                    let (span_lo, span_hi) = (lo_of(t0), pos0 as usize + t1 - 1);
                    let width = span_hi + 1 - span_lo;
                    // the scores, a row of width a query (query j's own keys lo_of(t)..=pos at their offsets)
                    let mut sc = vec![0.0f32; nq * width];
                    let mut bias = vec![0.0f32; nq * h.extent];
                    for j in 0..nq {
                        let (t, hh) = (tok(j), head(j));
                        let rh = &rel[(t * nh + hh) * nr..(t * nh + hh + 1) * nr];
                        for (d, b) in bias[j * h.extent..(j + 1) * h.extent].iter_mut().enumerate() {
                            *b = dot(&ly.rel_bias[d * nr..(d + 1) * nr], rh);
                        }
                    }
                    for p in span_lo..=span_hi {
                        let k = &ck[(p % cap) * kvd + g * hd..(p % cap) * kvd + (g + 1) * hd];
                        for j in 0..nq {
                            let (t, hh) = (tok(j), head(j));
                            let pos = pos0 as usize + t;
                            if p < lo_of(t) || p > pos {
                                continue;
                            }
                            let d = pos - p;
                            let b = if d < h.extent { bias[j * h.extent + d] } else { 0.0 };
                            let qh = &q[t * qd + hh * hd..t * qd + (hh + 1) * hd];
                            sc[j * width + (p - span_lo)] = dot(qh, k) * scale + (b - sink[t * nh + hh]);
                        }
                    }
                    // each query's softmax over its own keys, in position order, with the sink's zero beside them
                    let mut sums = vec![0.0f32; nq];
                    for j in 0..nq {
                        let t = tok(j);
                        let (lo, pos) = (lo_of(t), pos0 as usize + t);
                        let row = &mut sc[j * width + (lo - span_lo)..j * width + (pos + 1 - span_lo)];
                        let mut mx = 0.0f32;
                        for &v in row.iter() {
                            mx = mx.max(v);
                        }
                        let mut sum = (-mx).exp();
                        for v in row.iter_mut() {
                            *v = (*v - mx).exp();
                            sum += *v;
                        }
                        sums[j] = sum;
                    }
                    // the values, each row read once for the item; each query's sums stay in position order
                    for p in span_lo..=span_hi {
                        let v = &cv[(p % cap) * kvd + g * hd..(p % cap) * kvd + (g + 1) * hd];
                        for j in 0..nq {
                            let (t, hh) = (tok(j), head(j));
                            if p < lo_of(t) || p > pos0 as usize + t {
                                continue;
                            }
                            let wgt = sc[j * width + (p - span_lo)] / sums[j];
                            // SAFETY: item it owns heads h0..h0 + hpi of tokens t0..t1's rows of att
                            let oh = unsafe { ap.slice(t * qd + hh * hd, hd) };
                            for (a, b) in oh.iter_mut().zip(v) {
                                *a = wgt.mul_add(*b, *a);
                            }
                        }
                    }
                    // the output's gate
                    for j in 0..nq {
                        let (t, hh) = (tok(j), head(j));
                        // SAFETY: as above
                        let oh = unsafe { ap.slice(t * qd + hh * hd, hd) };
                        for (a, g) in oh.iter_mut().zip(&gate[t * qd + hh * hd..t * qd + (hh + 1) * hd]) {
                            *a *= 1.0 / (1.0 + (-g).exp());
                        }
                    }
                });
            }
        }
        lap.lap(4);
        let mut out = vec![0.0f32; n * dm];
        {
            let two = t_("attn_output.weight")?;
            apply_group_multi(&mut [(Mat::of(&two), &mut out[..])], quant(&att, n, qd).view(), n);
        }
        convolve(&mut out, n, dm, &ly.attn_conv, None, ha);
        // a block's first layer starts a new stream; the one it leaves joins the blocks
        if l.is_multiple_of(h.block) {
            blocks.push(std::mem::replace(x, out));
        } else {
            for (a, b) in x.iter_mut().zip(&out) {
                *a += b;
            }
        }
        lap.lap(5);
        // 4. the routed experts, on the mix of the blocks (with the pass before's message), normed
        let mut xn = mix(blocks, x, &ly.ffn_query, n, dm);
        if let (Some(_), Some(lp)) = (msg, &ly.looped) {
            control(&mut xn, &part(3), &lp.ctrl[3], n, dm, h.rank);
        }
        {
            let vp = SyncPtr::new(&mut xn);
            each(n, &|t| {
                // SAFETY: item t owns token t's row
                let r = unsafe { vp.slice(t * dm, dm) };
                let c = r.to_vec();
                rms_norm(r, &c, &ly.ffn_norm, h.eps);
            });
        }
        // the router: sigmoid scores, the top n_used by score plus bias (an equal one goes to the lower id), weighted
        // by the scores alone over their sum, times the scale
        let mut choice: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        {
            // every token's sigmoid scores, CTRL_SPLIT runs of the experts a token (the same dot per expert, so the
            // same bits), then the picks a token
            let ne = h.n_expert;
            let mut score = vec![0.0f32; n * ne];
            {
                let sp = SyncPtr::new(&mut score);
                let per = ne.div_ceil(CTRL_SPLIT);
                let xn = &xn;
                each(n * CTRL_SPLIT, &|i| {
                    let (t, s) = (i / CTRL_SPLIT, i % CTRL_SPLIT);
                    let (e0, e1) = ((s * per).min(ne), ((s + 1) * per).min(ne));
                    let xt = &xn[t * dm..(t + 1) * dm];
                    // SAFETY: item i owns experts e0..e1 of token t's scores
                    let st = unsafe { sp.slice(t * ne + e0, e1 - e0) };
                    for (o, e) in st.iter_mut().zip(e0..e1) {
                        *o = 1.0 / (1.0 + (-dot(&ly.router[e * dm..(e + 1) * dm], xt)).exp());
                    }
                });
            }
            let cp = SyncPtr::new(&mut choice);
            let score = &score;
            each(n, &|t| {
                let mut lg: Vec<(usize, f32, f32)> = (0..ne)
                    .map(|e| {
                        let p = score[t * ne + e];
                        (e, p + ly.pick_bias[e], p)
                    })
                    .collect();
                lg.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                lg.truncate(h.n_used);
                let sum: f32 = lg.iter().map(|c| c.2).sum::<f32>() + 1e-20;
                // SAFETY: item t owns choice[t]
                unsafe { cp.slice(t, 1)[0] = lg.iter().map(|&(e, _, p)| (e, p / sum * h.w_scale)).collect() };
            });
        }
        lap.lap(6);
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); h.n_expert];
        for (t, c) in choice.iter().enumerate() {
            for &(e, _) in c {
                users[e].push(t);
            }
        }
        let mut ys: Vec<Vec<(usize, f32, Vec<f32>)>> = vec![Vec::new(); n];
        let (ff, ffp) = (h.d_ff, h.d_ff_pad);
        if n == 1 {
            // one token: its experts' gates and ups in one sweep over the same input, then each expert's SITU and
            // down on a thread of its own (the same products as a block makes, so the same bits)
            let ids: Vec<u32> = choice[0].iter().map(|c| c.0 as u32).collect();
            let packs = self.w.experts_of(lw, ly.group, &ids)?;
            let k = ids.len();
            let piece = |pk: &'_ [u8], p: &format::Piece| Mat {
                ty: p.ty,
                rows: p.rows as usize,
                cols: p.cols as usize,
                data: unsafe { std::slice::from_raw_parts(pk.as_ptr().add(p.at as usize), p.bytes as usize) },
            };
            let xq = quant(&xn, 1, dm);
            let mut gu = vec![0.0f32; 2 * k * ffp];
            {
                let mut jobs: Vec<(Mat, &mut [f32])> = Vec::with_capacity(2 * k);
                for (i, part) in gu.chunks_mut(ffp).enumerate() {
                    let (pk, pack) = packs[i / 2];
                    let mut m = piece(pk, if i % 2 == 0 { &pack.gate } else { &pack.up });
                    // only the expert's ff real rows: the converter pads gate and up to ffp rows of zeros, whose
                    // outputs SITU never reads (each real row is the same dot product, so the same bits)
                    let rb = m.data.len() / m.rows;
                    m.data = &m.data[..ff * rb];
                    m.rows = ff;
                    jobs.push((m, part));
                }
                crate::moe::apply_group(&mut jobs, xq.view());
            }
            let mut y = vec![0.0f32; k * dm];
            {
                let yp = SyncPtr::new(&mut y);
                let (gu, packs) = (&gu, &packs);
                let (beta, lb) = (h.situ_beta, h.situ_lb);
                each(k, &|i| {
                    let (g, u) = (&gu[2 * i * ffp..(2 * i + 1) * ffp], &gu[(2 * i + 1) * ffp..(2 * i + 2) * ffp]);
                    let mut row = vec![0.0f32; ffp];
                    for j in 0..ff {
                        let act = beta * (g[j] / beta).tanh() * (1.0 / (1.0 + (-g[j]).exp()));
                        row[j] = act * if lb > 0.0 { lb * (u[j] / lb).tanh() } else { u[j] };
                    }
                    let mut hq = Q8::new(ffp / QK);
                    kops::q8k_quantize(&row, hq.view_mut());
                    let (pk, pack) = packs[i];
                    let d = piece(pk, &pack.down);
                    // SAFETY: item i owns expert i's row of y
                    crate::moe::dot_rows(d.ty, d.data, d.data.len() / d.rows, hq.view(), unsafe {
                        yp.slice(i * dm, dm)
                    });
                });
            }
            for (i, c) in choice[0].iter().enumerate() {
                ys[0].push((c.0, c.1, y[i * dm..(i + 1) * dm].to_vec()));
            }
        }
        for (e, toks) in users.iter().enumerate() {
            if toks.is_empty() || n == 1 {
                continue;
            }
            let m = toks.len();
            let mut xe = vec![0.0f32; m * dm];
            for (i, &t) in toks.iter().enumerate() {
                xe[i * dm..(i + 1) * dm].copy_from_slice(&xn[t * dm..(t + 1) * dm]);
            }
            let xeq = quant(&xe, m, dm);
            let packs = self.w.experts_of(lw, ly.group, &[e as u32])?;
            let (pk, pack) = packs[0];
            let piece = |p: &format::Piece| Mat {
                ty: p.ty,
                rows: p.rows as usize,
                cols: p.cols as usize,
                data: &pk[p.at as usize..(p.at + p.bytes) as usize],
            };
            let (mut g, mut u) = (vec![0.0f32; m * ffp], vec![0.0f32; m * ffp]);
            apply_group_multi(&mut [(piece(&pack.gate), &mut g[..]), (piece(&pack.up), &mut u[..])], xeq.view(), m);
            // SITU, a token per item, straight into that token's quantised row (zero beyond ff)
            let nbf = ffp / QK;
            let mut hq = Q8::new(m * nbf);
            {
                let qp = crate::pool::Q8Ptr::new(hq.view_mut());
                let (g, u) = (&g, &u);
                let (beta, lb) = (h.situ_beta, h.situ_lb);
                each(m, &|i| {
                    let mut row = vec![0.0f32; ffp];
                    for j in 0..ff {
                        let (gt, ut) = (g[i * ffp + j], u[i * ffp + j]);
                        let act = beta * (gt / beta).tanh() * (1.0 / (1.0 + (-gt).exp()));
                        row[j] = act * if lb > 0.0 { lb * (ut / lb).tanh() } else { ut };
                    }
                    // SAFETY: item i owns token i's blocks of hq
                    kops::q8k_quantize(&row, unsafe { qp.slice(i * nbf, nbf) });
                });
            }
            let mut y = vec![0.0f32; m * dm];
            apply_group_multi(&mut [(piece(&pack.down), &mut y[..])], hq.view(), m);
            for (i, &t) in toks.iter().enumerate() {
                let wgt = choice[t].iter().find(|c| c.0 == e).map_or(0.0, |c| c.1);
                ys[t].push((e, wgt, y[i * dm..(i + 1) * dm].to_vec()));
            }
        }
        lap.lap(7);
        let mut moe = vec![0.0f32; n * dm];
        for (t, list) in ys.iter_mut().enumerate() {
            // in the order the router ranked them, so a token's sum does not depend on the block
            let order: Vec<usize> = choice[t].iter().map(|c| c.0).collect();
            list.sort_by_key(|(e, _, _)| order.iter().position(|o| o == e));
            for (_, wgt, yt) in list.iter() {
                for i in 0..dm {
                    moe[t * dm + i] = wgt.mul_add(yt[i], moe[t * dm + i]);
                }
            }
        }
        convolve(&mut moe, n, dm, &ly.ffn_conv, None, hf);
        for (a, b) in x.iter_mut().zip(&moe) {
            *a += b;
        }
        lap.lap(8);
        Ok(())
    }
}
