//! gemma.rs - Gemma 4 (general.architecture gemma4: the 26B-A4B and its kin).
//!
//! How it differs from the Qwen models in forward.rs:
//! - Attention only, no DeltaNet. Most layers attend over a sliding window; every few layers one attends over the
//!   whole context. The two kinds differ in head size, KV head count and RoPE base, and the full layers also scale
//!   their angles by rope_freqs. A full layer with no attn_v uses its raw K as V. There's no 1/sqrt(head_dim) on the
//!   scores, since the q and k norms are trained to set their size. V gets a weightless RMS norm, and K and V are
//!   cached as 16-bit integers with one scale per row.
//! - Extra norms around each sub-block: post_attention_norm on attention's output, post_ffw_norm on the FFN's, then
//!   layer_output_scale on the result. The token embedding is multiplied by sqrt(d_model).
//! - The FFN adds two parts: a dense GELU feed-forward on the ffn_norm input (then post_ffw_norm_1), and the routed
//!   GELU experts on the residual through pre_ffw_norm_2 (then post_ffw_norm_2). The router sees the residual with a
//!   weightless norm, times 1 / sqrt(d_model) and ffn_gate_inp.scale. Each expert's output carries its
//!   ffn_down_exps.scale.
//! - The logits are soft-capped: c * tanh(l / c).
//!
//! A block of n tokens runs every step for all n at once (so each weight row is read once), and each token's numbers
//! are the same bits as a block of one.

use crate::attn::{AttnEngine, KvCache, quant16};
use crate::moe::{EJob, Mat, MoeLayer, NpuSide, apply_group_multi, moe_tokens, moe_tokens_side, run_ejobs};
use crate::ops::{rms_norm, rms_norm_inplace, rope_neox};
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::meta::{Meta, Value};
use format::{BlockKind, Result, err};
use kernels::{Q8, Q8m, QK, ops as kops};

/// One kind of attention layer's geometry.
#[derive(Clone, Debug)]
pub struct Kind {
    pub head_dim: usize,
    pub n_head_kv: usize,
    pub n_rot: usize,
    pub base: f32,
}

impl Kind {
    /// Refuses a geometry the code below can't run: the query heads must share the KV heads evenly, fattn works in
    /// 8-wide lanes, RoPE turns pairs inside the head, and the attention output is quantised in whole blocks.
    fn check(&self, name: &str, n_head: usize) -> Result<()> {
        let why = if self.n_head_kv == 0 || !n_head.is_multiple_of(self.n_head_kv) {
            "the query heads don't divide evenly among the KV heads"
        } else if !self.head_dim.is_multiple_of(8) {
            "the head size isn't a multiple of 8"
        } else if self.n_rot == 0 || !self.n_rot.is_multiple_of(2) || self.n_rot > self.head_dim {
            "the rotated width must be even, above 0 and no wider than the head"
        } else if !(n_head * self.head_dim).is_multiple_of(QK) {
            "the attention width isn't a whole number of quantisation blocks"
        } else {
            return Ok(());
        };
        err(format!("gemma4 {name} layers: {why}"))
    }
}

#[derive(Clone, Debug)]
pub struct GemmaHp {
    pub n_layer: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// final_logit_softcapping (0 for none)
    pub softcap: f32,
    pub window: u32,
    pub full: Kind,
    pub swa: Kind,
    /// for each layer, whether it is a sliding-window layer
    pub swa_layers: Vec<bool>,
}

/// A per-layer array from the metadata, as bools or integers.
fn per_layer(m: &Meta, key: &str) -> Option<Vec<i128>> {
    match m.get(key)? {
        Value::Array(_, v) => v
            .iter()
            .map(|x| match x {
                Value::Int(i) => Some(*i),
                Value::Bool(b) => Some(*b as i128),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

impl GemmaHp {
    /// The hyperparameters from the file's metadata, with the checks the forward pass relies on.
    pub fn from_meta(metadata: &[u8]) -> Result<GemmaHp> {
        let m = Meta::parse(metadata)?;
        let arch = m.str("general.architecture").unwrap_or("");
        if arch != "gemma4" {
            return err(format!("not a gemma4 file ({arch})"));
        }
        let need = |k: &str| m.arch_u32(k).ok_or_else(|| format::Error(format!("missing gemma4.{k}")));
        let needf = |k: &str| m.arch_f32(k).ok_or_else(|| format::Error(format!("missing gemma4.{k}")));
        let n_layer = need("block_count")? as usize;
        let pat = per_layer(&m, &m.arch_key("attention.sliding_window_pattern"))
            .filter(|p| p.len() >= n_layer)
            .ok_or_else(|| format::Error("missing gemma4.attention.sliding_window_pattern".into()))?;
        let swa_layers: Vec<bool> = pat[..n_layer].iter().map(|&v| v != 0).collect();
        // head_count_kv is a list with one entry per layer, or a single number for the whole model
        let kvs: Vec<i128> = match per_layer(&m, &m.arch_key("attention.head_count_kv")) {
            Some(v) if v.len() >= n_layer => v[..n_layer].to_vec(),
            _ => vec![need("attention.head_count_kv")? as i128; n_layer],
        };
        // every layer of one kind shares the same KV head count
        let kv_heads = |swa: bool| -> Result<usize> {
            let mut counts = kvs.iter().zip(&swa_layers).filter(|&(_, &s)| s == swa).map(|(&c, _)| c);
            let first = counts.next().unwrap_or(0);
            if first <= 0 || counts.any(|c| c != first) {
                let kind = if swa { "sliding-window" } else { "full-attention" };
                return err(format!("gemma4: the {kind} layers do not share one positive KV head count"));
            }
            Ok(first as usize)
        };
        let (kv_full, kv_window) = (kv_heads(false)?, kv_heads(true)?);
        let head_full = need("attention.key_length")? as usize;
        let full = Kind {
            head_dim: head_full,
            n_head_kv: kv_full,
            n_rot: m.arch_u32("rope.dimension_count").map_or(head_full, |v| v as usize),
            base: needf("rope.freq_base")?,
        };
        let swa = Kind {
            head_dim: need("attention.key_length_swa")? as usize,
            n_head_kv: kv_window,
            n_rot: need("rope.dimension_count_swa")? as usize,
            base: needf("rope.freq_base_swa")?,
        };
        let h = GemmaHp {
            n_layer,
            d_model: need("embedding_length")? as usize,
            n_head: need("attention.head_count")? as usize,
            n_vocab: m.len_of("tokenizer.ggml.tokens").unwrap_or(0),
            eps: needf("attention.layer_norm_rms_epsilon")?,
            softcap: m.arch_f32("final_logit_softcapping").unwrap_or(0.0),
            window: need("attention.sliding_window")?,
            full,
            swa,
            swa_layers,
        };
        h.full.check("full-attention", h.n_head)?;
        h.swa.check("sliding-window", h.n_head)?;
        if !h.d_model.is_multiple_of(QK) {
            return err(format!("gemma4: embedding_length {} is not a whole number of {QK}-value blocks", h.d_model));
        }
        if h.window == 0 {
            return err("gemma4: attention.sliding_window is 0");
        }
        if m.arch_u32("attention.shared_kv_layers").unwrap_or(0) != 0
            || m.arch_u32("embedding_length_per_layer_input").unwrap_or(0) != 0
        {
            return err("gemma4: shared KV layers / per-layer embeddings (E2B, E4B) not supported yet");
        }
        Ok(h)
    }
}

struct GLayer {
    swa: bool,
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// whether the file has attn_v (otherwise V is K, taken before K's norm: the full layers)
    has_v: bool,
    post_attn: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw: Vec<f32>,
    layer_scale: f32,
    pre_ffw_2: Vec<f32>,
    post_ffw_1: Vec<f32>,
    post_ffw_2: Vec<f32>,
    gate_inp_scale: Vec<f32>,
    /// the routed experts (GELU, with their output scales)
    moe: MoeLayer,
    kv: KvCache,
}

/// The time spent in each phase of forward, summed over calls (TORC_PROF=1; torc-gemma prints them).
pub static PROF: [std::sync::atomic::AtomicU64; 10] = [const { std::sync::atomic::AtomicU64::new(0) }; 10];
pub const PHASES: [&str; 10] =
    ["embed+rope", "norm", "qkv", "q/k/v prep", "attention", "wo+post", "dense", "moe in", "experts", "output"];

pub struct Gemma {
    pub h: GemmaHp,
    pub w: Weights,
    layers: Vec<GLayer>,
    eng_full: AttnEngine,
    eng_swa: AttnEngine,
    rope_freqs: Option<Vec<f32>>,
    out_norm: Vec<f32>,
    pub n_ctx: u32,
    /// The tied head as the CPU's output product reads it, requantised: TORC_HEAD=q4_0 (the default) or q8_0, both
    /// lighter to read than the file's Q6_K. None means the file's own head (TORC_HEAD=file, or the whole-NPU tier,
    /// which has a head of its own)
    head: Option<(u32, Vec<u8>)>,
    /// TORC_NPU=pure: the whole model runs on the NPU (npu_gemma.rs); forward does the embedding here and runs the rest
    /// there
    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
    pure: Option<crate::npu_gemma::GemmaNpu>,
    /// the last forward pass's hidden states after the final norm, one row (d_model) per row of logits: the drafter's h
    /// (gemma_mtp.rs)
    pub hid: Vec<f32>,
    /// the soft cap on the logits (on by default; a caller that only takes the largest can skip it)
    pub cap_logits: bool,
    /// TORC_NPU=pure with a drafter: the last two layers' keys and values are copied back from the NPU into their CPU
    /// caches after each forward pass (they are what the drafter attends to)
    pub mirror_kv: bool,
}

/// TORC_NPU=pure: the whole model on the NPU.
pub fn pure_npu() -> bool {
    crate::switches::is("TORC_NPU", "pure")
}

/// Cos and sin of every rotated pair's angle at `pos`. Pair i turns at base^(-2i / n_rot) radians per position; on the
/// full layers the file's rope_freqs slows each pair down by its own factor.
pub(crate) fn fill_rope(pos: u32, n_rot: usize, base: f32, slow: Option<&[f32]>, cos: &mut [f32], sin: &mut [f32]) {
    let p = pos as f32;
    for (i, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).take(n_rot / 2).enumerate() {
        let speed = base.powf(-2.0 * i as f32 / n_rot as f32);
        let angle = match slow {
            Some(f) => p * speed / f[i],
            None => p * speed,
        };
        *c = angle.cos();
        *s = angle.sin();
    }
}

const NPU_QKV: u32 = 1;
const NPU_WO: u32 = 2;
const NPU_FFN: u32 = 4;

/// TORC_GEMMA_NPU=mask (with TORC_NPU=1): which products go to the NPU - 1 for q/k/v, 2 for attn_output, 4 for the
/// dense branch (default 7, all of them)
fn npu_on(which: u32) -> bool {
    static M: std::sync::LazyLock<u32> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_GEMMA_NPU").unwrap_or(7));
    *M & which != 0
}

/// Runs f(i) for i in 0..n: on this thread for a single item, else over the pool (each item writes only its own rows,
/// so the bits are the same either way).
fn each(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n < 2 { (0..n).for_each(f) } else { crate::pool::par_for(n, f) }
}

/// The router's input for one token: its residual through norm_plain, then multiplied by 1 / sqrt(d_model) and by
/// ffn_gate_inp.scale, dimension by dimension.
fn router_input(out: &mut [f32], resid: &[f32], gate_scale: &[f32], eps: f32) {
    let shrink = 1.0f32 / (resid.len() as f32).sqrt();
    out.copy_from_slice(resid);
    norm_plain(out, eps);
    for (o, &g) in out.iter_mut().zip(gate_scale) {
        *o = *o * shrink * g;
    }
}

/// An RMS norm with no weights (Gemma's V, and the router's input). The squares are summed in f64.
fn norm_plain(v: &mut [f32], eps: f32) {
    let n = v.len();
    let mut ss = 0.0f64;
    for &x in v.iter() {
        ss += x as f64 * x as f64;
    }
    let sc = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
    for x in v.iter_mut() {
        *x *= sc;
    }
}

impl Gemma {
    pub fn load(path: &str, n_ctx: u32) -> Result<Gemma> {
        // the dense Q4_0 matrices are repacked for the SMMLA / SDOT kernels (repack.rs) when the CPU computes them: the
        // same bits, for about 1 GB more RAM. Not with an NPU tier, which takes those matrices
        crate::repack::set_default(crate::switches::get("TORC_NPU").is_none_or(|v| v.is_empty() || v == "0"));
        let w = Weights::load(path)?;
        let h = GemmaHp::from_meta(&w.file.metadata)?;
        let mut layers = Vec::with_capacity(h.n_layer);
        for l in 0..h.n_layer as u32 {
            let group = w
                .file
                .blocks
                .iter()
                .find(|b| b.layer == l && b.kind == BlockKind::Moe && b.experts.is_some())
                .and_then(|b| b.experts)
                .ok_or_else(|| format::Error(format!("layer {l}: no routed experts")))?;
            let f = |s: &str| -> Result<Vec<f32>> { w.layer(l, s)?.f32s() };
            let swa = h.swa_layers[l as usize];
            let k = if swa { &h.swa } else { &h.full };
            let mut moe = MoeLayer::load(&w, l, group)?;
            moe.gelu = true;
            moe.expert_scale = Some(f("ffn_down_exps.scale")?);
            if moe.shared.is_some() || moe.shexp_gate.is_some() {
                return err(format!("layer {l}: gemma4 with a shared expert?"));
            }
            for s in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                w.layer(l, s)?;
            }
            let os = w.tensor(&format!("blk.{l}.layer_output_scale.weight"));
            let layer_scale = match os {
                Some(t) if t.t.ty == 0 => t.f32s()?.first().copied().unwrap_or(1.0),
                _ => 1.0,
            };
            layers.push(GLayer {
                swa,
                attn_norm: f("attn_norm.weight")?,
                q_norm: f("attn_q_norm.weight")?,
                k_norm: f("attn_k_norm.weight")?,
                has_v: w.tensor(&format!("blk.{l}.attn_v.weight")).is_some(),
                post_attn: f("post_attention_norm.weight")?,
                ffn_norm: f("ffn_norm.weight")?,
                post_ffw: f("post_ffw_norm.weight")?,
                layer_scale,
                pre_ffw_2: f("pre_ffw_norm_2.weight")?,
                post_ffw_1: f("post_ffw_norm_1.weight")?,
                post_ffw_2: f("post_ffw_norm_2.weight")?,
                gate_inp_scale: f("ffn_gate_inp.scale")?,
                moe,
                kv: KvCache::with(k.n_head_kv, k.head_dim, n_ctx as usize, true),
            });
        }
        let rope_freqs = match w.tensor("rope_freqs.weight") {
            Some(t) => {
                let v = t.f32s()?;
                if v.len() != h.full.n_rot / 2 {
                    return err(format!(
                        "rope_freqs.weight has {} values, but the full layers rotate {} pairs",
                        v.len(),
                        h.full.n_rot / 2
                    ));
                }
                Some(v)
            }
            None => None,
        };
        // TORC_NPU=1: the dense products of big prompt blocks go on the NPU (npu.rs) - attention's q, k, v and output,
        // and the dense branch's gate, up and down; the routed experts and everything else stay on the CPU
        crate::npu::load(&w, h.n_layer as u32, &layers.iter().map(|ly| ly.moe.group).collect::<Vec<_>>());
        // its feed-forwards run there with GEGLU: the whole dense branch, beside the experts (forward), and the hot
        // experts
        if crate::npu::up() {
            eprintln!("npu: GEGLU feed-forwards {}", if crate::npu::set_geglu() { "on" } else { "not available" });
        }
        let out_norm = w.need("output_norm.weight")?.f32s()?;
        if w.tensor("output.weight").is_some() {
            return err("gemma4 with its own output head: not expected (tied embeddings)");
        }
        // attention scale 1.0, since the learned q and k norms already set the size of the scores; the sliding-window
        // layers get an engine of their own
        let eng_full =
            AttnEngine::create(h.n_head as u32, h.full.n_head_kv as u32, h.full.head_dim as u32, Some(1.0), 0)?;
        let eng_swa =
            AttnEngine::create(h.n_head as u32, h.swa.n_head_kv as u32, h.swa.head_dim as u32, Some(1.0), h.window)?;
        #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
        let pure = if pure_npu() { Some(crate::npu_gemma::GemmaNpu::load(&w, &h, n_ctx as usize)?) } else { None };
        // Q4_0 by default: fewer bytes a token than the file's Q6_K
        let head = match crate::switches::get("TORC_HEAD").as_deref() {
            Some("file") => None,
            _ if pure_npu() => None,
            v => {
                let ty = if v == Some("q8_0") { 8 } else { 2 };
                let e = w.need("token_embd.weight")?;
                let (cols, rows) = e.shape();
                let q = requant(e.t.ty, e.data, cols, rows, ty);
                // the Q4_0 head is repacked too (repack.rs; a second copy)
                #[cfg(target_arch = "aarch64")]
                if ty == 2 && crate::repack::on() && rows % 4 == 0 && crate::switches::on("TORC_Q4X4_HEAD") {
                    crate::repack::register(&q, rows, cols);
                }
                Some((ty, q))
            }
        };
        Ok(Gemma {
            head,
            h,
            w,
            layers,
            eng_full,
            eng_swa,
            rope_freqs,
            out_norm,
            n_ctx,
            #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
            pure,
            hid: Vec::new(),
            cap_logits: true,
            mirror_kv: false,
        })
    }

    /// The attention engine for a kind of layer (the drafter's queries run on the target's engines).
    pub(crate) fn eng(&self, swa: bool) -> &AttnEngine {
        if swa { &self.eng_swa } else { &self.eng_full }
    }

    /// Layer l's cache (the drafter reads the last two).
    pub(crate) fn kv_of(&self, l: usize) -> &KvCache {
        &self.layers[l].kv
    }

    /// The number of experts per layer (for the expert-use profile, see npu::write_profile).
    pub fn n_expert(&self) -> u32 {
        self.layers.first().map_or(0, |ly| self.w.layer_experts(ly.moe.group).n_expert)
    }

    pub fn n_vocab(&self) -> Result<usize> {
        Ok(self.w.need("token_embd.weight")?.shape().1)
    }

    /// Every layer's cache at positions 0..npos (KvCache::put_bytes): the model's whole state after its first npos
    /// tokens, since nothing else carries over from one token to the next (the drafter reads the last layers' caches
    /// and keeps none of its own). The CPU's caches, so not on the whole-NPU tier, whose caches are on the NPU. For
    /// torc-server's saved chats (saved.rs).
    pub fn state_bytes(&self, npos: usize) -> Vec<u8> {
        let mut b = Vec::new();
        for ly in &self.layers {
            let hd = if ly.swa { self.h.swa.head_dim } else { self.h.full.head_dim };
            ly.kv.put_bytes(hd, npos, &mut b);
        }
        b
    }

    /// Puts state_bytes back. Fails if they do not fit this model, leaving the cache partly written.
    pub fn set_state_bytes(&mut self, npos: usize, b: &[u8]) -> Result<()> {
        let (hd_swa, hd_full) = (self.h.swa.head_dim, self.h.full.head_dim);
        let mut at = 0;
        let mut take = |n: usize| {
            let e = (at + n).min(b.len());
            let r = b[at..e].to_vec();
            at = e;
            r
        };
        for ly in &mut self.layers {
            ly.kv.set_bytes(if ly.swa { hd_swa } else { hd_full }, npos, &mut take)?;
        }
        if at != b.len() {
            return err(format!("saved state: {} bytes left over", b.len() - at));
        }
        Ok(())
    }

    /// n tokens at positions pos0.. (n >= 1): writes the logits of the last one (or of all n when `all` is set, n x
    /// n_vocab) into logits.
    pub fn forward(&mut self, tokens: &[u32], pos0: u32, logits: &mut [f32], all: bool) -> Result<()> {
        let h = self.h.clone();
        let n = tokens.len();
        // a reply step (a token or a few drafts): the decode width (pool::Width); a prompt block keeps every thread
        let _w = (n <= 16).then(crate::pool::Width::decode);
        let mut lap = crate::Laps::start(&PROF);
        let (dm, nb) = (h.d_model, h.d_model / QK);
        if n == 0 || pos0 as usize + n > self.n_ctx as usize {
            return err("position past the context");
        }
        let emb = self.w.need("token_embd.weight")?;
        let (rb, nv) = (emb.row_bytes(), emb.shape().1);
        // the embedding, scaled by sqrt(d_model)
        let mut x = vec![0.0f32; n * dm];
        let s = (dm as f32).sqrt();
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= nv {
                return err("token out of range");
            }
            let xr = &mut x[t * dm..(t + 1) * dm];
            kops::dequantize(emb.t.ty, &emb.data[tok as usize * rb..(tok as usize + 1) * rb], xr);
            for v in xr.iter_mut() {
                *v *= s;
            }
        }
        // the angles of both kinds for every position of the block
        let (hf, hs) = (h.full.n_rot / 2, h.swa.n_rot / 2);
        let (mut csf, mut snf) = (vec![0.0f32; n * hf], vec![0.0f32; n * hf]);
        let (mut css, mut sns) = (vec![0.0f32; n * hs], vec![0.0f32; n * hs]);
        for t in 0..n {
            let pos = pos0 + t as u32;
            let (fc, fs) = (&mut csf[t * hf..(t + 1) * hf], &mut snf[t * hf..(t + 1) * hf]);
            fill_rope(pos, h.full.n_rot, h.full.base, self.rope_freqs.as_deref(), fc, fs);
            fill_rope(
                pos,
                h.swa.n_rot,
                h.swa.base,
                None,
                &mut css[t * hs..(t + 1) * hs],
                &mut sns[t * hs..(t + 1) * hs],
            );
        }
        #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
        if let Some(p) = self.pure.as_mut() {
            p.cap = self.cap_logits;
            p.forward(&x, n, pos0 as usize, logits, all)?;
            self.hid.clone_from(&p.hid);
            if self.mirror_kv {
                for l in h.n_layer.saturating_sub(2)..h.n_layer {
                    let (k, v) = p.kv_rows(l, pos0 as usize, n);
                    let ly = &mut self.layers[l];
                    let kind = if ly.swa { &h.swa } else { &h.full };
                    let (hd, nkv, cap) = (kind.head_dim, kind.n_head_kv, ly.kv.cap);
                    for t in 0..n {
                        let row = pos0 as usize + t;
                        for hh in 0..nkv {
                            let (at, sc, src) = ((hh * cap + row) * hd, hh * cap + row, (t * nkv + hh) * hd);
                            ly.kv.k_scale[sc] = quant16(&k[src..src + hd], &mut ly.kv.k16[at..at + hd]);
                            ly.kv.v_scale[sc] = quant16(&v[src..src + hd], &mut ly.kv.v16[at..at + hd]);
                        }
                    }
                }
            }
            lap.lap(9);
            return Ok(());
        }
        lap.lap(0);
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(n * nb);
        let eps = h.eps;
        let norm_quant = |xn: &mut [f32], xq: Q8m, x: &[f32], wn: &[f32]| {
            let (np, qp) = (SyncPtr::new(xn), crate::pool::Q8Ptr::new(xq));
            each(n, &|t| {
                // SAFETY: item t owns token t's rows of xn and xq
                let (a, b) = unsafe { (np.slice(t * dm, dm), qp.slice(t * nb, nb)) };
                rms_norm(a, &x[t * dm..(t + 1) * dm], wn, eps);
                kops::q8k_quantize(a, b);
            });
        };
        let trace = crate::switches::get("TORC_TRACE").is_some_and(|v| v != "0");
        for l in 0..h.n_layer {
            let ly = &mut self.layers[l];
            let lw = l as u32;
            // 1. attention
            norm_quant(&mut xn, xq.view_mut(), &x, &ly.attn_norm);
            lap.lap(1);
            let k = if ly.swa { &h.swa } else { &h.full };
            let (hd, nkv) = (k.head_dim, k.n_head_kv);
            let (qd, kvd) = (h.n_head * hd, nkv * hd);
            let mut q = vec![0.0f32; n * qd];
            let mut kk = vec![0.0f32; n * kvd];
            let mut vv = vec![0.0f32; n * kvd];
            {
                let (tq, tk) = (self.w.layer(lw, "attn_q.weight")?, self.w.layer(lw, "attn_k.weight")?);
                let tv = if ly.has_v { Some(self.w.layer(lw, "attn_v.weight")?) } else { None };
                let mut jobs = vec![(Mat::of(&tq), &mut q[..]), (Mat::of(&tk), &mut kk[..])];
                if let Some(tv) = &tv {
                    jobs.push((Mat::of(tv), &mut vv[..]));
                }
                if !(npu_on(NPU_QKV) && crate::npu::apply(&mut jobs, &xn, n)) {
                    apply_group_multi(&mut jobs, xq.view(), n);
                }
            }
            if !ly.has_v {
                vv.copy_from_slice(&kk); // global layers without attn_v reuse the raw (pre-norm) K as V
            }
            lap.lap(2);
            let (cs, sn, half) = if ly.swa { (&css, &sns, hs) } else { (&csf, &snf, hf) };
            let n_rot = k.n_rot;
            let angles = |t: usize| (&cs[t * half..(t + 1) * half], &sn[t * half..(t + 1) * half]);
            // queries: every head of every token gets its q_norm, then its RoPE turn
            {
                let qp = SyncPtr::new(&mut q);
                let qn = &ly.q_norm;
                each(n * h.n_head, &|j| {
                    let (c_t, s_t) = angles(j / h.n_head);
                    // SAFETY: j is one head of one token, a separate run of hd values in q
                    let head = unsafe { qp.slice(j * hd, hd) };
                    rms_norm_inplace(head, qn, eps);
                    rope_neox(head, n_rot, c_t, s_t);
                });
            }
            // keys and values: K gets k_norm and its RoPE turn, V a weightless norm; both go into the 16-bit cache at
            // the token's position (KvCache keeps each KV head's rows together, cap rows per head)
            {
                let (kp, vp) = (SyncPtr::new(&mut kk), SyncPtr::new(&mut vv));
                let (k16, v16) = (SyncPtr::new(&mut ly.kv.k16), SyncPtr::new(&mut ly.kv.v16));
                let (ksp, vsp) = (SyncPtr::new(&mut ly.kv.k_scale), SyncPtr::new(&mut ly.kv.v_scale));
                let (kn, cap) = (&ly.k_norm, ly.kv.cap);
                each(n * nkv, &|j| {
                    let (t, g) = (j / nkv, j % nkv);
                    let (c_t, s_t) = angles(t);
                    let slot = g * cap + pos0 as usize + t;
                    // SAFETY: j is one KV head of one token: its own hd values in kk and vv, and its own cache slot
                    unsafe {
                        let key = kp.slice(j * hd, hd);
                        rms_norm_inplace(key, kn, eps);
                        rope_neox(key, n_rot, c_t, s_t);
                        *ksp.at(slot) = quant16(key, k16.slice(slot * hd, hd));
                        let val = vp.slice(j * hd, hd);
                        norm_plain(val, eps);
                        *vsp.at(slot) = quant16(val, v16.slice(slot * hd, hd));
                    }
                });
            }
            // positions from pos0 on were just written: the GPU's copy is stale from there
            ly.kv.gpu_lo = ly.kv.gpu_lo.min(pos0 as usize);
            lap.lap(3);
            let mut att = vec![0.0f32; n * qd];
            let eng = if ly.swa { &self.eng_swa } else { &self.eng_full };
            eng.run_block_tiered(&mut ly.kv, &q, qd, n, pos0, &mut att);
            lap.lap(4);
            let nbq = qd / QK;
            let mut out = vec![0.0f32; n * dm];
            {
                let two = self.w.layer(lw, "attn_output.weight")?;
                let mut jobs = [(Mat::of(&two), &mut out[..])];
                if !(npu_on(NPU_WO) && crate::npu::apply(&mut jobs, &att, n)) {
                    let mut attq = Q8::new(n * nbq);
                    let ap = crate::pool::Q8Ptr::new(attq.view_mut());
                    // SAFETY: item t owns token t's blocks of attq
                    each(n, &|t| kops::q8k_quantize(&att[t * qd..(t + 1) * qd], unsafe { ap.slice(t * nbq, nbq) }));
                    apply_group_multi(&mut jobs, attq.view(), n);
                }
            }
            {
                let (op, xp, w) = (SyncPtr::new(&mut out), SyncPtr::new(&mut x), &ly.post_attn);
                each(n, &|t| {
                    // SAFETY: item t owns token t's rows of out and x
                    let (o, xr) = unsafe { (op.slice(t * dm, dm), xp.slice(t * dm, dm)) };
                    rms_norm_inplace(o, w, eps);
                    for i in 0..dm {
                        xr[i] += o[i];
                    }
                });
            }
            lap.lap(5);
            // 2. the two branches of the feed-forward
            norm_quant(&mut xn, xq.view_mut(), &x, &ly.ffn_norm);
            lap.lap(1);
            let resid = x.clone();
            // the dense GELU feed-forward, on the ffn_norm input quantised above
            let mut dall = vec![0.0f32; n * dm];
            let (g, u, d) = (
                self.w.layer(lw, "ffn_gate.weight")?,
                self.w.layer(lw, "ffn_up.weight")?,
                self.w.layer(lw, "ffn_down.weight")?,
            );
            // with the NPU's GEGLU (a big block, and no expert cache), the whole branch goes into the experts' graph
            // there (moe_tokens_side) and runs while the CPU does the other experts, so its input is kept for it
            let side_ids = if npu_on(NPU_FFN) && crate::npu::geglu() && self.w.cache.is_none() {
                crate::npu::ffn_ids(&[Mat::of(&g), Mat::of(&u), Mat::of(&d)], n)
            } else {
                None
            };
            let xd = if side_ids.is_some() { xn.clone() } else { Vec::new() };
            let mut dense_out = vec![0.0f32; n * dm];
            if side_ids.is_none() {
                let ff = g.shape().1;
                let (mut gb, mut ub) = (vec![0.0f32; n * ff], vec![0.0f32; n * ff]);
                // on the NPU (a big block, TORC_NPU=1): gate and up run there, GELU here in f32, and down there
                let npu = npu_on(NPU_FFN)
                    && crate::npu::apply(&mut [(Mat::of(&g), &mut gb[..]), (Mat::of(&u), &mut ub[..])], &xn, n)
                    && {
                        let hp = SyncPtr::new(&mut gb);
                        each(n, &|t| {
                            // SAFETY: item t owns token t's row of gb
                            let hr = unsafe { hp.slice(t * ff, ff) };
                            for (i, v) in hr.iter_mut().enumerate() {
                                *v = crate::act::glu(true, *v, ub[t * ff + i]);
                            }
                        });
                        crate::npu::apply(&mut [(Mat::of(&d), &mut dall[..])], &gb, n)
                    };
                if !npu {
                    let job = EJob { mats: [Mat::of(&g), Mat::of(&u), Mat::of(&d)], toks: (0..n).collect(), g0: 0 };
                    run_ejobs(false, true, &[job], xq.view(), nb, dm, &mut dall);
                }
                let (bp, w) = (SyncPtr::new(&mut dense_out), &ly.post_ffw_1);
                each(n, &|t| {
                    // SAFETY: item t owns token t's row of dense_out
                    rms_norm(unsafe { bp.slice(t * dm, dm) }, &dall[t * dm..(t + 1) * dm], w, eps);
                });
            }
            lap.lap(6);
            // the experts' input: the residual through pre_ffw_norm_2, quantised (and kept in f32 for the NPU's hot
            // experts when the dense branch runs there)
            let mut xe = if side_ids.is_some() { vec![0.0f32; n * dm] } else { Vec::new() };
            {
                let (np, qp, ep) =
                    (SyncPtr::new(&mut xn), crate::pool::Q8Ptr::new(xq.view_mut()), SyncPtr::new(&mut xe));
                let (w, resid, keep) = (&ly.pre_ffw_2, &resid, side_ids.is_some());
                each(n, &|t| {
                    // SAFETY: item t owns token t's rows of xn, xq and xe
                    let (xt, qt) = unsafe { (np.slice(t * dm, dm), qp.slice(t * nb, nb)) };
                    rms_norm(xt, &resid[t * dm..(t + 1) * dm], w, eps);
                    kops::q8k_quantize(xt, qt);
                    if keep {
                        unsafe { ep.slice(t * dm, dm) }.copy_from_slice(xt);
                    }
                });
            }
            // the router's input, made separately from the same residual
            let mut route = vec![0.0f32; n * dm];
            {
                let rp = SyncPtr::new(&mut route);
                let (gscale, resid) = (&ly.gate_inp_scale, &resid);
                each(n, &|t| {
                    // SAFETY: item t owns token t's row of route
                    router_input(unsafe { rp.slice(t * dm, dm) }, &resid[t * dm..(t + 1) * dm], gscale, eps);
                });
            }
            x.fill(0.0);
            lap.lap(7);
            if let Some(ids) = side_ids {
                let side = NpuSide {
                    xe: &xe,
                    ffn: Some(crate::npu::Ffn { ids, x: xd.as_ptr(), n, y: dall.as_mut_ptr() }),
                    done: std::cell::Cell::new(false),
                };
                moe_tokens_side(&self.w, &ly.moe, &route, xq.view(), &mut x, n, &side)?;
                if !side.done.get() {
                    // not started there: the CPU's path, from its input quantised again
                    let mut xqd = Q8::new(n * nb);
                    let qp = crate::pool::Q8Ptr::new(xqd.view_mut());
                    // SAFETY: item t owns token t's blocks of xqd
                    each(n, &|t| kops::q8k_quantize(&xd[t * dm..(t + 1) * dm], unsafe { qp.slice(t * nb, nb) }));
                    let job = EJob { mats: [Mat::of(&g), Mat::of(&u), Mat::of(&d)], toks: (0..n).collect(), g0: 0 };
                    run_ejobs(false, true, &[job], xqd.view(), nb, dm, &mut dall);
                }
                let (bp, w) = (SyncPtr::new(&mut dense_out), &ly.post_ffw_1);
                each(n, &|t| {
                    // SAFETY: item t owns token t's row of dense_out
                    rms_norm(unsafe { bp.slice(t * dm, dm) }, &dall[t * dm..(t + 1) * dm], w, eps);
                });
            } else {
                moe_tokens(&self.w, &ly.moe, &route, xq.view(), &mut x, n)?;
            }
            {
                let xp = SyncPtr::new(&mut x);
                let (w2, wf, os, dense_out, resid) = (&ly.post_ffw_2, &ly.post_ffw, ly.layer_scale, &dense_out, &resid);
                each(n, &|t| {
                    // SAFETY: item t owns token t's row of x
                    let xt = unsafe { xp.slice(t * dm, dm) };
                    rms_norm_inplace(xt, w2, eps);
                    let b = &dense_out[t * dm..(t + 1) * dm];
                    for i in 0..dm {
                        xt[i] += b[i];
                    }
                    // the sandwich: the FFN's output is normed, the residual added, and the layer's scale applied
                    rms_norm_inplace(xt, wf, eps);
                    let r = &resid[t * dm..(t + 1) * dm];
                    for i in 0..dm {
                        xt[i] = (r[i] + xt[i]) * os;
                    }
                });
            }
            lap.lap(8);
            if trace {
                crate::ops::trace_state(l, &x[(n - 1) * dm..n * dm]);
            }
        }
        // 3. the output: the final norm, the tied head and the soft cap
        let first = if all { 0 } else { n - 1 };
        let m = n - first;
        let mut xqo = Q8::new(m * nb);
        self.hid.resize(m * dm, 0.0);
        for t in first..n {
            let xo = &mut self.hid[(t - first) * dm..(t - first + 1) * dm];
            rms_norm(xo, &x[t * dm..(t + 1) * dm], &self.out_norm, h.eps);
            kops::q8k_quantize(xo, xqo.sub_mut((t - first) * nb..(t - first + 1) * nb));
        }
        if logits.len() < m * nv {
            return err("logits buffer too small");
        }
        let hm = match &self.head {
            Some((ty, d)) => Mat { ty: *ty, rows: nv, cols: dm, data: d },
            None => Mat::of(&emb),
        };
        apply_group_multi(&mut [(hm, &mut logits[..m * nv])], xqo.view(), m);
        if h.softcap > 0.0 && self.cap_logits {
            let cap = h.softcap;
            for v in logits[..m * nv].iter_mut() {
                *v = cap * (*v / cap).tanh();
            }
        }
        lap.lap(9);
        Ok(())
    }
}

fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let r = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | r as u16;
    }
    let r = m + 0xfff + ((m >> 13) & 1);
    let (mut e, mut m) = (e as u32, r >> 13);
    if m == 0x400 {
        m = 0;
        e += 1;
        if e >= 31 {
            return sign | 0x7c00;
        }
    }
    sign | ((e << 10) as u16) | m as u16
}

/// Rows of type `from` requantised to Q8_0 or Q4_0 (with ggml's reference rounding), the rows in parallel.
pub fn requant(from: u32, data: &[u8], cols: usize, rows: usize, to: u32) -> Vec<u8> {
    use rayon::prelude::*;
    let rb_in = data.len() / rows;
    let nb = cols / 32;
    let rb_out = nb * if to == 8 { 34 } else { 18 };
    let mut out = vec![0u8; rows * rb_out];
    out.par_chunks_mut(rb_out).enumerate().for_each(|(r, o)| {
        let mut x = vec![0f32; cols];
        kernels::ops::dequantize(from, &data[r * rb_in..(r + 1) * rb_in], &mut x);
        for (b, xb) in x.chunks_exact(32).enumerate() {
            if to == 8 {
                let amax = xb.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let ob = &mut o[b * 34..(b + 1) * 34];
                ob[..2].copy_from_slice(&f16_bits(d).to_le_bytes());
                for (j, v) in xb.iter().enumerate() {
                    ob[2 + j] = ((v * id).round() as i32).clamp(-127, 127) as i8 as u8;
                }
            } else {
                let (mut amax, mut max) = (0f32, 0f32);
                for &v in xb {
                    if v.abs() > amax {
                        amax = v.abs();
                        max = v;
                    }
                }
                let d = max / -8.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let ob = &mut o[b * 18..(b + 1) * 18];
                ob[..2].copy_from_slice(&f16_bits(d).to_le_bytes());
                for j in 0..16 {
                    let q0 = ((xb[j] * id + 8.5) as i32).min(15) as u8;
                    let q1 = ((xb[j + 16] * id + 8.5) as i32).min(15) as u8;
                    ob[2 + j] = q0 | (q1 << 4);
                }
            }
        }
    });
    out
}
