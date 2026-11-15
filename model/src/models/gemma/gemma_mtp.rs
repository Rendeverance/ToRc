//! gemma_mtp.rs - Gemma 4's assistant model (general.architecture gemma4-assistant), the drafter for speculative
//! decoding, run as llama.cpp runs it (src/models/gemma4-assistant.cpp; common/speculative.cpp, draft-mtp with the
//! memory shared):
//! - a row joins the target's embedding of a token (scaled by the square root of the target's width) with a hidden
//!   state of the target's width, `[embed(t); h]`, projected to the drafter's width (nextn.pre_projection);
//! - four layers. Each computes only queries and attends over keys and values the target already cached: layer
//!   n_layer - 2 (sliding window) serves the drafter's sliding layers and layer n_layer - 1 (full) its full one. So the
//!   drafter holds no cache of its own and has nothing to replay after the prompt. A dense GELU feed-forward follows,
//!   with the same extra norms and layer scale as the target;
//! - the final norm, then the drafter's own tied head (token_embd) for the draft, and nextn.post_projection back to the
//!   target's width for the next draft's h.
//!
//! The first draft's h is the target's hidden state after its final norm at the position before (Gemma::hid), and later
//! drafts use the drafter's own. Every draft sits at the same position, the position of the token the chain starts from
//! (transformers' gemma4_assistant: "the same position for all draft tokens"), and its attention reads the cache below
//! that position. Drafts only choose what the target scores in a verify pass, never the output (the target's greedy
//! pick decides every token), so this need not match llama.cpp bit for bit: its matrices (the file's Q8_0) are
//! requantised to Q4_0 at load, halving the bytes a draft step reads (TORC_DRAFT_HEAD=q8_0 keeps the file's types).

use crate::gemma::{Gemma, requant};
use crate::moe::{EJob, Mat, apply_group_multi, run_ejobs};
use crate::ops::{rms_norm, rms_norm_inplace, rope_neox};
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::meta::{Meta, Value};
use format::{Result, err};
use kernels::{Q8, QK, ops as kops};

/// The seconds spent in each part of a draft step, summed (the server prints them with TORC_DRAFT_LOG).
pub static PROF: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
pub const PARTS: [&str; 5] = ["input", "attention", "ffn", "head", "softmax"];

/// A matrix as the drafter runs it (requantised, or the file's bytes copied).
struct OwnMat {
    ty: u32,
    rows: usize,
    cols: usize,
    data: Vec<u8>,
}

impl OwnMat {
    fn of(m: Mat, keep: bool) -> OwnMat {
        let to = if keep { m.ty } else { 2 };
        let data = if to == m.ty { m.data.to_vec() } else { requant(m.ty, m.data, m.cols, m.rows, to) };
        OwnMat { ty: to, rows: m.rows, cols: m.cols, data }
    }

    fn mat(&self) -> Mat<'_> {
        Mat { ty: self.ty, rows: self.rows, cols: self.cols, data: &self.data }
    }
}

struct DLayer {
    swa: bool,
    wq: OwnMat,
    wo: OwnMat,
    /// the feed-forward: gate, up and down
    ffn: [OwnMat; 3],
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    post_attn: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw: Vec<f32>,
    layer_scale: f32,
}

pub struct GemmaDraft {
    pre: OwnMat,
    post: OwnMat,
    /// the drafter's width, and the target's
    dm: usize,
    dt: usize,
    eps: f32,
    layers: Vec<DLayer>,
    rope_freqs: Option<Vec<f32>>,
    out_norm: Vec<f32>,
    head: OwnMat,
    nv: usize,
}

impl GemmaDraft {
    /// The drafter for target g, checked against it: its attention must fit the target's cache.
    pub fn load(path: &str, g: &Gemma) -> Result<GemmaDraft> {
        let w = Weights::load(path)?;
        let m = Meta::parse(&w.file.metadata)?;
        if m.str("general.architecture") != Some("gemma4-assistant") {
            return err(format!("{path}: not a gemma4-assistant file"));
        }
        let need = |k: &str| m.arch_u32(k).ok_or_else(|| format::Error(format!("{path}: missing {k}")));
        let n_layer = need("block_count")? as usize;
        let dm = need("embedding_length")? as usize;
        let dt = need("embedding_length_out")? as usize;
        let n_head = need("attention.head_count")? as usize;
        let pat: Vec<bool> = match m.get(&m.arch_key("attention.sliding_window_pattern")) {
            Some(Value::Array(_, v)) if v.len() >= n_layer => {
                v[..n_layer].iter().map(|x| matches!(x, Value::Bool(true))).collect()
            }
            _ => return err(format!("{path}: missing sliding_window_pattern")),
        };
        let h = &g.h;
        let eps = m.arch_f32("attention.layer_norm_rms_epsilon").unwrap_or(h.eps);
        let fits = dt == h.d_model
            && n_head == h.n_head
            && need("attention.key_length")? as usize == h.full.head_dim
            && need("attention.key_length_swa")? as usize == h.swa.head_dim
            && need("rope.dimension_count")? as usize == h.full.n_rot
            && need("rope.dimension_count_swa")? as usize == h.swa.n_rot
            && m.arch_f32("rope.freq_base") == Some(h.full.base)
            && m.arch_f32("rope.freq_base_swa") == Some(h.swa.base)
            && h.n_layer >= 2
            && h.swa_layers[h.n_layer - 2]
            && !h.swa_layers[h.n_layer - 1]
            && dm.is_multiple_of(QK);
        if !fits {
            return err(format!("{path}: not a drafter of this model"));
        }
        let keep = crate::switches::get("TORC_DRAFT_HEAD").as_deref() == Some("q8_0");
        let own = |n: &str| -> Result<OwnMat> { Ok(OwnMat::of(Mat::of(&w.need(n)?), keep)) };
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer as u32 {
            let f = |s: &str| -> Result<Vec<f32>> { w.layer(l, s)?.f32s() };
            let layer_scale = match w.tensor(&format!("blk.{l}.layer_output_scale.weight")) {
                Some(t) if t.t.ty == 0 => t.f32s()?.first().copied().unwrap_or(1.0),
                _ => 1.0,
            };
            // the dense feed-forward (resident tensors, widened to whole 256s by the converter)
            let ffn_of = |s: &str| -> Result<OwnMat> { Ok(OwnMat::of(Mat::of(&w.layer(l, s)?), keep)) };
            let ffn = [ffn_of("ffn_gate.weight")?, ffn_of("ffn_up.weight")?, ffn_of("ffn_down.weight")?];
            layers.push(DLayer {
                swa: pat[l as usize],
                wq: own(&format!("blk.{l}.attn_q.weight"))?,
                wo: own(&format!("blk.{l}.attn_output.weight"))?,
                ffn,
                attn_norm: f("attn_norm.weight")?,
                q_norm: f("attn_q_norm.weight")?,
                post_attn: f("post_attention_norm.weight")?,
                ffn_norm: f("ffn_norm.weight")?,
                post_ffw: f("post_ffw_norm.weight")?,
                layer_scale,
            });
        }
        let rope_freqs = match w.tensor("rope_freqs.weight") {
            Some(t) => Some(t.f32s()?).filter(|v| v.len() == h.full.n_rot / 2),
            None => None,
        };
        let out_norm = w.need("output_norm.weight")?.f32s()?;
        let e = w.need("token_embd.weight")?;
        let (cols, nv) = e.shape();
        if cols != dm {
            return err(format!("{path}: head width {cols}"));
        }
        let head = OwnMat::of(Mat::of(&e), keep);
        let (pre, post) = (own("nextn.pre_projection.weight")?, own("nextn.post_projection.weight")?);
        Ok(GemmaDraft { pre, post, dm, dt, eps, layers, rope_freqs, out_norm, head, nv })
    }

    /// One draft: token `tok` with hidden state `hid` (of the target's width) at position pos (the target's cache holds
    /// the positions below it). Returns the drafter's pick, its probability (the softmax of the drafter's logits) and
    /// the next draft's hidden state; the probability is only computed `with_p` (otherwise it is 1).
    pub fn step(&self, g: &Gemma, tok: u32, hid: &[f32], pos: u32, with_p: bool) -> Result<(u32, f32, Vec<f32>)> {
        let (dm, dt, eps) = (self.dm, self.dt, self.eps);
        let h = &g.h;
        let mut lap = crate::Laps::start(&PROF);
        if hid.len() != dt || pos == 0 || pos as usize > g.n_ctx as usize {
            return err("draft: bad input");
        }
        // [embed(tok) * sqrt(dt); hid], projected to the drafter's width
        let emb = g.w.need("token_embd.weight")?;
        let rb = emb.row_bytes();
        if tok as usize >= emb.shape().1 {
            return err("draft: token out of range");
        }
        let mut cat = vec![0.0f32; 2 * dt];
        kops::dequantize(emb.t.ty, &emb.data[tok as usize * rb..(tok as usize + 1) * rb], &mut cat[..dt]);
        let s = (dt as f32).sqrt();
        for v in cat[..dt].iter_mut() {
            *v *= s;
        }
        cat[dt..].copy_from_slice(hid);
        let mut catq = Q8::new(2 * dt / QK);
        kops::q8k_quantize(&cat, catq.view_mut());
        let mut x = vec![0.0f32; dm];
        apply_group_multi(&mut [(self.pre.mat(), &mut x[..])], catq.view(), 1);
        lap.lap(0);
        let nb = dm / QK;
        let mut xn = vec![0.0f32; dm];
        let mut xq = Q8::new(nb);
        for ly in &self.layers {
            let k = if ly.swa { &h.swa } else { &h.full };
            let hd = k.head_dim;
            let qd = h.n_head * hd;
            rms_norm(&mut xn, &x, &ly.attn_norm, eps);
            kops::q8k_quantize(&xn, xq.view_mut());
            let mut q = vec![0.0f32; qd];
            apply_group_multi(&mut [(ly.wq.mat(), &mut q[..])], xq.view(), 1);
            let half = k.n_rot / 2;
            let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
            let slow = if ly.swa { None } else { self.rope_freqs.as_deref() };
            crate::gemma::fill_rope(pos, k.n_rot, k.base, slow, &mut cs, &mut sn);
            for hh in 0..h.n_head {
                let v = &mut q[hh * hd..(hh + 1) * hd];
                rms_norm_inplace(v, &ly.q_norm, eps);
                rope_neox(v, k.n_rot, &cs, &sn);
            }
            // the target's cache below pos: a query "at" pos - 1 reads rows 0..pos (the sliding window's far edge is
            // one row wider than llama.cpp's, which is fine for a draft)
            let tl = if ly.swa { h.n_layer - 2 } else { h.n_layer - 1 };
            let mut att = vec![0.0f32; qd];
            g.eng(ly.swa).run_block(g.kv_of(tl), &q, qd, 1, pos - 1, &mut att);
            let mut attq = Q8::new(qd / QK);
            kops::q8k_quantize(&att, attq.view_mut());
            let mut o = vec![0.0f32; dm];
            apply_group_multi(&mut [(ly.wo.mat(), &mut o[..])], attq.view(), 1);
            rms_norm_inplace(&mut o, &ly.post_attn, eps);
            for i in 0..dm {
                x[i] += o[i];
            }
            lap.lap(1);
            // the dense GELU feed-forward
            rms_norm(&mut xn, &x, &ly.ffn_norm, eps);
            kops::q8k_quantize(&xn, xq.view_mut());
            let job = EJob { mats: [ly.ffn[0].mat(), ly.ffn[1].mat(), ly.ffn[2].mat()], toks: vec![0], g0: 0 };
            let mut y = vec![0.0f32; dm];
            run_ejobs(false, true, &[job], xq.view(), nb, dm, &mut y);
            rms_norm_inplace(&mut y, &ly.post_ffw, eps);
            for i in 0..dm {
                x[i] = (x[i] + y[i]) * ly.layer_scale;
            }
            lap.lap(2);
        }
        rms_norm(&mut xn, &x, &self.out_norm, eps);
        kops::q8k_quantize(&xn, xq.view_mut());
        let mut logits = vec![0.0f32; self.nv];
        let mut hn = vec![0.0f32; dt];
        apply_group_multi(&mut [(self.head.mat(), &mut logits[..]), (self.post.mat(), &mut hn[..])], xq.view(), 1);
        lap.lap(3);
        let mut b = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[b] {
                b = i;
            }
        }
        if !with_p {
            return Ok((b as u32, 1.0, hn));
        }
        // the softmax's denominator, in chunks over the pool
        let mx = logits[b];
        const CH: usize = 8192;
        let nch = self.nv.div_ceil(CH);
        let mut part = vec![0.0f32; nch];
        let pp = SyncPtr::new(&mut part);
        crate::pool::par_for(nch, &|i| {
            let z: f32 = logits[i * CH..((i + 1) * CH).min(self.nv)].iter().map(|&v| (v - mx).exp()).sum();
            // SAFETY: chunk i only writes its own partial sum
            unsafe { *pp.at(i) = z };
        });
        lap.lap(4);
        Ok((b as u32, 1.0 / part.iter().sum::<f32>(), hn))
    }
}
