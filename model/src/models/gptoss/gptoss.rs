//! gptoss.rs - gpt-oss (general.architecture gpt-oss: the 20B and the 120B).
//!
//! How it differs from the other models here:
//! - The embedding is 2880 wide, not a whole number of 256-number blocks. The converter pads every quantised matrix
//!   that reads it with zero columns up to 3072, and this file pads its activations with zeros to match before it
//!   quantises them, so each product is the unpadded one. Norms and the residual stay 2880 wide.
//! - Attention: 64 query heads over 8 KV heads of 64, biases on q, k, v and the output, YaRN RoPE (NeoX pairs), every
//!   other layer (the even ones) over a sliding window of the last 128 positions, and a learned "sink" per head: an
//!   extra logit in each softmax that takes a share of the weight and points at nothing. Attention runs on the shared
//!   kernel (fattn.rs) with the layer's sinks, keys and values cached at 16 bits as Gemma's are.
//! - The routed experts (32, 4 a token) have biases on gate, up and down, and a clamped SwiGLU: gate at most 7, up
//!   between -7 and 7, then gate * sigmoid(1.702 gate) * (up + 1). The router has a bias too; the top 4 of its logits
//!   are softmaxed among themselves. The experts are MXFP4, through the expert packs like any other model's.
//!
//! A block of n tokens runs every product for all n at once (each weight row read once); each token's numbers are the
//! same bits as a block of one.

use crate::attn::{AttnEngine, KvCache, quant16};
use crate::moe::{Mat, apply_group_multi};
use crate::ops::{rms_norm, rope_neox};
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::meta::Meta;
use format::{BlockKind, Result, err};
use kernels::{Q8, QK, ops as kops};

#[derive(Clone, Debug)]
pub struct OssHp {
    pub n_layer: usize,
    /// the embedding width, and that padded to whole 256-number blocks
    pub d_model: usize,
    pub d_pad: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub window: usize,
    pub n_expert: usize,
    pub n_used: usize,
    /// the experts' width, and that padded (the converter pads gate's and up's rows to it)
    pub d_ff: usize,
    pub d_ff_pad: usize,
    pub rope_base: f32,
    /// YaRN: the context the model was trained at, the scale factor, beta_fast and beta_slow
    pub yarn_ctx: f32,
    pub yarn_factor: f32,
    pub yarn_fast: f32,
    pub yarn_slow: f32,
}

fn up256(n: usize) -> usize {
    n.div_ceil(QK) * QK
}

impl OssHp {
    pub fn from_meta(metadata: &[u8]) -> Result<OssHp> {
        let m = Meta::parse(metadata)?;
        let arch = m.str("general.architecture").unwrap_or("");
        if arch != "gpt-oss" {
            return err(format!("not a gpt-oss file ({arch})"));
        }
        let need =
            |k: &str| m.arch_u32(k).map(|v| v as usize).ok_or_else(|| format::Error(format!("missing gpt-oss.{k}")));
        let needf = |k: &str| m.arch_f32(k).ok_or_else(|| format::Error(format!("missing gpt-oss.{k}")));
        let d_model = need("embedding_length")?;
        let d_ff = need("expert_feed_forward_length")?;
        let h = OssHp {
            n_layer: need("block_count")?,
            d_model,
            d_pad: up256(d_model),
            n_head: need("attention.head_count")?,
            n_head_kv: need("attention.head_count_kv")?,
            head_dim: need("attention.key_length")?,
            n_vocab: m.len_of("tokenizer.ggml.tokens").unwrap_or(0),
            eps: needf("attention.layer_norm_rms_epsilon")?,
            window: need("attention.sliding_window")?,
            n_expert: need("expert_count")?,
            n_used: need("expert_used_count")?,
            d_ff,
            d_ff_pad: up256(d_ff),
            rope_base: needf("rope.freq_base")?,
            yarn_ctx: m.arch_u32("rope.scaling.original_context_length").unwrap_or(4096) as f32,
            yarn_factor: m.arch_f32("rope.scaling.factor").unwrap_or(1.0),
            yarn_fast: m.arch_f32("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            yarn_slow: m.arch_f32("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
        };
        if h.n_head_kv == 0
            || !h.n_head.is_multiple_of(h.n_head_kv)
            || !h.head_dim.is_multiple_of(2)
            || !(h.n_head * h.head_dim).is_multiple_of(QK)
        {
            return err("gpt-oss: an attention shape this file cannot run");
        }
        Ok(h)
    }

    /// Layer l attends over a sliding window (the even layers, as llama.cpp and the reference implementation have it).
    pub fn swa(&self, l: usize) -> bool {
        l.is_multiple_of(2)
    }
}

/// YaRN's cos and sin for every rotated pair of one position (llama.cpp's rope_yarn, NeoX pairs): each pair's angle
/// is blended between the interpolated (pos / factor) and the original one by a ramp over the pair index, and cos and
/// sin carry the attention factor 1 + 0.1 ln(factor).
fn yarn_row(h: &OssHp, pos: u32, cos: &mut [f32], sin: &mut [f32]) {
    let n = h.head_dim as f32;
    let corr = |beta: f32| n * (h.yarn_ctx / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * h.rope_base.ln());
    let low = corr(h.yarn_fast).floor().max(0.0);
    let high = corr(h.yarn_slow).ceil().min(n - 1.0);
    let scale = 1.0 / h.yarn_factor;
    let mscale = if h.yarn_factor > 1.0 { 1.0 + 0.1 * (1.0 / scale).ln() } else { 1.0 };
    for i in 0..h.head_dim / 2 {
        let extrap = pos as f32 * h.rope_base.powf(-2.0 * i as f32 / n);
        let interp = scale * extrap;
        let y = (i as f32 - low) / (high - low).max(0.001);
        let ramp = 1.0 - y.clamp(0.0, 1.0);
        let theta = interp * (1.0 - ramp) + extrap * ramp;
        cos[i] = theta.cos() * mscale;
        sin[i] = theta.sin() * mscale;
    }
}

struct OLayer {
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    qb: Vec<f32>,
    kb: Vec<f32>,
    vb: Vec<f32>,
    ob: Vec<f32>,
    /// the router (n_expert x d_model) and its bias
    router: Vec<f32>,
    router_b: Vec<f32>,
    /// each expert's gate, up and down biases (n_expert x d_ff, d_ff, d_model)
    gate_b: Vec<f32>,
    up_b: Vec<f32>,
    down_b: Vec<f32>,
    /// the layer's experts record
    group: u32,
    /// keys and values (after bias and RoPE), 16 bits with a scale per row
    kv: KvCache,
    /// attention with this layer's sinks (and window, on the even layers)
    eng: AttnEngine,
}

/// The time spent in each phase of forward, summed over calls (TORC_PROF=1).
pub static PROF: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
pub const PHASES: [&str; 9] =
    ["embed and head", "qkv", "attention", "output", "router", "gate and up", "SwiGLU", "down", "expert sums"];

pub struct Oss {
    pub h: OssHp,
    pub w: Weights,
    layers: Vec<OLayer>,
    out_norm: Vec<f32>,
    pub n_ctx: u32,
}

/// Runs f(i) for i in 0..n: on this thread for a single item, else over the pool (each item writes only its own rows,
/// so the bits are the same either way).
fn each(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n < 2 { (0..n).for_each(f) } else { crate::pool::par_for(n, f) }
}

/// The rows of x (n x d, d <= dp) quantised as n x dp, zero beyond d.
fn quant_padded(x: &[f32], n: usize, d: usize, dp: usize) -> Q8 {
    let nb = dp / QK;
    let mut q = Q8::new(n * nb);
    let qp = crate::pool::Q8Ptr::new(q.view_mut());
    each(n, &|t| {
        let mut row = vec![0.0f32; dp];
        row[..d].copy_from_slice(&x[t * d..(t + 1) * d]);
        // SAFETY: item t owns token t's blocks of q
        kops::q8k_quantize(&row, unsafe { qp.slice(t * nb, nb) });
    });
    q
}

impl Oss {
    pub fn load(path: &str, n_ctx: u32) -> Result<Oss> {
        let w = Weights::load(path)?;
        let h = OssHp::from_meta(&w.file.metadata)?;
        // the model's identity, so what is learned while it runs is kept in its tune file for the next start (gemma.rs)
        crate::pipe::set_ident(
            &w.file.header.model_id,
            w.file.header.file_bytes,
            h.n_layer as u32,
            h.d_model as u32,
            0,
            w.file.tensors.len(),
            n_ctx,
        );
        let mut layers = Vec::with_capacity(h.n_layer);
        for l in 0..h.n_layer as u32 {
            let group = w
                .file
                .blocks
                .iter()
                .find(|b| b.layer == l && b.kind == BlockKind::Moe && b.experts.is_some())
                .and_then(|b| b.experts)
                .ok_or_else(|| format::Error(format!("layer {l}: no routed experts")))?;
            let x = w.layer_experts(group);
            if x.n_expert as usize != h.n_expert || x.d_in as usize != h.d_pad || x.width as usize != h.d_ff_pad {
                return err(format!(
                    "layer {l}: experts of {} x {} x {}, expected {} x {} x {} (convert the GGUF again)",
                    x.n_expert, x.d_in, x.width, h.n_expert, h.d_pad, h.d_ff_pad
                ));
            }
            let f = |s: &str| -> Result<Vec<f32>> { w.layer(l, s)?.f32s() };
            let window = if h.swa(l as usize) { h.window as u32 } else { 0 };
            let eng = AttnEngine::create(h.n_head as u32, h.n_head_kv as u32, h.head_dim as u32, None, window)?
                .with_sinks(f("attn_sinks.weight")?);
            layers.push(OLayer {
                attn_norm: f("attn_norm.weight")?,
                post_norm: f("post_attention_norm.weight")?,
                qb: f("attn_q.bias")?,
                kb: f("attn_k.bias")?,
                vb: f("attn_v.bias")?,
                ob: f("attn_output.bias")?,
                router: f("ffn_gate_inp.weight")?,
                router_b: f("ffn_gate_inp.bias")?,
                gate_b: f("ffn_gate_exps.bias")?,
                up_b: f("ffn_up_exps.bias")?,
                down_b: f("ffn_down_exps.bias")?,
                group,
                kv: KvCache::with(h.n_head_kv, h.head_dim, n_ctx as usize, true),
                eng,
            });
        }
        let out_norm = w.need("output_norm.weight")?.f32s()?;
        Ok(Oss { h, w, layers, out_norm, n_ctx })
    }

    pub fn n_vocab(&self) -> Result<usize> {
        Ok(self.w.need("output.weight")?.shape().1)
    }

    /// Every layer's cache at positions 0..npos (KvCache::put_bytes): the model's whole state after its first npos
    /// tokens, since nothing else carries over from one token to the next. For torc-server's saved chats (saved.rs).
    pub fn state_bytes(&self, npos: usize) -> Vec<u8> {
        let mut b = Vec::new();
        for ly in &self.layers {
            ly.kv.put_bytes(self.h.head_dim, npos, &mut b);
        }
        b
    }

    /// Puts state_bytes back. Fails if they do not fit this model, leaving the cache partly written.
    pub fn set_state_bytes(&mut self, npos: usize, b: &[u8]) -> Result<()> {
        let hd = self.h.head_dim;
        let mut at = 0;
        let mut take = |n: usize| {
            let e = (at + n).min(b.len());
            let r = b[at..e].to_vec();
            at = e;
            r
        };
        for ly in &mut self.layers {
            ly.kv.set_bytes(hd, npos, &mut take)?;
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
        let (dm, dp, hd) = (h.d_model, h.d_pad, h.head_dim);
        let (nh, nkv) = (h.n_head, h.n_head_kv);
        let (qd, kvd) = (nh * hd, nkv * hd);
        if n == 0 || pos0 as usize + n > self.n_ctx as usize {
            return err("position past the context");
        }
        let mut lap = crate::Laps::start(&PROF);
        // the embedding (padded rows in the file: the first d_model of each)
        let emb = self.w.need("token_embd.weight")?;
        let (rb, nv) = (emb.row_bytes(), emb.shape().1);
        let mut x = vec![0.0f32; n * dm];
        let mut row = vec![0.0f32; emb.shape().0];
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= nv {
                return err("token out of range");
            }
            kops::dequantize(emb.t.ty, &emb.data[tok as usize * rb..(tok as usize + 1) * rb], &mut row);
            x[t * dm..(t + 1) * dm].copy_from_slice(&row[..dm]);
        }
        // the angles for every position of the block
        let half = hd / 2;
        let (mut cs, mut sn) = (vec![0.0f32; n * half], vec![0.0f32; n * half]);
        for t in 0..n {
            yarn_row(&h, pos0 + t as u32, &mut cs[t * half..(t + 1) * half], &mut sn[t * half..(t + 1) * half]);
        }
        lap.lap(0);
        let mut xn = vec![0.0f32; n * dm];
        let norm = |xn: &mut [f32], x: &[f32], wn: &[f32]| {
            let np = SyncPtr::new(xn);
            // SAFETY: item t owns token t's row of xn
            each(n, &|t| rms_norm(unsafe { np.slice(t * dm, dm) }, &x[t * dm..(t + 1) * dm], wn, h.eps));
        };
        for l in 0..h.n_layer {
            let lw = l as u32;
            // 1. attention
            norm(&mut xn, &x, &self.layers[l].attn_norm);
            let xq = quant_padded(&xn, n, dm, dp);
            let (mut q, mut kk, mut vv) = (vec![0.0f32; n * qd], vec![0.0f32; n * kvd], vec![0.0f32; n * kvd]);
            {
                let (tq, tk, tv) = (
                    self.w.layer(lw, "attn_q.weight")?,
                    self.w.layer(lw, "attn_k.weight")?,
                    self.w.layer(lw, "attn_v.weight")?,
                );
                apply_group_multi(
                    &mut [(Mat::of(&tq), &mut q[..]), (Mat::of(&tk), &mut kk[..]), (Mat::of(&tv), &mut vv[..])],
                    xq.view(),
                    n,
                );
            }
            let ly = &mut self.layers[l];
            {
                // biases, then RoPE on every query and key head; keys and values into the cache
                let (qp, kp, vp) = (SyncPtr::new(&mut q), SyncPtr::new(&mut kk), SyncPtr::new(&mut vv));
                let (k16, v16) = (SyncPtr::new(&mut ly.kv.k16), SyncPtr::new(&mut ly.kv.v16));
                let (ksp, vsp) = (SyncPtr::new(&mut ly.kv.k_scale), SyncPtr::new(&mut ly.kv.v_scale));
                let cap = ly.kv.cap;
                let (qb, kb, vb, cs, sn) = (&ly.qb, &ly.kb, &ly.vb, &cs, &sn);
                each(n, &|t| {
                    let (c_t, s_t) = (&cs[t * half..(t + 1) * half], &sn[t * half..(t + 1) * half]);
                    // SAFETY: item t owns token t's rows of q, kk, vv and its position's rows of the cache
                    let (qt, kt, vt) =
                        unsafe { (qp.slice(t * qd, qd), kp.slice(t * kvd, kvd), vp.slice(t * kvd, kvd)) };
                    for (a, b) in qt.iter_mut().zip(qb) {
                        *a += b;
                    }
                    for (a, b) in kt.iter_mut().zip(kb) {
                        *a += b;
                    }
                    for (a, b) in vt.iter_mut().zip(vb) {
                        *a += b;
                    }
                    for hh in 0..nh {
                        rope_neox(&mut qt[hh * hd..(hh + 1) * hd], hd, c_t, s_t);
                    }
                    for hh in 0..nkv {
                        rope_neox(&mut kt[hh * hd..(hh + 1) * hd], hd, c_t, s_t);
                    }
                    // KvCache keeps each KV head's rows together, cap rows a head
                    for g in 0..nkv {
                        let slot = g * cap + pos0 as usize + t;
                        unsafe {
                            *ksp.at(slot) = quant16(&kt[g * hd..(g + 1) * hd], k16.slice(slot * hd, hd));
                            *vsp.at(slot) = quant16(&vt[g * hd..(g + 1) * hd], v16.slice(slot * hd, hd));
                        }
                    }
                });
            }
            // positions from pos0 on were just written: the GPU's copy is stale from there
            ly.kv.gpu_lo = ly.kv.gpu_lo.min(pos0 as usize);
            lap.lap(1);
            let mut att = vec![0.0f32; n * qd];
            ly.eng.run_block_tiered(&mut ly.kv, &q, qd, n, pos0, &mut att);
            lap.lap(2);
            {
                let nbq = qd / QK;
                let mut attq = Q8::new(n * nbq);
                let ap = crate::pool::Q8Ptr::new(attq.view_mut());
                // SAFETY: item t owns token t's blocks of attq
                each(n, &|t| kops::q8k_quantize(&att[t * qd..(t + 1) * qd], unsafe { ap.slice(t * nbq, nbq) }));
                let mut out = vec![0.0f32; n * dm];
                let two = self.w.layer(lw, "attn_output.weight")?;
                apply_group_multi(&mut [(Mat::of(&two), &mut out[..])], attq.view(), n);
                let ob = &self.layers[l].ob;
                for t in 0..n {
                    for i in 0..dm {
                        x[t * dm + i] += out[t * dm + i] + ob[i];
                    }
                }
            }
            lap.lap(3);
            // 2. the routed experts
            norm(&mut xn, &x, &self.layers[l].post_norm);
            let ly = &self.layers[l];
            // the router in f32: logits, the top n_used (an equal logit goes to the lower id), their softmax
            let mut choice: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
            {
                let cp = SyncPtr::new(&mut choice);
                let (rt, rb, xn) = (&ly.router, &ly.router_b, &xn);
                each(n, &|t| {
                    let xt = &xn[t * dm..(t + 1) * dm];
                    let mut lg: Vec<(usize, f32)> = (0..h.n_expert)
                        .map(|e| {
                            let r = &rt[e * dm..(e + 1) * dm];
                            let mut d = 0.0f32;
                            for (a, b) in r.iter().zip(xt) {
                                d = a.mul_add(*b, d);
                            }
                            (e, d + rb[e])
                        })
                        .collect();
                    lg.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    lg.truncate(h.n_used);
                    let mx = lg[0].1;
                    let sum: f32 = lg.iter().map(|&(_, v)| (v - mx).exp()).sum();
                    // SAFETY: item t owns choice[t]
                    unsafe { cp.slice(t, 1)[0] = lg.iter().map(|&(e, v)| (e, (v - mx).exp() / sum)).collect() };
                });
            }
            lap.lap(4);
            // each expert used in the block: its tokens' inputs, gate and up, the clamped SwiGLU, down, then each
            // token's share added in token order
            let mut ys: Vec<Vec<(usize, f32, Vec<f32>)>> = vec![Vec::new(); n];
            let mut users: Vec<Vec<usize>> = vec![Vec::new(); h.n_expert];
            for (t, c) in choice.iter().enumerate() {
                for &(e, _) in c {
                    users[e].push(t);
                }
            }
            let (ff, ffp) = (h.d_ff, h.d_ff_pad);
            for (e, toks) in users.iter().enumerate() {
                if toks.is_empty() {
                    continue;
                }
                let m = toks.len();
                let mut xe = vec![0.0f32; m * dm];
                for (i, &t) in toks.iter().enumerate() {
                    xe[i * dm..(i + 1) * dm].copy_from_slice(&xn[t * dm..(t + 1) * dm]);
                }
                let xeq = quant_padded(&xe, m, dm, dp);
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
                lap.lap(5);
                let (gb, ub) = (&ly.gate_b[e * ff..(e + 1) * ff], &ly.up_b[e * ff..(e + 1) * ff]);
                // the clamped SwiGLU, a token per item, straight into that token's quantised row (zero beyond ff)
                let nbf = ffp / QK;
                let mut hq = Q8::new(m * nbf);
                {
                    let qp = crate::pool::Q8Ptr::new(hq.view_mut());
                    let (g, u) = (&g, &u);
                    each(m, &|i| {
                        let mut row = vec![0.0f32; ffp];
                        for j in 0..ff {
                            let gt = (g[i * ffp + j] + gb[j]).min(7.0);
                            let ut = (u[i * ffp + j] + ub[j]).clamp(-7.0, 7.0);
                            row[j] = gt / (1.0 + (-1.702 * gt).exp()) * (ut + 1.0);
                        }
                        // SAFETY: item i owns token i's blocks of hq
                        kops::q8k_quantize(&row, unsafe { qp.slice(i * nbf, nbf) });
                    });
                }
                lap.lap(6);
                let mut y = vec![0.0f32; m * dm];
                apply_group_multi(&mut [(piece(&pack.down), &mut y[..])], hq.view(), m);
                lap.lap(7);
                let db = &ly.down_b[e * dm..(e + 1) * dm];
                for (i, &t) in toks.iter().enumerate() {
                    let wgt = choice[t].iter().find(|c| c.0 == e).map_or(0.0, |c| c.1);
                    let mut yt = y[i * dm..(i + 1) * dm].to_vec();
                    for (a, b) in yt.iter_mut().zip(db) {
                        *a += b;
                    }
                    ys[t].push((e, wgt, yt));
                }
            }
            for (t, list) in ys.iter_mut().enumerate() {
                // in the order the router ranked them, so a token's sum does not depend on the block
                let order: Vec<usize> = choice[t].iter().map(|c| c.0).collect();
                list.sort_by_key(|(e, _, _)| order.iter().position(|o| o == e));
                for (_, wgt, yt) in list.iter() {
                    for i in 0..dm {
                        x[t * dm + i] = wgt.mul_add(yt[i], x[t * dm + i]);
                    }
                }
            }
            lap.lap(8);
        }
        // 3. the output: the final norm and the head
        let first = if all { 0 } else { n - 1 };
        let m = n - first;
        let mut xo = vec![0.0f32; m * dm];
        for t in first..n {
            rms_norm(&mut xo[(t - first) * dm..(t - first + 1) * dm], &x[t * dm..(t + 1) * dm], &self.out_norm, h.eps);
        }
        let xoq = quant_padded(&xo, m, dm, dp);
        if logits.len() < m * nv {
            return err("logits buffer too small");
        }
        let head = self.w.need("output.weight")?;
        apply_group_multi(&mut [(Mat::of(&head), &mut logits[..m * nv])], xoq.view(), m);
        lap.lap(0);
        Ok(())
    }
}
