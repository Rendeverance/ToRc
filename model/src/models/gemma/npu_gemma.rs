//! npu_gemma.rs - Gemma 4 run whole on the NPU (TORC_NPU=pure).
//!
//! Every layer is one ggml graph on GenieX's HTP backend (kernels::ggml), with the weights and the KV cache resident
//! there. The graph follows llama.cpp's gemma4 builder (MIT, ggml-org: src/models/gemma4.cpp, and build_moe_ffn in
//! src/llama-graph.cpp) op for op, so the HTP runs the same kernels it runs for GenieX. A graph is built once per
//! (block size, cache spans, rows out) and replayed: each call only sets its inputs (the embedded tokens, positions,
//! cache rows and masks) and reads the logits.
//!
//! The layout on the NPU is set by its two limits: one call (an "op batch") maps at most about 2.9 GB of buffers, and a
//! session maps about 16 GB in all. So every layer's weights are their own buffer (a call then holds about 6 layers;
//! buffers of about 1 GB split a token into 19 calls), every layer's cache is too, and a sliding-window layer's cache
//! is sized to its window plus one block, with positions wrapping round in it, rather than to the context.
//!
//! It is not bit-identical to the CPU path (the HTP uses f16/f32 arithmetic), so it is checked against that path by
//! tolerance.

use crate::gemma::GemmaHp;
use crate::weights::Weights;
use format::meta::Meta;
use format::{BlockKind, Result, err};
use kernels::ggml::{self, Ctx, Ggml, T};
use std::collections::HashMap;

/// One layer's tensors on the NPU.
struct NLayer {
    swa: bool,
    attn_norm: T,
    wq: T,
    wk: T,
    wv: Option<T>,
    q_norm: T,
    k_norm: T,
    wo: T,
    post_attn: T,
    ffn_norm: T,
    gate: T,
    up: T,
    down: T,
    post_ffw_1: T,
    pre_ffw_2: T,
    router: T,
    gate_inp_scale: T,
    gate_up_exps: T,
    down_exps: T,
    expert_scale: T,
    post_ffw_2: T,
    post_ffw: T,
    layer_scale: f32,
    kc: T,
    vc: T,
}

/// A built graph, with its inputs and output.
struct Built {
    ctx: Ctx,
    graph: ggml::Graph,
    alloc: ggml::Alloc,
    x: T,
    pos: T,
    kidx_full: T,
    kidx_swa: T,
    mask_full: T,
    mask_swa: T,
    oid: T,
    logits: T,
    /// the output rows after the final norm (Gemma::hid, the drafter's h)
    hid: T,
}

/// A metadata context, and the NPU buffer behind its tensors.
struct Part {
    ctx: Ctx,
    buf: ggml::Buffer,
}

pub struct GemmaNpu {
    g: Ggml,
    h: GemmaHp,
    layers: Vec<NLayer>,
    emb: T,
    out_norm: T,
    rope_ff: Option<T>,
    n_expert: usize,
    n_used: usize,
    ff_exp: usize,
    n_ctx: usize,
    /// the rows in a sliding-window layer's ring: the window plus the longest block
    ring: usize,
    block: usize,
    parts: Vec<Part>,
    graphs: HashMap<(usize, usize, usize, usize, bool), Built>,
    /// the soft cap on the logits (Gemma::cap_logits)
    pub cap: bool,
    /// the last run's hidden rows after the final norm, one row (d_model) per row of logits
    pub hid: Vec<f32>,
}

const SPAN: usize = 256;
const GRAPH_NODES: usize = 8192;

fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap()
}

/// TORC_NPU_BLOCK: the longest block one graph takes (default 512).
fn npu_block() -> usize {
    crate::switches::num("TORC_NPU_BLOCK").unwrap_or(512).max(1)
}

impl GemmaNpu {
    /// Uploads the model in `w` (with hyperparameters h) to the NPU, with a KV cache of n_ctx positions.
    pub fn load(w: &Weights, h: &GemmaHp, n_ctx: usize) -> Result<GemmaNpu> {
        let g = Ggml::open().map_err(format::Error)?;
        let t0 = std::time::Instant::now();
        let meta = Meta::parse(&w.file.metadata)?;
        let n_used =
            meta.arch_u32("expert_used_count").ok_or_else(|| format::Error("no expert_used_count".into()))? as usize;
        let block = npu_block();
        let ring = (h.window as usize + block).min(n_ctx);
        let f = &g.f;
        let mut parts: Vec<Part> = Vec::new();
        let (mut wbytes, mut kvbytes) = (0usize, 0usize);
        // a part: tensors made by `make` in a fresh context, one buffer for them, then `fill` sets their data
        let mut part = |n: usize, weights: bool, make: &mut dyn FnMut(Ctx) -> Result<()>| -> Result<ggml::Buffer> {
            let ctx = g.meta_ctx(n, false);
            make(ctx)?;
            // SAFETY: ctx holds only metadata tensors, and the buffer backs them all
            let buf = unsafe { (f.ggml_backend_alloc_ctx_tensors_from_buft)(ctx, g.buft) };
            if buf.is_null() {
                return err("NPU: buffer allocation failed");
            }
            unsafe {
                if weights {
                    // before the copies, so the weights are repacked for the HTP as they go in
                    (f.ggml_backend_buffer_set_usage)(buf, ggml::USAGE_WEIGHTS);
                } else {
                    // the cache is zeroed, because a masked row is still read and must not hold a NaN
                    (f.ggml_backend_buffer_clear)(buf, 0);
                }
            }
            parts.push(Part { ctx, buf });
            Ok(buf)
        };
        let set = |t: T, d: &[u8]| -> Result<usize> {
            // SAFETY: t was allocated, and its size is checked against the data
            unsafe {
                if (f.ggml_nbytes)(t) != d.len() {
                    return err("NPU: a tensor's size does not match its data");
                }
                (f.ggml_backend_tensor_set)(t, d.as_ptr().cast(), 0, d.len());
            }
            Ok(d.len())
        };
        // a matrix or vector from the file, as it is (in ggml's types)
        let mat = |ctx: Ctx, name: &str, list: &mut Vec<(T, String)>| -> Result<T> {
            let t = w.need(name)?;
            let (cols, rows) = t.shape();
            // SAFETY: ctx has room
            let x = unsafe { (f.ggml_new_tensor_2d)(ctx, t.t.ty as i32, cols as i64, rows as i64) };
            list.push((x, name.to_string()));
            Ok(x)
        };
        // the embedding (the tied head), the final norm, and rope_freqs
        let (mut emb, mut out_norm, mut rope_ff) = (std::ptr::null_mut(), std::ptr::null_mut(), None);
        let mut list: Vec<(T, String)> = Vec::new();
        // the head (the tied embedding, Q6_K) requantised for the NPU: TORC_NPU_HEAD=q4_0 (the default) or q8_0; file
        // keeps it as it is
        let head_ty = match crate::switches::get("TORC_NPU_HEAD").as_deref() {
            Some("q8_0") => Some(ggml::Q8_0),
            Some("file") => None,
            _ => Some(ggml::Q4_0),
        };
        part(8, true, &mut |c| {
            emb = match head_ty {
                Some(ty) => {
                    let t = w.need("token_embd.weight")?;
                    let (cols, rows) = t.shape();
                    // SAFETY: c has room
                    unsafe { (f.ggml_new_tensor_2d)(c, ty, cols as i64, rows as i64) }
                }
                None => mat(c, "token_embd.weight", &mut list)?,
            };
            out_norm = mat(c, "output_norm.weight", &mut list)?;
            if w.tensor("rope_freqs.weight").is_some() {
                rope_ff = Some(mat(c, "rope_freqs.weight", &mut list)?);
            }
            Ok(())
        })?;
        for (t, d) in list.drain(..) {
            wbytes += set(t, w.need(&d)?.data)?;
        }
        if let Some(ty) = head_ty {
            let t = w.need("token_embd.weight")?;
            let q = crate::gemma::requant(t.t.ty, t.data, t.shape().0, t.shape().1, ty as u32);
            wbytes += set(emb, &q)?;
        }
        let mut layers: Vec<NLayer> = Vec::with_capacity(h.n_layer);
        let (mut n_expert, mut ff_exp) = (0, 0);
        let (mut gu, mut dn) = (Vec::new(), Vec::new());
        // TORC_NPU_GROUP: how many layers a weight buffer (and a cache buffer) holds (default 2)
        let per: usize = crate::switches::num("TORC_NPU_GROUP").unwrap_or(2).max(1);
        let group_of = |l: usize| -> Result<u32> {
            w.file
                .blocks
                .iter()
                .find(|b| b.layer == l as u32 && b.kind == BlockKind::Moe && b.experts.is_some())
                .and_then(|b| b.experts)
                .ok_or_else(|| format::Error(format!("layer {l}: no routed experts")))
        };
        for l0 in (0..h.n_layer).step_by(per) {
            let l1 = (l0 + per).min(h.n_layer);
            // the group's tensors (the expert stacks shaped from expert 0), with one buffer for them
            let mut made: Vec<(NLayer, u32)> = Vec::new();
            part(32 * (l1 - l0), true, &mut |c| {
                for l in l0..l1 {
                    let name = |s: &str| format!("blk.{l}.{s}");
                    let group = group_of(l)?;
                    n_expert = w.layer_experts(group).n_expert as usize;
                    let (_, lay) = w.expert(group, 0)?;
                    let (mg, mu, md) = (&lay.gate, &lay.up, &lay.down);
                    if mg.ty != mu.ty || mg.rows != mu.rows || md.cols != mg.rows {
                        return err(format!("layer {l}: experts gate/up/down do not stack"));
                    }
                    let (ff, dm) = (mg.rows as usize, mg.cols as usize);
                    ff_exp = ff;
                    // the experts as llama.cpp holds them: gate and up stacked per expert ([d_model, 2 ff, n_expert]),
                    // and down as [ff, d_model, n_expert] SAFETY: c has room
                    let (tgu, tdn) = unsafe {
                        (
                            (f.ggml_new_tensor_3d)(c, mg.ty as i32, dm as i64, 2 * ff as i64, n_expert as i64),
                            (f.ggml_new_tensor_3d)(c, md.ty as i32, ff as i64, dm as i64, n_expert as i64),
                        )
                    };
                    let os = match w.tensor(&name("layer_output_scale.weight")) {
                        Some(t) if t.t.ty == 0 => t.f32s()?.first().copied().unwrap_or(1.0),
                        _ => 1.0,
                    };
                    let mut m = |s: &str| mat(c, &name(s), &mut list);
                    let ly = NLayer {
                        swa: h.swa_layers[l],
                        attn_norm: m("attn_norm.weight")?,
                        wq: m("attn_q.weight")?,
                        wk: m("attn_k.weight")?,
                        wv: if w.tensor(&name("attn_v.weight")).is_some() { Some(m("attn_v.weight")?) } else { None },
                        q_norm: m("attn_q_norm.weight")?,
                        k_norm: m("attn_k_norm.weight")?,
                        wo: m("attn_output.weight")?,
                        post_attn: m("post_attention_norm.weight")?,
                        ffn_norm: m("ffn_norm.weight")?,
                        gate: m("ffn_gate.weight")?,
                        up: m("ffn_up.weight")?,
                        down: m("ffn_down.weight")?,
                        post_ffw_1: m("post_ffw_norm_1.weight")?,
                        pre_ffw_2: m("pre_ffw_norm_2.weight")?,
                        router: m("ffn_gate_inp.weight")?,
                        gate_inp_scale: m("ffn_gate_inp.scale")?,
                        gate_up_exps: tgu,
                        down_exps: tdn,
                        expert_scale: m("ffn_down_exps.scale")?,
                        post_ffw_2: m("post_ffw_norm_2.weight")?,
                        post_ffw: m("post_ffw_norm.weight")?,
                        layer_scale: os,
                        kc: std::ptr::null_mut(),
                        vc: std::ptr::null_mut(),
                    };
                    made.push((ly, group));
                }
                Ok(())
            })?;
            for (t, d) in list.drain(..) {
                wbytes += set(t, w.need(&d)?.data)?;
            }
            // the stacks, a layer at a time, from the per-expert slots
            for (ly, group) in &made {
                gu.clear();
                dn.clear();
                for e in 0..n_expert as u32 {
                    let (slot, lay) = w.expert(*group, e)?;
                    for x in [&lay.gate, &lay.up] {
                        gu.extend_from_slice(x.of(slot));
                    }
                    dn.extend_from_slice(lay.down.of(slot));
                }
                wbytes += set(ly.gate_up_exps, &gu)?;
                wbytes += set(ly.down_exps, &dn)?;
            }
            // the group's caches: the whole context, or a sliding-window layer's ring
            part(2 * (l1 - l0), false, &mut |c| {
                for (ly, _) in made.iter_mut() {
                    let k = if ly.swa { &h.swa } else { &h.full };
                    let (kvd, rows) = ((k.n_head_kv * k.head_dim) as i64, if ly.swa { ring } else { n_ctx } as i64);
                    // SAFETY: c has room
                    unsafe {
                        ly.kc = (f.ggml_new_tensor_2d)(c, ggml::F16, kvd, rows);
                        ly.vc = (f.ggml_new_tensor_2d)(c, ggml::F16, kvd, rows);
                    }
                    kvbytes += 4 * (kvd * rows) as usize;
                }
                Ok(())
            })?;
            layers.extend(made.into_iter().map(|(ly, _)| ly));
        }
        drop((gu, dn));
        eprintln!(
            "npu (pure): {:.0} MiB of weights and {:.0} MiB of cache on the NPU in {:.1} s ({} experts, {} used, \
            sliding-window ring {ring})",
            wbytes as f64 / 1048576.0,
            kvbytes as f64 / 1048576.0,
            t0.elapsed().as_secs_f64(),
            n_expert,
            n_used
        );
        Ok(GemmaNpu {
            g,
            h: h.clone(),
            layers,
            emb,
            out_norm,
            rope_ff,
            n_expert,
            n_used,
            ff_exp,
            n_ctx,
            ring,
            block,
            parts,
            graphs: HashMap::new(),
            cap: true,
            hid: Vec::new(),
        })
    }

    /// The graph for n tokens attending to the first nkf rows of the full layers' caches and the first nks rows of the
    /// sliding-window rings, with m rows of logits out.
    fn build(&self, n: usize, nkf: usize, nks: usize, m: usize, cap: bool) -> Result<Built> {
        let f = &self.g.f;
        let h = &self.h;
        let ctx = self.g.meta_ctx(GRAPH_NODES, true);
        let (dm, eps) = (h.d_model as i64, h.eps);
        let n_ = n as i64;
        // SAFETY: every call builds metadata in ctx from tensors of this model and graph
        unsafe {
            let gf = (f.ggml_new_graph_custom)(ctx, GRAPH_NODES, false);
            let input = |t: T, name: &str| {
                (f.ggml_set_input)(t);
                (f.ggml_set_name)(t, cstr(name).as_ptr());
                t
            };
            let x0 = input((f.ggml_new_tensor_2d)(ctx, ggml::F32, dm, n_), "x");
            let pos = input((f.ggml_new_tensor_1d)(ctx, ggml::I32, n_), "pos");
            let kidx_full = input((f.ggml_new_tensor_1d)(ctx, ggml::I64, n_), "kidx_full");
            let kidx_swa = input((f.ggml_new_tensor_1d)(ctx, ggml::I64, n_), "kidx_swa");
            let mask_full = input((f.ggml_new_tensor_2d)(ctx, ggml::F16, nkf as i64, n_), "mask_full");
            let mask_swa = input((f.ggml_new_tensor_2d)(ctx, ggml::F16, nks as i64, n_), "mask_swa");
            let oid = input((f.ggml_new_tensor_1d)(ctx, ggml::I32, m as i64), "oid");
            let norm_w = |t: T, w: T| (f.ggml_mul)(ctx, (f.ggml_rms_norm)(ctx, t, eps), w);
            let (ne, k_used, ff) = (self.n_expert as i64, self.n_used as i64, self.ff_exp as i64);
            let mut x = x0;
            for ly in self.layers.iter() {
                let kind = if ly.swa { &h.swa } else { &h.full };
                let (hd, nkv, nh) = (kind.head_dim as i64, kind.n_head_kv as i64, h.n_head as i64);
                let (kvd, qd) = (hd * nkv, hd * nh);
                let ffac = if ly.swa { std::ptr::null_mut() } else { self.rope_ff.unwrap_or(std::ptr::null_mut()) };
                let rope = |t: T| {
                    (f.ggml_rope_ext)(
                        ctx,
                        t,
                        pos,
                        ffac,
                        kind.n_rot as i32,
                        ggml::ROPE_NEOX,
                        0,
                        kind.base,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    )
                };
                // attention
                let a = norm_w(x, ly.attn_norm);
                let q = (f.ggml_mul_mat)(ctx, ly.wq, a);
                let q = rope(norm_w((f.ggml_reshape_3d)(ctx, q, hd, nh, n_), ly.q_norm));
                let kr = (f.ggml_mul_mat)(ctx, ly.wk, a);
                // the full layers have no attn_v: V is K, taken before K's norm
                let vr = match ly.wv {
                    Some(wv) => (f.ggml_mul_mat)(ctx, wv, a),
                    None => kr,
                };
                let k = rope(norm_w((f.ggml_reshape_3d)(ctx, kr, hd, nkv, n_), ly.k_norm));
                let v = (f.ggml_rms_norm)(ctx, (f.ggml_reshape_3d)(ctx, vr, hd, nkv, n_), eps);
                let kidx = if ly.swa { kidx_swa } else { kidx_full };
                let ks = (f.ggml_set_rows)(ctx, ly.kc, (f.ggml_reshape_2d)(ctx, k, kvd, n_), kidx);
                let vs = (f.ggml_set_rows)(ctx, ly.vc, (f.ggml_reshape_2d)(ctx, v, kvd, n_), kidx);
                (f.ggml_build_forward_expand)(gf, ks);
                (f.ggml_build_forward_expand)(gf, vs);
                let span = if ly.swa { nks } else { nkf } as i64;
                let view = |c: T| {
                    let t = (f.ggml_view_3d)(ctx, c, hd, nkv, span, (hd * 2) as usize, (kvd * 2) as usize, 0);
                    (f.ggml_permute)(ctx, t, 0, 2, 1, 3)
                };
                let qp = (f.ggml_permute)(ctx, q, 0, 2, 1, 3);
                let mask = if ly.swa { mask_swa } else { mask_full };
                // scale 1.0: the learned q and k norms already set the size of the scores
                let fa = (f.ggml_flash_attn_ext)(ctx, qp, view(ly.kc), view(ly.vc), mask, 1.0, 0.0, 0.0);
                (f.ggml_prec_set_acc)(fa, ggml::PREC_F32);
                let o = (f.ggml_mul_mat)(ctx, ly.wo, (f.ggml_reshape_2d)(ctx, fa, qd, n_));
                x = (f.ggml_add)(ctx, norm_w(o, ly.post_attn), x);
                let resid = x;
                // the dense branch
                let h1 = norm_w(x, ly.ffn_norm);
                let act =
                    (f.ggml_geglu_split)(ctx, (f.ggml_mul_mat)(ctx, ly.gate, h1), (f.ggml_mul_mat)(ctx, ly.up, h1));
                let br = norm_w((f.ggml_mul_mat)(ctx, ly.down, act), ly.post_ffw_1);
                // router input: weightless RMS norm of the residual, times 1 / sqrt(d_model) and ffn_gate_inp.scale
                let r = (f.ggml_scale)(ctx, (f.ggml_rms_norm)(ctx, x, eps), 1.0 / (h.d_model as f32).sqrt());
                let lg = (f.ggml_mul_mat)(ctx, ly.router, (f.ggml_mul)(ctx, r, ly.gate_inp_scale));
                let probs = (f.ggml_soft_max)(ctx, lg);
                let sel = (f.ggml_argsort_top_k)(ctx, probs, k_used as i32);
                let wts = (f.ggml_get_rows)(ctx, (f.ggml_reshape_3d)(ctx, probs, 1, ne, n_), sel);
                let w2 = (f.ggml_reshape_2d)(ctx, wts, k_used, n_);
                let w2 = (f.ggml_div)(ctx, w2, (f.ggml_sum_rows)(ctx, w2));
                // ffn_down_exps.scale: multiplying the routing weight by it scales that expert's output the same way
                let es = (f.ggml_repeat_4d)(ctx, (f.ggml_reshape_3d)(ctx, ly.expert_scale, 1, ne, 1), 1, ne, n_, 1);
                let wts =
                    (f.ggml_mul)(ctx, (f.ggml_reshape_3d)(ctx, w2, 1, k_used, n_), (f.ggml_get_rows)(ctx, es, sel));
                // the experts
                let h2 = (f.ggml_reshape_3d)(ctx, norm_w(x, ly.pre_ffw_2), dm, 1, n_);
                let gu = (f.ggml_mul_mat_id)(ctx, ly.gate_up_exps, h2, sel);
                // the gate and up halves of gu as 3-D strided views (as llama.cpp does), then GELU(gate) * up
                let (nb1, nb2) = ((2 * ff * 4) as usize, (2 * ff * k_used * 4) as usize);
                let gv = (f.ggml_view_3d)(ctx, gu, ff, k_used, n_, nb1, nb2, 0);
                let uv = (f.ggml_view_3d)(ctx, gu, ff, k_used, n_, nb1, nb2, (ff * 4) as usize);
                let act = (f.ggml_geglu_split)(ctx, gv, uv);
                let ex = (f.ggml_mul_mat_id)(ctx, ly.down_exps, act, sel);
                // each expert's output times its weight, as the 2-D broadcast [dm, k n] x [1, k n]. The HTP's 3-D
                // broadcast MUL of [dm, k, n] by [1, k, n] (llama.cpp's form) gives wrong results for tokens 3.. of a
                // block on the HTP.
                let e2 = (f.ggml_reshape_2d)(ctx, ex, dm, k_used * n_);
                let w2 = (f.ggml_reshape_2d)(ctx, wts, 1, k_used * n_);
                let ex = (f.ggml_reshape_3d)(ctx, (f.ggml_mul)(ctx, e2, w2), dm, k_used, n_);
                (f.ggml_build_forward_expand)(gf, ex);
                let one = |i: i64| (f.ggml_view_2d)(ctx, ex, dm, n_, (dm * k_used * 4) as usize, (i * dm * 4) as usize);
                let mut moe = one(0);
                for i in 1..k_used {
                    moe = (f.ggml_add)(ctx, moe, one(i));
                }
                let y = norm_w((f.ggml_add)(ctx, br, norm_w(moe, ly.post_ffw_2)), ly.post_ffw);
                x = (f.ggml_add)(ctx, y, resid);
                if ly.layer_scale != 1.0 {
                    x = (f.ggml_scale)(ctx, x, ly.layer_scale);
                }
            }
            // the output: the rows asked for, the final norm, the tied head and the soft cap
            let xl = norm_w((f.ggml_get_rows)(ctx, x, oid), self.out_norm);
            (f.ggml_set_output)(xl);
            let mut lo = (f.ggml_mul_mat)(ctx, self.emb, xl);
            if h.softcap > 0.0 && cap {
                lo = (f.ggml_scale)(ctx, (f.ggml_tanh)(ctx, (f.ggml_scale)(ctx, lo, 1.0 / h.softcap)), h.softcap);
            }
            (f.ggml_set_output)(lo);
            (f.ggml_build_forward_expand)(gf, lo);
            let alloc = (f.ggml_gallocr_new)(self.g.buft);
            if alloc.is_null() || !(f.ggml_gallocr_alloc_graph)(alloc, gf) {
                return err("NPU: graph allocation failed");
            }
            if crate::switches::get("TORC_NPU_VERBOSE").is_some_and(|v| v != "0") {
                eprintln!("npu (pure): graph n={n} spans {nkf}/{nks} m={m}: {} nodes", (f.ggml_graph_n_nodes)(gf));
            }
            Ok(Built {
                ctx,
                graph: gf,
                alloc,
                x: x0,
                pos,
                kidx_full,
                kidx_swa,
                mask_full,
                mask_swa,
                oid,
                logits: lo,
                hid: xl,
            })
        }
    }

    /// n tokens at positions pos0.., whose embeddings (scaled by sqrt(d_model)) are x (n x d_model): writes the logits
    /// of the last one (or of all n when `all` is set) into logits. Blocks longer than TORC_NPU_BLOCK (default 512) go
    /// a block at a time.
    pub fn forward(&mut self, x: &[f32], n: usize, pos0: usize, logits: &mut [f32], all: bool) -> Result<()> {
        let (dm, nv) = (self.h.d_model, self.h.n_vocab);
        let mut t0 = 0;
        while t0 < n {
            let b = self.block.min(n - t0);
            let last = t0 + b == n;
            let m = if all { b } else { 1 };
            let out: &mut [f32] = if all {
                &mut logits[t0 * nv..(t0 + b) * nv]
            } else if last {
                &mut logits[..nv]
            } else {
                &mut []
            };
            self.run(&x[t0 * dm..(t0 + b) * dm], b, pos0 + t0, out, m)?;
            t0 += b;
        }
        Ok(())
    }

    fn run(&mut self, x: &[f32], n: usize, pos0: usize, out: &mut [f32], m: usize) -> Result<()> {
        let end = pos0 + n;
        if end > self.n_ctx {
            return err("position past the context");
        }
        let (r, win) = (self.ring, self.h.window as usize);
        let nkf = (end.div_ceil(SPAN) * SPAN).min(self.n_ctx);
        let nks = (end.div_ceil(SPAN) * SPAN).min(r);
        let key = (n, nkf, nks, m, self.cap);
        if !self.graphs.contains_key(&key) {
            let b = self.build(n, nkf, nks, m, self.cap)?;
            self.graphs.insert(key, b);
        }
        let b = &self.graphs[&key];
        let f = &self.g.f;
        let pos: Vec<i32> = (pos0..end).map(|p| p as i32).collect();
        let kf: Vec<i64> = (pos0..end).map(|p| p as i64).collect();
        let ks: Vec<i64> = (pos0..end).map(|p| (p % r) as i64).collect();
        // the masks (f16): 0 where token t may see the row, -inf elsewhere. For the full layers that is rows 0..=p. In
        // the rings, slot j holds the newest position q < end with q = j mod r, which is seen when q <= p and p - q <
        // window (r >= window + n, so a block never overwrites a position one of its own tokens still needs)
        let (zero, ninf) = (0u16, 0xFC00u16);
        let mut mf = vec![ninf; n * nkf];
        let mut ms = vec![ninf; n * nks];
        for t in 0..n {
            let p = pos0 + t;
            mf[t * nkf..t * nkf + p + 1].fill(zero);
            for j in 0..nks {
                let back = (end - 1 + r - j) % r;
                if back <= end - 1 {
                    let q = end - 1 - back;
                    if q <= p && p - q < win {
                        ms[t * nks + j] = zero;
                    }
                }
            }
        }
        let oid: Vec<i32> = if m == n { (0..n as i32).collect() } else { vec![n as i32 - 1] };
        // SAFETY: the inputs were allocated with these sizes, and the graph and its buffers live in b
        unsafe {
            let set = |t: T, p: *const u8, len: usize| (f.ggml_backend_tensor_set)(t, p.cast(), 0, len);
            set(b.x, x.as_ptr().cast(), x.len() * 4);
            set(b.pos, pos.as_ptr().cast(), pos.len() * 4);
            set(b.kidx_full, kf.as_ptr().cast(), kf.len() * 8);
            set(b.kidx_swa, ks.as_ptr().cast(), ks.len() * 8);
            set(b.mask_full, mf.as_ptr().cast(), mf.len() * 2);
            set(b.mask_swa, ms.as_ptr().cast(), ms.len() * 2);
            set(b.oid, oid.as_ptr().cast(), oid.len() * 4);
            if (f.ggml_backend_graph_compute)(self.g.be, b.graph) != 0 {
                return err("NPU: graph compute failed");
            }
            if !out.is_empty() {
                let nv = self.h.n_vocab;
                if out.len() < m * nv {
                    return err("logits buffer too small");
                }
                (f.ggml_backend_tensor_get)(b.logits, out.as_mut_ptr().cast(), 0, m * nv * 4);
                let dm = self.h.d_model;
                self.hid.resize(m * dm, 0.0);
                (f.ggml_backend_tensor_get)(b.hid, self.hid.as_mut_ptr().cast(), 0, m * dm * 4);
            }
        }
        Ok(())
    }

    /// Layer l's keys and values for positions pos0..pos0 + n, as f32 (n x n_head_kv x head_dim each), read back from
    /// its cache on the NPU (a ring's rows at position mod ring): what the CPU's drafter attends to (gemma_mtp.rs).
    pub fn kv_rows(&self, l: usize, pos0: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let ly = &self.layers[l];
        let kind = if ly.swa { &self.h.swa } else { &self.h.full };
        let kvd = kind.n_head_kv * kind.head_dim;
        let f = &self.g.f;
        let mut raw = vec![0u16; n * kvd];
        let mut out = [vec![0.0f32; n * kvd], vec![0.0f32; n * kvd]];
        for (t, o) in [ly.kc, ly.vc].into_iter().zip(out.iter_mut()) {
            for i in 0..n {
                let row = if ly.swa { (pos0 + i) % self.ring } else { pos0 + i };
                // SAFETY: row is below the tensor's rows (ring or context), with kvd f16 values per row
                unsafe {
                    (f.ggml_backend_tensor_get)(t, raw[i * kvd..].as_mut_ptr().cast(), row * kvd * 2, kvd * 2);
                }
            }
            for (d, &b) in o.iter_mut().zip(&raw) {
                *d = f16_f32(b);
            }
        }
        let [k, v] = out;
        (k, v)
    }
}

/// An IEEE half's value.
fn f16_f32(b: u16) -> f32 {
    let (s, e, m) = ((b >> 15) as u32, ((b >> 10) & 0x1f) as u32, (b & 0x3ff) as u32);
    let bits = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => {
            // subnormal: normalise it
            let mut e = 113u32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, _) => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

impl Drop for GemmaNpu {
    fn drop(&mut self) {
        let f = &self.g.f;
        // SAFETY: everything here was made by this model and is freed once
        unsafe {
            for (_, b) in self.graphs.drain() {
                (f.ggml_gallocr_free)(b.alloc);
                (f.ggml_free)(b.ctx);
            }
            for p in self.parts.drain(..) {
                (f.ggml_backend_buffer_free)(p.buf);
                (f.ggml_free)(p.ctx);
            }
        }
    }
}
