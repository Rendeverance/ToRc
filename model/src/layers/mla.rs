//! mla.rs - DeepSeek's multi-head latent attention (deepseek2: GLM-4.7-Flash, Moonlight, DeepSeek-V2-Lite), run the
//! way llama.cpp runs it ("absorbed"): every head attends over one shared cache row per position, the layer's
//! compressed latent (kv_lora_rank, RMS-normed) followed by its rotary key (n_rot). That is multi-query attention with
//! a head of kv_lora_rank + n_rot, so it goes through the ordinary attention kernel (fattn.rs) with one KV head.
//!
//! Per token: q = q_b(norm(q_a(x))) (or attn_q for a model without a q LoRA), split per head into a no-rope part and a
//! rope part; the no-rope part is taken into latent space through the head's attn_k_b, and the rope part rotated. The
//! cache row is kv_a_mqa(x): its latent part normed, its rope part rotated. After attention each head's output (the
//! first kv_lora_rank values) goes back out through its attn_v_b, and the heads together through attn_output.
//!
//! The rows are written to both K and V of the cache (V reads the same row; the rope tail of the output is unused).

use crate::hparams::Hparams;
use crate::moe::Mat;
use crate::ops::{rms_norm_inplace, rope_pairs};
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::{Result, err};
use kernels::{Q8s, QK, ops as kops};

/// The sizes of a deepseek2 attention layer (from the file's metadata).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaDims {
    /// the q LoRA's rank (0: no q LoRA, attn_q directly)
    pub q_lora: usize,
    /// the compressed latent's width (kv_lora_rank)
    pub kv_lora: usize,
    /// a head's key width without rope (key_length_mla - n_rot) and its value width (value_length_mla)
    pub nope: usize,
    pub vd: usize,
}

pub struct Mla {
    pub l: u32,
    pub d: MlaDims,
    pub q_a_norm: Vec<f32>,
    pub kv_a_norm: Vec<f32>,
    /// attn_k_b per head as f32: [head][latent row][nope], so a head's absorbed query is k_b[h] q_nope (empty when
    /// kb_q8 is used)
    pub k_b: Vec<f32>,
    /// attn_v_b per head as f32: [head][value row][latent] (empty when the file's Q8_0 rows are used)
    pub v_b: Vec<f32>,
    /// attn_k_b kept as Q8_0 with each row widened from nope to kpad (a multiple of 256) by zero blocks, so a head's
    /// absorption is one Q8 row product per latent dim over a quantised q_nope (empty: the f32 path)
    pub kb_q8: Vec<u8>,
    pub kpad: usize,
    /// attn_v_b is Q8_0 with whole 256-blocks per row: its rows are used from the file as they are
    pub vb_q8: bool,
}

const Q8_0: u32 = 8;
const Q8_0_BLOCK: usize = 34;

/// A whole matrix dequantised to f32, row by row.
fn f32_matrix(w: &Weights, name: &str) -> Result<(Vec<f32>, [u64; 4])> {
    let t = w.need(name)?;
    let dims = t.t.dims;
    let cols = dims[0] as usize;
    let rows = (dims[1] * dims[2].max(1)) as usize;
    let rb = t.data.len() / rows;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        kops::dequantize(t.t.ty, &t.data[r * rb..(r + 1) * rb], &mut out[r * cols..(r + 1) * cols]);
    }
    Ok((out, dims))
}

impl Mla {
    pub fn load(w: &Weights, h: &Hparams, l: u32) -> Result<Mla> {
        let Some(d) = h.mla else { return err("not an MLA model") };
        let nh = h.n_head as usize;
        let lay = |s: &str| format!("blk.{l}.{s}");
        let (tk, tv) = (w.need(&lay("attn_k_b.weight"))?, w.need(&lay("attn_v_b.weight"))?);
        let (kd, vdims) = (tk.t.dims, tv.t.dims);
        // attn_k_b: [nope, kv_lora, n_head] (a head's rows are its latent dims); attn_v_b: [kv_lora, vd, n_head]
        if kd[0] as usize != d.nope
            || kd[1] as usize != d.kv_lora
            || kd[2] as usize != nh
            || vdims[0] as usize != d.kv_lora
            || vdims[1] as usize != d.vd
            || vdims[2] as usize != nh
        {
            return err(format!("layer {l}: attn_k_b / attn_v_b shapes {kd:?} / {vdims:?} do not match the metadata"));
        }
        let q_a_norm = if d.q_lora > 0 { w.need(&lay("attn_q_a_norm.weight"))?.f32s()? } else { Vec::new() };
        // Q8_0 rows (GLM-4.7-Flash's): attn_k_b widened by zero blocks to whole 256s, attn_v_b used in place. Other
        // types go through f32 copies
        let kpad = d.nope.div_ceil(256) * 256;
        let (k_b, kb_q8) = if tk.t.ty == Q8_0 && d.nope % 32 == 0 {
            let (src_rb, dst_rb) = (d.nope / 32 * Q8_0_BLOCK, kpad / 32 * Q8_0_BLOCK);
            let rows = nh * d.kv_lora;
            let mut out = vec![0u8; rows * dst_rb];
            for r in 0..rows {
                out[r * dst_rb..r * dst_rb + src_rb].copy_from_slice(&tk.data[r * src_rb..(r + 1) * src_rb]);
            }
            (Vec::new(), out)
        } else {
            (f32_matrix(w, &lay("attn_k_b.weight"))?.0, Vec::new())
        };
        let vb_q8 = tv.t.ty == Q8_0 && d.kv_lora % 256 == 0;
        let v_b = if vb_q8 { Vec::new() } else { f32_matrix(w, &lay("attn_v_b.weight"))?.0 };
        Ok(Mla {
            l,
            d,
            q_a_norm,
            kv_a_norm: w.need(&lay("attn_kv_a_norm.weight"))?.f32s()?,
            k_b,
            v_b,
            kb_q8,
            kpad,
            vb_q8,
        })
    }
}

/// The layer for n tokens at positions pos0.. (n = 1 for a decoded token): the products of all n at once, each token's
/// absorbed query and cache row, attention over the cache, then each head back through attn_v_b and attn_output, added
/// to x.
#[allow(clippy::too_many_arguments)]
pub fn mla_block(
    w: &Weights,
    h: &Hparams,
    m: &Mla,
    eng: &crate::attn::AttnEngine,
    kv: &mut crate::attn::KvCache,
    pos0: u32,
    xq: Q8s,
    x: &mut [f32],
    n: usize,
) -> Result<()> {
    let d = m.d;
    let (nh, dm, nr) = (h.n_head as usize, h.d_model as usize, h.n_rot as usize);
    let hq = d.nope + nr;
    let lat = d.kv_lora + nr;
    if pos0 as usize + n > kv.cap {
        return err("position past the KV cache");
    }
    let lay = |s: &str| format!("blk.{}.{s}", m.l);
    let buf = crate::scratch::buf::<f32>;
    // the profile (TORC_PROF; torc-run prints attn::ATTN_NS): products, preparation, core, output as attn_block's
    let ns = |i: usize, t: std::time::Instant| {
        crate::attn::ATTN_NS[i].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    };
    let t0 = std::time::Instant::now();
    // q (n x nh hq) and the cache rows' source (n x lat)
    let mut q = buf("mla.q", n * nh * hq);
    let mut kvr = buf("mla.kv", n * lat);
    let tkv = w.need(&lay("attn_kv_a_mqa.weight"))?;
    if d.q_lora > 0 {
        let mut qa = buf("mla.qa", n * d.q_lora);
        let tqa = w.need(&lay("attn_q_a.weight"))?;
        crate::moe::apply_group_multi(&mut [(Mat::of(&tqa), &mut qa[..]), (Mat::of(&tkv), &mut kvr[..])], xq, n);
        let nb = d.q_lora / QK;
        let mut qaq = crate::scratch::q8("mla.qaq", n * nb);
        for t in 0..n {
            let r = &mut qa[t * d.q_lora..(t + 1) * d.q_lora];
            rms_norm_inplace(r, &m.q_a_norm, h.eps);
            kops::q8k_quantize(r, qaq.sub_mut(t * nb..(t + 1) * nb));
        }
        let tqb = w.need(&lay("attn_q_b.weight"))?;
        crate::moe::apply_group_multi(&mut [(Mat::of(&tqb), &mut q[..])], qaq.view(), n);
    } else {
        let tq = w.need(&lay("attn_q.weight"))?;
        crate::moe::apply_group_multi(&mut [(Mat::of(&tq), &mut q[..]), (Mat::of(&tkv), &mut kvr[..])], xq, n);
    }
    ns(0, t0);
    let t0 = std::time::Instant::now();
    // each token: its absorbed query (nh x lat: k_b q_nope, then the rotated q_pe) and its cache row
    let mut qa = buf("mla.qabs", n * nh * lat);
    let half = nr / 2;
    kv.gpu_lo = kv.gpu_lo.min(pos0 as usize);
    let (cap, wide) = (kv.cap, !kv.k16.is_empty());
    let (kp, vp) = (SyncPtr::new(&mut kv.k[..]), SyncPtr::new(&mut kv.v[..]));
    let (k16p, v16p) = (SyncPtr::new(&mut kv.k16[..]), SyncPtr::new(&mut kv.v16[..]));
    let (ksp, vsp) = (SyncPtr::new(&mut kv.k_scale[..]), SyncPtr::new(&mut kv.v_scale[..]));
    let (qap, kvp) = (SyncPtr::new(&mut qa[..]), SyncPtr::new(&mut kvr[..]));
    crate::pool::par_for(n * nh, &|i| {
        let (t, hh) = (i / nh, i % nh);
        let pos = pos0 + t as u32;
        let (mut cs, mut sn) = (vec![0.0f32; half], vec![0.0f32; half]);
        crate::ops::rope_row(&mut cs, &mut sn, pos, h.n_rot, h.rope_base);
        let qh = &q[(t * nh + hh) * hq..(t * nh + hh + 1) * hq];
        // SAFETY: item i owns token t's head hh of qa
        let o = unsafe { qap.slice((t * nh + hh) * lat, lat) };
        if m.kb_q8.is_empty() {
            let kb = &m.k_b[hh * d.kv_lora * d.nope..(hh + 1) * d.kv_lora * d.nope];
            for (c, ov) in o[..d.kv_lora].iter_mut().enumerate() {
                *ov = kops::dot_f32(&kb[c * d.nope..(c + 1) * d.nope], &qh[..d.nope]);
            }
        } else {
            // q_nope widened with zeros to kpad and quantised, then one Q8_0 row product per latent dim
            let mut qn = vec![0f32; m.kpad];
            qn[..d.nope].copy_from_slice(&qh[..d.nope]);
            let mut q8 = kernels::Q8::new(m.kpad / QK);
            kops::q8k_quantize(&qn, q8.view_mut());
            let rb = m.kpad / 32 * Q8_0_BLOCK;
            let kb = &m.kb_q8[hh * d.kv_lora * rb..(hh + 1) * d.kv_lora * rb];
            crate::moe::dot_rows(Q8_0, kb, rb, q8.view(), &mut o[..d.kv_lora]);
        }
        o[d.kv_lora..].copy_from_slice(&qh[d.nope..]);
        rope_pairs(&mut o[d.kv_lora..], nr, &cs, &sn);
        if hh == 0 {
            // SAFETY: token t's cache row (one KV head, position pos) and its kvr row belong to the item of its head 0
            unsafe {
                let r = kvp.slice(t * lat, lat);
                rms_norm_inplace(&mut r[..d.kv_lora], &m.kv_a_norm, h.eps);
                rope_pairs(&mut r[d.kv_lora..], nr, &cs, &sn);
                let p = pos as usize;
                if !wide {
                    let s = crate::attn::quant8(r, kp.slice(p * lat, lat));
                    *ksp.at(p) = s;
                    *vsp.at(p) = crate::attn::quant8(r, vp.slice(p * lat, lat));
                } else {
                    *ksp.at(p) = crate::attn::quant16(r, k16p.slice(p * lat, lat));
                    *vsp.at(p) = crate::attn::quant16(r, v16p.slice(p * lat, lat));
                }
            }
        }
    });
    let _ = cap;
    ns(4, t0);
    let t0 = std::time::Instant::now();
    // attention: nh query heads of width lat over one KV head
    let mut att = buf("mla.att", n * nh * lat);
    eng.run_block_tiered(kv, &qa, nh * lat, n, pos0, &mut att);
    ns(1, t0);
    let t0 = std::time::Instant::now();
    // each head's latent output back out through its attn_v_b, all heads into one row of nh x vd
    let vw = nh * d.vd;
    let mut hv = buf("mla.hv", n * vw);
    let hp = SyncPtr::new(&mut hv[..]);
    let tv = w.need(&lay("attn_v_b.weight"))?;
    let vrb = d.kv_lora / 32 * Q8_0_BLOCK;
    crate::pool::par_for(n * nh, &|i| {
        let (t, hh) = (i / nh, i % nh);
        let a = &att[(t * nh + hh) * lat..(t * nh + hh) * lat + d.kv_lora];
        // SAFETY: item i owns token t's head hh of hv
        let o = unsafe { hp.slice(t * vw + hh * d.vd, d.vd) };
        if m.vb_q8 {
            let mut q8 = kernels::Q8::new(d.kv_lora / QK);
            kops::q8k_quantize(a, q8.view_mut());
            crate::moe::dot_rows(Q8_0, &tv.data[hh * d.vd * vrb..(hh + 1) * d.vd * vrb], vrb, q8.view(), o);
        } else {
            let vb = &m.v_b[hh * d.vd * d.kv_lora..(hh + 1) * d.vd * d.kv_lora];
            for (r, ov) in o.iter_mut().enumerate() {
                *ov = kops::dot_f32(&vb[r * d.kv_lora..(r + 1) * d.kv_lora], a);
            }
        }
    });
    let nb = vw / QK;
    let mut hq8 = crate::scratch::q8("mla.hq", n * nb);
    for t in 0..n {
        kops::q8k_quantize(&hv[t * vw..(t + 1) * vw], hq8.sub_mut(t * nb..(t + 1) * nb));
    }
    let mut out = buf("mla.out", n * dm);
    let two = w.need(&lay("attn_output.weight"))?;
    crate::moe::apply_group_multi(&mut [(Mat::of(&two), &mut out[..])], hq8.view(), n);
    ns(2, t0);
    for (xr, o) in x.chunks_exact_mut(dm).zip(out.chunks_exact(dm)) {
        for i in 0..dm {
            xr[i] += o[i];
        }
    }
    Ok(())
}
