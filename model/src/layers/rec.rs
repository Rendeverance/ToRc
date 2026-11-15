//! rec.rs - the gated DeltaNet layer.
//!
//! For qwen35moe (where value head h reads key head h % n_key_heads, and the state is fp16): the q|k|v and z products,
//! the beta and alpha dots, the causal convolution with SiLU, the L2 norms of the q and k heads, the recurrence
//! (kernels::delta_rule_h, four state rows at a time), the gated RMS norm, and ssm_out. qwen3next and qwen4exp use the
//! same layer with their own products and an f32 state. A layer takes three sweeps of the pool, for a token or a block:
//! the products together with the beta and alpha dots; then the convolution (one item per key head) and the recurrence
//! (one item per 32 state rows of a value head), as stages of one sweep where waiting is safe; then ssm_out, which for
//! a token is that sweep's third stage.

use crate::act::silu;
use crate::hparams::Hparams;
use crate::moe::Mat;
use crate::pool::SyncPtr;
use crate::weights::Weights;
use format::{Result, err};
use kernels::{Q8m, Q8s, QK, ops as kops};

/// The fewest state rows a recurrence item takes (the step kernels work on rows in fours, the state on 16-wide lanes).
const MIN_ROWS: usize = 16;

/// TORC_REC_F32=1: the DeltaNet state in f32 (as qwen4exp keeps it) instead of fp16, which rounds every state number
/// back to an 11-bit mantissa after each token. A test of how much that rounding moves the model; the bits change.
pub static REC_F32: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_REC_F32"));

/// The rows of a value head's state per recurrence item in rec_core, for `nv` heads of `ds` rows. On ARM64, the whole
/// head. On x86 the head is halved until there are four items per pool thread or a half would drop below MIN_ROWS:
/// heads whose alphas are near zero make every product a denormal (each a microcode assist on x86), and smaller items
/// spread those heads over the threads. TORC_REC_ROWS=r (a multiple of MIN_ROWS that divides the head) overrides it.
/// The rows are independent, so the bits are the same.
fn rec_rows(ds: usize, nv: usize) -> usize {
    static R: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| crate::switches::num("TORC_REC_ROWS"));
    match *R {
        Some(r) if r >= MIN_ROWS && r % MIN_ROWS == 0 && ds.is_multiple_of(r) => r,
        _ if cfg!(target_arch = "aarch64") => ds,
        _ => {
            let want = 4 * crate::pool::get().threads.max(1);
            let mut r = ds;
            while r.is_multiple_of(2) && r / 2 >= MIN_ROWS && nv * (ds / r) < want {
                r /= 2;
            }
            r
        }
    }
}

/// One recurrence step over rows of a head's state (v.len() rows of ds): kernels::delta_rule_h four rows at a time, or
/// else (with TORC_GDN=0, or for a shape it does not take) the portable step, with the same bits.
#[allow(clippy::too_many_arguments)]
fn gdn_head(
    state: &mut [u16],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    ds: usize,
) {
    static NATIVE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| crate::switches::get("TORC_GDN").is_none_or(|v| v != "0"));
    if *NATIVE && kernels::delta_rule_h(state, q, k, v, alpha, beta, q_scale, o, ds) {
        return;
    }
    kernels::delta_rule_h_scalar(state, q, k, v, alpha, beta, q_scale, o, ds)
}

pub struct RecLayer {
    pub l: u32,
    pub conv_taps: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a: Vec<f32>,
    pub beta_w: Vec<f32>,
    pub alpha_w: Vec<f32>,
    pub norm_w: Vec<f32>,
    /// qwen4exp (hc.rs): the betas and alphas are summed in f64 over the bf16 mix, softplus is ggml's, the output is
    /// gated by sigmoid(z) after a plain RMS norm, and ssm_out's input is quantised as ggml does it
    pub hc: bool,
    /// lfm2moe: a short convolution layer instead of DeltaNet (short_block): in_proj makes b, c and x; the convolution
    /// runs over b x on each channel (conv_taps, [tap][channel]), then is gated by c and goes through out_proj
    pub short: bool,
}

/// The recurrent state of one layer: the convolution's last d_conv - 1 inputs, the fp16 DeltaNet state (for qwen4exp it
/// is f32, held as pairs of u16 in the same buffer, twice as long and 4-aligned: see rec_core), and the log that makes
/// a block revertible (for MTP verification). During a revertible block, `conv` and `ssm` stay as the state before it
/// (the base), and the block's state is the log's tip.
pub struct RecState {
    pub conv: Vec<f32>,
    pub ssm: Vec<u16>,
    log: RecLog,
}

/// A revertible block: the block's state (the tip - the state before it stays in RecState as the base), and each of its
/// tokens' inputs to the state (the pre-convolution q|k|v row, beta and alpha). The block's first call reads the base
/// and writes the tip (each head's rows are copied over before their first step), and later calls work on the tip in
/// place. If everything is kept, the tip becomes the state (a swap). Rolling back to the first `kept` tokens means
/// taking the base, untouched, and running those tokens' convolution and recurrence again from the log, with the same
/// kernels in the same order, which gives the same bits as a block of `kept` tokens. The buffers are kept between uses.
#[derive(Default)]
struct RecLog {
    active: bool,
    /// whether the tip holds the block's state (a call has run since begin)
    live: bool,
    n: usize,
    conv: Vec<f32>,
    ssm: Vec<u16>,
    mixed: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
}

/// A tensor's rows as f32: F32 as it is, f16/bf16 widened, and a quantised type dequantised whole.
fn f32_rows(w: &Weights, l: u32, suffix: &str, d0: usize, d1: usize) -> Result<Vec<f32>> {
    let t = w.layer(l, suffix)?;
    if t.t.dims[0] as usize != d0 || t.t.dims[1] as usize != d1 {
        return err(format!("blk.{l}.{suffix}: unexpected shape"));
    }
    match t.t.ty {
        0 => t.f32s(),
        1 => Ok(t.data.chunks_exact(2).map(|b| kernels::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()),
        30 => Ok(t
            .data
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect()),
        ty => {
            let mut y = vec![0.0; d0 * d1];
            if kops::block_size(ty).is_none() || !d0.is_multiple_of(QK) {
                return err(format!("blk.{l}.{suffix}: type {ty} not readable"));
            }
            kops::dequantize(ty, t.data, &mut y);
            Ok(y)
        }
    }
}

impl RecLayer {
    pub fn load(w: &Weights, h: &Hparams, l: u32) -> Result<RecLayer> {
        if h.short_conv > 0 {
            return Self::load_short(w, h, l);
        }
        let s = h.ssm.as_ref().ok_or_else(|| format::Error("no SSM hyperparameters".into()))?;
        let (dm, nv) = (h.d_model as usize, s.n_value_heads as usize);
        let cc = conv_width(h);
        let dc = s.d_conv as usize;
        let conv = w.layer(l, "ssm_conv1d.weight")?.f32s()?;
        if conv.len() != dc * cc {
            return err(format!("blk.{l}.ssm_conv1d.weight: unexpected size"));
        }
        let mut conv_taps = vec![0.0; dc * cc];
        for ch in 0..cc {
            for i in 0..dc {
                conv_taps[i * cc + ch] = conv[ch * dc + i];
            }
        }
        Ok(RecLayer {
            l,
            conv_taps,
            dt_bias: w.layer(l, "ssm_dt.bias")?.f32s()?,
            a: w.layer(l, "ssm_a")?.f32s()?,
            beta_w: if h.next { Vec::new() } else { f32_rows(w, l, "ssm_beta.weight", dm, nv)? },
            alpha_w: if h.next { Vec::new() } else { f32_rows(w, l, "ssm_alpha.weight", dm, nv)? },
            norm_w: w.layer(l, "ssm_norm.weight")?.f32s()?,
            hc: h.hc > 0,
            short: false,
        })
    }

    /// An lfm2moe short-convolution layer: its taps (shortconv.conv.weight, [channel][tap] in the file) laid out as
    /// [tap][channel].
    fn load_short(w: &Weights, h: &Hparams, l: u32) -> Result<RecLayer> {
        let (dm, nt) = (h.d_model as usize, h.short_conv as usize);
        let conv = w.layer(l, "shortconv.conv.weight")?.f32s()?;
        if conv.len() != nt * dm {
            return err(format!("blk.{l}.shortconv.conv.weight: unexpected size"));
        }
        let mut conv_taps = vec![0.0; nt * dm];
        for ch in 0..dm {
            for i in 0..nt {
                conv_taps[i * dm + ch] = conv[ch * nt + i];
            }
        }
        Ok(RecLayer {
            l,
            conv_taps,
            dt_bias: Vec::new(),
            a: Vec::new(),
            beta_w: Vec::new(),
            alpha_w: Vec::new(),
            norm_w: Vec::new(),
            hc: false,
            short: true,
        })
    }
}

/// lfm2moe's short convolution for n tokens in order (n = 1 for a decoded token): in_proj for all n at once, giving
/// b, c and x per token; then on each channel, token by token, the convolution over the last l_cache values of b x
/// (the state holds the l_cache - 1 before them, oldest first), gated by c; then out_proj for all n, added to x. As
/// llama.cpp's build_shortconv_block (ggml_ssm_conv). No revertible blocks: the model has no MTP block.
pub fn short_block(
    w: &Weights,
    h: &Hparams,
    ly: &RecLayer,
    st: &mut RecState,
    xq: Q8s,
    x: &mut [f32],
    n: usize,
) -> Result<()> {
    let (dm, nt) = (h.d_model as usize, h.short_conv as usize);
    let keep = nt - 1;
    let mut bcx = crate::scratch::buf::<f32>("short.bcx", n * 3 * dm);
    let tin = w.layer(ly.l, "shortconv.in_proj.weight")?;
    crate::moe::apply_group_multi(&mut [(Mat::of(&tin), &mut bcx[..])], xq, n);
    let mut y = crate::scratch::buf::<f32>("short.y", n * dm);
    let (yp, cp) = (SyncPtr::new(&mut y[..]), SyncPtr::new(&mut st.conv[..]));
    // channels in chunks of 64: a channel's state and steps are its own, so the order across chunks does not matter
    crate::pool::par_for(dm.div_ceil(64), &|c| {
        for ch in c * 64..(c * 64 + 64).min(dm) {
            // SAFETY: chunk c owns its channels' state rows and outputs
            unsafe {
                let mut win = [0f32; 8];
                for k in 0..keep {
                    win[k] = *cp.at(k * dm + ch);
                }
                for t in 0..n {
                    let row = &bcx[t * 3 * dm..(t + 1) * 3 * dm];
                    win[keep] = row[ch] * row[2 * dm + ch];
                    let mut acc = 0f32;
                    for k in 0..nt {
                        acc += ly.conv_taps[k * dm + ch] * win[k];
                    }
                    *yp.at(t * dm + ch) = row[dm + ch] * acc;
                    win.copy_within(1..nt, 0);
                }
                for k in 0..keep {
                    *cp.at(k * dm + ch) = win[k];
                }
            }
        }
    });
    let nb = dm / QK;
    let mut yq = crate::scratch::q8("short.yq", n * nb);
    for t in 0..n {
        kops::q8k_quantize(&y[t * dm..(t + 1) * dm], yq.sub_mut(t * nb..(t + 1) * nb));
    }
    let mut out = crate::scratch::buf::<f32>("short.out", n * dm);
    let tout = w.layer(ly.l, "shortconv.out_proj.weight")?;
    crate::moe::apply_group_multi(&mut [(Mat::of(&tout), &mut out[..])], yq.view(), n);
    for (xr, o) in x.chunks_exact_mut(dm).zip(out.chunks_exact(dm)) {
        for i in 0..dm {
            xr[i] += o[i];
        }
    }
    Ok(())
}

/// The q, k and v channels: 2 x the key dimension + d_inner.
pub fn conv_width(h: &Hparams) -> usize {
    let s = h.ssm.as_ref().unwrap();
    2 * (s.n_key_heads * s.ds) as usize + s.d_inner as usize
}

impl RecState {
    /// The tokens of the revertible block in progress, if one is: the most `commit` may keep.
    pub fn logged(&self) -> Option<usize> {
        self.log.active.then_some(self.log.n)
    }

    pub fn new(h: &Hparams) -> RecState {
        // lfm2moe's short convolution: the last l_cache - 1 values of b x per channel, and no recurrent state
        if h.short_conv > 0 {
            return RecState {
                conv: vec![0.0; (h.short_conv as usize - 1) * h.d_model as usize],
                ssm: Vec::new(),
                log: RecLog::default(),
            };
        }
        let s = h.ssm.as_ref().unwrap();
        RecState {
            conv: vec![0.0; (s.d_conv as usize - 1) * conv_width(h)],
            ssm: vec![0; (s.n_value_heads * s.ds * s.ds) as usize * if h.hc > 0 || *REC_F32 { 2 } else { 1 }],
            log: RecLog::default(),
        }
    }

    /// Goes back to the start of a sequence (a pipeline worker's next run): a zero state.
    pub fn clear(&mut self) {
        self.conv.fill(0.0);
        self.ssm.fill(0);
        // and no revertible block left open by a run that ended in the middle of a verify
        self.log.active = false;
        self.log.live = false;
    }

    /// Starts a revertible block: the state as it is now stays as the base, and the block writes the tip and logs its
    /// inputs (rec_block only). There is no copy, since the tip's contents are rewritten before they are read.
    pub fn begin(&mut self) {
        let lg = &mut self.log;
        lg.conv.resize(self.conv.len(), 0.0);
        lg.ssm.resize(self.ssm.len(), 0);
        lg.mixed.clear();
        lg.beta.clear();
        lg.alpha.clear();
        lg.n = 0;
        lg.live = false;
        lg.active = true;
    }
}

/// The longest revertible block (an MTP pass is at most 9 tokens).
pub const LOG_MAX: usize = 64;

/// Ends a revertible block, keeping its first `kept` tokens. If all are kept, the tip becomes the state (a swap of
/// buffers); if fewer, the tip is dropped and the kept tokens' convolution and recurrence run again from the log on the
/// base, in place (with their outputs thrown away).
pub fn commit(h: &Hparams, ly: &RecLayer, st: &mut RecState, kept: usize) {
    if !st.log.active || ly.short {
        return;
    }
    st.log.active = false;
    let live = std::mem::replace(&mut st.log.live, false);
    let n = st.log.n;
    assert!(kept <= n, "commit {kept} of a block of {n}");
    let RecState { conv, ssm, log } = st;
    if kept == n {
        if live {
            std::mem::swap(conv, &mut log.conv);
            std::mem::swap(ssm, &mut log.ssm);
        }
        return;
    }
    if kept > 0 {
        let s = h.ssm.as_ref().unwrap();
        let (cc, nv, d_inner) = (conv_width(h), s.n_value_heads as usize, s.d_inner as usize);
        let z = vec![0.0f32; kept * d_inner];
        let mut att = vec![0.0f32; kept * d_inner];
        let mut prof = crate::Laps::start(&REC_NS);
        rec_core(
            h,
            ly,
            None,
            conv,
            None,
            ssm,
            &log.mixed[..kept * cc],
            &z,
            &log.beta[..kept * nv],
            &log.alpha[..kept * nv],
            &mut att,
            kept,
            None,
            &mut prof,
        );
    }
}

/// ((0+4) + (2+6)) + ((1+5) + (3+7)) over two 4-lane accumulators.
fn hsum4x2(a: [f32; 4], b: [f32; 4]) -> f32 {
    let s: [f32; 4] = std::array::from_fn(|l| a[l] + b[l]);
    (s[0] + s[2]) + (s[1] + s[3])
}

/// A sum of squares in the NEON norm kernels' order: lanes i..i+4 in a0 and i+4..i+8 in a1, fused, 8 at a time.
fn sumsq8(x: &[f32]) -> f32 {
    let (mut a0, mut a1) = ([0.0f32; 4], [0.0f32; 4]);
    for i in (0..x.len()).step_by(8) {
        for l in 0..4 {
            a0[l] = x[i + l].mul_add(x[i + l], a0[l]);
            a1[l] = x[i + 4 + l].mul_add(x[i + 4 + l], a1[l]);
        }
    }
    hsum4x2(a0, a1)
}

/// qwen3next's products for n tokens (ssm_in). ssm_in has rows 2 n_key_heads ds + 2 d_inner, grouped per key head g as
/// [q_g, k_g, v of its rf value heads, z of them], and ssm_ba has rows 2 n_value_heads, per g: rf betas, then rf
/// alphas. Both are quantised products of xq (one vector as in rec_token, more as in rec_block, with the same bits).
/// Then each token's row is unpacked into the convolution's order (q block, k block, v block) and z by value head,
/// along with the betas and decays.
#[allow(clippy::too_many_arguments)]
fn next_products(
    w: &Weights,
    h: &Hparams,
    ly: &RecLayer,
    xq: Q8s,
    n: usize,
    mixed: &mut [f32],
    z: &mut [f32],
    beta: &mut [f32],
    alpha: &mut [f32],
) -> Result<()> {
    let s = h.ssm.as_ref().unwrap();
    let (ds, nkh, nv, d_inner) = (s.ds as usize, s.n_key_heads as usize, s.n_value_heads as usize, s.d_inner as usize);
    let (rf, kd, cc) = (nv / nkh, nkh * ds, conv_width(h));
    let (gsz, qkvz) = (2 * ds + 2 * ds * rf, 2 * nkh * ds + 2 * d_inner);
    let mut big = crate::scratch::buf::<f32>("rec.big", n * qkvz);
    let mut ba = crate::scratch::buf::<f32>("rec.ba", n * 2 * nv);
    let (ti, tb) = (w.layer(ly.l, "ssm_in.weight")?, w.layer(ly.l, "ssm_ba.weight")?);
    let mut jobs = [(Mat::of(&ti), &mut big[..]), (Mat::of(&tb), &mut ba[..])];
    if n == 1 {
        crate::moe::apply_group(&mut jobs, xq);
    } else {
        crate::moe::apply_group_multi(&mut jobs, xq, n);
    }
    for j in 0..n {
        let src = &big[j * qkvz..(j + 1) * qkvz];
        let mx = &mut mixed[j * cc..(j + 1) * cc];
        let b = &ba[j * 2 * nv..(j + 1) * 2 * nv];
        for g in 0..nkh {
            let sg = &src[g * gsz..(g + 1) * gsz];
            mx[g * ds..(g + 1) * ds].copy_from_slice(&sg[..ds]);
            mx[kd + g * ds..kd + (g + 1) * ds].copy_from_slice(&sg[ds..2 * ds]);
            mx[2 * kd + g * ds * rf..2 * kd + (g + 1) * ds * rf].copy_from_slice(&sg[2 * ds..2 * ds + ds * rf]);
            for r in 0..rf {
                let hh = g * rf + r;
                z[j * d_inner + hh * ds..j * d_inner + (hh + 1) * ds]
                    .copy_from_slice(&sg[2 * ds + ds * rf + r * ds..2 * ds + ds * rf + (r + 1) * ds]);
                let al = b[2 * g * rf + rf + r] + ly.dt_bias[hh];
                let sp = if al > 20.0 { al } else { al.exp().ln_1p() };
                beta[j * nv + hh] = crate::ops::sigmoid(b[2 * g * rf + r]);
                alpha[j * nv + hh] = (sp * ly.a[hh]).exp();
            }
        }
    }
    Ok(())
}

/// Item i = t * nv + h of the betas and decays for n tokens: head h's two f32 dots over token t's normed row (xn holds
/// n rows). Each item makes the same calls on whichever thread runs it, so they can ride along in the products' sweep
/// of the pool.
fn beta_alpha(ly: &RecLayer, xn: &[f32], dm: usize, nv: usize, beta: SyncPtr<f32>, alpha: SyncPtr<f32>, i: usize) {
    let (t, hh) = (i / nv, i % nv);
    let xt = &xn[t * dm..(t + 1) * dm];
    // qwen4exp: summed as ggml sums its bf16 matrices, with ggml's softplus, log(1 + exp)
    let dot = if ly.hc { crate::hc::dot_dbl } else { kops::dot_f32 };
    let b = dot(&ly.beta_w[hh * dm..(hh + 1) * dm], xt);
    let al = dot(&ly.alpha_w[hh * dm..(hh + 1) * dm], xt) + ly.dt_bias[hh];
    let sp = if al > 20.0 {
        al
    } else if ly.hc {
        (1.0 + al.exp()).ln()
    } else {
        al.exp().ln_1p()
    };
    // SAFETY: item i owns entry i of beta and alpha
    unsafe {
        beta.slice(i, 1)[0] = crate::ops::sigmoid(b);
        alpha.slice(i, 1)[0] = (sp * ly.a[hh]).exp();
    }
}

/// rec_token's ssm_out, run inside rec_core's sweep: the product, its Q8 input (each block is quantised by whichever
/// head item finishes it last) and its output.
struct SsmOut<'a> {
    m: Mat<'a>,
    attq: Q8m<'a>,
    y: &'a mut [f32],
}

/// x += the layer's output for one token, where xn is the token normed and xq is xn as Q8. Where waiting is safe
/// (pool::waits_ok) this takes two sweeps of the pool: the products with the betas and decays; then the convolution
/// with the recurrence, followed by ssm_out (rec_core). Otherwise it takes three, with ssm_out on its own.
pub fn rec_token(
    w: &Weights,
    h: &Hparams,
    ly: &RecLayer,
    st: &mut RecState,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
) -> Result<()> {
    if ly.short {
        return short_block(w, h, ly, st, xq, x, 1);
    }
    let s = h.ssm.as_ref().unwrap();
    let (dm, d_inner, nv) = (h.d_model as usize, s.d_inner as usize, s.n_value_heads as usize);
    let cc = conv_width(h);
    let buf = crate::scratch::buf::<f32>;
    let mut prof = crate::Laps::start(&REC_NS);

    // 1. q|k|v and z, and the betas and alphas (f32 dots, one head per item)
    let (mut mixed, mut z) = (buf("rec.mixed", cc), buf("rec.z", d_inner));
    let (mut beta, mut alpha) = (buf("rec.beta", nv), buf("rec.alpha", nv));
    let (bp, dp) = (SyncPtr::new(&mut beta), SyncPtr::new(&mut alpha));
    if h.next {
        next_products(w, h, ly, xq, 1, &mut mixed, &mut z, &mut beta, &mut alpha)?;
    } else {
        let (tq, tz) = (w.layer(ly.l, "attn_qkv.weight")?, w.layer(ly.l, "attn_gate.weight")?);
        crate::moe::apply_group_and(&mut [(Mat::of(&tq), &mut mixed[..]), (Mat::of(&tz), &mut z[..])], xq, nv, &|i| {
            beta_alpha(ly, xn, dm, nv, bp, dp, i)
        });
    }
    prof.lap(0);
    // 2-4. the token's step through the state: the convolution with SiLU and the q/k L2 norms, the recurrence, and the
    // gated norm (rec_core with n = 1: the same calls as a block of one)
    // 5. ssm_out on the quantised heads (inside rec_core's sweep), added to the residual
    let (mut att, mut out) = (buf("rec.att", d_inner), buf("rec.out", dm));
    let mut attq = crate::scratch::q8("rec.attq", d_inner / QK);
    let so = SsmOut { m: Mat::of(&w.layer(ly.l, "ssm_out.weight")?), attq: attq.view_mut(), y: &mut out };
    let RecState { conv, ssm, log: lg } = st;
    if lg.active {
        // a revertible block is open (a speculative token in the two-machine pipeline's speculation, or a verify pass
        // of one token): the token is logged, the tip is written from the base as rec_steps does, and commit later
        // keeps or drops it
        lg.mixed.extend_from_slice(&mixed[..cc]);
        lg.beta.extend_from_slice(&beta);
        lg.alpha.extend_from_slice(&alpha);
        lg.n += 1;
        if std::mem::replace(&mut lg.live, true) {
            rec_core(
                h,
                ly,
                None,
                &mut lg.conv,
                None,
                &mut lg.ssm,
                &mixed,
                &z,
                &beta,
                &alpha,
                &mut att,
                1,
                Some(so),
                &mut prof,
            );
        } else {
            rec_core(
                h,
                ly,
                Some(&conv[..]),
                &mut lg.conv,
                Some(&ssm[..]),
                &mut lg.ssm,
                &mixed,
                &z,
                &beta,
                &alpha,
                &mut att,
                1,
                Some(so),
                &mut prof,
            );
        }
    } else {
        rec_core(h, ly, None, conv, None, ssm, &mixed, &z, &beta, &alpha, &mut att, 1, Some(so), &mut prof);
    }
    for i in 0..dm {
        x[i] += out[i];
    }
    Ok(())
}

/// The time spent in each phase of rec_block, and of rec_token's rec_core (TORC_PROF=1).
pub static REC_NS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
pub const REC_PHASES: [&str; 8] = ["qkv+gate+betas", "-", "-", "conv+gdn+norm", "quant", "ssm_out", "add", "-"];

/// The layer for n tokens in order (a prompt block): the products of all n at once, so each weight row is read once
/// (with the same bits as one at a time), then the steps through the state token by token. xn, xq and x hold n rows
/// each. The result has the same bits as rec_token on each token in turn.
pub fn rec_block(
    w: &Weights,
    h: &Hparams,
    ly: &RecLayer,
    st: &mut RecState,
    xn: &[f32],
    xq: Q8s,
    x: &mut [f32],
    n: usize,
) -> Result<()> {
    if ly.short {
        return short_block(w, h, ly, st, xq, x, n);
    }
    let s = h.ssm.as_ref().unwrap();
    let (dm, d_inner) = (h.d_model as usize, s.d_inner as usize);
    let cc = conv_width(h);
    let mut prof = crate::Laps::start(&REC_NS);
    let mut mixed = crate::scratch::buf::<f32>("rec.mixed", n * cc);
    let mut z = crate::scratch::buf::<f32>("rec.z", n * d_inner);
    // the products and the betas and decays (one item per token and head) in one sweep of the pool
    let nv = s.n_value_heads as usize;
    let mut beta = crate::scratch::buf::<f32>("rec.beta", n * nv);
    let mut alpha = crate::scratch::buf::<f32>("rec.alpha", n * nv);
    let (bp, dp) = (SyncPtr::new(&mut beta), SyncPtr::new(&mut alpha));
    let bd = |i: usize| beta_alpha(ly, xn, dm, nv, bp, dp, i);
    if h.next {
        next_products(w, h, ly, xq, n, &mut mixed, &mut z, &mut beta, &mut alpha)?;
    } else {
        let (tq, tz) = (w.layer(ly.l, "attn_qkv.weight")?, w.layer(ly.l, "attn_gate.weight")?);
        let mut jobs = [(Mat::of(&tq), &mut mixed[..]), (Mat::of(&tz), &mut z[..])];
        if crate::npu::apply(&mut jobs, xn, n) {
            crate::pool::par_for(n * nv, &bd);
        } else {
            crate::moe::apply_group_multi_and(&mut jobs, xq, n, n * nv, &bd);
        }
    }
    prof.lap(0);
    let mut att = crate::scratch::buf::<f32>("rec.att", n * d_inner);
    rec_steps(h, ly, st, &mixed, &z, &beta, &alpha, &mut att, n, &mut prof);
    let nbi = d_inner / QK;
    let mut out = crate::scratch::buf::<f32>("rec.out", n * dm);
    let tout = w.layer(ly.l, "ssm_out.weight")?;
    let mut jobs = [(Mat::of(&tout), &mut out[..])];
    if !crate::npu::apply(&mut jobs, &att, n) {
        let mut attq = crate::scratch::q8("rec.attq", n * nbi);
        let quant = if ly.hc { kops::q8k_quantize_gg } else { kops::q8k_quantize };
        for t in 0..n {
            quant(&att[t * d_inner..(t + 1) * d_inner], attq.sub_mut(t * nbi..(t + 1) * nbi));
        }
        prof.lap(4);
        crate::moe::apply_group_multi(&mut jobs, attq.view(), n);
    }
    prof.lap(5);
    for (xr, o) in x.chunks_exact_mut(dm).zip(out.chunks_exact(dm)) {
        for i in 0..dm {
            xr[i] += o[i];
        }
    }
    prof.lap(6);
    Ok(())
}

/// The n tokens' steps through the state, from their products, betas and decays (rec_core), logged when a revertible
/// block is open. mixed, z and att hold n rows each; beta and alpha hold n x n_value_heads.
#[allow(clippy::too_many_arguments)]
fn rec_steps(
    h: &Hparams,
    ly: &RecLayer,
    st: &mut RecState,
    mixed: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    att: &mut [f32],
    n: usize,
    prof: &mut crate::Laps,
) {
    let cc = conv_width(h);
    let RecState { conv, ssm, log: lg } = st;
    if !lg.active {
        rec_core(h, ly, None, conv, None, ssm, mixed, z, beta, alpha, att, n, None, prof);
        return;
    }
    lg.mixed.extend_from_slice(&mixed[..n * cc]);
    lg.beta.extend_from_slice(beta);
    lg.alpha.extend_from_slice(alpha);
    lg.n += n;
    if std::mem::replace(&mut lg.live, true) {
        // a later call of the block: work on the tip in place
        rec_core(h, ly, None, &mut lg.conv, None, &mut lg.ssm, mixed, z, beta, alpha, att, n, None, prof);
    } else {
        // the block's first call: from the base into the tip
        rec_core(
            h,
            ly,
            Some(&conv[..]),
            &mut lg.conv,
            Some(&ssm[..]),
            &mut lg.ssm,
            mixed,
            z,
            beta,
            alpha,
            att,
            n,
            None,
            prof,
        );
    }
}

/// The convolution, the state's new window, and the recurrence with the gated norm, for n tokens in order, from their
/// q|k|v rows, betas and decays (the second half of rec_steps, and also the replay of a rolled-back block). The state
/// is read from conv_src / ssm_src when they are given (a revertible block's base) and the new state is written to conv
/// / ssm (its tip); otherwise conv / ssm are read and written in place. The values are the same either way. With out
/// (one token), ssm_out runs on att in the same sweep.
#[allow(clippy::too_many_arguments)]
fn rec_core(
    h: &Hparams,
    ly: &RecLayer,
    conv_src: Option<&[f32]>,
    conv: &mut [f32],
    ssm_src: Option<&[u16]>,
    ssm: &mut [u16],
    mixed: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    att: &mut [f32],
    n: usize,
    out: Option<SsmOut>,
    prof: &mut crate::Laps,
) {
    let s = h.ssm.as_ref().unwrap();
    let (ds, nv, nkh) = (s.ds as usize, s.n_value_heads as usize, s.n_key_heads as usize);
    let (kd, dc, d_inner) = (nkh * ds, s.d_conv as usize, s.d_inner as usize);
    let cc = conv_width(h);
    let eps = h.eps;
    assert!(out.is_none() || n == 1, "ssm_out in the sweep is for one token");
    assert!(nv % nkh == 0 && ds % MIN_ROWS == 0 && dc >= 1);
    use std::sync::atomic::{AtomicUsize, Ordering};
    let scale = 1.0f32 / (ds as f32).sqrt();
    // One sweep of the pool in stages (pool::par_stages). Stage 1, one item per key head g, its tokens in order: the
    // convolution of q and k for g and of v for each value head that reads g (g, g + n_key_heads, ..), then the window
    // shift of those channels (only this item reads or writes them). Stage 2, one item per (value head, rec_rows rows
    // of its state): the rows' recurrence steps for the n tokens, and the head's last item to finish runs the head's
    // gated norms. Stage 3 (one token only): ssm_out. Every value comes from the same calls on the same inputs as if a
    // value head's convolution, steps and norm ran in one item (the rows are independent), so the bits are the same.
    // The rows are split over threads because one head can be far slower than the rest: an alpha near 0 makes its
    // products denormal, and on x86 every one of them is a microcode assist.
    let mut cv = crate::scratch::buf::<f32>("rec.qkv", n * cc);
    let cw = &ly.conv_taps;
    let wp = SyncPtr::new(conv);
    // the window to read: the base when it is given, else conv itself (each item reads and then rewrites only its own
    // channels' rows)
    let stc = SyncPtr(conv_src.map_or(wp.0, |s| s.as_ptr().cast_mut()));
    // entry k of the sequence [window rows 0..dc - 1, mixed_0, mixed_1, ..], for channels c0..c0 + ds. Token t's window
    // is the sequence from t on, as a token step shifts it. The branch is per row rather than per element, so the loops
    // below run over contiguous channels and vectorise; each channel's steps are unchanged
    let seq = |k: usize, c0: usize| -> &[f32] {
        // SAFETY: row k < dc - 1 of the window (dc - 1 rows of cc channels); only the item owning channels c0..c0 + ds
        // writes them, and only after its last read
        if k + 1 < dc {
            unsafe { std::slice::from_raw_parts(stc.at(k * cc + c0), ds) }
        } else {
            &mixed[(k + 1 - dc) * cc + c0..][..ds]
        }
    };
    let cp = SyncPtr::new(&mut cv);
    // token t's causal convolution of channels c0..c0 + ds (s = win0 * w0, then fused steps) with SiLU, then, for q and
    // k, its L2 norm, into row t of cv
    let conv_head = |t: usize, c0: usize| {
        // SAFETY: channels c0..c0 + ds of row t belong to the item of their key head
        let out = unsafe { cp.slice(t * cc + c0, ds) };
        for ((o, &s), &w) in out.iter_mut().zip(seq(t, c0)).zip(&cw[c0..c0 + ds]) {
            *o = s * w;
        }
        for i in 1..dc {
            for ((o, &s), &w) in out.iter_mut().zip(seq(t + i, c0)).zip(&cw[i * cc + c0..i * cc + c0 + ds]) {
                *o = s.mul_add(w, *o);
            }
        }
        for e in out.iter_mut() {
            *e = silu(*e);
        }
        if c0 < 2 * kd {
            let inv = 1.0 / (sumsq8(out) + eps).sqrt();
            for e in out.iter_mut() {
                *e *= inv;
            }
        }
    };
    // the window after the block, for channels c0..c0 + ds: the last dc - 1 entries of the sequence. In place, row r
    // takes row n + r > r, so working upwards never reads a row that has already been written
    let shift = |c0: usize| {
        for r in 0..dc - 1 {
            let from = seq(n + r, c0).as_ptr();
            // SAFETY: this item's channels of row r; the source is a later row, the base, or mixed
            unsafe { std::ptr::copy_nonoverlapping(from, wp.at(r * cc + c0), ds) };
        }
    };
    // the value heads that read key head g: g, g + n_key_heads, .. (qwen35moe), or g rf .. g rf + rf - 1 (qwen3next)
    let rf = nv / nkh;
    let convs = |g: usize| {
        let vh = if h.next { (g * rf..(g + 1) * rf).step_by(1) } else { (g..nv).step_by(nkh) };
        for t in 0..n {
            conv_head(t, g * ds);
            conv_head(t, kd + g * ds);
            for hh in vh.clone() {
                conv_head(t, 2 * kd + hh * ds);
            }
        }
        shift(g * ds);
        shift(kd + g * ds);
        for hh in vh {
            shift(2 * kd + hh * ds);
        }
    };
    // the u16 elements per state number: 1 (fp16), or 2 (qwen4exp's f32)
    let f32s = ly.hc || *REC_F32;
    let el = if f32s { 2 } else { 1 };
    assert!(!f32s || (ssm.as_ptr() as usize).is_multiple_of(4), "f32 DeltaNet state not 4-aligned");
    let (mp, ap) = (SyncPtr::new(ssm), SyncPtr::new(att));
    let rows = rec_rows(ds, nv);
    let rb = ds / rows;
    let rows_left: Vec<AtomicUsize> = (0..nv).map(|_| AtomicUsize::new(rb)).collect();
    // with ssm_out (one token): each Q8 block of the output counts down the heads that write it, and the head that
    // finishes it last quantises it (the same call on the same values as quantising the whole row at once, since blocks
    // are quantised independently)
    let nb = d_inner / QK;
    let left: Vec<AtomicUsize> = (0..nb)
        .map(|b| AtomicUsize::new((0..nv).filter(|&hh| hh * ds < (b + 1) * QK && b * QK < (hh + 1) * ds).count()))
        .collect();
    let wo = out.as_ref().map(|o| o.m);
    let (qp, yp) = match out {
        Some(o) => (crate::pool::Q8Ptr::new(o.attq), SyncPtr::new(o.y)),
        None => (crate::pool::Q8Ptr::null(), SyncPtr(std::ptr::null_mut())),
    };
    let steps = |i: usize| {
        // item i: value head vh, its band of `rows` state rows from r0, and the key head that value head reads
        let vh = i / rb;
        let r0 = (i - vh * rb) * rows;
        let kh = if h.next { vh / rf } else { vh % nkh };
        // SAFETY: the item owns these state rows and these rows of the head's outputs, and cv is complete (stage 1)
        let m = unsafe { mp.slice((vh * ds + r0) * ds * el, rows * ds * el) };
        let src_rows = ssm_src.map(|src| &src[(vh * ds + r0) * ds * el..(vh * ds + r0 + rows) * ds * el]);
        if let (Some(src), false) = (src_rows, f32s) {
            // copy on first write: the rows' base goes into the tip, then their steps run there
            m.copy_from_slice(src);
        }
        let cv = unsafe { std::slice::from_raw_parts(cp.at(0), n * cc) };
        for t in 0..n {
            let row = &cv[t * cc..(t + 1) * cc];
            let (q, k) = (&row[kh * ds..(kh + 1) * ds], &row[kd + kh * ds..kd + (kh + 1) * ds]);
            let v = &row[2 * kd + vh * ds + r0..][..rows];
            let o = unsafe { ap.slice(t * d_inner + vh * ds + r0, rows) };
            if f32s {
                // SAFETY: the buffer is 4-aligned (asserted), and these rows are rows * ds f32s (2 u16 each)
                let m32 = unsafe { std::slice::from_raw_parts_mut(m.as_mut_ptr() as *mut f32, rows * ds) };
                // a tip's first step reads the base and writes the tip, with no copy first SAFETY: the base buffer is
                // the state's twin (4-aligned, with the same rows)
                let s32 = (t == 0)
                    .then_some(src_rows)
                    .flatten()
                    .map(|s| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, rows * ds) });
                kernels::delta_rule_f32(s32, m32, q, k, v, alpha[t * nv + vh], beta[t * nv + vh], scale, o, ds);
            } else {
                gdn_head(m, q, k, v, alpha[t * nv + vh], beta[t * nv + vh], scale, o, ds);
            }
        }
        // AcqRel: the head's other rows become visible to whichever item runs its norms
        if rows_left[vh].fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        for t in 0..n {
            // the gated RMS norm: (o * sc) * w * silu(z) SAFETY: every item of head vh is done, and only this one
            // touches its outputs now
            let o = unsafe { ap.slice(t * d_inner + vh * ds, ds) };
            let zz = &z[t * d_inner + vh * ds..t * d_inner + (vh + 1) * ds];
            if ly.hc {
                // qwen4exp: rms_norm, then * sigmoid(z)
                crate::ops::rms_norm_inplace(o, &ly.norm_w[..ds], eps);
                for (e, &zi) in o.iter_mut().zip(zz) {
                    *e *= crate::ops::sigmoid(zi);
                }
                continue;
            }
            let sc = 1.0 / (sumsq8(o) / ds as f32 + eps).sqrt();
            for ((e, &w), &zi) in o.iter_mut().zip(&ly.norm_w[..ds]).zip(zz) {
                *e = (*e * sc * w) * silu(zi);
            }
        }
        if wo.is_some() {
            for b in vh * ds / QK..((vh + 1) * ds).div_ceil(QK) {
                // AcqRel: the other heads' outputs in this block become visible to whichever item quantises it
                if left[b].fetch_sub(1, Ordering::AcqRel) == 1 {
                    // SAFETY: every head writing block b is done, and only this item touches attq[b]
                    let (a, y) = unsafe { (&*ap.slice(b * QK, QK), qp.slice(b, 1)) };
                    if ly.hc { kops::q8k_quantize_gg(a, y) } else { kops::q8k_quantize(a, y) }
                }
            }
        }
    };
    match wo {
        None => crate::pool::par_stages(&[(nkh, &convs), (nv * rb, &steps)]),
        // ssm_out's row chunks, once every head (and so every block) is done
        Some(m) => crate::pool::par_stages(&[
            (nkh, &convs),
            (nv * rb, &steps),
            // SAFETY: attq is complete and no longer written (the stage before is done), and chunk i owns its rows
            (m.chunks(), &|i| unsafe { m.apply_chunk(qp.part(0, nb), yp, i) }),
        ]),
    }
    prof.lap(3);
}
