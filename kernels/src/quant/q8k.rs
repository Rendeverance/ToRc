//! q8k.rs - Q8_K activations, vectorised (SPEC.md section 4).
//!
//! These give the same numbers as crate::q8k_quantize_scalar: each block's largest magnitude, d = amax / 127, q = x
//! times the reciprocal of d rounded half to even and clamped to +-127, sums of 16, and the same again for each 32 with
//! its own scale. Every operation is per number, a max (which is order-free) or an integer sum, so the vector forms
//! give the same bits.

use crate::QK;
use crate::quant::q8_k::Q8m;

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::*;

    /// The largest |x| (FMAX: a NaN wins, as in the scalar form).
    #[inline(always)]
    pub fn amax(x: &[f32]) -> f32 {
        // SAFETY: four at a time inside x (whose length is a multiple of 16)
        unsafe {
            let mut m = vdupq_n_f32(0.0);
            for i in (0..x.len()).step_by(4) {
                m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x.as_ptr().add(i))));
            }
            vmaxvq_f32(m)
        }
    }

    /// Sixteen numbers times id, rounded half to even and clamped to +-127, into q; returns their sum.
    #[inline(always)]
    pub fn group16(x: &[f32], id: f32, q: &mut [i8]) -> i16 {
        // SAFETY: x and q hold 16
        unsafe {
            let p = x.as_ptr();
            let (lo, hi) = (vdupq_n_s32(-127), vdupq_n_s32(127));
            let c = |o: usize| vmaxq_s32(vminq_s32(vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(p.add(o)), id)), hi), lo);
            let a = vcombine_s16(vmovn_s32(c(0)), vmovn_s32(c(4)));
            let b = vcombine_s16(vmovn_s32(c(8)), vmovn_s32(c(12)));
            let v = vcombine_s8(vmovn_s16(a), vmovn_s16(b));
            vst1q_s8(q.as_mut_ptr(), v);
            vaddlvq_s8(v)
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod imp {
    use std::arch::x86_64::*;

    /// The largest |x|.
    #[inline(always)]
    pub fn amax(x: &[f32]) -> f32 {
        // SAFETY: eight at a time inside x (whose length is a multiple of 16)
        unsafe {
            let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
            let mut m = _mm256_setzero_ps();
            for i in (0..x.len()).step_by(8) {
                m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(x.as_ptr().add(i)), abs));
            }
            let mut l = [0f32; 8];
            _mm256_storeu_ps(l.as_mut_ptr(), m);
            l.iter().fold(0.0f32, |a, &v| a.max(v))
        }
    }

    /// Sixteen numbers times id, rounded half to even (the default rounding) and clamped to +-127, into q; returns
    /// their sum.
    #[inline(always)]
    pub fn group16(x: &[f32], id: f32, q: &mut [i8]) -> i16 {
        // SAFETY: x and q hold 16
        unsafe {
            let (lo, hi, v) = (_mm256_set1_epi32(-127), _mm256_set1_epi32(127), _mm256_set1_ps(id));
            let c = |o: usize| {
                let r = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(o)), v));
                _mm256_max_epi32(_mm256_min_epi32(r, hi), lo)
            };
            let mut t = [0i32; 16];
            _mm256_storeu_si256(t.as_mut_ptr().cast(), c(0));
            _mm256_storeu_si256(t.as_mut_ptr().add(8).cast(), c(8));
            let mut s = 0i32;
            for (o, &v) in q.iter_mut().zip(&t) {
                *o = v as i8;
                s += v;
            }
            s as i16
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2"))))]
mod imp {
    /// The largest |x| (a NaN wins).
    pub fn amax(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |a, &v| if a.is_nan() || v.is_nan() { f32::NAN } else { a.max(v.abs()) })
    }

    /// Sixteen numbers times id, rounded half to even and clamped to +-127, into q; returns their sum.
    pub fn group16(x: &[f32], id: f32, q: &mut [i8]) -> i16 {
        let mut s = 0i32;
        for (o, &v) in q.iter_mut().zip(x) {
            let r = ((v * id).round_ties_even() as i32).clamp(-127, 127);
            *o = r as i8;
            s += r;
        }
        s as i16
    }
}

/// Blocks of 256 with the scales given by s256 and s32 (amax -> (stored scale, multiplier)).
fn blocks(x: &[f32], y: Q8m, s256: impl Fn(f32) -> (f32, f32), s32: impl Fn(f32) -> (f32, f32)) {
    assert!(x.len().is_multiple_of(QK) && y.len() >= x.len() / QK && y.g.len() == y.k.len());
    for ((xb, yb), gb) in x.chunks_exact(QK).zip(y.k.iter_mut()).zip(y.g.iter_mut()) {
        let (d, id) = s256(imp::amax(xb));
        yb.d = d;
        for g in 0..QK / 16 {
            yb.bsums[g] = imp::group16(&xb[16 * g..16 * g + 16], id, &mut yb.qs[16 * g..16 * g + 16]);
        }
        for k in 0..8 {
            let (d, id) = s32(imp::amax(&xb[32 * k..32 * k + 32]));
            gb.d[k] = d;
            let mut sum = 0i16;
            for h in 0..2 {
                let o = 32 * k + 16 * h;
                sum += imp::group16(&xb[o..o + 16], id, &mut gb.qs[o..o + 16]);
            }
            gb.sums[k] = sum;
        }
    }
}

/// d = amax / 127, with the multiplier 1 / d (0 when d = 0).
fn plain(a: f32) -> (f32, f32) {
    let d = a / 127.0;
    (d, if d > 0.0 { 1.0 / d } else { 0.0 })
}

/// The default quantiser (with crate::q8k_quantize_scalar's bits).
pub fn quantize(x: &[f32], y: Q8m) {
    blocks(x, y, plain, plain);
}

/// f32 to IEEE half, rounding to nearest even (overflow becomes infinity, and a NaN stays a NaN).
pub fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 0x1f {
        return s | 0x7c00;
    }
    if e16 <= 0 {
        // a subnormal half (or zero): the 24-bit significand shifted right, rounded half to even
        if e16 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let sh = (14 - e16) as u32;
        let (q, r, half) = (m >> sh, m & ((1 << sh) - 1), 1u32 << (sh - 1));
        let q = if r > half || (r == half && q & 1 == 1) { q + 1 } else { q };
        return s | q as u16;
    }
    let (q, r) = (m >> 13, m & 0x1fff);
    let v = ((e16 as u32) << 10) | q;
    let v = if r > 0x1000 || (r == 0x1000 && v & 1 == 1) { v + 1 } else { v };
    s | v as u16
}

/// qwen4exp's quantiser (ggml's Q8_K scales): the 256-block's multiplier is 127 / amax, with d its reciprocal; each
/// 32's d = amax / 127 is stored through fp16, with the multiplier 1 / d (of the f32 d).
pub fn quantize_gg(x: &[f32], y: Q8m) {
    let s256 = |a: f32| {
        if a == 0.0 {
            (0.0, 0.0)
        } else {
            let id = 127.0 / a;
            (1.0 / id, id)
        }
    };
    let s32 = |a: f32| {
        let d = a / 127.0;
        (crate::f16_to_f32(f32_to_f16(d)), if d != 0.0 { 1.0 / d } else { 0.0 })
    };
    blocks(x, y, s256, s32);
}
