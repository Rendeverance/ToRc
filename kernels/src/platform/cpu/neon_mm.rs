//! neon_mm.rs - Q4_0 rows against a few vectors (MTP verify, short blocks) on ARM64 with SMMLA (FEAT_I8MM).
//!
//! SMMLA takes 8 weights from each of two rows and the same 8 activations from each of two vectors, and adds the four
//! sums into [r0v0, r0v1, r1v0, r1v1]. The two vectors' activations are interleaved once and serve every pair of rows,
//! and Q4_0's -8 goes in as a starting value (-8 x the activations' sums), so the integers are exact. The float steps
//! are the single-vector kernel's, lane for lane (for each block k of 256: acc[k] = fma(d_w[k] x g.d[k], dot, acc[k]),
//! then fold8), so every result matches the reference bit for bit.

use crate::neon::{q4_0_matmul, q4_0_scales, smmla};
use crate::{Groups32, Q4_0_BLOCK};
use core::arch::aarch64::*;

/// The most blocks of 256 a row may have (8192 columns).
pub const MAX_NB: usize = 32;

/// The activations of vectors a and b arranged for SMMLA: for each block of 32, four groups of 16 bytes, 8 from a and
/// then the same 8 from b, for weights 0-7, 16-23, 8-15 and 24-31 (the order the 4-bit blocks unpack in).
unsafe fn interleave(a: &[Groups32], b: &[Groups32], nb: usize, xi: &mut [i8]) {
    unsafe {
        for s in 0..nb {
            for k in 0..8 {
                let (p, q) = (a[s].qs.as_ptr().add(32 * k), b[s].qs.as_ptr().add(32 * k));
                let a0 = vreinterpretq_s64_s8(vld1q_s8(p));
                let a1 = vreinterpretq_s64_s8(vld1q_s8(p.add(16)));
                let b0 = vreinterpretq_s64_s8(vld1q_s8(q));
                let b1 = vreinterpretq_s64_s8(vld1q_s8(q.add(16)));
                let o = xi.as_mut_ptr().add((s * 8 + k) * 64);
                vst1q_s8(o, vreinterpretq_s8_s64(vzip1q_s64(a0, b0)));
                vst1q_s8(o.add(16), vreinterpretq_s8_s64(vzip1q_s64(a1, b1)));
                vst1q_s8(o.add(32), vreinterpretq_s8_s64(vzip2q_s64(a0, b0)));
                vst1q_s8(o.add(48), vreinterpretq_s8_s64(vzip2q_s64(a1, b1)));
            }
        }
    }
}

/// Q4_0's -8 for SMMLA to start from: for each block of 32 of a and b, [-8 sa, -8 sb, -8 sa, -8 sb].
fn bias(a: &[Groups32], b: &[Groups32], nb: usize, xb: &mut [i32]) {
    for s in 0..nb {
        for k in 0..8 {
            let sa = a[s].sums[k] as i32;
            let sb = b[s].sums[k] as i32;
            let o = &mut xb[(s * 8 + k) * 4..(s * 8 + k) * 4 + 4];
            o[0] = -8 * sa;
            o[2] = -8 * sa;
            o[1] = -8 * sb;
            o[3] = -8 * sb;
        }
    }
}

/// A block's two rows' nibbles as SMMLA's four operands.
#[inline(always)]
unsafe fn operands(q0p: *const u8, q1p: *const u8) -> [int8x16_t; 4] {
    unsafe {
        let m4 = vdupq_n_u8(15);
        let q0 = vreinterpretq_u64_u8(vld1q_u8(q0p));
        let q1 = vreinterpretq_u64_u8(vld1q_u8(q1p));
        let lo = vreinterpretq_u8_u64(vzip1q_u64(q0, q1));
        let hi = vreinterpretq_u8_u64(vzip2q_u64(q0, q1));
        [
            vreinterpretq_s8_u8(vandq_u8(lo, m4)),
            vreinterpretq_s8_u8(vshrq_n_u8(lo, 4)),
            vreinterpretq_s8_u8(vandq_u8(hi, m4)),
            vreinterpretq_s8_u8(vshrq_n_u8(hi, 4)),
        ]
    }
}

/// The eight blocks' dot products for a group of 256 (s[k] = [r0v0 r0v1 r1v0 r1v1]) into acc[pair][half], where lane k
/// of a half is block k (0-3) or 4 + k: fma(d_w x g.d, dot, acc). Pairs 0-3 are r0v0, r0v1, r1v0 and r1v1.
#[inline(always)]
unsafe fn accumulate(
    s: &[int32x4_t; 8],
    d0: [float32x4_t; 2],
    d1: [float32x4_t; 2],
    xa: [float32x4_t; 2],
    xb: [float32x4_t; 2],
    acc: &mut [[float32x4_t; 2]; 4],
) {
    unsafe {
        for h in 0..2 {
            let q = &s[4 * h..4 * h + 4];
            let t0 = vreinterpretq_s64_s32(vtrn1q_s32(q[0], q[1]));
            let t1 = vreinterpretq_s64_s32(vtrn2q_s32(q[0], q[1]));
            let t2 = vreinterpretq_s64_s32(vtrn1q_s32(q[2], q[3]));
            let t3 = vreinterpretq_s64_s32(vtrn2q_s32(q[2], q[3]));
            let p0 = vreinterpretq_s32_s64(vzip1q_s64(t0, t2));
            let p1 = vreinterpretq_s32_s64(vzip1q_s64(t1, t3));
            let p2 = vreinterpretq_s32_s64(vzip2q_s64(t0, t2));
            let p3 = vreinterpretq_s32_s64(vzip2q_s64(t1, t3));
            acc[0][h] = vfmaq_f32(acc[0][h], vmulq_f32(d0[h], xa[h]), vcvtq_f32_s32(p0));
            acc[1][h] = vfmaq_f32(acc[1][h], vmulq_f32(d0[h], xb[h]), vcvtq_f32_s32(p1));
            acc[2][h] = vfmaq_f32(acc[2][h], vmulq_f32(d1[h], xa[h]), vcvtq_f32_s32(p2));
            acc[3][h] = vfmaq_f32(acc[3][h], vmulq_f32(d1[h], xb[h]), vcvtq_f32_s32(p3));
        }
    }
}

#[inline(always)]
unsafe fn fold(a: float32x4_t, b: float32x4_t) -> f32 {
    unsafe {
        let s = vaddq_f32(a, b);
        let t = vadd_f32(vget_low_f32(s), vget_high_f32(s));
        vget_lane_f32::<0>(t) + vget_lane_f32::<1>(t)
    }
}

fn group_scales(x: &Groups32) -> [float32x4_t; 2] {
    // SAFETY: g.d holds 8 floats
    unsafe { [vld1q_f32(x.d.as_ptr()), vld1q_f32(x.d.as_ptr().add(4))] }
}

/// Rows w0 and w1 against U pairs of vectors (pair u: vectors v[2u] and v[2u + 1], interleaved in xi[u], with the -8s
/// in xb[u]). The rows' nibbles are taken apart once for all the pairs, and out[4u..4u + 4] gets r0v0, r0v1, r1v0 and
/// r1v1 of pair u.
#[inline]
#[target_feature(enable = "i8mm")]
unsafe fn rows2<const U: usize>(
    w0: *const u8,
    w1: *const u8,
    v: [&[Groups32]; 4],
    xi: [&[i8]; 2],
    xb: [&[i32]; 2],
    nb: usize,
    out: &mut [f32; 8],
) {
    unsafe {
        let mut acc = [[[vdupq_n_f32(0.0); 2]; 4]; U];
        for s in 0..nb {
            let (r0, r1) = (w0.add(s * 8 * Q4_0_BLOCK), w1.add(s * 8 * Q4_0_BLOCK));
            let mut dots = [[vdupq_n_s32(0); 8]; U];
            for k in 0..8 {
                let i = s * 8 + k;
                let o = operands(r0.add(k * Q4_0_BLOCK + 2), r1.add(k * Q4_0_BLOCK + 2));
                for u in 0..U {
                    let g = xi[u].as_ptr().add(i * 64);
                    let d = smmla(vld1q_s32(xb[u].as_ptr().add(4 * i)), o[0], vld1q_s8(g));
                    let d = smmla(d, o[1], vld1q_s8(g.add(16)));
                    let d = smmla(d, o[2], vld1q_s8(g.add(32)));
                    dots[u][k] = smmla(d, o[3], vld1q_s8(g.add(48)));
                }
            }
            let (a0, a1) = q4_0_scales(r0);
            let (b0, b1) = q4_0_scales(r1);
            for u in 0..U {
                accumulate(
                    &dots[u],
                    [a0, a1],
                    [b0, b1],
                    group_scales(v[2 * u].get_unchecked(s)),
                    group_scales(v[2 * u + 1].get_unchecked(s)),
                    &mut acc[u],
                );
            }
        }
        for u in 0..U {
            for p in 0..4 {
                out[4 * u + p] = fold(acc[u][p][0], acc[u][p][1]);
            }
        }
    }
}

/// Q4_0 weights (row r begins at byte r * row_bytes of w) times nv vectors (vector v = the nb blocks from x[v * nb]);
/// output (v, r) goes to y[v * y_stride + r]. Four vectors at a time (two pairs per pass of a pair of rows), then two,
/// by SMMLA; an odd row, and any vectors left over, go through the SDOT kernel q4_0_matmul (with the same bits).
///
/// # Safety
/// FEAT_I8MM and FEAT_DotProd; y is valid for writes at v * y_stride + r, and nothing else touches those elements
/// during the call. (w holds rows * row_bytes, x holds nv * nb blocks, and nb <= MAX_NB: asserted.)
#[target_feature(enable = "i8mm,dotprod")]
pub unsafe fn q4_0_rows_multi(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(
        nv > 0 && x.len() == nv * nb && nb <= MAX_NB && row_bytes >= nb * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes
    );
    let vec = |i: usize| &x[i * nb..(i + 1) * nb];
    // the interleaved activations and -8s of two pairs, in each thread's own buffers, made once (there is a call per
    // chunk of rows, so allocating them every time would add up)
    thread_local! {
        static BUF: std::cell::RefCell<(Vec<i8>, Vec<i8>, Vec<i32>, Vec<i32>)> = const {
            std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new()))
        };
    }
    BUF.with_borrow_mut(|(xi0, xi1, xb0, xb1)| {
        for b in [&mut *xi0, &mut *xi1] {
            if b.len() < MAX_NB * 512 {
                b.resize(MAX_NB * 512, 0);
            }
        }
        for b in [&mut *xb0, &mut *xb1] {
            if b.len() < MAX_NB * 32 {
                b.resize(MAX_NB * 32, 0);
            }
        }
        let xi = [&mut xi0[..nb * 512], &mut xi1[..nb * 512]];
        let xb = [&mut xb0[..nb * 32], &mut xb1[..nb * 32]];
        let r2 = rows / 2 * 2;
        let mut v = 0;
        unsafe {
            while nv >= 2 && rows >= 2 && v + 1 < nv {
                let four = v + 3 < nv;
                let np = if four { 2 } else { 1 };
                for u in 0..np {
                    let (a, b) = (vec(v + 2 * u), vec(v + 2 * u + 1));
                    interleave(a, b, nb, xi[u]);
                    bias(a, b, nb, xb[u]);
                }
                let v4 = [vec(v), vec(v + 1), vec(if four { v + 2 } else { v }), vec(if four { v + 3 } else { v + 1 })];
                let mut o = [0f32; 8];
                for i in (0..r2).step_by(2) {
                    let (w0, w1) = (w.as_ptr().add(i * row_bytes), w.as_ptr().add((i + 1) * row_bytes));
                    if four {
                        rows2::<2>(w0, w1, v4, [&*xi[0], &*xi[1]], [&*xb[0], &*xb[1]], nb, &mut o);
                    } else {
                        rows2::<1>(w0, w1, v4, [&*xi[0], &*xi[0]], [&*xb[0], &*xb[0]], nb, &mut o);
                    }
                    for u in 0..np {
                        let yp = y.add(i + (v + 2 * u) * y_stride);
                        *yp = o[4 * u];
                        *yp.add(y_stride) = o[4 * u + 1];
                        *yp.add(1) = o[4 * u + 2];
                        *yp.add(1 + y_stride) = o[4 * u + 3];
                    }
                }
                let g = 2 * np;
                if r2 < rows {
                    // an odd row: these vectors against that one row
                    q4_0_matmul(
                        &w[r2 * row_bytes..],
                        row_bytes,
                        1,
                        &x[v * nb..(v + g) * nb],
                        g,
                        y.add(r2 + v * y_stride),
                        y_stride,
                    );
                }
                v += g;
            }
            if v < nv {
                q4_0_matmul(w, row_bytes, rows, &x[v * nb..], nv - v, y.add(v * y_stride), y_stride);
            }
        }
    });
}
