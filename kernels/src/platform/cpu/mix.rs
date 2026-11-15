//! mix.rs - products for the hyper-connection mixers, when the file keeps them quantised.
//!
//! ggml-org's Flash Next IQ4_NL keeps Q8_0 attention mixers and IQ4_NL FFN mixers. Widening such a matrix to bf16 and
//! multiplying bf16 by bf16 in f64 costs two to 3.6 times the bytes per token, and rounds every weight and every input
//! to 8 bits of mantissa. Here the weights stay as stored (exact: d * q per 32-weight block), and the input goes in as
//! three int8 planes per block, x = c * (65536 q1 + 256 q2 + q3) with c = amax / 127 / 65536. A value's error is then
//! at most amax / 16.6M, about 24 bits relative to the block's largest value. (Two planes would give 16 bits, which is
//! worse than bf16 where a block's outlier sets its scale, since bf16 keeps an exponent per value.) The integer
//! products are exact (SDOT), and a block's are scaled as f(dot q1) * d c 65536 + f(256 dot q2 + dot q3) * d c into
//! four f32 lanes (fused), with the lanes added at the end as (0 + 1) + (2 + 3), i.e. FADDP twice.
//!
//! Lanes: sdot puts weights 4k..4k+3 (and 16+4k..16+4k+3, from the block's second half) into lane k. The portable
//! version keeps the same lanes and the same float steps, so both give the same bits (the tests check this on ARM).

use crate::f16_to_f32;

/// The ggml types the mixers' quantised path takes, and their 32-weight block sizes in bytes (an fp16 scale, then the
/// codes).
pub const Q8_0: u32 = 8;
pub const IQ4_NL: u32 = 20;

pub fn block_bytes(ty: u32) -> Option<usize> {
    match ty {
        Q8_0 => Some(34),
        IQ4_NL => Some(18),
        _ => None,
    }
}

/// IQ4_NL's sixteen code values (ggml's kvalues_iq4nl).
pub const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// One block of 32 inputs: x[i] ~= c * (65536 q1[i] + 256 q2[i] + q3[i]), with c = amax / 127 / 65536 (q1 in -127..127,
/// q2 and q3 in -128..127).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockX {
    pub c: f32,
    pub q1: [i8; 32],
    pub q2: [i8; 32],
    pub q3: [i8; 32],
}

impl Default for BlockX {
    fn default() -> Self {
        BlockX { c: 0.0, q1: [0; 32], q2: [0; 32], q3: [0; 32] }
    }
}

/// Quantises x (a multiple of 32 long) into blocks: q1 = round(x / s), then the remainder in 256ths of s, then that
/// remainder's remainder in 65536ths (the remainders are taken in f64, which is exact enough not to add to the error).
pub fn quantize(x: &[f32], out: &mut [BlockX]) {
    assert!(x.len().is_multiple_of(32) && out.len() >= x.len() / 32);
    fn step(v: f64) -> (f64, f64) {
        let q = v.round_ties_even();
        (q, v - q)
    }
    for (xb, o) in x.chunks_exact(32).zip(out.iter_mut()) {
        let amax = xb.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        if !(amax > 0.0) || !amax.is_finite() {
            // all zero (or not finite, in which case the product is garbage either way, as it would be in f32)
            *o = BlockX { c: if amax.is_finite() { 0.0 } else { f32::NAN }, ..Default::default() };
            continue;
        }
        let inv = 127.0 / amax as f64;
        for i in 0..32 {
            let v = xb[i] as f64 * inv;
            let (mut q1, r1) = step(v);
            let (mut q2, r2) = step(r1 * 256.0);
            let (mut q3, _) = step(r2 * 256.0);
            // a plane at +128 (a remainder is at most half a step, so at most 128) is carried into the plane above (q1
            // never needs this, since |v| <= 127)
            if q3 > 127.0 {
                q3 -= 256.0;
                q2 += 1.0;
            }
            if q2 > 127.0 && q1 < 127.0 {
                q2 -= 256.0;
                q1 += 1.0;
            }
            o.q1[i] = q1.clamp(-127.0, 127.0) as i8;
            o.q2[i] = q2.clamp(-128.0, 127.0) as i8;
            o.q3[i] = q3.clamp(-128.0, 127.0) as i8;
        }
        o.c = (amax as f64 / 127.0 / 65536.0) as f32;
    }
}

/// A block's 32 weights as int8 codes (Q8_0: as stored; IQ4_NL: the table's values), and its fp16 scale.
#[inline]
fn codes(ty: u32, blk: &[u8]) -> ([i8; 32], f32) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let mut q = [0i8; 32];
    match ty {
        Q8_0 => {
            for i in 0..32 {
                q[i] = blk[2 + i] as i8;
            }
        }
        _ => {
            for i in 0..16 {
                q[i] = KVALUES_IQ4NL[(blk[2 + i] & 15) as usize];
                q[16 + i] = KVALUES_IQ4NL[(blk[2 + i] >> 4) as usize];
            }
        }
    }
    (q, d)
}

/// One row (nb blocks of type ty) against nb input blocks: the portable version, with the same bits as the NEON kernel.
pub fn dot_scalar(ty: u32, w: &[u8], x: &[BlockX]) -> f32 {
    let bs = block_bytes(ty).expect("mixer type");
    let mut acc = [0.0f32; 4];
    for (b, xb) in x.iter().enumerate() {
        let (q, d) = codes(ty, &w[b * bs..(b + 1) * bs]);
        let mut i1 = [0i32; 4];
        let mut i2 = [0i32; 4];
        let mut i3 = [0i32; 4];
        for k in 0..4 {
            for j in 0..4 {
                for half in [0, 16] {
                    let i = half + 4 * k + j;
                    i1[k] += q[i] as i32 * xb.q1[i] as i32;
                    i2[k] += q[i] as i32 * xb.q2[i] as i32;
                    i3[k] += q[i] as i32 * xb.q3[i] as i32;
                }
            }
        }
        let sc = d * xb.c;
        let sc1 = sc * 65536.0;
        for k in 0..4 {
            let c = (i2[k] << 8) + i3[k];
            acc[k] = (i1[k] as f32).mul_add(sc1, acc[k]);
            acc[k] = (c as f32).mul_add(sc, acc[k]);
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// One row against f32 inputs, in f64 with the weights exact: the reference for the quantised products.
pub fn dot_exact(ty: u32, w: &[u8], x: &[f32]) -> f64 {
    let bs = block_bytes(ty).expect("mixer type");
    let mut acc = 0.0f64;
    for (b, xb) in x.chunks_exact(32).enumerate() {
        let (q, d) = codes(ty, &w[b * bs..(b + 1) * bs]);
        for i in 0..32 {
            acc += d as f64 * q[i] as f64 * xb[i] as f64;
        }
    }
    acc
}

/// One row, using the fastest kernel this CPU has.
pub fn dot(ty: u32, w: &[u8], x: &[BlockX]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: the feature is present, and neon::dot checks the lengths
        return unsafe { neon::dot(ty, w, x) };
    }
    dot_scalar(ty, w, x)
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::{BlockX, IQ4_NL, KVALUES_IQ4NL, Q8_0, block_bytes};
    use core::arch::aarch64::*;
    use core::arch::asm;

    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        unsafe {
            asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
                options(pure, nomem, nostack));
        }
        acc
    }

    /// fp16 to fp32 (FCVTL, base Advanced SIMD; exact, as f16_to_f32 is).
    #[inline]
    unsafe fn h2f(p: *const u8) -> f32 {
        unsafe {
            let h = vdup_n_u16(u16::from_le_bytes([*p, *p.add(1)]));
            let o: float32x4_t;
            asm!("fcvtl {o:v}.4s, {h:v}.4h", h = in(vreg) h, o = out(vreg) o, options(pure, nomem, nostack));
            vgetq_lane_f32(o, 0)
        }
    }

    #[inline(always)]
    unsafe fn row<const TY: u32>(w: *const u8, x: &[BlockX]) -> f32 {
        unsafe { hsum(blocks::<TY>(vdupq_n_f32(0.0), w, x)) }
    }

    #[inline(always)]
    unsafe fn hsum(acc: float32x4_t) -> f32 {
        unsafe {
            let s = vpaddq_f32(acc, acc);
            vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1)
        }
    }

    /// Adds a row's blocks x (with the row's bytes from w) into the row's accumulator acc, so a row done in pieces
    /// gives the bits of the whole row.
    #[inline(always)]
    unsafe fn blocks<const TY: u32>(mut acc: float32x4_t, w: *const u8, x: &[BlockX]) -> float32x4_t {
        unsafe {
            let bs = if TY == Q8_0 { 34 } else { 18 };
            let kv = vld1q_s8(KVALUES_IQ4NL.as_ptr());
            let m4 = vdupq_n_u8(15);
            let z = vdupq_n_s32(0);
            for (b, xb) in x.iter().enumerate() {
                let p = w.add(b * bs);
                let (w0, w1) = if TY == Q8_0 {
                    (vld1q_s8(p.add(2) as *const i8), vld1q_s8(p.add(18) as *const i8))
                } else {
                    let q = vld1q_u8(p.add(2));
                    (vqtbl1q_s8(kv, vandq_u8(q, m4)), vqtbl1q_s8(kv, vshrq_n_u8(q, 4)))
                };
                let i1 = sdot(sdot(z, w0, vld1q_s8(xb.q1.as_ptr())), w1, vld1q_s8(xb.q1.as_ptr().add(16)));
                let i2 = sdot(sdot(z, w0, vld1q_s8(xb.q2.as_ptr())), w1, vld1q_s8(xb.q2.as_ptr().add(16)));
                let i3 = sdot(sdot(z, w0, vld1q_s8(xb.q3.as_ptr())), w1, vld1q_s8(xb.q3.as_ptr().add(16)));
                let c = vaddq_s32(vshlq_n_s32(i2, 8), i3);
                let sc = h2f(p) * xb.c;
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(i1), sc * 65536.0);
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(c), sc);
            }
            acc
        }
    }

    /// One row: as dot_scalar, with the same bits.
    ///
    /// # Safety
    /// FEAT_DotProd.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot(ty: u32, w: &[u8], x: &[BlockX]) -> f32 {
        let bs = block_bytes(ty).expect("mixer type");
        assert!(w.len() >= bs * x.len());
        unsafe {
            match ty {
                Q8_0 => row::<Q8_0>(w.as_ptr(), x),
                IQ4_NL => row::<IQ4_NL>(w.as_ptr(), x),
                _ => unreachable!(),
            }
        }
    }

    /// Rows r0..r1 (row r at w[r * rb..]) against nv input vectors of nb blocks (vector j = x[j * nb..]), into y[j *
    /// ys + r]. Each is a row's dot product, with the same bits. With kb > 0 and at most 256 (row, vector) pairs, the
    /// rows are walked kb blocks at a time, with every pair's accumulator kept between pieces, so each piece's inputs
    /// and rows stay in L1 instead of every row re-reading all nv inputs from L2 (a prompt's 16 inputs to a 10240-wide
    /// row are 512 KB).
    ///
    /// # Safety
    /// FEAT_DotProd; y must be valid for those writes, with nothing else touching them in the meantime.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "dotprod")]
    pub unsafe fn rows(
        ty: u32,
        w: &[u8],
        rb: usize,
        r0: usize,
        r1: usize,
        x: &[BlockX],
        nb: usize,
        nv: usize,
        y: *mut f32,
        ys: usize,
        kb: usize,
    ) {
        let bs = block_bytes(ty).expect("mixer type");
        assert!(rb >= bs * nb && w.len() >= rb * r1 && x.len() >= nb * nv);
        const MAXP: usize = 256;
        if kb > 0 && kb < nb && (r1 - r0) * nv <= MAXP {
            unsafe {
                let mut acc = [vdupq_n_f32(0.0); MAXP];
                let mut b0 = 0;
                while b0 < nb {
                    let b1 = (b0 + kb).min(nb);
                    for r in r0..r1 {
                        let p = w.as_ptr().add(r * rb + b0 * bs);
                        let a = &mut acc[(r - r0) * nv..(r - r0 + 1) * nv];
                        for (j, a) in a.iter_mut().enumerate() {
                            let xs = &x[j * nb + b0..j * nb + b1];
                            *a = if ty == Q8_0 { blocks::<Q8_0>(*a, p, xs) } else { blocks::<IQ4_NL>(*a, p, xs) };
                        }
                    }
                    b0 = b1;
                }
                for r in r0..r1 {
                    for j in 0..nv {
                        *y.add(j * ys + r) = hsum(acc[(r - r0) * nv + j]);
                    }
                }
            }
            return;
        }
        unsafe {
            for r in r0..r1 {
                let p = w.as_ptr().add(r * rb);
                for j in 0..nv {
                    let xs = &x[j * nb..(j + 1) * nb];
                    *y.add(j * ys + r) = if ty == Q8_0 { row::<Q8_0>(p, xs) } else { row::<IQ4_NL>(p, xs) };
                }
            }
        }
    }

    /// Bytes per block of a vector pair in interleave_pair's layout: three planes, four groups of 16.
    pub const PAIR_BLOCK: usize = 192;

    /// Vectors a and c (nb blocks each) arranged for SMMLA into o (nb x PAIR_BLOCK bytes). For each block and each
    /// plane (q1, q2, q3), group l holds a's values 4l..4l+3 and 16+4l..16+4l+3, then c's same eight: exactly the 8
    /// products that SDOT adds into lane l in `blocks`.
    pub fn interleave_pair(a: &[BlockX], c: &[BlockX], o: &mut [i8]) {
        let nb = a.len();
        assert!(c.len() == nb && o.len() >= nb * PAIR_BLOCK);
        // per plane: the 4-byte words of values 0-15 and 16-31 zipped (each vector's groups 0,1 and 2,3, 8 bytes a
        // group), then the two vectors' groups zipped as 8-byte halves: the layout pair_weights gives the rows
        // SAFETY: base Advanced SIMD; every load is 16 bytes inside a block's plane and every store 16 inside o's block
        unsafe {
            for b in 0..nb {
                let (xa, xc) = (&a[b], &c[b]);
                let op = o.as_mut_ptr().add(b * PAIR_BLOCK);
                for (s, (qa, qc)) in [(&xa.q1, &xc.q1), (&xa.q2, &xc.q2), (&xa.q3, &xc.q3)].into_iter().enumerate() {
                    let (al, ah) = (vld1q_s32(qa.as_ptr() as *const i32), vld1q_s32(qa.as_ptr().add(16) as *const i32));
                    let (cl, ch) = (vld1q_s32(qc.as_ptr() as *const i32), vld1q_s32(qc.as_ptr().add(16) as *const i32));
                    let (a1, a2) =
                        (vreinterpretq_s64_s32(vzip1q_s32(al, ah)), vreinterpretq_s64_s32(vzip2q_s32(al, ah)));
                    let (c1, c2) =
                        (vreinterpretq_s64_s32(vzip1q_s32(cl, ch)), vreinterpretq_s64_s32(vzip2q_s32(cl, ch)));
                    let p = op.add(s * 64);
                    vst1q_s8(p, vreinterpretq_s8_s64(vzip1q_s64(a1, c1)));
                    vst1q_s8(p.add(16), vreinterpretq_s8_s64(vzip2q_s64(a1, c1)));
                    vst1q_s8(p.add(32), vreinterpretq_s8_s64(vzip1q_s64(a2, c2)));
                    vst1q_s8(p.add(48), vreinterpretq_s8_s64(vzip2q_s64(a2, c2)));
                }
            }
        }
    }

    /// interleave_pair for vectors (0, 1), (2, 3), .. of x (n vectors of nb blocks), pair after pair. An odd last
    /// vector is left out (rows_mm takes it from x).
    pub fn interleave_pairs(x: &[BlockX], nb: usize, n: usize) -> Vec<i8> {
        let mut o = vec![0i8; n / 2 * nb * PAIR_BLOCK];
        for p in 0..n / 2 {
            interleave_pair(
                &x[2 * p * nb..(2 * p + 1) * nb],
                &x[(2 * p + 1) * nb..(2 * p + 2) * nb],
                &mut o[p * nb * PAIR_BLOCK..(p + 1) * nb * PAIR_BLOCK],
            );
        }
        o
    }

    /// A block's 32 weights as two int8x16 (0-15, 16-31), as `blocks` reads them.
    #[inline(always)]
    unsafe fn weights<const TY: u32>(p: *const u8) -> (int8x16_t, int8x16_t) {
        unsafe {
            if TY == Q8_0 {
                (vld1q_s8(p.add(2) as *const i8), vld1q_s8(p.add(18) as *const i8))
            } else {
                let kv = vld1q_s8(KVALUES_IQ4NL.as_ptr());
                let q = vld1q_u8(p.add(2));
                (vqtbl1q_s8(kv, vandq_u8(q, vdupq_n_u8(15))), vqtbl1q_s8(kv, vshrq_n_u8(q, 4)))
            }
        }
    }

    /// The SMMLA operands of one block of rows ra and rb: g[l] = ra's lane-l eight values, then rb's (values 4l..4l+3
    /// and 16+4l..16+4l+3, the 8 that SDOT adds into lane l in `blocks`).
    #[inline(always)]
    unsafe fn pair_weights(ty: u32, pa: *const u8, pb: *const u8) -> [int8x16_t; 4] {
        unsafe {
            let ((a0, a1), (c0, c1)) = if ty == Q8_0 {
                (weights::<Q8_0>(pa), weights::<Q8_0>(pb))
            } else {
                (weights::<IQ4_NL>(pa), weights::<IQ4_NL>(pb))
            };
            let za1 = vreinterpretq_s64_s32(vzip1q_s32(vreinterpretq_s32_s8(a0), vreinterpretq_s32_s8(a1)));
            let za2 = vreinterpretq_s64_s32(vzip2q_s32(vreinterpretq_s32_s8(a0), vreinterpretq_s32_s8(a1)));
            let zb1 = vreinterpretq_s64_s32(vzip1q_s32(vreinterpretq_s32_s8(c0), vreinterpretq_s32_s8(c1)));
            let zb2 = vreinterpretq_s64_s32(vzip2q_s32(vreinterpretq_s32_s8(c0), vreinterpretq_s32_s8(c1)));
            [
                vreinterpretq_s8_s64(vzip1q_s64(za1, zb1)),
                vreinterpretq_s8_s64(vzip2q_s64(za1, zb1)),
                vreinterpretq_s8_s64(vzip1q_s64(za2, zb2)),
                vreinterpretq_s8_s64(vzip2q_s64(za2, zb2)),
            ]
        }
    }

    /// rows with SMMLA: the same results bit for bit, two rows and two vectors at a time. xi is interleave_pair of
    /// these nv vectors, pair after pair (from the first). Each of the four (row, vector) pairs of a tile keeps its 4
    /// SDOT lanes as lane k of 4 accumulators acc[l] (k = ra.va, ra.vb, rb.va, rb.vb; l = SDOT lane), which is the
    /// order SMMLA gives, so nothing is transposed: SMMLA's sum of 8 products is the integer SDOT puts in lane l, the
    /// scale is the same f32 product lane by lane, the two FMAs are `blocks`' own in the same order, and the end sum
    /// (acc0 + acc1) + (acc2 + acc3) is hsum's association. Blocks go in order, kb at a time (all at once with kb = 0)
    /// with the accumulators kept between pieces, so every bit matches rows. An odd last row or vector goes to rows.
    ///
    /// # Safety
    /// FEAT_I8MM and FEAT_DotProd; y as for rows.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "i8mm,dotprod")]
    pub unsafe fn rows_mm(
        ty: u32,
        w: &[u8],
        rb: usize,
        r0: usize,
        r1: usize,
        x: &[BlockX],
        xi: &[i8],
        nb: usize,
        nv: usize,
        y: *mut f32,
        ys: usize,
        kb: usize,
    ) {
        let nr = r1 - r0;
        if nv < 2 || nr < 2 {
            return unsafe { rows(ty, w, rb, r0, r1, x, nb, nv, y, ys, kb) };
        }
        let bs = block_bytes(ty).expect("mixer type");
        assert!(rb >= bs * nb && w.len() >= rb * r1 && x.len() >= nb * nv && xi.len() >= nv / 2 * nb * PAIR_BLOCK);
        let (nrp, nvp) = (nr / 2, nv / 2);
        // a piece's decoded row pair: at most KBMAX blocks
        const KBMAX: usize = 64;
        let piece = if kb == 0 { nb } else { kb }.min(KBMAX);
        // the accumulators of a group of row pairs (4 per row pair and vector pair)
        const MAXQ: usize = 1024;
        let gp = (MAXQ / (4 * nvp)).max(1);
        unsafe {
            let z = vdupq_n_s32(0);
            let mut acc = [vdupq_n_f32(0.0); MAXQ];
            let mut wg = [[vdupq_n_s8(0); 4]; KBMAX];
            let mut dg = [[0.0f32; 2]; KBMAX];
            let mut g0 = 0;
            while g0 < nrp {
                let g1 = (g0 + gp).min(nrp);
                for a in acc[..(g1 - g0) * nvp * 4].iter_mut() {
                    *a = vdupq_n_f32(0.0);
                }
                let mut b0 = 0;
                while b0 < nb {
                    let b1 = (b0 + piece).min(nb);
                    for rp in g0..g1 {
                        let (ra, rb_) = (r0 + 2 * rp, r0 + 2 * rp + 1);
                        for b in b0..b1 {
                            let (pa, pb) = (w.as_ptr().add(ra * rb + b * bs), w.as_ptr().add(rb_ * rb + b * bs));
                            wg[b - b0] = pair_weights(ty, pa, pb);
                            dg[b - b0] = [h2f(pa), h2f(pb)];
                        }
                        for vp in 0..nvp {
                            let (ja, jb) = (2 * vp, 2 * vp + 1);
                            let ai = ((rp - g0) * nvp + vp) * 4;
                            let mut q = [acc[ai], acc[ai + 1], acc[ai + 2], acc[ai + 3]];
                            for b in b0..b1 {
                                let g = &wg[b - b0];
                                let d = dg[b - b0];
                                let xp = xi.as_ptr().add((vp * nb + b) * PAIR_BLOCK);
                                let (ca, cb) = (x[ja * nb + b].c, x[jb * nb + b].c);
                                // [da ca, da cb, db ca, db cb]: each lane the f32 product blocks makes for its pair
                                let dv = vcombine_f32(vdup_n_f32(d[0]), vdup_n_f32(d[1]));
                                let cv = vcombine_f32(
                                    vset_lane_f32::<1>(cb, vdup_n_f32(ca)),
                                    vset_lane_f32::<1>(cb, vdup_n_f32(ca)),
                                );
                                let scv = vmulq_f32(dv, cv);
                                let sc1 = vmulq_n_f32(scv, 65536.0);
                                for (l, ql) in q.iter_mut().enumerate() {
                                    let i1 = crate::neon::smmla(z, g[l], vld1q_s8(xp.add(l * 16)));
                                    let i2 = crate::neon::smmla(z, g[l], vld1q_s8(xp.add(64 + l * 16)));
                                    let i3 = crate::neon::smmla(z, g[l], vld1q_s8(xp.add(128 + l * 16)));
                                    let c = vaddq_s32(vshlq_n_s32(i2, 8), i3);
                                    *ql = vfmaq_f32(*ql, vcvtq_f32_s32(i1), sc1);
                                    *ql = vfmaq_f32(*ql, vcvtq_f32_s32(c), scv);
                                }
                            }
                            acc[ai..ai + 4].copy_from_slice(&q);
                        }
                    }
                    b0 = b1;
                }
                for rp in g0..g1 {
                    let (ra, rb_) = (r0 + 2 * rp, r0 + 2 * rp + 1);
                    for vp in 0..nvp {
                        let ai = ((rp - g0) * nvp + vp) * 4;
                        let s = vaddq_f32(vaddq_f32(acc[ai], acc[ai + 1]), vaddq_f32(acc[ai + 2], acc[ai + 3]));
                        let (ja, jb) = (2 * vp, 2 * vp + 1);
                        *y.add(ja * ys + ra) = vgetq_lane_f32::<0>(s);
                        *y.add(jb * ys + ra) = vgetq_lane_f32::<1>(s);
                        *y.add(ja * ys + rb_) = vgetq_lane_f32::<2>(s);
                        *y.add(jb * ys + rb_) = vgetq_lane_f32::<3>(s);
                    }
                }
                g0 = g1;
            }
            // an odd last vector (the paired rows) and an odd last row (every vector), as rows does them
            if nv % 2 == 1 {
                rows(ty, w, rb, r0, r0 + 2 * nrp, &x[(nv - 1) * nb..nv * nb], nb, 1, y.add((nv - 1) * ys), ys, kb);
            }
            if nr % 2 == 1 {
                rows(ty, w, rb, r1 - 1, r1, x, nb, nv, y, ys, kb);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// The quantised input is within amax / 16.6M of every value (plus rounding in the check itself).
    #[test]
    fn input_error_bound() {
        let mut s = 7u64;
        let x: Vec<f32> = (0..32 * 64).map(|i| lcg(&mut s) * (1.0 + (i % 7) as f32)).collect();
        let mut b = vec![BlockX::default(); 64];
        quantize(&x, &mut b);
        for (xb, ob) in x.chunks_exact(32).zip(&b) {
            let amax = xb.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            for i in 0..32 {
                let y = ob.c as f64 * (65536.0 * ob.q1[i] as f64 + 256.0 * ob.q2[i] as f64 + ob.q3[i] as f64);
                // half a step of the last plane, plus c rounded to f32 (one f32 ulp of the value, as x itself has)
                let bound = amax as f64 / 127.0 / 65536.0 / 2.0 * 1.001 + xb[i].abs() as f64 * 1.2e-7;
                assert!((y - xb[i] as f64).abs() <= bound, "{} vs {}", y, xb[i]);
            }
        }
    }

    /// Against f64 arithmetic on the exact weights and inputs: far inside what bf16 rounding gives.
    #[test]
    fn dot_close_to_exact() {
        let mut s = 11u64;
        for ty in [Q8_0, IQ4_NL] {
            let bs = block_bytes(ty).unwrap();
            let nb = 80;
            let mut w = vec![0u8; nb * bs];
            for b in 0..nb {
                // a normal fp16 value: the f32's top mantissa bits (truncated), with the exponent rebased
                let f = (0.01 + 0.002 * (b % 5) as f32).to_bits();
                let d = (((f >> 23) & 0xff) + 15 - 127) << 10 | ((f >> 13) & 0x3ff);
                let d = d as u16;
                w[b * bs..b * bs + 2].copy_from_slice(&d.to_le_bytes());
                for i in 2..bs {
                    w[b * bs + i] = (lcg(&mut s) * 127.0) as i8 as u8;
                }
            }
            let x: Vec<f32> = (0..nb * 32).map(|_| lcg(&mut s)).collect();
            let mut xb = vec![BlockX::default(); nb];
            quantize(&x, &mut xb);
            let mut exact = 0.0f64;
            for b in 0..nb {
                let (q, d) = codes(ty, &w[b * bs..(b + 1) * bs]);
                for i in 0..32 {
                    exact += d as f64 * q[i] as f64 * x[b * 32 + i] as f64;
                }
            }
            let got = dot_scalar(ty, &w, &xb) as f64;
            let scale: f64 = (0..nb)
                .map(|b| {
                    let (q, d) = codes(ty, &w[b * bs..(b + 1) * bs]);
                    (0..32).map(|i| (d as f64 * q[i] as f64 * x[b * 32 + i] as f64).abs()).sum::<f64>()
                })
                .sum();
            assert!((got - exact).abs() < 2e-6 * scale, "ty {ty}: {got} vs {exact} (scale {scale})");
            assert_eq!(dot(ty, &w, &xb).to_bits(), dot_scalar(ty, &w, &xb).to_bits());
        }
    }

    /// neon::rows gives every row the same bits as dot_scalar, for both types, a row range that is not a multiple of
    /// four, and two input vectors.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rows_same_bits_as_scalar() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut s = 23u64;
        for ty in [Q8_0, IQ4_NL] {
            let bs = block_bytes(ty).unwrap();
            for (nb, nr, nv, kb) in [(10, 11, 2, 0), (37, 5, 17, 8), (37, 3, 4, 16)] {
                let rb = nb * bs + 6; // a row stride wider than the row, as a padded tensor would have
                let mut w = vec![0u8; nr * rb];
                for r in 0..nr {
                    for b in 0..nb {
                        let o = r * rb + b * bs;
                        let f = (0.005 + 0.003 * ((r + b) % 7) as f32).to_bits();
                        let d = ((((f >> 23) & 0xff) - 127 + 15) << 10 | ((f >> 13) & 0x3ff)) as u16;
                        w[o..o + 2].copy_from_slice(&d.to_le_bytes());
                        for i in 2..bs {
                            w[o + i] = (lcg(&mut s) * 127.0) as i8 as u8;
                        }
                    }
                }
                let x: Vec<f32> = (0..nv * nb * 32).map(|_| lcg(&mut s) * 3.0).collect();
                let mut xb = vec![BlockX::default(); nv * nb];
                quantize(&x, &mut xb);
                let (r0, r1) = (1, nr);
                let mut y = vec![f32::NAN; nv * nr];
                // SAFETY: dotprod was checked above, and y holds nv x nr outputs
                unsafe { neon::rows(ty, &w, rb, r0, r1, &xb, nb, nv, y.as_mut_ptr(), nr, kb) };
                for j in 0..nv {
                    for r in r0..r1 {
                        let want = dot_scalar(ty, &w[r * rb..r * rb + nb * bs], &xb[j * nb..(j + 1) * nb]);
                        assert_eq!(y[j * nr + r].to_bits(), want.to_bits(), "ty {ty} kb {kb} row {r} vector {j}");
                    }
                }
            }
        }
    }

    /// neon::rows_mm gives the same bits as neon::rows: both types, odd and even row and vector counts, pieces of kb
    /// blocks that do and do not divide the row, and the shapes of Flash Next's mixers (10240 x 320 both ways).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn rows_mm_same_bits_as_rows() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") || !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }
        let mut s = 29u64;
        for ty in [Q8_0, IQ4_NL] {
            let bs = block_bytes(ty).unwrap();
            // (the up product: 10 blocks, 64-row tasks, 16 tokens, kb >= nb; and kb = 0, and a group split of the rows)
            for (nb, nr, nv, kb) in [
                (37, 6, 4, 8),
                (37, 7, 5, 16),
                (40, 9, 16, 16),
                (320, 10, 16, 16),
                (10, 64, 3, 4),
                (12, 2, 2, 5),
                (10, 64, 16, 16),
                (10, 63, 16, 0),
                (37, 7, 5, 0),
                (100, 300, 16, 16),
                (70, 4, 2, 64),
            ] {
                let rb = nb * bs + 6;
                let mut w = vec![0u8; (nr + 1) * rb];
                for r in 0..nr + 1 {
                    for b in 0..nb {
                        let o = r * rb + b * bs;
                        let f = (0.005 + 0.003 * ((r + b) % 7) as f32).to_bits();
                        let d = ((((f >> 23) & 0xff) - 127 + 15) << 10 | ((f >> 13) & 0x3ff)) as u16;
                        w[o..o + 2].copy_from_slice(&d.to_le_bytes());
                        for i in 2..bs {
                            w[o + i] = (lcg(&mut s) * 127.0) as i8 as u8;
                        }
                    }
                }
                let x: Vec<f32> = (0..nv * nb * 32).map(|_| lcg(&mut s) * 3.0).collect();
                let mut xb = vec![BlockX::default(); nv * nb];
                quantize(&x, &mut xb);
                let xi = neon::interleave_pairs(&xb, nb, nv);
                let (r0, r1) = (1, nr + 1);
                let ys = nr + 1;
                let mut want = vec![f32::NAN; nv * ys];
                let mut got = vec![f32::NAN; nv * ys];
                // SAFETY: both features were checked above, and each y holds nv x ys outputs
                unsafe {
                    neon::rows(ty, &w, rb, r0, r1, &xb, nb, nv, want.as_mut_ptr(), ys, kb);
                    neon::rows_mm(ty, &w, rb, r0, r1, &xb, &xi, nb, nv, got.as_mut_ptr(), ys, kb);
                }
                for j in 0..nv {
                    for r in r0..r1 {
                        assert_eq!(
                            got[j * ys + r].to_bits(),
                            want[j * ys + r].to_bits(),
                            "ty {ty} nb {nb} kb {kb} row {r} vector {j} of {nv}"
                        );
                    }
                }
            }
        }
    }
}
