//! x86mm.rs - rows against several vectors on x86-64 (AVX2, FMA, F16C, and VNNI where the CPU has it: AVX-VNNI, or
//! AVX512-VNNI on 256-bit registers).
//!
//! These are the NEON passes of neon.rs (q8_0_dotv, q4_1_dotv, iq4nl_dotv, q45k_vec_dot, q3k_dotv, iq4xs_dotv,
//! iq3s_dotv, q6k_matmul) in 256-bit registers. Each row's block is unpacked once for several vectors. For each vector,
//! each group's dot product is formed exactly as an integer, the group sums are gathered one per lane (the NEON vpaddq
//! tree, 8 wide) and weighted by the group scales as int32 (as the NEON vmulq / vmlaq do), and then the float steps the
//! bits require are applied.
//!
//! Q8_0 / Q4_1 / Q5_1 / IQ4_NL (q8_0_dotv, q4_1_dotv, iq4nl_dotv): lane k holds block position k and its exact sum,
//! acc[k] = fma(d_w[k] x g.d[k], sum, acc[k]) (Q4_1 and Q5_1 also form accm[k] = fma(m[k] x g.d[k], the activations'
//! sum, accm[k]), added afterwards), and the eight lanes are folded as ((0+4)+(2+6))+((1+5)+(3+7)) (hsum4x2). The
//! K-quants and the IQ types (q45k_vec_dot, q3k_dotv, q6k_matmul, iq4xs_dotv, iq3s_dotv): for each block, the exact sum
//! of scale_g x (group g's dot), less the offset term (for Q3_K 4, and for Q6_K 32, times the scale-weighted activation
//! sums; for Q4_K / Q5_K the mins are kept apart), then acc = fma(d_w x d8, float(sum), acc) for each vector. Integer
//! sums are exact whatever instructions compute them, so the bits are those of the NEON passes and of the reference
//! (parity test).
//!
//! References (MIT: ggml / llama.cpp, github.com/ggml-org/ggml): the block layouts and reference dot products of these
//! types; the x86 signed-by-signed byte product as |w| (unsigned) times a with w's sign put onto a (_mm256_sign_epi8,
//! as in ggml's mul_sum_i8_pairs); Q3_K's scale applied to the int16 pair sums by madd (as in ggml's AVX2
//! ggml_vec_dot_q3_K_q8_K, and likewise for Q4_K, Q5_K, Q6_K, IQ4_XS and IQ3_S); and the IQ3_S grid (iq3s_grid.rs).

use super::x86dot::fold;
use crate::{BlockQ8K, Groups32, Q8s};
use core::arch::x86_64::*;

const Q8_0_BLOCK: usize = 34;
const Q4_1_BLOCK: usize = 20;
const Q5_1_BLOCK: usize = 24;
const Q5_0_BLOCK: usize = 22;
const IQ4NL_BLOCK: usize = 18;
const Q6_K_BLOCK: usize = 210;
const Q4_K_BLOCK: usize = 144;
const Q5_K_BLOCK: usize = 176;
const IQ4NL_LEVELS: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Lane k: the sum of the eight int32 lanes of s[k] (exact).
#[inline(always)]
unsafe fn sums8(s: &[__m256i; 8]) -> __m256i {
    unsafe {
        let a = _mm256_hadd_epi32(_mm256_hadd_epi32(s[0], s[1]), _mm256_hadd_epi32(s[2], s[3]));
        let b = _mm256_hadd_epi32(_mm256_hadd_epi32(s[4], s[5]), _mm256_hadd_epi32(s[6], s[7]));
        let lo = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
        let hi = _mm_add_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256::<1>(b));
        _mm256_set_m128i(hi, lo)
    }
}

/// Lane v: the total of t[v]'s eight int32 lanes for v < NV (0 beyond; exact). This is sums8 across the vectors, so a
/// block's float steps for all NV vectors take one instruction each, with the same per-lane steps as one vector at a
/// time.
#[inline(always)]
unsafe fn totals<const NV: usize>(t: &[__m256i; NV]) -> __m256i {
    unsafe { sums8(&std::array::from_fn(|v| if v < NV { t[v] } else { _mm256_setzero_si256() })) }
}

/// Lane v: vector v's block-b scale (x[v * nb + b].d) for v < NV (0 beyond).
#[inline(always)]
unsafe fn d8s<const NV: usize>(x: *const BlockQ8K, nb: usize, b: usize) -> __m256 {
    unsafe {
        let d: [f32; 8] = std::array::from_fn(|v| if v < NV { (*x.add(v * nb + b)).d } else { 0.0 });
        _mm256_loadu_ps(d.as_ptr())
    }
}

/// Writes lane v of a to y[v * ys], for v < NV.
#[inline(always)]
unsafe fn store_nv<const NV: usize>(a: __m256, y: *mut f32, ys: usize) {
    unsafe {
        let mut r = [0f32; 8];
        _mm256_storeu_ps(r.as_mut_ptr(), a);
        for v in 0..NV {
            *y.add(v * ys) = r[v];
        }
    }
}

/// Unsigned bytes u times signed bytes s, with each four adjacent products summed into an int32 lane (SDOT's shape).
/// This is exact: the pairs fit int16 for every operand used here. VNNI: 0 = AVX2 (maddubs, then madd by ones), 1 =
/// AVX-VNNI's VPDPBUSD, 2 = the same instruction from AVX512-VNNI on 256-bit registers (AVX512VL), for the CPUs that
/// have the AVX-512 form and not the AVX one (AMD Zen 4; Intel Ice Lake to Sapphire Rapids).
#[inline(always)]
unsafe fn dot4<const VNNI: u8>(u: __m256i, s: __m256i) -> __m256i {
    unsafe {
        match VNNI {
            1 => _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s),
            2 => _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s),
            _ => _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)),
        }
    }
}

/// acc plus the int16 pair products of p and s, summed in pairs into int32 lanes (VPDPWSSD's shape; on AVX2, madd
/// then add). Exact: every caller's pair sums fit int32. VNNI as for dot4.
#[inline(always)]
unsafe fn wssd<const VNNI: u8>(acc: __m256i, p: __m256i, s: __m256i) -> __m256i {
    unsafe {
        match VNNI {
            1 => _mm256_dpwssd_avx_epi32(acc, p, s),
            2 => _mm256_dpwssd_epi32(acc, p, s),
            _ => _mm256_add_epi32(acc, _mm256_madd_epi16(p, s)),
        }
    }
}

/// The 8 fp16 values at p + off, p + stride + off, .., widened exactly (neon.rs fcvtl_x, 8 at once).
#[inline(always)]
unsafe fn halves8(p: *const u8, stride: usize, off: usize) -> __m256 {
    unsafe {
        let h: [u16; 8] = std::array::from_fn(|k| core::ptr::read_unaligned(p.add(k * stride + off) as *const u16));
        _mm256_cvtph_ps(_mm_loadu_si128(h.as_ptr().cast()))
    }
}

/// Software prefetch, for products over a single vector (decode) only: there each weight byte carries too little
/// arithmetic to cover the wait for memory, so while a block is multiplied, the block this many blocks further along
/// the row is asked for. With several vectors the arithmetic covers it. TORC_X86_PF_BLOCKS=<n> sets the distance
/// (default 16; 0 turns it off).
fn lookahead_blocks(nv: usize) -> usize {
    static N: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_X86_PF_BLOCKS").unwrap_or(16));
    if nv == 1 { *N } else { 0 }
}

/// Asks for the block of `len` bytes that starts `blocks` blocks after `blk`, every 64-byte line of it (nothing for 0
/// blocks). Past the row's end this is harmless: a prefetch never faults, so the address is formed with wrapping_add.
#[inline(always)]
unsafe fn ask_ahead(blk: *const u8, blocks: usize, len: usize) {
    if blocks == 0 {
        return;
    }
    let p = blk.wrapping_add(blocks * len);
    unsafe {
        let mut off = 0;
        while off < len {
            _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(off) as *const i8);
            off += 64;
        }
    }
}

/// Q5_1's or IQ3_S's sign or fifth bits spread over 32 bytes: byte l takes byte l / 8 of the u32 and keeps bit l % 8
/// (as neon.rs SPREAD / BIT do), giving all ones where the bit is set.
#[inline(always)]
unsafe fn bits32(m: u32) -> __m256i {
    unsafe {
        let spread = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        );
        let bit = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
        _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(m as i32), spread), bit), bit)
    }
}

/// One row (w) of Q8_0 (K = 0), Q4_1 (1), IQ4_NL (2), Q5_1 (3), Q5_0 (4) or MXFP4 (5) against NV vectors (x, nb blocks
/// each), into y[v * ys]: q8_0_dotv / q4_1_dotv / iq4nl_dotv, with both halves of a block of 256 in one register.
/// Q5_0's weights are unpacked to q - 16 (exactly a Q8_0 weight under the same scale) and then go Q8_0's way. MXFP4's
/// weights are unpacked as w + 12 (0..24, unsigned), so the activations go in as they are (no sign step for each
/// vector), and 12 x each group's activation sum is taken off its integer sum: the same integer as the sum of w x a.
#[inline(always)]
unsafe fn lanes_pass<const NV: usize, const VNNI: u8, const K: u8>(
    w: *const u8,
    x: *const Groups32,
    nb: usize,
    y: *mut f32,
    ys: usize,
) {
    unsafe {
        let bs = match K {
            0 => Q8_0_BLOCK,
            1 => Q4_1_BLOCK,
            3 => Q5_1_BLOCK,
            4 => Q5_0_BLOCK,
            5 => crate::MXFP4_BLOCK,
            _ => IQ4NL_BLOCK,
        };
        let dm = K == 1 || K == 3;
        let off = K == 5;
        let m15 = _mm_set1_epi8(15);
        // MXFP4's levels are its doubled E2M1 values (-12..12, here plus 12), as IQ4_NL's are its own (a table of 16
        // bytes either way)
        let levels = if off { crate::MXFP4_LEVELS.as_ptr() } else { IQ4NL_LEVELS.as_ptr() };
        let lv = _mm256_broadcastsi128_si256(_mm_loadu_si128(levels.cast()));
        let lv = if off { _mm256_add_epi8(lv, _mm256_set1_epi8(12)) } else { lv };
        let mut a = [_mm256_setzero_ps(); NV];
        let mut e = [_mm256_setzero_ps(); NV];
        let ahead = lookahead_blocks(NV);
        for b in 0..nb {
            let wb = w.add(b * 8 * bs);
            ask_ahead(wb, ahead, 8 * bs);
            // MXFP4: each block's scale is one E8M0 byte, halved (crate::e8m0_half, exact)
            let d = if K == 5 {
                let e: [f32; 8] = std::array::from_fn(|k| crate::e8m0_half(*wb.add(k * bs)));
                _mm256_loadu_ps(e.as_ptr())
            } else {
                halves8(wb, bs, 0)
            };
            let m = if dm { halves8(wb, bs, 2) } else { _mm256_setzero_ps() };
            // for each block position k: the unsigned operand (the weight, w + 12 for MXFP4, or |w| for the other
            // signed types) and, for those, w itself (whose sign goes onto a)
            let mut u = [_mm256_setzero_si256(); 8];
            let mut q = [_mm256_setzero_si256(); 8];
            for k in 0..8 {
                let p = wb.add(k * bs);
                let nib = |o: usize| {
                    let b16 = _mm_loadu_si128(p.add(o).cast());
                    _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16::<4>(b16), m15), _mm_and_si128(b16, m15))
                };
                match K {
                    0 => {
                        q[k] = _mm256_loadu_si256(p.add(2).cast());
                        u[k] = _mm256_abs_epi8(q[k]);
                    }
                    1 => u[k] = nib(4),
                    3 => {
                        let fifth = bits32(core::ptr::read_unaligned(p.add(4) as *const u32));
                        u[k] = _mm256_or_si256(nib(8), _mm256_and_si256(fifth, _mm256_set1_epi8(16)));
                    }
                    4 => {
                        let fifth = bits32(core::ptr::read_unaligned(p.add(2) as *const u32));
                        let q5 = _mm256_or_si256(nib(6), _mm256_and_si256(fifth, _mm256_set1_epi8(16)));
                        q[k] = _mm256_sub_epi8(q5, _mm256_set1_epi8(16));
                        u[k] = _mm256_abs_epi8(q[k]);
                    }
                    5 => u[k] = _mm256_shuffle_epi8(lv, nib(1)),
                    _ => {
                        q[k] = _mm256_shuffle_epi8(lv, nib(2));
                        u[k] = _mm256_abs_epi8(q[k]);
                    }
                }
            }
            for v in 0..NV {
                let xb = &*x.add(v * nb + b);
                let s: [__m256i; 8] = std::array::from_fn(|k| {
                    let av = _mm256_loadu_si256(xb.qs.as_ptr().add(32 * k).cast());
                    dot4::<VNNI>(u[k], if dm || off { av } else { _mm256_sign_epi8(av, q[k]) })
                });
                let mut t = sums8(&s);
                if off {
                    let sa = _mm256_cvtepi16_epi32(_mm_loadu_si128(xb.sums.as_ptr().cast()));
                    t = _mm256_sub_epi32(t, _mm256_mullo_epi32(sa, _mm256_set1_epi32(12)));
                }
                let dx = _mm256_loadu_ps(xb.d.as_ptr());
                a[v] = _mm256_fmadd_ps(_mm256_mul_ps(d, dx), _mm256_cvtepi32_ps(t), a[v]);
                if dm {
                    // the activations' sum for each block position (exact integers)
                    let sf = _mm256_cvtepi16_epi32(_mm_loadu_si128(xb.sums.as_ptr().cast()));
                    e[v] = _mm256_fmadd_ps(_mm256_mul_ps(m, dx), _mm256_cvtepi32_ps(sf), e[v]);
                }
            }
        }
        for v in 0..NV {
            *y.add(v * ys) = if dm { fold(a[v]) + fold(e[v]) } else { fold(a[v]) };
        }
    }
}

/// Register k's scale words for 16-weight groups (Q3_K, Q6_K): words 0-7 hold scale_2k and 8-15 hold scale_2k+1, made
/// by one byte shuffle of the int16 scales 0-7 (lo) or 8-15 (hi), each held in both lanes.
#[inline(always)]
unsafe fn sv16<const K: i8>(lo: __m256i, hi: __m256i) -> __m256i {
    unsafe {
        let (e, o) = (4 * (K % 4), 4 * (K % 4) + 2);
        _mm256_shuffle_epi8(
            if K < 4 { lo } else { hi },
            _mm256_setr_epi8(
                e,
                e + 1,
                e,
                e + 1,
                e,
                e + 1,
                e,
                e + 1,
                e,
                e + 1,
                e,
                e + 1,
                e,
                e + 1,
                e,
                e + 1,
                o,
                o + 1,
                o,
                o + 1,
                o,
                o + 1,
                o,
                o + 1,
                o,
                o + 1,
                o,
                o + 1,
                o,
                o + 1,
                o,
                o + 1,
            ),
        )
    }
}

/// One row (w) of Q6_K against NV vectors (x, nb blocks each), into y[v * ys] (q6k_matmul' sum). The 256 values are
/// 0..63 (the low four bits from ql, the top two from qh), and register k = 4n + j holds weights 128n + 32j .. + 32
/// (groups 2k and 2k + 1). Register by register, each vector's int16 pair sums of q x a are multiplied by the group
/// scale and added pairwise into its int32 accumulator (ggml's AVX2 Q6_K form, MIT; exact, since the pairs are at most
/// 16128 and are multiplied by at most 128), less 32 x the sum of scale_g x the group's activation sum; then acc =
/// fma(d_w x d8, sum, acc).
#[inline(always)]
unsafe fn q6k_pass<const NV: usize, const VNNI: u8>(
    w: *const u8,
    x: *const BlockQ8K,
    nb: usize,
    y: *mut f32,
    ys: usize,
) {
    unsafe {
        let (low4, low2) = (_mm256_set1_epi8(0x0f), _mm256_set1_epi8(0x03));
        let mut acc = _mm256_setzero_ps();
        let ahead = lookahead_blocks(NV);
        for b in 0..nb {
            let blk = w.add(b * Q6_K_BLOCK);
            ask_ahead(blk, ahead, Q6_K_BLOCK);
            let sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(blk.add(192).cast()));
            let s_lo = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sc16));
            let s_hi = _mm256_broadcastsi128_si256(_mm256_extracti128_si256::<1>(sc16));
            let mut t = [_mm256_setzero_si256(); NV];
            macro_rules! reg {
                ($k:literal, $l:expr, $h:expr) => {{
                    let q =
                        _mm256_or_si256(_mm256_and_si256($l, low4), _mm256_slli_epi16::<4>(_mm256_and_si256($h, low2)));
                    let sv = sv16::<$k>(s_lo, s_hi);
                    for (v, tv) in t.iter_mut().enumerate() {
                        let a = _mm256_loadu_si256((*x.add(v * nb + b)).qs.as_ptr().add(32 * $k).cast());
                        let p = _mm256_maddubs_epi16(q, a);
                        *tv = wssd::<VNNI>(*tv, p, sv);
                    }
                }};
            }
            let ld = |o: usize| _mm256_loadu_si256(blk.add(o).cast());
            reg!(0, ld(0), ld(128));
            reg!(1, ld(32), _mm256_srli_epi16::<2>(ld(128)));
            reg!(2, _mm256_srli_epi16::<4>(ld(0)), _mm256_srli_epi16::<4>(ld(128)));
            reg!(3, _mm256_srli_epi16::<4>(ld(32)), _mm256_srli_epi16::<6>(ld(128)));
            reg!(4, ld(64), ld(160));
            reg!(5, ld(96), _mm256_srli_epi16::<2>(ld(160)));
            reg!(6, _mm256_srli_epi16::<4>(ld(64)), _mm256_srli_epi16::<4>(ld(160)));
            reg!(7, _mm256_srli_epi16::<4>(ld(96)), _mm256_srli_epi16::<6>(ld(160)));
            let dw = crate::f16_to_f32(core::ptr::read_unaligned(blk.add(208) as *const u16));
            for (v, tv) in t.iter_mut().enumerate() {
                // the -32 offset: 32 x the sum over groups of scale x the group's activation sum
                let c = _mm256_madd_epi16(_mm256_loadu_si256((*x.add(v * nb + b)).bsums.as_ptr().cast()), sc16);
                *tv = _mm256_sub_epi32(*tv, _mm256_slli_epi32::<5>(c));
            }
            acc = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_set1_ps(dw), d8s::<NV>(x, nb, b)),
                _mm256_cvtepi32_ps(totals(&t)),
                acc,
            );
        }
        store_nv::<NV>(acc, y, ys);
    }
}

/// Unpacks the 12 scale bytes at q of a Q4_K / Q5_K block (ggml's get_scale_min_k4 layout) into one register: the
/// eight 6-bit scales in bytes 0-7, the eight 6-bit mins in bytes 8-15. Same result as lib.rs q45k_scales.
#[inline(always)]
unsafe fn q45k_scales_x(q: *const u8) -> __m128i {
    unsafe {
        let w = |i: usize| core::ptr::read_unaligned(q.add(i) as *const u32);
        let (a, b, c) = (w(0), w(4), w(8));
        let (low6, low4, top2) = (0x3f3f3f3fu32, 0x0f0f0f0fu32, 0x03030303u32);
        _mm_setr_epi32(
            (a & low6) as i32,
            ((c & low4) | (((a >> 6) & top2) << 4)) as i32,
            (b & low6) as i32,
            (((c >> 4) & low4) | (((b >> 6) & top2) << 4)) as i32,
        )
    }
}

/// One row (w) of Q4_K (FIVE = false) or Q5_K (true) against NV vectors (x, nb blocks each), into y[v * ys]
/// (q45k_vec_dot's sums). Sub-block by sub-block, each vector's int16 pair sums of q x a are multiplied by scale_k and
/// added pairwise into its int32 accumulator (the scale goes in the madd, as in ggml's AVX2 Q4_K product, MIT; exact,
/// since the pairs are at most 7936 and are multiplied by at most 63). The summs (the mins against the bsums) and the
/// two f32 chains, d d8 sumi and dmin d8 summs, are as in q45k_vec_dot. It goes register by register so AVX2's 16
/// registers can hold the vectors' accumulators.
#[inline(always)]
unsafe fn q45k_pass<const NV: usize, const VNNI: u8, const FIVE: bool>(
    w: *const u8,
    x: *const BlockQ8K,
    nb: usize,
    y: *mut f32,
    ys: usize,
) {
    unsafe {
        let bs = if FIVE { Q5_K_BLOCK } else { Q4_K_BLOCK };
        let (low4, one) = (_mm256_set1_epi8(0x0f), _mm256_set1_epi8(1));
        let (mut acc, mut accm) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let ahead = lookahead_blocks(NV);
        for b in 0..nb {
            let blk = w.add(b * bs);
            ask_ahead(blk, ahead, bs);
            let qs = blk.add(if FIVE { 48 } else { 16 });
            let sm = q45k_scales_x(blk.add(4));
            let s16 = _mm_cvtepu8_epi16(sm);
            let m16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(sm));
            let qh = if FIVE { _mm256_loadu_si256(blk.add(16).cast()) } else { _mm256_setzero_si256() };
            let mut t = [_mm256_setzero_si256(); NV];
            // sub-block 2j is the low nibbles of qs[32j..32j + 32] and 2j + 1 the high ones; Q5_K's fifth bit for
            // weight l of sub-block k is bit k of qh[l]
            macro_rules! reg {
                ($k:literal, $n:expr) => {{
                    let q = if FIVE {
                        _mm256_or_si256($n, _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<$k>(qh), one)))
                    } else {
                        $n
                    };
                    let sv = _mm256_broadcastw_epi16(_mm_srli_si128::<{ 2 * $k }>(s16));
                    for (v, tv) in t.iter_mut().enumerate() {
                        let a = _mm256_loadu_si256((*x.add(v * nb + b)).qs.as_ptr().add(32 * $k).cast());
                        let p = _mm256_maddubs_epi16(q, a);
                        *tv = wssd::<VNNI>(*tv, p, sv);
                    }
                }};
            }
            let ld = |j: usize| _mm256_loadu_si256(qs.add(32 * j).cast());
            reg!(0, _mm256_and_si256(ld(0), low4));
            reg!(1, _mm256_and_si256(_mm256_srli_epi16::<4>(ld(0)), low4));
            reg!(2, _mm256_and_si256(ld(1), low4));
            reg!(3, _mm256_and_si256(_mm256_srli_epi16::<4>(ld(1)), low4));
            reg!(4, _mm256_and_si256(ld(2), low4));
            reg!(5, _mm256_and_si256(_mm256_srli_epi16::<4>(ld(2)), low4));
            reg!(6, _mm256_and_si256(ld(3), low4));
            reg!(7, _mm256_and_si256(_mm256_srli_epi16::<4>(ld(3)), low4));
            let fd = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16));
            let fm = crate::f16_to_f32(core::ptr::read_unaligned(blk.add(2) as *const u16));
            // min_j x the sub-block's activation sum
            let pm: [__m256i; NV] = std::array::from_fn(|v| {
                let bs = &(*x.add(v * nb + b)).bsums;
                let s32: [i16; 8] = std::array::from_fn(|k| bs[2 * k].wrapping_add(bs[2 * k + 1]));
                _mm256_castsi128_si256(_mm_madd_epi16(m16, _mm_loadu_si128(s32.as_ptr().cast())))
            });
            let pm: [__m256i; NV] = std::array::from_fn(|v| _mm256_inserti128_si256::<1>(pm[v], _mm_setzero_si128()));
            let d8 = d8s::<NV>(x, nb, b);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(fd), d8), _mm256_cvtepi32_ps(totals(&t)), acc);
            accm = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(fm), d8), _mm256_cvtepi32_ps(totals(&pm)), accm);
        }
        store_nv::<NV>(_mm256_sub_ps(acc, accm), y, ys);
    }
}

/// Unpacks the 12 scale bytes at q of a Q3_K block into one register of sixteen signed scales, -32..31. In ggml's
/// layout bytes 0-7 hold the low nibbles and bytes 8-11 the high two bits; the shuffle follows ggml's AVX2 Q3_K (its
/// aux words), then 32 is subtracted. Same result as lib.rs q3k_unpack_scales.
#[inline(always)]
unsafe fn q3k_scales_x(q: *const u8) -> __m128i {
    unsafe {
        let w = |i: usize| core::ptr::read_unaligned(q.add(i) as *const u32);
        let (a0, a1, a2) = (w(0), w(4), w(8));
        let (low4, top2) = (0x0f0f0f0fu32, 0x03030303u32);
        let o = _mm_setr_epi32(
            ((a0 & low4) | ((a2 & top2) << 4)) as i32,
            ((a1 & low4) | (((a2 >> 2) & top2) << 4)) as i32,
            (((a0 >> 4) & low4) | (((a2 >> 4) & top2) << 4)) as i32,
            (((a1 >> 4) & low4) | (((a2 >> 6) & top2) << 4)) as i32,
        );
        _mm_sub_epi8(o, _mm_set1_epi8(32))
    }
}

/// One row (w) of Q3_K against NV vectors (x, nb blocks each), into y[v * ys] (q3k_dotv's sum), with u = value + 4
/// (0..7). Register k = 4n + j holds groups 2k and 2k + 1: qs[32n..32n + 32] >> 2j, and bit k of hmask[0..32]. For each
/// vector, the int16 pair sums of u x a are multiplied by their group's scale and added pairwise into int32 lanes
/// (ggml's AVX2 Q3_K product, MIT: _mm256_madd_epi16(scale, maddubs(q, a)); exact, since the pairs fit int16 and their
/// scaled sums fit int32), less 4 x the sum of scale_g x the group's activation sum; then acc = fma(d x d8, sum, acc).
#[inline(always)]
unsafe fn q3k_pass<const NV: usize, const VNNI: u8>(
    w: *const u8,
    x: *const BlockQ8K,
    nb: usize,
    y: *mut f32,
    ys: usize,
) {
    unsafe {
        let (three, one) = (_mm256_set1_epi8(3), _mm256_set1_epi8(1));
        let mut acc = _mm256_setzero_ps();
        let ahead = lookahead_blocks(NV);
        for b in 0..nb {
            let blk = w.add(b * crate::Q3_K_BLOCK);
            ask_ahead(blk, ahead, crate::Q3_K_BLOCK);
            let sc16 = _mm256_cvtepi8_epi16(q3k_scales_x(blk.add(96)));
            // the int16 scales 0-7 and 8-15, each in both lanes
            let s_lo = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sc16));
            let s_hi = _mm256_broadcastsi128_si256(_mm256_extracti128_si256::<1>(sc16));
            let hm = _mm256_loadu_si256(blk.cast());
            let qs = [_mm256_loadu_si256(blk.add(32).cast()), _mm256_loadu_si256(blk.add(64).cast())];
            // register by register, with each vector's int32 sum in its own accumulator: AVX2's 16 registers cannot
            // hold the block's 8 unpacked registers and 8 scale vectors across the vectors (neon.rs has 32)
            let mut t = [_mm256_setzero_si256(); NV];
            macro_rules! reg {
                ($k:literal, $n:literal, $j2:literal, $hs:expr) => {{
                    let u = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<$j2>(qs[$n]), three),
                        _mm256_slli_epi16::<2>(_mm256_and_si256($hs, one)),
                    );
                    let sv = sv16::<$k>(s_lo, s_hi);
                    for (v, tv) in t.iter_mut().enumerate() {
                        let a = _mm256_loadu_si256((*x.add(v * nb + b)).qs.as_ptr().add(32 * $k).cast());
                        let p = _mm256_maddubs_epi16(u, a);
                        *tv = wssd::<VNNI>(*tv, p, sv);
                    }
                }};
            }
            reg!(0, 0, 0, hm);
            reg!(1, 0, 2, _mm256_srli_epi16::<1>(hm));
            reg!(2, 0, 4, _mm256_srli_epi16::<2>(hm));
            reg!(3, 0, 6, _mm256_srli_epi16::<3>(hm));
            reg!(4, 1, 0, _mm256_srli_epi16::<4>(hm));
            reg!(5, 1, 2, _mm256_srli_epi16::<5>(hm));
            reg!(6, 1, 4, _mm256_srli_epi16::<6>(hm));
            reg!(7, 1, 6, _mm256_srli_epi16::<7>(hm));
            let d = crate::f16_to_f32(core::ptr::read_unaligned(blk.add(108) as *const u16));
            for (v, tv) in t.iter_mut().enumerate() {
                let c = _mm256_madd_epi16(_mm256_loadu_si256((*x.add(v * nb + b)).bsums.as_ptr().cast()), sc16);
                *tv = _mm256_sub_epi32(*tv, _mm256_slli_epi32::<2>(c));
            }
            acc = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_set1_ps(d), d8s::<NV>(x, nb, b)),
                _mm256_cvtepi32_ps(totals(&t)),
                acc,
            );
        }
        store_nv::<NV>(acc, y, ys);
    }
}

/// One row (w) of IQ4_XS against NV vectors (x, nb blocks each), into y[v * ys] (iq4xs_dotv's sum). Group by group,
/// each vector's int16 pair sums of |level| x (a with the level's sign) are multiplied by scale_g and added pairwise
/// into its int32 accumulator (exact, since the pairs are at most 32258 and are multiplied by at most 32); then acc =
/// fma(d x d8, sum, acc).
#[inline(always)]
unsafe fn iq4xs_pass<const NV: usize, const VNNI: u8>(
    w: *const u8,
    x: *const BlockQ8K,
    nb: usize,
    y: *mut f32,
    ys: usize,
) {
    unsafe {
        let m15 = _mm_set1_epi8(15);
        let lv = _mm256_broadcastsi128_si256(_mm_loadu_si128(crate::IQ4_LEVELS.as_ptr().cast()));
        let mut acc = _mm256_setzero_ps();
        let ahead = lookahead_blocks(NV);
        for b in 0..nb {
            let blk = w.add(b * crate::IQ4_XS_BLOCK);
            ask_ahead(blk, ahead, crate::IQ4_XS_BLOCK);
            let sc = crate::iq4xs_scales(core::slice::from_raw_parts(blk, 8));
            let s16 = _mm_packs_epi32(_mm_loadu_si128(sc.as_ptr().cast()), _mm_loadu_si128(sc.as_ptr().add(4).cast()));
            let mut t = [_mm256_setzero_si256(); NV];
            macro_rules! grp {
                ($g:literal) => {{
                    let b16 = _mm_loadu_si128(blk.add(8 + 16 * $g).cast());
                    let nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16::<4>(b16), m15), _mm_and_si128(b16, m15));
                    let q = _mm256_shuffle_epi8(lv, nib);
                    let u = _mm256_abs_epi8(q);
                    let sv = _mm256_broadcastw_epi16(_mm_srli_si128::<{ 2 * $g }>(s16));
                    for (v, tv) in t.iter_mut().enumerate() {
                        let a = _mm256_loadu_si256((*x.add(v * nb + b)).qs.as_ptr().add(32 * $g).cast());
                        let p = _mm256_maddubs_epi16(u, _mm256_sign_epi8(a, q));
                        *tv = wssd::<VNNI>(*tv, p, sv);
                    }
                }};
            }
            grp!(0);
            grp!(1);
            grp!(2);
            grp!(3);
            grp!(4);
            grp!(5);
            grp!(6);
            grp!(7);
            let d = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16));
            acc = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_set1_ps(d), d8s::<NV>(x, nb, b)),
                _mm256_cvtepi32_ps(totals(&t)),
                acc,
            );
        }
        store_nv::<NV>(acc, y, ys);
    }
}

/// One row (w) of IQ3_S against NV vectors (x, nb blocks each), into y[v * ys] (iq3s_dotv's sum). Group by group, its
/// 32 unsigned levels (crate::iq3s_group) and signs are made once, and each vector's int16 pair sums of level x (a with
/// the weight's sign) are multiplied by scale_g and added pairwise into its int32 accumulator (exact); then acc = fma(d
/// x d8, sum, acc).
#[inline(always)]
unsafe fn iq3s_pass<const NV: usize, const VNNI: u8>(
    w: *const u8,
    x: *const BlockQ8K,
    nb: usize,
    y: *mut f32,
    ys: usize,
) {
    unsafe {
        let one = _mm256_set1_epi8(1);
        let mut acc = _mm256_setzero_ps();
        let ahead = lookahead_blocks(NV);
        for b in 0..nb {
            let blk = w.add(b * crate::IQ3_S_BLOCK);
            ask_ahead(blk, ahead, crate::IQ3_S_BLOCK);
            let bs = core::slice::from_raw_parts(blk, crate::IQ3_S_BLOCK);
            let mut t = [_mm256_setzero_si256(); NV];
            for g in 0..8 {
                let (l, s) = crate::iq3s_group(bs, g);
                let lv = _mm256_loadu_si256(l.as_ptr().cast());
                // -1 where the weight is negative, else 1
                let sg = _mm256_or_si256(bits32(s), one);
                let sv = _mm256_set1_epi16(crate::iq3s_group_scale(bs, g) as i16);
                for (v, tv) in t.iter_mut().enumerate() {
                    let a = _mm256_loadu_si256((*x.add(v * nb + b)).qs.as_ptr().add(32 * g).cast());
                    let p = _mm256_maddubs_epi16(lv, _mm256_sign_epi8(a, sg));
                    *tv = wssd::<VNNI>(*tv, p, sv);
                }
            }
            let d = crate::f16_to_f32(core::ptr::read_unaligned(blk as *const u16));
            acc = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_set1_ps(d), d8s::<NV>(x, nb, b)),
                _mm256_cvtepi32_ps(totals(&t)),
                acc,
            );
        }
        store_nv::<NV>(acc, y, ys);
    }
}

/// Rows of type ty against nv vectors: the K and IQ types (Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S) up to 8 a pass, with
/// the float steps done across the vectors (TORC_X86_WIDE=0 makes it 4), and the block-position types up to 4.
#[inline(always)]
unsafe fn each_row<const VNNI: u8>(
    ty: u32,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: Q8s,
    nv: usize,
    y: *mut f32,
    ys: usize,
) {
    let nb = x.len() / nv;
    // TORC_X86_WIDE=0: the K and IQ types take 4 vectors a pass
    static WIDE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_X86_WIDE", "0"));
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            let mut v0 = 0;
            while *WIDE && matches!(ty, 11 | 12 | 13 | 14 | 21 | 23) && v0 < nv {
                let (xs, yr) = (x.k.as_ptr().add(v0 * nb), y.add(v0 * ys + r));
                macro_rules! go8 {
                    ($n:literal) => {
                        match ty {
                            11 => q3k_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                            23 => iq4xs_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                            21 => iq3s_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                            12 => q45k_pass::<$n, VNNI, false>(wr, xs, nb, yr, ys),
                            13 => q45k_pass::<$n, VNNI, true>(wr, xs, nb, yr, ys),
                            _ => q6k_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                        }
                    };
                }
                match nv - v0 {
                    1 => go8!(1),
                    2 => go8!(2),
                    3 => go8!(3),
                    4 => go8!(4),
                    5 => go8!(5),
                    6 => go8!(6),
                    7 => go8!(7),
                    _ => go8!(8),
                }
                v0 += 8;
            }
            while v0 < nv {
                let (xs, xg, yr) = (x.k.as_ptr().add(v0 * nb), x.g.as_ptr().add(v0 * nb), y.add(v0 * ys + r));
                macro_rules! go {
                    ($n:literal) => {
                        match ty {
                            8 => lanes_pass::<$n, VNNI, 0>(wr, xg, nb, yr, ys),
                            3 => lanes_pass::<$n, VNNI, 1>(wr, xg, nb, yr, ys),
                            20 => lanes_pass::<$n, VNNI, 2>(wr, xg, nb, yr, ys),
                            7 => lanes_pass::<$n, VNNI, 3>(wr, xg, nb, yr, ys),
                            6 => lanes_pass::<$n, VNNI, 4>(wr, xg, nb, yr, ys),
                            39 => lanes_pass::<$n, VNNI, 5>(wr, xg, nb, yr, ys),
                            11 => q3k_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                            23 => iq4xs_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                            21 => iq3s_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                            12 => q45k_pass::<$n, VNNI, false>(wr, xs, nb, yr, ys),
                            13 => q45k_pass::<$n, VNNI, true>(wr, xs, nb, yr, ys),
                            _ => q6k_pass::<$n, VNNI>(wr, xs, nb, yr, ys),
                        }
                    };
                }
                match nv - v0 {
                    1 => go!(1),
                    2 => go!(2),
                    3 => go!(3),
                    _ => go!(4),
                }
                v0 += 4;
            }
        }
    }
}

#[target_feature(enable = "avxvnni")]
unsafe fn rows_vnni(ty: u32, w: &[u8], row_bytes: usize, rows: usize, x: Q8s, nv: usize, y: *mut f32, ys: usize) {
    unsafe { each_row::<1>(ty, w, row_bytes, rows, x, nv, y, ys) }
}

#[target_feature(enable = "avx512vnni,avx512vl")]
unsafe fn rows_vnni512(ty: u32, w: &[u8], row_bytes: usize, rows: usize, x: Q8s, nv: usize, y: *mut f32, ys: usize) {
    unsafe { each_row::<2>(ty, w, row_bytes, rows, x, nv, y, ys) }
}

/// The VNNI form matmul uses: AVX-VNNI where the CPU has it, else AVX512-VNNI's on 256-bit registers where it has
/// that (with AVX512VL), else none (0, 1, 2 as for dot4). TORC_VNNI=0 turns both off; the integers are the same in
/// all three.
pub fn vnni() -> u8 {
    static V: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| {
        if crate::switches::is("TORC_VNNI", "0") {
            0
        } else if std::arch::is_x86_feature_detected!("avxvnni") {
            1
        } else if std::arch::is_x86_feature_detected!("avx512vnni") && std::arch::is_x86_feature_detected!("avx512vl") {
            2
        } else {
            0
        }
    });
    *V
}

/// Whether matmul takes this type.
pub fn covers(ty: u32) -> bool {
    matches!(ty, 3 | 6 | 7 | 8 | 11 | 12 | 13 | 14 | 20 | 21 | 23 | 39)
}

/// Rows of Q8_0 (ty 8), Q4_1 (3), Q5_0 (6), Q5_1 (7), IQ4_NL (20), MXFP4 (39), Q6_K (14), Q3_K (11), Q4_K (12), Q5_K
/// (13), IQ4_XS (23) or IQ3_S (21), (weight row r beginning at byte r * row_bytes of w) times nv vectors of x.len() /
/// nv blocks of 256, with vector v's outputs going to y[v * ys + r]. It uses VNNI where the CPU has it, in either form
/// (vnni(); TORC_VNNI=0 turns it off; the integers are the same either way).
///
/// # Safety
/// The CPU must have AVX2, FMA and F16C (the build's x86-64-v3), and y[v * ys + r] must be writable for v < nv and r <
/// rows.
pub unsafe fn matmul(ty: u32, w: &[u8], row_bytes: usize, rows: usize, x: Q8s, nv: usize, y: *mut f32, ys: usize) {
    let nb = x.len() / nv.max(1);
    let bpb = match ty {
        8 => 8 * Q8_0_BLOCK,
        3 => 8 * Q4_1_BLOCK,
        7 => 8 * Q5_1_BLOCK,
        6 => 8 * Q5_0_BLOCK,
        20 => 8 * IQ4NL_BLOCK,
        39 => 8 * crate::MXFP4_BLOCK,
        14 => Q6_K_BLOCK,
        12 => Q4_K_BLOCK,
        11 => crate::Q3_K_BLOCK,
        23 => crate::IQ4_XS_BLOCK,
        21 => crate::IQ3_S_BLOCK,
        13 => Q5_K_BLOCK,
        _ => panic!("x86mm: type {ty}"),
    };
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * bpb && w.len() >= rows * row_bytes);
    unsafe {
        match vnni() {
            1 => rows_vnni(ty, w, row_bytes, rows, x, nv, y, ys),
            2 => rows_vnni512(ty, w, row_bytes, rows, x, nv, y, ys),
            _ => each_row::<0>(ty, w, row_bytes, rows, x, nv, y, ys),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q5_0 rows give exactly the bits of the same matrix stored as Q8_0 (each weight q - 16 under the block's scale),
    /// for 1 to 9 vectors: after the unpack Q5_0 follows Q8_0's path, and Q8_0 is parity-tested against the reference.
    #[test]
    fn q5_0_as_q8_0_bit_identical() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) {
            return;
        }
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for nb in [1usize, 3] {
            for rows in [1usize, 5, 16] {
                let mut w5 = vec![0u8; rows * nb * 8 * Q5_0_BLOCK];
                for blk in w5.chunks_exact_mut(Q5_0_BLOCK) {
                    let h = (next() as u16 & 0x83ff) | ((((next() % 20) + 4) as u16) << 10);
                    blk[..2].copy_from_slice(&h.to_le_bytes());
                    blk[2..].iter_mut().for_each(|b| *b = next() as u8);
                }
                let w8: Vec<u8> = w5
                    .chunks_exact(Q5_0_BLOCK)
                    .flat_map(|s| {
                        let qh = u32::from_le_bytes([s[2], s[3], s[4], s[5]]);
                        let mut o = [0u8; Q8_0_BLOCK];
                        o[..2].copy_from_slice(&s[..2]);
                        for j in 0..16 {
                            o[2 + j] = (((s[6 + j] & 15) | ((qh >> j) as u8 & 1) << 4) as i8 - 16) as u8;
                            o[18 + j] = (((s[6 + j] >> 4) | ((qh >> (j + 16)) as u8 & 1) << 4) as i8 - 16) as u8;
                        }
                        o
                    })
                    .collect();
                for nv in 1..=9usize {
                    let x: Vec<f32> =
                        (0..nv * nb * 256).map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
                    let mut xq = crate::Q8::new(nv * nb);
                    crate::q8k::quantize(&x, xq.view_mut());
                    let stride = rows + 1;
                    let (mut a, mut b) = (vec![0f32; nv * stride], vec![0f32; nv * stride]);
                    unsafe {
                        matmul(6, &w5, nb * 8 * Q5_0_BLOCK, rows, xq.view(), nv, a.as_mut_ptr(), stride);
                        matmul(8, &w8, nb * 8 * Q8_0_BLOCK, rows, xq.view(), nv, b.as_mut_ptr(), stride);
                    }
                    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&a), bits(&b), "nb {nb} rows {rows} nv {nv}");
                }
            }
        }
    }
}
