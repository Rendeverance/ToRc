//! x86.rs - the x86-64 kernels: AVX2 + FMA + F16C, with AVX-VNNI where the CPU has it. Each one computes the same exact
//! integer sums as the scalar reference and keeps its float order, so the bits match.

use crate::{BlockQ8K, Groups32, Q4_0_BLOCK};
use core::arch::x86_64::*;

/// How far ahead q4_0_dot reads a row's weights into cache: 16 blocks of 256 (2304 bytes), a block's three lines at
/// a time.
const PF_BLOCKS: usize = 16;

/// Asks for the three cache lines of the 256-weight Q4_0 block PF_BLOCKS blocks after the one at `wb` (8 x 18 = 144
/// bytes, so its first and last bytes and the line between them). Reading past the row's end is harmless: prefetch
/// never faults.
#[inline(always)]
unsafe fn prefetch_block(wb: *const u8) {
    unsafe {
        let p = wb.wrapping_add(PF_BLOCKS * 8 * Q4_0_BLOCK) as *const i8;
        _mm_prefetch::<_MM_HINT_T0>(p);
        _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(64));
        _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(8 * Q4_0_BLOCK - 1));
    }
}

/// The NEON kernel's method (neon::q4_0_dot) in AVX2 registers. The weights stay unsigned (the nibble, 0..15): for the
/// 256-weight block at `wb`, ul[k] holds the low nibbles (weights 0-15) of sub-block k in its low half and of
/// sub-block k + 4 in its high half, uh[k] the high nibbles (weights 16-31) the same way.
#[inline(always)]
unsafe fn nibbles_paired(wb: *const u8) -> ([__m256i; 4], [__m256i; 4]) {
    unsafe {
        let m15 = _mm256_set1_epi8(15);
        let mut ul = [_mm256_setzero_si256(); 4];
        let mut uh = [_mm256_setzero_si256(); 4];
        for k in 0..4 {
            let lo = _mm_loadu_si128(wb.add(k * Q4_0_BLOCK + 2) as *const __m128i);
            let hi = _mm_loadu_si128(wb.add((k + 4) * Q4_0_BLOCK + 2) as *const __m128i);
            let b = _mm256_set_m128i(hi, lo);
            ul[k] = _mm256_and_si256(b, m15);
            uh[k] = _mm256_and_si256(_mm256_srli_epi16::<4>(b), m15);
        }
        (ul, uh)
    }
}

/// Each sub-block's exact sum of (q - 8) a, sub-block k in lane k: the unsigned products (maddubs, at most 2 x 2 x 15
/// x 127 a pair, no saturation), four pairs of sub-blocks reduced together so lanes 0-3 come from the low halves and
/// 4-7 from the high ones, then 8 x the sub-block's activation sum (g.sums[k]) taken off, as the NEON
/// kernel does.
#[inline(always)]
unsafe fn q4_0_sums(ul: &[__m256i; 4], uh: &[__m256i; 4], xb: &Groups32) -> __m256i {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let a = xb.qs.as_ptr();
        let mut r = [_mm256_setzero_si256(); 4];
        for k in 0..4 {
            let al = _mm256_set_m128i(
                _mm_loadu_si128(a.add(32 * (k + 4)) as *const __m128i),
                _mm_loadu_si128(a.add(32 * k) as *const __m128i),
            );
            let ah = _mm256_set_m128i(
                _mm_loadu_si128(a.add(32 * (k + 4) + 16) as *const __m128i),
                _mm_loadu_si128(a.add(32 * k + 16) as *const __m128i),
            );
            let p = _mm256_add_epi16(_mm256_maddubs_epi16(ul[k], al), _mm256_maddubs_epi16(uh[k], ah));
            r[k] = _mm256_madd_epi16(p, ones);
        }
        let s = _mm256_hadd_epi32(_mm256_hadd_epi32(r[0], r[1]), _mm256_hadd_epi32(r[2], r[3]));
        let bias = _mm256_cvtepi16_epi32(_mm_loadu_si128(xb.sums.as_ptr() as *const __m128i));
        _mm256_sub_epi32(s, _mm256_slli_epi32::<3>(bias))
    }
}

/// The block's eight fp16 scales as floats, sub-block k in lane k.
#[inline(always)]
unsafe fn q4_0_scales(wb: *const u8) -> __m256 {
    unsafe {
        let h = |k: usize| i16::from_le_bytes([*wb.add(k * Q4_0_BLOCK), *wb.add(k * Q4_0_BLOCK + 1)]);
        _mm256_cvtph_ps(_mm_setr_epi16(h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7)))
    }
}

/// Q4_0 weights times nv vectors: for each of the `rows` rows (weight row r begins at byte r * row_bytes of w and holds
/// nb blocks of 256), output (v, r) goes to y[v * y_stride + r], vector v being the nb blocks from x[v * nb]. Each
/// block's weights are unpacked once for all the vectors, and every vector's sums and float steps are q4_0_dot's, so
/// every result has the same bits as the single-vector product, without a call per row.
///
/// # Safety
/// AVX2, FMA and F16C; w holds rows * row_bytes and x holds nv * nb blocks; y is valid for writes at v * y_stride + r
/// for v < nv and r < rows, with no other access to those elements during the call.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q4_0_dot_multi(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv;
    assert!(nv > 0 && x.len() == nv * nb && row_bytes >= nb * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes);
    const G: usize = 8;
    unsafe {
        for r in 0..rows {
            let wr = w.as_ptr().add(r * row_bytes);
            for v0 in (0..nv).step_by(G) {
                let g = (nv - v0).min(G);
                let mut acc = [_mm256_setzero_ps(); G];
                for b in 0..nb {
                    let wb = wr.add(b * 8 * Q4_0_BLOCK);
                    if g <= 2 {
                        prefetch_block(wb);
                    }
                    let dw = q4_0_scales(wb);
                    let (ul, uh) = nibbles_paired(wb);
                    for (vi, a) in acc[..g].iter_mut().enumerate() {
                        let xb = &x[(v0 + vi) * nb + b];
                        let s = q4_0_sums(&ul, &uh, xb);
                        *a = _mm256_fmadd_ps(
                            _mm256_mul_ps(dw, _mm256_loadu_ps(xb.d.as_ptr())),
                            _mm256_cvtepi32_ps(s),
                            *a,
                        );
                    }
                }
                for (vi, a) in acc[..g].iter().enumerate() {
                    *y.add((v0 + vi) * y_stride + r) = hsum8(*a);
                }
            }
        }
    }
}

/// Rows of Q4_0 against many vectors (a prompt block), 8 rows at a time with one row in each lane. Each 32-weight
/// sub-block of the 8 rows is unpacked (the nibbles as unsigned 0..15) and transposed once; then, for every vector,
/// AVX-VNNI's vpdpbusd builds each row's sub-block sum in its own lane (the sum of (q + 8) a, less 8 x the sum of a:
/// exact), with no horizontal reduction. The float steps for each (row, sub-block) are q4_0_dot's: d_w * g.d, then
/// fused onto that lane's accumulator block after block, with the eight accumulators summed as hsum8. So every result
/// has the same bits as q4_0_dot. rows must be a multiple of 8; y as for q4_0_dot_multi.
///
/// # Safety
/// AVX2, FMA, F16C and AVX-VNNI; otherwise as for q4_0_dot_multi.
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
pub unsafe fn q4_0_gemm8(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { gemm8::<false>(w, row_bytes, rows, x, nv, y, y_stride) }
}

/// Q8_0 (34-byte blocks: an fp16 scale and 32 signed weights) as in q4_0_gemm8: the weights are made unsigned for
/// vpdpbusd (w + 128, an xor of the sign bit), and 128 x the sum of a is taken off, which is exact. The float steps are
/// the same as Q4_0's, so every result has the same bits as the single-vector product.
///
/// # Safety
/// As for q4_0_gemm8.
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
pub unsafe fn q8_0_gemm8(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { gemm8::<true>(w, row_bytes, rows, x, nv, y, y_stride) }
}

pub const Q8_0_BLOCK: usize = 34;

#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
unsafe fn gemm8<const Q8: bool>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let blk = if Q8 { Q8_0_BLOCK } else { Q4_0_BLOCK };
    let nb = x.len() / nv;
    assert!(nv > 0 && x.len() == nv * nb && rows.is_multiple_of(8) && row_bytes >= nb * 8 * blk);
    assert!(w.len() >= rows * row_bytes);
    unsafe {
        let low4 = _mm_set1_epi8(0x0f);
        let sign = _mm256_set1_epi8(-128);
        // on the stack for up to 64 vectors (there is a call per 16-row task, so no allocation each time), else on the
        // heap
        let mut stack = core::mem::MaybeUninit::<[__m256; 64 * 8]>::uninit();
        let mut heap = Vec::new();
        let acc: &mut [__m256] = if nv <= 64 {
            crate::zeroed_prefix(&mut stack, nv * 8)
        } else {
            heap.resize(nv * 8, _mm256_setzero_ps());
            &mut heap
        };
        for r0 in (0..rows).step_by(8) {
            acc.fill(_mm256_setzero_ps());
            for b in 0..nb {
                for k in 0..8 {
                    let at = |i: usize| w.as_ptr().add((r0 + i) * row_bytes + (b * 8 + k) * blk);
                    let h = |i: usize| i16::from_le_bytes([*at(i), *at(i).add(1)]);
                    let dk = _mm256_cvtph_ps(_mm_setr_epi16(h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7)));
                    // row i: its 32 weights as unsigned nibbles, weights 0-15 then 16-31 (dword j holds weights
                    // 4j..4j+3)
                    let row = |i: usize| {
                        if Q8 {
                            _mm256_xor_si256(_mm256_loadu_si256(at(i).add(2) as *const __m256i), sign)
                        } else {
                            let q = _mm_loadu_si128(at(i).add(2) as *const __m128i);
                            _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16::<4>(q), low4), _mm_and_si128(q, low4))
                        }
                    };
                    let rr = [row(0), row(1), row(2), row(3), row(4), row(5), row(6), row(7)];
                    // an 8 x 8 dword transpose: tj = dword j of rows 0..7
                    let t0 = _mm256_unpacklo_epi32(rr[0], rr[1]);
                    let t1 = _mm256_unpackhi_epi32(rr[0], rr[1]);
                    let t2 = _mm256_unpacklo_epi32(rr[2], rr[3]);
                    let t3 = _mm256_unpackhi_epi32(rr[2], rr[3]);
                    let t4 = _mm256_unpacklo_epi32(rr[4], rr[5]);
                    let t5 = _mm256_unpackhi_epi32(rr[4], rr[5]);
                    let t6 = _mm256_unpacklo_epi32(rr[6], rr[7]);
                    let t7 = _mm256_unpackhi_epi32(rr[6], rr[7]);
                    let u0 = _mm256_unpacklo_epi64(t0, t2);
                    let u1 = _mm256_unpackhi_epi64(t0, t2);
                    let u2 = _mm256_unpacklo_epi64(t1, t3);
                    let u3 = _mm256_unpackhi_epi64(t1, t3);
                    let u4 = _mm256_unpacklo_epi64(t4, t6);
                    let u5 = _mm256_unpackhi_epi64(t4, t6);
                    let u6 = _mm256_unpacklo_epi64(t5, t7);
                    let u7 = _mm256_unpackhi_epi64(t5, t7);
                    let tj = [
                        _mm256_permute2x128_si256::<0x20>(u0, u4),
                        _mm256_permute2x128_si256::<0x20>(u1, u5),
                        _mm256_permute2x128_si256::<0x20>(u2, u6),
                        _mm256_permute2x128_si256::<0x20>(u3, u7),
                        _mm256_permute2x128_si256::<0x31>(u0, u4),
                        _mm256_permute2x128_si256::<0x31>(u1, u5),
                        _mm256_permute2x128_si256::<0x31>(u2, u6),
                        _mm256_permute2x128_si256::<0x31>(u3, u7),
                    ];
                    for v in 0..nv {
                        let xb = &x[v * nb + b];
                        let a = xb.qs.as_ptr().add(32 * k) as *const i32;
                        let mut s = _mm256_setzero_si256();
                        for (j, t) in tj.iter().enumerate() {
                            s = _mm256_dpbusd_avx_epi32(s, *t, _mm256_set1_epi32(a.add(j).read_unaligned()));
                        }
                        let suma = xb.sums[k] as i32;
                        s = _mm256_sub_epi32(s, _mm256_set1_epi32(if Q8 { suma << 7 } else { suma << 3 }));
                        let ac = acc.get_unchecked_mut(v * 8 + k);
                        *ac = _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_set1_ps(xb.d[k])), _mm256_cvtepi32_ps(s), *ac);
                    }
                }
            }
            for v in 0..nv {
                let a = &acc[v * 8..v * 8 + 8];
                // hsum8 for each row: ((0+4) + (2+6)) + ((1+5) + (3+7))
                let r = _mm256_add_ps(
                    _mm256_add_ps(_mm256_add_ps(a[0], a[4]), _mm256_add_ps(a[2], a[6])),
                    _mm256_add_ps(_mm256_add_ps(a[1], a[5]), _mm256_add_ps(a[3], a[7])),
                );
                let mut o = [0.0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), r);
                for (i, &val) in o.iter().enumerate() {
                    *y.add(v * y_stride + r0 + i) = val;
                }
            }
        }
    }
}

/// One row of Q4_0 against one vector: neon::q4_0_dot in AVX2 (q4_0_sums), one float lane per block position, each
/// block's sum exact, the lanes summed as hsum8 at the end.
///
/// # Safety
/// The CPU must have AVX2, FMA and F16C, and w must hold x.len() * 8 Q4_0 blocks.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q4_0_dot(w: &[u8], x: &[Groups32]) -> f32 {
    let nb = x.len();
    assert!(w.len() >= nb * 8 * Q4_0_BLOCK);
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (b, xb) in x.iter().enumerate() {
            let wb = w.as_ptr().add(b * 8 * Q4_0_BLOCK);
            prefetch_block(wb);
            let (ul, uh) = nibbles_paired(wb);
            let s = q4_0_sums(&ul, &uh, xb);
            acc = _mm256_fmadd_ps(
                _mm256_mul_ps(q4_0_scales(wb), _mm256_loadu_ps(xb.d.as_ptr())),
                _mm256_cvtepi32_ps(s),
                acc,
            );
        }
        hsum8(acc)
    }
}

/// ((0+4) + (2+6)) + ((1+5) + (3+7)) over eight lanes: ggml's hsum_float_8 (ggml-cpu, MIT), the usual SSE reduction.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum8(a: __m256) -> f32 {
    let mut s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

/// The gated delta rule step on 4 fp16 state rows at a time. Each row's steps are its own and unchanged; they are only
/// interleaved, so that four rows' dependent sums run side by side. The bits are the same. `state` holds rows x dim
/// fp16 bits, v and o hold rows values; dim must be a multiple of 16 (and <= 256), rows a multiple of 4.
///
/// # Safety
/// AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn delta_rule_h(
    state: &mut [u16],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) {
    let rows = v.len();
    assert!(
        dim.is_multiple_of(16) && dim <= 256 && rows.is_multiple_of(4) && o.len() == rows && state.len() >= rows * dim
    );
    assert!(q.len() >= dim && k.len() >= dim);
    unsafe {
        let alpha_v = _mm256_set1_ps(alpha);
        let mut decayed = [[0.0f32; 256]; 4];
        for j0 in (0..rows).step_by(4) {
            let mut a = [[_mm256_setzero_ps(); 2]; 4];
            for i in (0..dim).step_by(16) {
                let (k0, k1) = (_mm256_loadu_ps(k.as_ptr().add(i)), _mm256_loadu_ps(k.as_ptr().add(i + 8)));
                for (rr, row) in decayed.iter_mut().enumerate() {
                    let mp = state.as_ptr().add((j0 + rr) * dim + i);
                    let x0 = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(mp as *const __m128i)), alpha_v);
                    let x1 = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(mp.add(8) as *const __m128i)), alpha_v);
                    _mm256_storeu_ps(row.as_mut_ptr().add(i), x0);
                    _mm256_storeu_ps(row.as_mut_ptr().add(i + 8), x1);
                    a[rr][0] = _mm256_fmadd_ps(x0, k0, a[rr][0]);
                    a[rr][1] = _mm256_fmadd_ps(x1, k1, a[rr][1]);
                }
            }
            let mut delta = [_mm256_setzero_ps(); 4];
            for rr in 0..4 {
                let d = (v[j0 + rr] - hsum8(_mm256_add_ps(a[rr][0], a[rr][1]))) * beta;
                delta[rr] = _mm256_set1_ps(d);
            }
            let mut a = [[_mm256_setzero_ps(); 2]; 4];
            for i in (0..dim).step_by(16) {
                let (k0, k1) = (_mm256_loadu_ps(k.as_ptr().add(i)), _mm256_loadu_ps(k.as_ptr().add(i + 8)));
                let (q0, q1) = (_mm256_loadu_ps(q.as_ptr().add(i)), _mm256_loadu_ps(q.as_ptr().add(i + 8)));
                for (rr, row) in decayed.iter().enumerate() {
                    let mp = state.as_mut_ptr().add((j0 + rr) * dim + i);
                    let x0 = _mm256_fmadd_ps(delta[rr], k0, _mm256_loadu_ps(row.as_ptr().add(i)));
                    let x1 = _mm256_fmadd_ps(delta[rr], k1, _mm256_loadu_ps(row.as_ptr().add(i + 8)));
                    _mm_storeu_si128(mp as *mut __m128i, _mm256_cvtps_ph::<0>(x0));
                    _mm_storeu_si128(mp.add(8) as *mut __m128i, _mm256_cvtps_ph::<0>(x1));
                    a[rr][0] = _mm256_fmadd_ps(x0, q0, a[rr][0]);
                    a[rr][1] = _mm256_fmadd_ps(x1, q1, a[rr][1]);
                }
            }
            for rr in 0..4 {
                o[j0 + rr] = hsum8(_mm256_add_ps(a[rr][0], a[rr][1])) * q_scale;
            }
        }
    }
}

/// Advances a gated DeltaNet's f32 state (qwen4exp; v.len() rows) by one token: neon::delta_rule_f32 in AVX2. Four rows
/// a pass so their dependent sums overlap, k and q loaded once a step for all four, the state read from `src` (a
/// revertible block's base) when one is given, else from `state`, and written to `state`. Each pair of the NEON
/// kernel's 4-lane accumulators is one 8-lane register here (acc[t][0]: elements i..i+8, acc[t][1]: i+8..i+16), summed
/// as hsum8, which is hsum4x2's order, so the bits are crate::delta_rule_f32_scalar's. dim must be a multiple of 16.
///
/// # Safety
/// AVX2 and FMA.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn delta_rule_f32(
    src: Option<&[f32]>,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    q_scale: f32,
    o: &mut [f32],
    dim: usize,
) {
    let rows = v.len();
    assert!(dim.is_multiple_of(16) && o.len() >= rows && state.len() >= rows * dim && q.len() >= dim && k.len() >= dim);
    assert!(src.is_none_or(|s| s.len() >= rows * dim));
    const R: usize = 4;
    let rr = rows / R * R;
    unsafe {
        let alpha_v = _mm256_set1_ps(alpha);
        let (kp, qp) = (k.as_ptr(), q.as_ptr());
        let mp = state.as_mut_ptr();
        let sp: *const f32 = src.map_or(mp as *const f32, |s| s.as_ptr());
        let mut j0 = 0;
        while j0 < rows {
            let n = if j0 < rr { R } else { 1 };
            let mut acc = [[_mm256_setzero_ps(); 2]; R];
            for i in (0..dim).step_by(16) {
                let kv = [_mm256_loadu_ps(kp.add(i)), _mm256_loadu_ps(kp.add(i + 8))];
                for t in 0..n {
                    let (rs, rw) = (sp.add((j0 + t) * dim + i), mp.add((j0 + t) * dim + i));
                    for h in 0..2 {
                        let x = _mm256_mul_ps(_mm256_loadu_ps(rs.add(8 * h)), alpha_v);
                        _mm256_storeu_ps(rw.add(8 * h), x);
                        acc[t][h] = _mm256_fmadd_ps(x, kv[h], acc[t][h]);
                    }
                }
            }
            let mut delta = [_mm256_setzero_ps(); R];
            for t in 0..n {
                let sk = hsum8(_mm256_add_ps(acc[t][0], acc[t][1]));
                delta[t] = _mm256_set1_ps((*v.get_unchecked(j0 + t) - sk) * beta);
            }
            let mut acc = [[_mm256_setzero_ps(); 2]; R];
            for i in (0..dim).step_by(16) {
                let kv = [_mm256_loadu_ps(kp.add(i)), _mm256_loadu_ps(kp.add(i + 8))];
                let qv = [_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(qp.add(i + 8))];
                for t in 0..n {
                    let rw = mp.add((j0 + t) * dim + i);
                    for h in 0..2 {
                        let x = _mm256_fmadd_ps(delta[t], kv[h], _mm256_loadu_ps(rw.add(8 * h)));
                        _mm256_storeu_ps(rw.add(8 * h), x);
                        acc[t][h] = _mm256_fmadd_ps(x, qv[h], acc[t][h]);
                    }
                }
            }
            for t in 0..n {
                *o.get_unchecked_mut(j0 + t) = hsum8(_mm256_add_ps(acc[t][0], acc[t][1])) * q_scale;
            }
            j0 += n;
        }
    }
}

/// Q6_K: 210-byte blocks of 256 weights (ql[128], qh[64], sixteen int8 scales and an fp16 d), where a weight is d *
/// scales[g] * (q - 32) for its 16-weight group g.
pub const Q6K_BLOCK: usize = 210;

/// An 8 x 8 dword transpose: lane i of out[j] is dword j of rr[i].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose8(rr: &[__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(rr[0], rr[1]);
    let t1 = _mm256_unpackhi_epi32(rr[0], rr[1]);
    let t2 = _mm256_unpacklo_epi32(rr[2], rr[3]);
    let t3 = _mm256_unpackhi_epi32(rr[2], rr[3]);
    let t4 = _mm256_unpacklo_epi32(rr[4], rr[5]);
    let t5 = _mm256_unpackhi_epi32(rr[4], rr[5]);
    let t6 = _mm256_unpacklo_epi32(rr[6], rr[7]);
    let t7 = _mm256_unpackhi_epi32(rr[6], rr[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

/// Rows of Q6_K against many vectors, 8 rows at a time with one row in each int32 lane (rows must be a multiple of 8;
/// nb = x.len() / nv blocks per row; y as for q4_0_dot_multi). Each block of the 8 rows is unpacked once (6-bit q,
/// unsigned, in natural weight order) and transposed. Then, for each vector and each 16-weight group, vpdpbusd builds
/// each row's sum of q * a (a being the Q8_K qs) in its lane, which is multiplied by that row's group scale (vpmulld)
/// and summed over the sixteen groups; the -32 offset, 32 * sum_g scale_g * bsums[g] (vpdpwssd), is subtracted. All of
/// this is exact: the block's integer sum is the reference's. There is one float step per block, acc = fma(d_w * x.d,
/// (float)isum, acc), lane-wise, with the blocks in order and no reduction. So every result has the reference's bits.
///
/// # Safety
/// AVX2, FMA, F16C and AVX-VNNI; w holds rows * row_bytes (row_bytes >= nb * 210) and x holds nv * nb blocks; y is
/// valid for writes at v * y_stride + r for v < nv and r < rows, with no other access to those elements during the
/// call.
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
pub unsafe fn q6k_gemm8(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv > 0 && x.len() == nv * nb && rows.is_multiple_of(8) && row_bytes >= nb * Q6K_BLOCK);
    assert!(w.len() >= rows * row_bytes);
    unsafe {
        let low4 = _mm256_set1_epi8(0x0f);
        let low2 = _mm256_set1_epi8(0x03);
        // on the stack for up to 64 vectors, else on the heap
        let mut stack = core::mem::MaybeUninit::<[__m256; 64]>::uninit();
        let mut heap = Vec::new();
        let acc: &mut [__m256] = if nv <= 64 {
            crate::zeroed_prefix(&mut stack, nv)
        } else {
            heap.resize(nv, _mm256_setzero_ps());
            &mut heap
        };
        // the block of the 8 rows, transposed: lane r of wt[i] holds weights 4i..4i+3 of row r (qs dword i, group i
        // / 4)
        let mut wt = [_mm256_setzero_si256(); 64];
        for r0 in (0..rows).step_by(8) {
            acc.fill(_mm256_setzero_ps());
            for b in 0..nb {
                let at = |i: usize| w.as_ptr().add((r0 + i) * row_bytes + b * Q6K_BLOCK);
                let h = |i: usize| i16::from_le_bytes([*at(i).add(208), *at(i).add(209)]);
                let dw = _mm256_cvtph_ps(_mm_setr_epi16(h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7)));
                for n in 0..2 {
                    let mut q = [[_mm256_setzero_si256(); 8]; 4];
                    for i in 0..8 {
                        let p = at(i);
                        let ql0 = _mm256_loadu_si256(p.add(64 * n) as *const __m256i);
                        let ql1 = _mm256_loadu_si256(p.add(64 * n + 32) as *const __m256i);
                        let qh = _mm256_loadu_si256(p.add(128 + 32 * n) as *const __m256i);
                        q[0][i] = _mm256_or_si256(
                            _mm256_and_si256(ql0, low4),
                            _mm256_slli_epi16::<4>(_mm256_and_si256(qh, low2)),
                        );
                        q[1][i] = _mm256_or_si256(
                            _mm256_and_si256(ql1, low4),
                            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(qh), low2)),
                        );
                        q[2][i] = _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16::<4>(ql0), low4),
                            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(qh), low2)),
                        );
                        q[3][i] = _mm256_or_si256(
                            _mm256_and_si256(_mm256_srli_epi16::<4>(ql1), low4),
                            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(qh), low2)),
                        );
                    }
                    for (part, qp) in q.iter().enumerate() {
                        let t = transpose8(qp);
                        let c = 4 * n + part; // weights 32c..32c+31
                        wt[8 * c..8 * c + 8].copy_from_slice(&t);
                    }
                }
                // the scales: lane r of sp[p] holds (scales[2p], scales[2p+1]) of row r as an int16 pair, and sc[g]
                // holds them as int32
                let sr: [__m256i; 8] =
                    core::array::from_fn(|i| _mm256_cvtepi8_epi16(_mm_loadu_si128(at(i).add(192) as *const __m128i)));
                let sp = transpose8(&sr);
                let mut sc = [_mm256_setzero_si256(); 16];
                for p in 0..8 {
                    sc[2 * p] = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(sp[p]));
                    sc[2 * p + 1] = _mm256_srai_epi32::<16>(sp[p]);
                }
                let mut v = 0;
                while v + 4 <= nv {
                    q6k_vecs::<4>(&wt, &sc, &sp, dw, x, v, nb, b, acc);
                    v += 4;
                }
                while v < nv {
                    q6k_vecs::<1>(&wt, &sc, &sp, dw, x, v, nb, b, acc);
                    v += 1;
                }
            }
            for (v, a) in acc.iter().enumerate() {
                let mut o = [0.0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), *a);
                for (i, &val) in o.iter().enumerate() {
                    *y.add(v * y_stride + r0 + i) = val;
                }
            }
        }
    }
}

/// Block b of vectors v0..v0+V against the 8 transposed rows: the exact block sums, then the float step.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
#[allow(clippy::too_many_arguments)]
unsafe fn q6k_vecs<const V: usize>(
    wt: &[__m256i; 64],
    sc: &[__m256i; 16],
    sp: &[__m256i; 8],
    dw: __m256,
    x: &[BlockQ8K],
    v0: usize,
    nb: usize,
    b: usize,
    acc: &mut [__m256],
) {
    unsafe {
        let xb: [&BlockQ8K; V] = core::array::from_fn(|k| x.get_unchecked((v0 + k) * nb + b));
        let qa: [*const i32; V] = core::array::from_fn(|k| xb[k].qs.as_ptr() as *const i32);
        let mut isum = [_mm256_setzero_si256(); V];
        for g in 0..16 {
            let wg = [wt[4 * g], wt[4 * g + 1], wt[4 * g + 2], wt[4 * g + 3]];
            for k in 0..V {
                let mut s = _mm256_setzero_si256();
                for (j, wj) in wg.iter().enumerate() {
                    s = _mm256_dpbusd_avx_epi32(s, *wj, _mm256_set1_epi32(qa[k].add(4 * g + j).read_unaligned()));
                }
                isum[k] = _mm256_add_epi32(isum[k], _mm256_mullo_epi32(s, sc[g]));
            }
        }
        for k in 0..V {
            let bs = xb[k].bsums.as_ptr() as *const i32;
            let mut off = _mm256_setzero_si256();
            for (p, spp) in sp.iter().enumerate() {
                off = _mm256_dpwssd_avx_epi32(off, *spp, _mm256_set1_epi32(bs.add(p).read_unaligned()));
            }
            let is = _mm256_sub_epi32(isum[k], _mm256_slli_epi32::<5>(off));
            let a = acc.get_unchecked_mut(v0 + k);
            *a = _mm256_fmadd_ps(_mm256_mul_ps(dw, _mm256_set1_ps(xb[k].d)), _mm256_cvtepi32_ps(is), *a);
        }
    }
}

/// The bytes of a Q4_K block (fp16 d and dmin, 12 bytes of 6-bit scales and mins, and qs[128]) and of a Q5_K block (the
/// same, with qh[32], the fifth bits, before qs).
pub const Q4K_BLOCK: usize = 144;
pub const Q5K_BLOCK: usize = 176;

/// Rows of Q4_K (five = false) or Q5_K against many vectors, 8 rows at a time with one row in each int32 lane (rows
/// must be a multiple of 8; nb = x.len() / nv blocks per row; y as for q4_0_dot_multi), using q6k_gemm8's scheme. Each
/// block of the 8 rows is unpacked once (weights 0..15 or 0..31, unsigned) and transposed per 32-weight sub-block.
/// Then, for each vector and sub-block k, vpdpbusd builds each row's sum of q * a in its lane, multiplied by that row's
/// scale_k (vpmulld) to give isum; the mins term is imin = sum_k min_k * the sub-block's activation sum (vpdpwssd on
/// (min, min) pairs). All of this is exact: these are the reference's integers. The two float steps per block are acc =
/// fma(d_w * x.d, isum, acc) and accm = fma(dmin_w * x.d, imin, accm), lane-wise, with the blocks in order, and the
/// result is acc - accm, the same bits as x86mm's q45k_pass.
///
/// # Safety
/// AVX2, FMA, F16C and AVX-VNNI; w holds rows * row_bytes (row_bytes >= nb x the block size) and x holds nv * nb
/// blocks; y is valid for writes at v * y_stride + r for v < nv and r < rows, with no other access to those elements
/// during the call.
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
pub unsafe fn q45k_gemm8(
    five: bool,
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe {
        if five {
            q45k_gemm8_t::<true>(w, row_bytes, rows, x, nv, y, y_stride)
        } else {
            q45k_gemm8_t::<false>(w, row_bytes, rows, x, nv, y, y_stride)
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
unsafe fn q45k_gemm8_t<const FIVE: bool>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let bs = if FIVE { Q5K_BLOCK } else { Q4K_BLOCK };
    let nb = x.len() / nv.max(1);
    assert!(nv > 0 && x.len() == nv * nb && rows.is_multiple_of(8) && row_bytes >= nb * bs);
    assert!(w.len() >= rows * row_bytes);
    unsafe {
        let (low4, one) = (_mm256_set1_epi8(0x0f), _mm256_set1_epi8(1));
        // acc[v] and then accm[v]: on the stack for up to 64 vectors, else on the heap
        let mut stack = core::mem::MaybeUninit::<[__m256; 128]>::uninit();
        let mut heap = Vec::new();
        let all: &mut [__m256] = if nv <= 64 {
            crate::zeroed_prefix(&mut stack, 2 * nv)
        } else {
            heap.resize(2 * nv, _mm256_setzero_ps());
            &mut heap
        };
        // lane r of wt[8k + j] holds weights 4j..4j+3 of sub-block k of row r
        let mut wt = [_mm256_setzero_si256(); 64];
        for r0 in (0..rows).step_by(8) {
            all.fill(_mm256_setzero_ps());
            for b in 0..nb {
                let at = |i: usize| w.as_ptr().add((r0 + i) * row_bytes + b * bs);
                let h = |i: usize, o: usize| i16::from_le_bytes([*at(i).add(o), *at(i).add(o + 1)]);
                let dw = _mm256_cvtph_ps(_mm_setr_epi16(
                    h(0, 0),
                    h(1, 0),
                    h(2, 0),
                    h(3, 0),
                    h(4, 0),
                    h(5, 0),
                    h(6, 0),
                    h(7, 0),
                ));
                let dm = _mm256_cvtph_ps(_mm_setr_epi16(
                    h(0, 2),
                    h(1, 2),
                    h(2, 2),
                    h(3, 2),
                    h(4, 2),
                    h(5, 2),
                    h(6, 2),
                    h(7, 2),
                ));
                let sm: [([u8; 8], [u8; 8]); 8] =
                    core::array::from_fn(|i| crate::q45k_scales(&*(at(i).add(4) as *const [u8; 12])));
                let sc: [__m256i; 8] = core::array::from_fn(|k| {
                    let s = |i: usize| sm[i].0[k] as i32;
                    _mm256_setr_epi32(s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7))
                });
                // lane r of mp[p] holds (min_2p, min_2p+1) of row r as an int16 pair
                let mp: [__m256i; 4] = core::array::from_fn(|p| {
                    let m = |i: usize| (sm[i].1[2 * p] as i32) | ((sm[i].1[2 * p + 1] as i32) << 16);
                    _mm256_setr_epi32(m(0), m(1), m(2), m(3), m(4), m(5), m(6), m(7))
                });
                let qs = |i: usize| at(i).add(if FIVE { 48 } else { 16 });
                for j in 0..4 {
                    let mut lo = [_mm256_setzero_si256(); 8];
                    let mut hi = [_mm256_setzero_si256(); 8];
                    for i in 0..8 {
                        let q = _mm256_loadu_si256(qs(i).add(32 * j) as *const __m256i);
                        lo[i] = _mm256_and_si256(q, low4);
                        hi[i] = _mm256_and_si256(_mm256_srli_epi16::<4>(q), low4);
                        if FIVE {
                            // the fifth bit of weight n of sub-block k is bit k of qh[n]
                            let qh = _mm256_loadu_si256(at(i).add(16) as *const __m256i);
                            let (k0, k1) = (2 * j as i32, 2 * j as i32 + 1);
                            let b0 = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(k0)), one);
                            let b1 = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(k1)), one);
                            lo[i] = _mm256_or_si256(lo[i], _mm256_slli_epi16::<4>(b0));
                            hi[i] = _mm256_or_si256(hi[i], _mm256_slli_epi16::<4>(b1));
                        }
                    }
                    wt[8 * (2 * j)..8 * (2 * j) + 8].copy_from_slice(&transpose8(&lo));
                    wt[8 * (2 * j + 1)..8 * (2 * j + 1) + 8].copy_from_slice(&transpose8(&hi));
                }
                let mut v = 0;
                while v + 4 <= nv {
                    q45k_vecs::<4>(&wt, &sc, &mp, dw, dm, x, v, nv, nb, b, all);
                    v += 4;
                }
                while v < nv {
                    q45k_vecs::<1>(&wt, &sc, &mp, dw, dm, x, v, nv, nb, b, all);
                    v += 1;
                }
            }
            for v in 0..nv {
                let mut o = [0.0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), _mm256_sub_ps(all[v], all[nv + v]));
                for (i, &val) in o.iter().enumerate() {
                    *y.add(v * y_stride + r0 + i) = val;
                }
            }
        }
    }
}

/// Block b of vectors v0..v0+V against the 8 transposed Q4_K / Q5_K rows: the exact block sums, then the two float
/// steps (all[v] = acc, all[nv + v] = accm).
#[inline]
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
#[allow(clippy::too_many_arguments)]
unsafe fn q45k_vecs<const V: usize>(
    wt: &[__m256i; 64],
    sc: &[__m256i; 8],
    mp: &[__m256i; 4],
    dw: __m256,
    dm: __m256,
    x: &[BlockQ8K],
    v0: usize,
    nv: usize,
    nb: usize,
    b: usize,
    all: &mut [__m256],
) {
    unsafe {
        let xb: [&BlockQ8K; V] = core::array::from_fn(|k| x.get_unchecked((v0 + k) * nb + b));
        let qa: [*const i32; V] = core::array::from_fn(|k| xb[k].qs.as_ptr() as *const i32);
        let mut isum = [_mm256_setzero_si256(); V];
        for k in 0..8 {
            for u in 0..V {
                let mut s = _mm256_setzero_si256();
                for j in 0..8 {
                    s = _mm256_dpbusd_avx_epi32(
                        s,
                        wt[8 * k + j],
                        _mm256_set1_epi32(qa[u].add(8 * k + j).read_unaligned()),
                    );
                }
                isum[u] = _mm256_add_epi32(isum[u], _mm256_mullo_epi32(s, sc[k]));
            }
        }
        for u in 0..V {
            // the sum of each 32 activations (two sums of 16), paired up as i32 words for dpwssd
            let s32: [i16; 8] = std::array::from_fn(|k| xb[u].bsums[2 * k].wrapping_add(xb[u].bsums[2 * k + 1]));
            let bs = s32.as_ptr() as *const i32;
            let mut im = _mm256_setzero_si256();
            for (p, mpp) in mp.iter().enumerate() {
                im = _mm256_dpwssd_avx_epi32(im, *mpp, _mm256_set1_epi32(bs.add(p).read_unaligned()));
            }
            let xd = _mm256_set1_ps(xb[u].d);
            let a = all.get_unchecked_mut(v0 + u);
            *a = _mm256_fmadd_ps(_mm256_mul_ps(dw, xd), _mm256_cvtepi32_ps(isum[u]), *a);
            let m = all.get_unchecked_mut(nv + v0 + u);
            *m = _mm256_fmadd_ps(_mm256_mul_ps(dm, xd), _mm256_cvtepi32_ps(im), *m);
        }
    }
}
