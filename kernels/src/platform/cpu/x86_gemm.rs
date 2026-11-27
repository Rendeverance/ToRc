//! x86_gemm.rs - packed GEMM for the 32-weight block types (Q4_0, Q8_0, MXFP4, IQ4_NL) against many vectors, with VNNI:
//! AVX-VNNI, or AVX512-VNNI's form of the same instruction on 256-bit registers (x86mm::vnni).
//!
//! The weights of 8 rows are packed once per call into a panel: every 32-weight block of the 8 rows is
//! transposed so that each dword lane of a register holds one row's four weights, unsigned (Q4_0 as its
//! nibbles 0..15, Q8_0 plus 128, MXFP4 as its doubled level plus 12, IQ4_NL as its level plus 128), beside the 8
//! rows' scales as floats. A
//! register-blocked microkernel then computes 8 rows x MR vectors with its accumulators in registers: per
//! block, eight `vpdpbusd` leave each row's integer sum in its own lane, the offset (8, 128, 12 or 128 times the
//! block's activation sum) is taken off, exactly, and one fused multiply-add puts the block on that row's
//! float accumulator.
//!
//! The float steps per row and vector are the scalar reference's (`q4_0_dot_scalar`): block position k (0..7)
//! of every 256 has its own accumulator, acc_k = fma(d_w * d_x, isum, acc_k), blocks in order, and the eight
//! are summed pairwise at the end, ((0+4) + (2+6)) + ((1+5) + (3+7)). The microkernel runs position k across
//! all blocks before moving to k + 1, so it needs one accumulator per vector instead of eight. Every result has
//! the same bits as the single-vector kernels (MXFP4's and IQ4_NL's: x86mm's lanes_pass, whose lane k is block
//! position k with the same fma and the same fold).

use crate::Groups32;
use core::arch::x86_64::*;

const Q4_0_BYTES: usize = 18;
const Q8_0_BYTES: usize = 34;
/// Vectors a microkernel call holds in registers: 8 for the packed weights, MR accumulators, the rest scratch.
const MR: usize = 4;

/// The weight types (gemm's T): Q4_0, Q8_0, MXFP4, IQ4_NL.
const Q4_0: u8 = 0;
const Q8_0: u8 = 1;
const MXFP4: u8 = 2;
const IQ4_NL: u8 = 3;

/// One 32-weight block of a panel's 8 rows: w[j] lane r = weights 4j..4j+3 of row r (unsigned bytes), d lane
/// r = row r's scale.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct PackedBlock {
    w: [__m256i; 8],
    d: __m256,
}

thread_local! {
    static PANEL: core::cell::RefCell<Vec<PackedBlock>> = const { core::cell::RefCell::new(Vec::new()) };
    static OFFSETS: core::cell::RefCell<Vec<i32>> = const { core::cell::RefCell::new(Vec::new()) };
}

/// Rows of Q4_0 against nv vectors: y[v * y_stride + r] for row r < rows (a multiple of 8, row r at
/// w[r * row_bytes..]) and vector v at x[v * nb..] (nb = x.len() / nv blocks of 256).
///
/// # Safety
/// AVX2, FMA and F16C, and VNNI in either form (x86mm::vnni() > 0); w holds rows * row_bytes; y is valid for
/// writes at v * y_stride + r for v < nv, r < rows, with no other access to those elements during the call.
pub unsafe fn gemm_q4_0(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { run::<Q4_0>(w, row_bytes, rows, x, nv, y, y_stride) }
}

/// Rows of Q8_0 against nv vectors, as [`gemm_q4_0`].
///
/// # Safety
/// As [`gemm_q4_0`].
pub unsafe fn gemm_q8_0(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { run::<Q8_0>(w, row_bytes, rows, x, nv, y, y_stride) }
}

/// Rows of MXFP4 (17-byte blocks: an E8M0 scale and 32 four-bit levels) against nv vectors, as [`gemm_q4_0`].
///
/// # Safety
/// As [`gemm_q4_0`].
pub unsafe fn gemm_mxfp4(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { run::<MXFP4>(w, row_bytes, rows, x, nv, y, y_stride) }
}

/// Rows of IQ4_NL (18-byte blocks: an fp16 scale and 32 four-bit indices into IQ4_LEVELS) against nv vectors, as
/// [`gemm_q4_0`] (the Inkling and Flash Next experts' type).
///
/// # Safety
/// As [`gemm_q4_0`].
pub unsafe fn gemm_iq4_nl(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    unsafe { run::<IQ4_NL>(w, row_bytes, rows, x, nv, y, y_stride) }
}

/// The thread's panel and offsets, then the product in the VNNI form the CPU has.
unsafe fn run<const T: u8>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
) {
    let bytes = match T {
        Q4_0 | IQ4_NL => Q4_0_BYTES,
        Q8_0 => Q8_0_BYTES,
        _ => crate::MXFP4_BLOCK,
    };
    let nb = x.len() / nv.max(1);
    assert!(nv > 0 && x.len() == nv * nb && rows.is_multiple_of(8) && row_bytes >= nb * 8 * bytes);
    assert!(w.len() >= rows * row_bytes);
    // the offsets the unsigned weights add: per vector, block of 256 and position k, 8, 128 or 12 times the sum
    // of the block's 32 activations (exact in i32)
    let times = match T {
        Q4_0 => 8,
        Q8_0 | IQ4_NL => 128,
        _ => 12,
    };
    OFFSETS.with_borrow_mut(|off| {
        off.clear();
        off.extend(x.iter().flat_map(|xb| (0..8).map(move |k| xb.sums[k] as i32 * times)));
        PANEL.with_borrow_mut(|panel| {
            // SAFETY: the zeroed __m256 fields are plain data; the CPU's features are the caller's promise
            unsafe {
                panel.resize(nb * 8, PackedBlock { w: [_mm256_setzero_si256(); 8], d: _mm256_setzero_ps() });
                match crate::x86mm::vnni() {
                    1 => gemm_avx::<T>(w, row_bytes, rows, x, nv, y, y_stride, panel, off),
                    2 => gemm_512::<T>(w, row_bytes, rows, x, nv, y, y_stride, panel, off),
                    _ => panic!("x86_gemm: the CPU has no VNNI"),
                }
            }
        })
    })
}

#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
unsafe fn gemm_avx<const T: u8>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
    panel: &mut [PackedBlock],
    off: &[i32],
) {
    unsafe { gemm::<T, 1>(w, row_bytes, rows, x, nv, y, y_stride, panel, off) }
}

#[target_feature(enable = "avx2,fma,f16c,avx512vnni,avx512vl")]
unsafe fn gemm_512<const T: u8>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
    panel: &mut [PackedBlock],
    off: &[i32],
) {
    unsafe { gemm::<T, 2>(w, row_bytes, rows, x, nv, y, y_stride, panel, off) }
}

/// The product, 8 rows at a time; E is the VNNI form (1 AVX-VNNI, 2 AVX512-VNNI on 256-bit registers).
#[inline(always)]
unsafe fn gemm<const T: u8, const E: u8>(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: *mut f32,
    y_stride: usize,
    panel: &mut [PackedBlock],
    off: &[i32],
) {
    let nb = x.len() / nv;
    for r0 in (0..rows).step_by(8) {
        unsafe {
            pack::<T>(w, row_bytes, r0, nb, panel);
            let mut v0 = 0;
            while v0 < nv {
                let ys = y.add(v0 * y_stride + r0);
                match nv - v0 {
                    1 => micro::<1, E>(panel, x, off, v0, nb, ys, y_stride),
                    2 => micro::<2, E>(panel, x, off, v0, nb, ys, y_stride),
                    3 => micro::<3, E>(panel, x, off, v0, nb, ys, y_stride),
                    _ => micro::<MR, E>(panel, x, off, v0, nb, ys, y_stride),
                }
                v0 += MR;
            }
        }
    }
}

/// 8 x 8 dword transpose: out[j] lane i = dword j of r[i].
#[inline(always)]
unsafe fn transpose8(r: &[__m256i; 8]) -> [__m256i; 8] {
    unsafe {
        let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
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
}

/// Packs rows r0..r0 + 8 into the panel: entry b * 8 + k is block k of 256-block b.
#[inline(always)]
unsafe fn pack<const T: u8>(w: &[u8], row_bytes: usize, r0: usize, nb: usize, panel: &mut [PackedBlock]) {
    let bytes = match T {
        Q4_0 | IQ4_NL => Q4_0_BYTES,
        Q8_0 => Q8_0_BYTES,
        _ => crate::MXFP4_BLOCK,
    };
    unsafe {
        let low4 = _mm_set1_epi8(0x0f);
        let sign = _mm256_set1_epi8(-128);
        // MXFP4's doubled levels plus 12 (0..24), looked up by nibble
        let lv = _mm256_add_epi8(
            _mm256_broadcastsi128_si256(_mm_loadu_si128(crate::MXFP4_LEVELS.as_ptr().cast())),
            _mm256_set1_epi8(12),
        );
        // IQ4_NL's levels plus 128 (1..241, unsigned), looked up by nibble
        let lq = _mm256_add_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128(crate::IQ4_LEVELS.as_ptr().cast())), sign);
        // the 32 weights' nibbles, weights 0-15 then 16-31, from the 16 bytes at p
        let nibbles = |p: *const u8| {
            let q = _mm_loadu_si128(p as *const __m128i);
            _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16::<4>(q), low4), _mm_and_si128(q, low4))
        };
        for (i, pb) in panel.iter_mut().enumerate().take(nb * 8) {
            let at = |r: usize| w.as_ptr().add((r0 + r) * row_bytes + i * bytes);
            pb.d = if T == MXFP4 {
                // one E8M0 byte each, halved (e8m0_half, exact)
                let e: [f32; 8] = std::array::from_fn(|r| crate::e8m0_half(*at(r)));
                _mm256_loadu_ps(e.as_ptr())
            } else {
                let h = |r: usize| i16::from_le_bytes([*at(r), *at(r).add(1)]);
                _mm256_cvtph_ps(_mm_setr_epi16(h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7)))
            };
            let row = |r: usize| match T {
                Q8_0 => _mm256_xor_si256(_mm256_loadu_si256(at(r).add(2) as *const __m256i), sign),
                Q4_0 => nibbles(at(r).add(2)),
                IQ4_NL => _mm256_shuffle_epi8(lq, nibbles(at(r).add(2))),
                _ => _mm256_shuffle_epi8(lv, nibbles(at(r).add(1))),
            };
            pb.w = transpose8(&[row(0), row(1), row(2), row(3), row(4), row(5), row(6), row(7)]);
        }
    }
}

/// s + the four byte products of each lane of w (unsigned) and a (signed), in VNNI form E.
#[inline(always)]
unsafe fn dp<const E: u8>(s: __m256i, w: __m256i, a: __m256i) -> __m256i {
    unsafe { if E == 2 { _mm256_dpbusd_epi32(s, w, a) } else { _mm256_dpbusd_avx_epi32(s, w, a) } }
}

/// The microkernel: the panel's 8 rows against vectors v0..v0 + V, results at ys[v * y_stride + r].
#[inline(always)]
unsafe fn micro<const V: usize, const E: u8>(
    panel: &[PackedBlock],
    x: &[Groups32],
    off: &[i32],
    v0: usize,
    nb: usize,
    ys: *mut f32,
    y_stride: usize,
) {
    unsafe {
        let mut part = [[_mm256_setzero_ps(); 8]; V];
        for k in 0..8 {
            let mut acc = [_mm256_setzero_ps(); V];
            for b in 0..nb {
                let pb = panel.get_unchecked(b * 8 + k);
                for (v, a) in acc.iter_mut().enumerate() {
                    let xb = x.get_unchecked((v0 + v) * nb + b);
                    let q = xb.qs.as_ptr().add(32 * k) as *const i32;
                    let mut s = _mm256_setzero_si256();
                    for (j, wj) in pb.w.iter().enumerate() {
                        s = dp::<E>(s, *wj, _mm256_set1_epi32(q.add(j).read_unaligned()));
                    }
                    s = _mm256_sub_epi32(s, _mm256_set1_epi32(*off.get_unchecked(((v0 + v) * nb + b) * 8 + k)));
                    *a = _mm256_fmadd_ps(_mm256_mul_ps(pb.d, _mm256_set1_ps(xb.d[k])), _mm256_cvtepi32_ps(s), *a);
                }
            }
            for v in 0..V {
                part[v][k] = acc[v];
            }
        }
        for (v, p) in part.iter().enumerate() {
            let r = _mm256_add_ps(
                _mm256_add_ps(_mm256_add_ps(p[0], p[4]), _mm256_add_ps(p[2], p[6])),
                _mm256_add_ps(_mm256_add_ps(p[1], p[5]), _mm256_add_ps(p[3], p[7])),
            );
            _mm256_storeu_ps(ys.add(v * y_stride), r);
        }
    }
}
