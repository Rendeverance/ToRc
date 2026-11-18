//! x86dot.rs - Q4_0 rows against one vector on x86-64 (AVX2, FMA, F16C): the NEON kernel's design (neon::q4_0_dot_rows)
//! in 8-lane registers. For each row and block of 256, lane k holds sub-block k: its exact integer sum of nibble times
//! activation, less 8 times the activations' sum; then acc[k] = fma(d_w[k] * g.d[k], sum, acc[k]), and the eight lanes
//! are folded as ((0+4)+(2+6))+((1+5)+(3+7)). That is the reference's order, so the bits are the reference's.

use super::x86mm::{ask_ahead, lookahead_blocks};
use crate::{Groups32, Q4_0_BLOCK};
use core::arch::x86_64::*;

/// The eight lane sums of s[0..8], as the eight lanes of one register (exact).
#[inline(always)]
unsafe fn sums8(s: &[__m256i; 8]) -> __m256i {
    unsafe {
        let a = _mm256_hadd_epi32(_mm256_hadd_epi32(s[0], s[1]), _mm256_hadd_epi32(s[2], s[3]));
        let b = _mm256_hadd_epi32(_mm256_hadd_epi32(s[4], s[5]), _mm256_hadd_epi32(s[6], s[7]));
        // a: lanes 0-3 hold the low halves' sums of s0..s3, and lanes 4-7 their high halves'
        let lo = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
        let hi = _mm_add_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256::<1>(b));
        _mm256_set_m128i(hi, lo)
    }
}

/// The eight lanes added as ((0+4)+(2+6))+((1+5)+(3+7)), in neon.rs hsum4x2's order (fold8 of a register); x86mm
/// uses it too.
#[inline(always)]
pub(crate) unsafe fn fold(a: __m256) -> f32 {
    unsafe {
        let s4 = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        _mm_cvtss_f32(s2) + _mm_cvtss_f32(_mm_shuffle_ps::<1>(s2, s2))
    }
}

/// R rows from w (row_bytes apart) into y[0..R], asking for each row's block `ahead` blocks on as it goes (x86mm's
/// lookahead_blocks: one vector gives the memory too little time otherwise). With VNNI, each sub-block's four-product
/// sums come from vpdpbusd (AVX-VNNI) instead of pmaddubsw + pmaddwd (the integers are the same).
#[inline(always)]
unsafe fn rows_pass<const R: usize, const VNNI: bool>(
    w: *const u8,
    row_bytes: usize,
    x: &[Groups32],
    y: *mut f32,
    ahead: usize,
) {
    unsafe {
        let (m15, ones) = (_mm_set1_epi8(15), _mm256_set1_epi16(1));
        let mut acc = [_mm256_setzero_ps(); R];
        for (b, xb) in x.iter().enumerate() {
            // 8 x the activations' sum for each sub-block, and g.d, shared by the rows
            let bias = _mm256_slli_epi32::<3>(_mm256_cvtepi16_epi32(_mm_loadu_si128(xb.sums.as_ptr().cast())));
            let gd = _mm256_loadu_ps(xb.d.as_ptr());
            let a: [__m256i; 8] = std::array::from_fn(|k| _mm256_loadu_si256(xb.qs.as_ptr().add(32 * k).cast()));
            for r in 0..R {
                let wb = w.add(r * row_bytes + b * 8 * Q4_0_BLOCK);
                ask_ahead(wb, ahead, 8 * Q4_0_BLOCK);
                let s: [__m256i; 8] = std::array::from_fn(|k| {
                    let q = _mm_loadu_si128(wb.add(k * Q4_0_BLOCK + 2).cast());
                    let nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16::<4>(q), m15), _mm_and_si128(q, m15));
                    if VNNI {
                        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), nib, a[k])
                    } else {
                        _mm256_madd_epi16(_mm256_maddubs_epi16(nib, a[k]), ones)
                    }
                });
                let dot = _mm256_sub_epi32(sums8(&s), bias);
                let h: [u16; 8] =
                    std::array::from_fn(|k| core::ptr::read_unaligned(wb.add(k * Q4_0_BLOCK) as *const u16));
                let dw = _mm256_cvtph_ps(_mm_loadu_si128(h.as_ptr().cast()));
                acc[r] = _mm256_fmadd_ps(_mm256_mul_ps(dw, gd), _mm256_cvtepi32_ps(dot), acc[r]);
            }
        }
        for r in 0..R {
            *y.add(r) = fold(acc[r]);
        }
    }
}

#[inline(always)]
unsafe fn each_row<const VNNI: bool>(w: &[u8], row_bytes: usize, rows: usize, x: &[Groups32], y: &mut [f32]) {
    let (r2, ahead) = (rows / 2 * 2, lookahead_blocks(1));
    unsafe {
        for r in (0..r2).step_by(2) {
            rows_pass::<2, VNNI>(w.as_ptr().add(r * row_bytes), row_bytes, x, y.as_mut_ptr().add(r), ahead);
        }
        if r2 < rows {
            rows_pass::<1, VNNI>(w.as_ptr().add(r2 * row_bytes), row_bytes, x, y.as_mut_ptr().add(r2), ahead);
        }
    }
}

#[target_feature(enable = "avxvnni")]
unsafe fn rows_vnni(w: &[u8], row_bytes: usize, n: usize, x: &[Groups32], y: &mut [f32]) {
    unsafe { each_row::<true>(w, row_bytes, n, x, y) }
}

/// Q4_0 matrix times one vector of x.len() blocks of 256 (weight row r begins at byte r * row_bytes of w), result y[r],
/// two rows per pass (with the activations loaded once for both). It uses AVX-VNNI where the CPU has it (TORC_VNNI=0
/// turns that off).
///
/// # Safety
/// The CPU must have AVX2, FMA and F16C (the build's x86-64-v3).
pub unsafe fn q4_0_rows(w: &[u8], row_bytes: usize, rows: usize, x: &[Groups32], y: &mut [f32]) {
    assert!(row_bytes >= x.len() * 8 * Q4_0_BLOCK && w.len() >= rows * row_bytes && y.len() >= rows);
    static VNNI: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::arch::is_x86_feature_detected!("avxvnni") && !crate::switches::is("TORC_VNNI", "0")
    });
    unsafe { if *VNNI { rows_vnni(w, row_bytes, rows, x, y) } else { each_row::<false>(w, row_bytes, rows, x, y) } }
}
