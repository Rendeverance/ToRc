//! fattn.rs - ToRc's attention, built from SPEC.md section 1.
//!
//! Split-K attention in two stages. The keys and values are cut into pages of PAGE positions at absolute positions. In
//! the first stage every (KV head, page, query-head subgroup, token tile) is a work item that computes, for each of
//! its slots (a token and a query head), the page's partial result: the largest score, the sum of the weights and the
//! weighted sum of the values. In the second stage every (token, query head) row combines its pages' partials, in page
//! order, into the output. A partial depends only on its own slot and page, never on which item or thread made it, so a
//! token's output has the same bits alone or in any block, on any number of threads.
//!
//! On ToRc's own pool both stages run in one round: threads take first-stage items from a shared cursor, meet at a
//! barrier when the last one is done, then take output rows from a second cursor. Under rayon, where a thread may not
//! wait for another's work, each stage is a round of its own.

use crate::act::expf;
use crate::attn::KvCache;
use crate::pool::SyncPtr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Positions per page (absolute: page p holds positions PAGE p .. PAGE (p + 1) - 1).
const PAGE: usize = 256;

pub struct Fattn {
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    scale: f32,
    window: usize,
    /// gpt-oss: each query head's sink, a logit that takes a share of the softmax and points at no value (None for
    /// every other model)
    sinks: Option<Vec<f32>>,
}

/// A selection of positions for each token (QSA: qwen4exp's indexer picks, for each token, the pools of `kp` positions
/// it attends to). Token j's row of flags is at rows[j stride..], one flag per pool, and nv[j] is the number of pools
/// the flags cover. The positions past them (the tail) are all seen; u32::MAX means the token has no selection and sees
/// every position.
pub struct Sel<'a> {
    pub rows: &'a [u8],
    pub nv: &'a [u32],
    pub stride: usize,
    pub kp: usize,
}

/// The first stage's results for one call: per (page, token, query head), with pages counted from the call's first
/// page, the largest score and the weight sum (`stat`, two numbers) and the weighted value sum (`acc`, head_dim
/// numbers).
struct Partials {
    stat: SyncPtr<f32>,
    acc: SyncPtr<f32>,
    n: usize,
    n_head: usize,
    head_dim: usize,
}

impl Partials {
    fn index(&self, page: usize, j: usize, h: usize) -> usize {
        (page * self.n + j) * self.n_head + h
    }

    /// The (largest score, weight sum) pair of a slot's page.
    ///
    /// # Safety
    /// No other thread may use the same (page, j, h) at the same time.
    #[allow(clippy::mut_from_ref)]
    unsafe fn stat(&self, page: usize, j: usize, h: usize) -> &mut [f32] {
        unsafe { self.stat.slice(self.index(page, j, h) * 2, 2) }
    }

    /// The weighted value sum of a slot's page.
    ///
    /// # Safety
    /// As for stat.
    #[allow(clippy::mut_from_ref)]
    unsafe fn acc(&self, page: usize, j: usize, h: usize) -> &mut [f32] {
        unsafe { self.acc.slice(self.index(page, j, h) * self.head_dim, self.head_dim) }
    }
}

/// TORC_ATTN_F32 (ARM, and x86-64 with AVX2; on by default, =0 turns it off): an item's page of keys and values is
/// converted to f32 once (exactly, since they are small integers), so the loops then run without conversions, with the
/// scores two positions a pass (neon::scores_f32). It applies from TORC_ATTN_F32_MIN slots (default 64; below 9 the
/// per-batch conversion already happens once a row). The same numbers in the same order give the same bits.
fn f32_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        cfg!(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))
            && crate::switches::get("TORC_ATTN_F32").is_none_or(|v| v != "0")
    });
    *ON
}

/// TORC_ATTN_FAST=1: the scores over each query at sixteen bits (fast::scores). Off by default: the scores are then no
/// longer the exact path's bits (they differ by the query's rounding, about 3e-5 of the largest score).
fn fast_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_ATTN_FAST"));
    *ON
}

/// Whether fast::scores takes this cache and head width: int8 keys, and a head of at most 256 numbers in whole
/// sixteens (so a score's integer sum stays inside 32 bits: 256 x 127 x 32767 is under 2^31).
fn fast_fits(kv: &KvCache, hd: usize) -> bool {
    kv.k16.is_empty() && hd <= fast::HD_MAX && hd.is_multiple_of(16)
}

/// The fast scores (TORC_ATTN_FAST=1). A query row is rounded to sixteen-bit integers with a step of its own (its
/// largest magnitude goes to 32767), and its product with a row of int8 keys is then an exact integer sum, which the
/// processor does sixteen numbers to an instruction where the float path does four (NEON) or eight (AVX2). The score
/// is that sum times the step, the scale and the key's scale. A slot's scores depend only on its own query and the
/// keys, so a token gives the same bits alone or in any block; every kernel here gives the portable one's bits.
mod fast {
    use super::*;

    /// The widest head the integer sums allow.
    pub(super) const HD_MAX: usize = 256;

    /// q at sixteen bits: q[i] is about out[i] x the step returned (0 for a row of zeros).
    pub(super) fn quantize(q: &[f32], out: &mut [i16]) -> f32 {
        let m = q.iter().fold(0f32, |a, &x| a.max(x.abs()));
        if m == 0.0 {
            out.fill(0);
            return 0.0;
        }
        let up = 32767.0 / m;
        for (o, &x) in out.iter_mut().zip(q) {
            *o = (x * up).round_ties_even() as i16;
        }
        m / 32767.0
    }

    /// The integer sums of one key row against NS queries: the definition.
    #[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
    pub(super) fn dots_portable<const NS: usize>(k: &[i8], q: &[[i16; HD_MAX]; NS], hd: usize) -> [i32; NS] {
        std::array::from_fn(|s| k[..hd].iter().zip(&q[s][..hd]).map(|(&a, &b)| a as i32 * b as i32).sum())
    }

    #[cfg(target_arch = "x86_64")]
    mod x86 {
        use super::HD_MAX;
        use std::arch::x86_64::*;

        #[inline(always)]
        unsafe fn sum8(v: __m256i) -> i32 {
            unsafe {
                let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b0100_1110>(s));
                _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32::<0b1011_0001>(s)))
            }
        }

        /// AVX2: sixteen keys widened to 16 bits, multiplied and added in pairs (vpmaddwd), then added in.
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn dots_avx2<const NS: usize>(k: *const i8, q: &[[i16; HD_MAX]; NS], hd: usize) -> [i32; NS] {
            unsafe {
                let mut a = [_mm256_setzero_si256(); NS];
                let mut i = 0;
                while i < hd {
                    let kw = _mm256_cvtepi8_epi16(_mm_loadu_si128(k.add(i).cast()));
                    for s in 0..NS {
                        let qv = _mm256_loadu_si256(q[s].as_ptr().add(i).cast());
                        a[s] = _mm256_add_epi32(a[s], _mm256_madd_epi16(kw, qv));
                    }
                    i += 16;
                }
                a.map(|v| sum8(v))
            }
        }

        /// AVX-VNNI: the multiply, the pair sum and the add in one instruction (vpdpwssd).
        #[target_feature(enable = "avx2,avxvnni")]
        pub(super) unsafe fn dots_vnni<const NS: usize>(k: *const i8, q: &[[i16; HD_MAX]; NS], hd: usize) -> [i32; NS] {
            unsafe {
                let mut a = [_mm256_setzero_si256(); NS];
                let mut i = 0;
                while i < hd {
                    let kw = _mm256_cvtepi8_epi16(_mm_loadu_si128(k.add(i).cast()));
                    for s in 0..NS {
                        let qv = _mm256_loadu_si256(q[s].as_ptr().add(i).cast());
                        a[s] = _mm256_dpwssd_avx_epi32(a[s], kw, qv);
                    }
                    i += 16;
                }
                a.map(|v| sum8(v))
            }
        }

        /// AVX512-VNNI at 256 bits (the processors that have it without AVX-VNNI).
        #[target_feature(enable = "avx2,avx512vnni,avx512vl")]
        pub(super) unsafe fn dots_vnni512<const NS: usize>(
            k: *const i8,
            q: &[[i16; HD_MAX]; NS],
            hd: usize,
        ) -> [i32; NS] {
            unsafe {
                let mut a = [_mm256_setzero_si256(); NS];
                let mut i = 0;
                while i < hd {
                    let kw = _mm256_cvtepi8_epi16(_mm_loadu_si128(k.add(i).cast()));
                    for s in 0..NS {
                        let qv = _mm256_loadu_si256(q[s].as_ptr().add(i).cast());
                        a[s] = _mm256_dpwssd_epi32(a[s], kw, qv);
                    }
                    i += 16;
                }
                a.map(|v| sum8(v))
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod arm {
        use super::HD_MAX;
        use std::arch::aarch64::*;
        use std::arch::asm;

        #[inline]
        #[target_feature(enable = "dotprod")]
        unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
            unsafe {
                asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
                    options(pure, nomem, nostack));
            }
            acc
        }

        /// USDOT: acc.4s += u.16b (unsigned) . s.16b (signed).
        #[inline]
        #[target_feature(enable = "i8mm")]
        unsafe fn usdot(mut acc: int32x4_t, u: uint8x16_t, s: int8x16_t) -> int32x4_t {
            unsafe {
                asm!("usdot {acc:v}.4s, {u:v}.16b, {s:v}.16b", acc = inout(vreg) acc, u = in(vreg) u, s = in(vreg) s,
                    options(pure, nomem, nostack));
            }
            acc
        }

        /// A query's sixteen-bit numbers as their high bytes (signed) and low bytes (unsigned): q = 256 hi + lo, so
        /// the sum with the keys is 256 x SDOT(keys, hi) + USDOT(lo, keys), sixteen keys to each instruction.
        pub(super) fn split(q: &[i16; HD_MAX], hi: &mut [i8; HD_MAX], lo: &mut [u8; HD_MAX]) {
            for i in 0..HD_MAX {
                hi[i] = (q[i] >> 8) as i8;
                lo[i] = q[i] as u8;
            }
        }

        #[target_feature(enable = "dotprod,i8mm")]
        pub(super) unsafe fn dots<const NS: usize>(
            k: *const i8,
            hi: &[[i8; HD_MAX]; NS],
            lo: &[[u8; HD_MAX]; NS],
            hd: usize,
        ) -> [i32; NS] {
            unsafe {
                let mut ah = [vdupq_n_s32(0); NS];
                let mut al = [vdupq_n_s32(0); NS];
                let mut i = 0;
                while i < hd {
                    let kv = vld1q_s8(k.add(i));
                    for s in 0..NS {
                        ah[s] = sdot(ah[s], kv, vld1q_s8(hi[s].as_ptr().add(i)));
                        al[s] = usdot(al[s], vld1q_u8(lo[s].as_ptr().add(i)), kv);
                    }
                    i += 16;
                }
                std::array::from_fn(|s| (256 * vaddvq_s32(ah[s]) as i64 + vaddvq_s32(al[s]) as i64) as i32)
            }
        }
    }

    /// portable::scores' arguments, over the queries at sixteen bits. The caller has checked fast_fits.
    pub(super) fn scores<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [&[f32]; NS],
        sc: &mut [f32],
    ) {
        assert!(sc.len() >= NS * PAGE && nt <= PAGE && hd <= HD_MAX && hd.is_multiple_of(16));
        assert!(kv.k16.is_empty() && kv.k.len() >= (base + nt) * hd && qs.iter().all(|q| q.len() >= hd));
        let mut qi = [[0i16; HD_MAX]; NS];
        let step: [f32; NS] = std::array::from_fn(|s| quantize(&qs[s][..hd], &mut qi[s][..hd]));
        let ks = &kv.k_scale[base..base + nt];
        let k = &kv.k[base * hd..(base + nt) * hd];
        let put = |t: usize, sums: [i32; NS], sc: &mut [f32]| {
            for s in 0..NS {
                sc[s * PAGE + t] = ((sums[s] as f32 * step[s]) * scale) * ks[t];
            }
        };
        #[cfg(target_arch = "x86_64")]
        {
            // 0: AVX2, 1: AVX-VNNI, 2: AVX512-VNNI at 256 bits
            static KIND: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| {
                if std::arch::is_x86_feature_detected!("avxvnni") {
                    1
                } else if std::arch::is_x86_feature_detected!("avx512vnni")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                {
                    2
                } else if std::arch::is_x86_feature_detected!("avx2") {
                    0
                } else {
                    3
                }
            });
            if *KIND < 3 {
                for t in 0..nt {
                    let row = k[t * hd..].as_ptr();
                    // SAFETY: the row holds hd keys and each query hd numbers (checked), and the feature was detected
                    let sums = unsafe {
                        match *KIND {
                            1 => x86::dots_vnni::<NS>(row, &qi, hd),
                            2 => x86::dots_vnni512::<NS>(row, &qi, hd),
                            _ => x86::dots_avx2::<NS>(row, &qi, hd),
                        }
                    };
                    put(t, sums, sc);
                }
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            static OK: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                std::arch::is_aarch64_feature_detected!("dotprod") && std::arch::is_aarch64_feature_detected!("i8mm")
            });
            if *OK {
                let (mut hi, mut lo) = ([[0i8; HD_MAX]; NS], [[0u8; HD_MAX]; NS]);
                for s in 0..NS {
                    arm::split(&qi[s], &mut hi[s], &mut lo[s]);
                }
                for t in 0..nt {
                    // SAFETY: as above
                    let sums = unsafe { arm::dots::<NS>(k[t * hd..].as_ptr(), &hi, &lo, hd) };
                    put(t, sums, sc);
                }
                return;
            }
        }
        #[allow(unreachable_code)]
        for t in 0..nt {
            put(t, dots_portable::<NS>(&k[t * hd..], &qi, hd), sc);
        }
    }

    /// For the tests: the portable sums whatever the processor.
    #[cfg(test)]
    pub(super) fn scores_portable<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [&[f32]; NS],
        sc: &mut [f32],
    ) {
        let mut qi = [[0i16; HD_MAX]; NS];
        let step: [f32; NS] = std::array::from_fn(|s| quantize(&qs[s][..hd], &mut qi[s][..hd]));
        let ks = &kv.k_scale[base..base + nt];
        for t in 0..nt {
            let sums = dots_portable::<NS>(&kv.k[(base + t) * hd..], &qi, hd);
            for s in 0..NS {
                sc[s * PAGE + t] = ((sums[s] as f32 * step[s]) * scale) * ks[t];
            }
        }
    }
}

/// scores::<NS>, or over the item's f32 copy (kf: rows base..base + nt) when f32c is set; with TORC_ATTN_FAST=1, the
/// fast scores (fast::scores) where they apply.
#[allow(unused_variables, clippy::too_many_arguments)]
#[inline(always)]
fn scores_c<const NS: usize>(
    f32c: bool,
    kf: &[f32],
    kv: &KvCache,
    base: usize,
    nt: usize,
    hd: usize,
    scale: f32,
    qs: [&[f32]; NS],
    sc: &mut [f32],
) {
    if fast_on() && fast_fits(kv, hd) {
        return fast::scores::<NS>(kv, base, nt, hd, scale, qs, sc);
    }
    #[cfg(target_arch = "aarch64")]
    if f32c {
        return neon::scores_f32::<NS>(kf, &kv.k_scale[base..base + nt], nt, hd, scale, qs, sc);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    if f32c {
        return x86::scores_f32::<NS>(kf, &kv.k_scale[base..base + nt], nt, hd, scale, qs, sc);
    }
    scores::<NS>(kv, base, nt, hd, scale, qs, sc)
}

/// values::<NS>, or over the item's f32 copy (vf) when f32c is set.
#[allow(unused_variables, clippy::too_many_arguments)]
#[inline(always)]
fn values_c<const NS: usize>(
    f32c: bool,
    vf: &[f32],
    kv: &KvCache,
    base: usize,
    nt: usize,
    hd: usize,
    w: &[f32],
    out: [&mut [f32]; NS],
) {
    #[cfg(target_arch = "aarch64")]
    if f32c {
        return neon::values_f32::<NS>(vf, nt, hd, w, out);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    if f32c {
        return x86::values_f32::<NS>(vf, nt, hd, w, out);
    }
    values::<NS>(kv, base, nt, hd, w, out)
}

/// The decode gather (Fattn::item, ARM and x86-64 with AVX2): on unless TORC_QSA_GATHER=0.
#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
fn gather_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_QSA_GATHER", "0"));
    *ON
}

/// The model's default for the tiled path when TORC_ATTN_TILE is unset (set_tile_default).
static TILE_DEFAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets the model's default for the tiled path (tile_on): on for the model it was measured on, the Qwen3.6-35B's
/// architecture, where a draft-checking pass at a long context is faster through it with the same bits. Call it
/// before the model's first attention pass. It changes nothing off ARM, where there is no tiled path.
pub fn set_tile_default(on: bool) {
    TILE_DEFAULT.store(on, Ordering::Relaxed);
}

/// The tiled f32 path of item (ARM), for 16 slots or more (TORC_ATTN_TILE_MIN: another count) but below
/// TORC_ATTN_F32_MIN: TORC_ATTN_TILE=1 / 0, else the model's default. Fixed at the first call.
#[cfg(target_arch = "aarch64")]
fn tile_on() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| match crate::switches::get("TORC_ATTN_TILE").as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => TILE_DEFAULT.load(Ordering::Relaxed),
        });
    *ON
}

/// The fewest slots the tiled path takes: TORC_ATTN_TILE_MIN, else 16.
#[cfg(target_arch = "aarch64")]
fn tile_min() -> usize {
    static MIN: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| crate::switches::num("TORC_ATTN_TILE_MIN").unwrap_or(16).max(1));
    *MIN
}

/// Eight lanes summed as a pairwise tree: ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7)).
#[inline(always)]
fn fold8(a: &[f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// A cache row as f32 (exact), without its scale applied (the tests' reference and run_block_exact).
fn row_f32(kv: &KvCache, v: bool, at: usize, hd: usize, out: &mut [f32]) {
    if kv.k16.is_empty() {
        let b = if v { &kv.v } else { &kv.k };
        for (o, &x) in out.iter_mut().zip(&b[at..at + hd]) {
            *o = x as f32;
        }
    } else {
        let b = if v { &kv.v16 } else { &kv.k16 };
        for (o, &x) in out.iter_mut().zip(&b[at..at + hd]) {
            *o = x as f32;
        }
    }
}

/// The slot batches the score and value kernels take, largest first: a batch of 8 keeps 8 rows of scores (8 x 1
/// KiB) in L1 and 8 accumulators in registers.
const BATCHES: [usize; 4] = [8, 4, 2, 1];

impl Fattn {
    /// n_head query heads over n_head_kv KV heads of head_dim (a multiple of 8). The scores are scaled by `scale` (None
    /// means 1 / sqrt(head_dim)), and `window` gives a sliding window of that many positions (0 for none).
    pub fn new(n_head: usize, n_head_kv: usize, head_dim: usize, scale: Option<f32>, window: usize) -> Fattn {
        assert!(n_head_kv > 0 && n_head.is_multiple_of(n_head_kv) && head_dim.is_multiple_of(8));
        Fattn {
            n_head,
            n_head_kv,
            head_dim,
            scale: scale.unwrap_or(1.0 / (head_dim as f32).sqrt()),
            window,
            sinks: None,
        }
    }

    /// The same engine with a sink per query head (gpt-oss): in the second stage each row starts from its sink as a
    /// page of its own, with that score, weight 1 and no value.
    pub fn with_sinks(mut self, sinks: Vec<f32>) -> Fattn {
        assert_eq!(sinks.len(), self.n_head);
        self.sinks = Some(sinks);
        self
    }

    /// The query heads, KV heads, head dim, score scale and window it was made with.
    pub fn shape(&self) -> (usize, usize, usize, f32, usize) {
        (self.n_head, self.n_head_kv, self.head_dim, self.scale, self.window)
    }

    /// Each query head's sink (gpt-oss), in the scores' units, or None.
    pub fn sinks(&self) -> Option<&[f32]> {
        self.sinks.as_deref()
    }

    /// run_block's definition in f64 (scores, weights and sums; the result rounded to f32 once), no selection: the
    /// judge for TORC_ATTN_EXACT=1, which measures how far a faster path's rounding carries through the whole model.
    /// Slow; each query head runs on a thread of its own.
    pub fn run_block_exact(&self, kv: &KvCache, q: &[f32], qd: usize, n: usize, pos0: u32, out: &mut [f32]) {
        let (hd, g) = (self.head_dim, self.n_head / self.n_head_kv);
        let lo = |p: usize| if self.window > 0 { (p + 1).saturating_sub(self.window) } else { 0 };
        let heads: Vec<Vec<f64>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..self.n_head)
                .map(|h| {
                    s.spawn(move || {
                        let kh = h / g;
                        let mut row = vec![0.0f32; hd];
                        let mut o = vec![0.0f64; n * hd];
                        for t in 0..n {
                            let p = pos0 as usize + t;
                            let qv = &q[t * qd + h * hd..t * qd + (h + 1) * hd];
                            let sc: Vec<f64> = (lo(p)..=p)
                                .map(|j| {
                                    row_f32(kv, false, (kh * kv.cap + j) * hd, hd, &mut row);
                                    qv.iter().zip(&row).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>()
                                        * self.scale as f64
                                        * kv.k_scale[kh * kv.cap + j] as f64
                                })
                                .collect();
                            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let w: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                            let z: f64 = w.iter().sum();
                            let ot = &mut o[t * hd..(t + 1) * hd];
                            for (k, j) in (lo(p)..=p).enumerate() {
                                row_f32(kv, true, (kh * kv.cap + j) * hd, hd, &mut row);
                                let c = w[k] / z * kv.v_scale[kh * kv.cap + j] as f64;
                                for d in 0..hd {
                                    ot[d] += c * row[d] as f64;
                                }
                            }
                        }
                        o
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (h, o) in heads.iter().enumerate() {
            for t in 0..n {
                for d in 0..hd {
                    out[t * qd + h * hd + d] = o[t * hd + d] as f32;
                }
            }
        }
    }

    /// n queries at positions pos0.. (q is n x qd, with qd = n_head x head_dim and query head h at h x head_dim), with
    /// out the same shape, over the keys and values in `kv` (the caller has written positions up to pos0 + n - 1).
    pub fn run_block(
        &self,
        kv: &KvCache,
        q: &[f32],
        qd: usize,
        n: usize,
        pos0: u32,
        out: &mut [f32],
        sel: Option<&Sel>,
    ) {
        let (hd, nh, nkv) = (self.head_dim, self.n_head, self.n_head_kv);
        let group = nh / nkv;
        let pos0 = pos0 as usize;
        assert!(qd == nh * hd && q.len() >= n * qd && out.len() >= n * qd && pos0 + n <= kv.cap);
        if n == 0 {
            return;
        }
        // the pages anyone in the block can see: from the window start of the first token to the last token's page
        let first_page = if self.window > 0 { (pos0 + 1).saturating_sub(self.window) / PAGE } else { 0 };
        let pages = (pos0 + n - 1) / PAGE + 1 - first_page;
        let threads = crate::pool::get().threads.max(1);
        // shape the first stage: split each group's query heads into subgroups until every thread has two items, then
        // cut the tokens into tiles while there are fewer than four (in a long block a page is seen only by the tokens
        // from its first position on, so whole-block items would be uneven). Subgroups: the least power of two giving
        // 2 * threads items over the (kv head, page) pairs, capped by the largest power of two dividing the group
        let need = (2 * threads).div_ceil((nkv * pages).max(1));
        let subgroups = need.next_power_of_two().min(1usize << group.max(1).trailing_zeros());
        let heads_per_item = group / subgroups;
        let mut tiles = 1;
        while nkv * pages * subgroups * tiles < 4 * threads && n >= 16 * tiles {
            tiles *= 2;
        }
        let tile_len = n.div_ceil(tiles);
        let tiles = n.div_ceil(tile_len);
        let items = nkv * pages * subgroups * tiles;

        let slots = pages * n * nh;
        let mut stat = crate::scratch::buf::<f32>("fattn.stat", slots * 2);
        let mut acc = crate::scratch::buf::<f32>("fattn.acc", slots * hd);
        let parts =
            Partials { stat: SyncPtr::new(&mut stat), acc: SyncPtr::new(&mut acc), n, n_head: nh, head_dim: hd };
        let first = |it: usize| {
            let page = it % pages;
            let rest = it / pages;
            let (kvh, rest) = (rest % nkv, rest / nkv);
            let (sub, tile) = (rest % subgroups, rest / subgroups);
            let (j0, j1) = (tile * tile_len, ((tile + 1) * tile_len).min(n));
            self.item(
                kv,
                q,
                qd,
                j0,
                j1,
                pos0,
                kvh,
                first_page + page,
                sub * heads_per_item,
                heads_per_item,
                &parts,
                page,
                sel,
            );
        };
        let op = SyncPtr::new(out);
        let second = |row: usize| {
            let (j, h) = (row / nh, row % nh);
            // SAFETY: row (j, h) of out belongs to this call alone
            let o = unsafe { op.slice(j * qd + h * hd, hd) };
            // SAFETY: the first stage is complete, so the partials are only read from here on
            unsafe { self.combine(&parts, (pos0 + j) / PAGE + 1 - first_page, j, h, o) };
        };
        let rows = n * nh;
        if crate::pool::waits_ok() {
            let (next_item, items_done, next_row) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
            crate::pool::par_for(threads, &|_| {
                loop {
                    let it = next_item.fetch_add(1, Ordering::Relaxed);
                    if it >= items {
                        break;
                    }
                    first(it);
                    items_done.fetch_add(1, Ordering::Release);
                }
                crate::pool::wait_for(&items_done, items);
                loop {
                    let r = next_row.fetch_add(1, Ordering::Relaxed);
                    if r >= rows {
                        break;
                    }
                    second(r);
                }
            });
        } else {
            crate::pool::par_for_dynamic(items, &first);
            crate::pool::par_for_dynamic(rows, &second);
        }
    }

    /// The second stage for one (token j, query head h): the pages 0..pages of its partials combined in page order.
    /// With M the largest of their scores, each page counts with e = expf(m - M): the weight sums add up as fma(e, sum,
    /// total), and the value sums as fma(e, acc[i], o[i]); then each o[i] is divided by the total. A row that saw
    /// nothing (every page empty) is 0.
    ///
    /// # Safety
    /// The partials of (j, h) must all be written, and nothing may write them during the call.
    unsafe fn combine(&self, parts: &Partials, pages: usize, j: usize, h: usize, o: &mut [f32]) {
        let sink = self.sinks.as_ref().map(|s| s[h]);
        let mut top = sink.unwrap_or(f32::NEG_INFINITY);
        for p in 0..pages {
            let m = unsafe { parts.stat(p, j, h)[0] };
            if m > top {
                top = m;
            }
        }
        o.fill(0.0);
        if top == f32::NEG_INFINITY {
            return;
        }
        let mut total = match sink {
            Some(s) => expf(s - top),
            None => 0.0f32,
        };
        for p in 0..pages {
            let (st, acc) = unsafe { (parts.stat(p, j, h), parts.acc(p, j, h)) };
            let e = expf(st[0] - top);
            total = e.mul_add(st[1], total);
            for (y, &x) in o.iter_mut().zip(acc.iter()) {
                *y = e.mul_add(x, *y);
            }
        }
        for y in o.iter_mut() {
            *y /= total;
        }
    }

    /// One first-stage item: KV head kvh, absolute page `page` (stored as page `slot_page` of the call), the block's
    /// tokens j0..j1, and query heads g0..g0 + heads of the KV head's group. A (token, head) pair is a slot, and each
    /// slot's partial goes to `parts`. A token whose own position comes before the page has no slot here: the second
    /// stage never reads a page past a token's own. The kernels take slots in batches (BATCHES), so that a batch's
    /// scores fit in L1 and its accumulators in registers; a slot's numbers do not depend on its batch.
    #[allow(clippy::too_many_arguments)]
    fn item(
        &self,
        kv: &KvCache,
        q: &[f32],
        qd: usize,
        j0: usize,
        j1: usize,
        pos0: usize,
        kvh: usize,
        page: usize,
        g0: usize,
        heads: usize,
        parts: &Partials,
        slot_page: usize,
        sel: Option<&Sel>,
    ) {
        let (hd, group) = (self.head_dim, self.n_head / self.n_head_kv);
        // SAFETY (both): the slots (j, g) of this item's (KV head, page) are its own partials and no other item's
        let stat = |j: usize, g: usize| unsafe { parts.stat(slot_page, j, kvh * group + g) };
        let acc = |j: usize, g: usize| unsafe { parts.acc(slot_page, j, kvh * group + g) };
        // the page's positions: from page_lo, and as far as the last token of the block can see (len of them)
        let page_lo = page * PAGE;
        let page_hi = (page_lo + PAGE).min(pos0 + j1);
        if page_hi <= page_lo {
            return;
        }
        let len = page_hi - page_lo;
        // the tokens with a slot here: from the first whose own position is on this page or past it
        let first = j0.max(page_lo.saturating_sub(pos0));
        let slot_count = (j1 - first) * heads;
        // slots go token by token, each token's heads together
        let slot = |s: usize| (first + s / heads, g0 + s % heads);
        // the part of the page token j may look at, as offsets into it: up to its own position, and from where its
        // sliding window begins; None when the window has already left the page behind
        let seen = |j: usize| -> Option<(usize, usize)> {
            let next = pos0 + j + 1;
            let from = if self.window > 0 && next > self.window { next - self.window } else { 0 };
            (from < page_hi).then(|| (from.saturating_sub(page_lo), next.min(page_hi) - page_lo))
        };
        let batch = |left: usize| BATCHES.into_iter().find(|&n| n <= left).unwrap_or(1);
        let qrow = |s: usize| {
            let (j, g) = slot(s);
            let h = kvh * group + g;
            &q[j * qd + h * hd..j * qd + (h + 1) * hd]
        };
        let base = kvh * kv.cap + page_lo;
        // for each slot s: the mask, the largest visible score, the weights and their sum, then times the values'
        // scales (row holds the slot's scores, PAGE wide)
        let post = |s: usize, row: &mut [f32]| {
            let (j, g) = slot(s);
            let p = stat(j, g);
            // a slot with nothing to see here leaves its weighted values at 0
            acc(j, g).fill(0.0);
            let Some((from, to)) = seen(j) else {
                p[0] = f32::NEG_INFINITY;
                p[1] = 0.0;
                row[..len].fill(0.0);
                return;
            };
            row[..from].fill(f32::NEG_INFINITY);
            // with a selection, the positions of the pools the token did not pick weigh nothing, and a page with none
            // picked contributes nothing at all (the scores were computed for every row; a masked one gets a weight
            // of 0)
            if let Some(se) = sel.filter(|se| se.nv[j] != u32::MAX) {
                let lim = se.nv[j] as usize * se.kp;
                let picked = &se.rows[j * se.stride..];
                let mut any = false;
                for (i, x) in row[..to].iter_mut().enumerate() {
                    let t = page_lo + i;
                    if t < lim && picked[t / se.kp] == 0 {
                        *x = f32::NEG_INFINITY;
                    } else {
                        any = true;
                    }
                }
                if !any {
                    p[0] = f32::NEG_INFINITY;
                    p[1] = 0.0;
                    row.fill(0.0);
                    return;
                }
            }
            let (m, l) = weights(row, len, to, &kv.v_scale[base..base + len]);
            p[0] = m;
            p[1] = l;
        };
        // from TORC_ATTN_F32_MIN slots (default 64): the copy pays for a page's conversion once, which is only worth
        // it over many slots (a prompt block's, not an MTP verify pass's)
        static MIN: std::sync::LazyLock<usize> =
            std::sync::LazyLock::new(|| crate::switches::num("TORC_ATTN_F32_MIN").unwrap_or(64));
        let f32c = slot_count >= (*MIN).max(9) && f32_on();
        // TORC_ATTN_TILE=1 (ARM): from 16 slots up to where the whole-page copy takes over (an MTP
        // verify pass), the page is converted to f32 a tile of TP positions at a time (32 KB, held in L1). First every
        // slot's scores over the key tile; then each slot's weights over the page; then the value tiles into the
        // partials, with each slot's chain carried from tile to tile. The scores are per position and the values are
        // chained in position order, so the bits are the same
        #[cfg(target_arch = "aarch64")]
        if !f32c && slot_count >= tile_min() && f32_on() && tile_on() {
            const TP: usize = 32;
            let mut st = crate::scratch::buf::<f32>("fattn.st", slot_count * PAGE);
            let mut tf = crate::scratch::buf::<f32>("fattn.tf", TP * hd);
            let ks = &kv.k_scale[base..base + len];
            // with TORC_ATTN_FAST=1 the scores come from the int8 keys as they are (fast::scores), with no tile of
            // keys to convert; the values below are tiled either way
            let fast = fast_on() && fast_fits(kv, hd);
            let mut s0 = 0;
            while fast && s0 < slot_count {
                let step = batch(slot_count - s0);
                let sc = &mut st[s0 * PAGE..];
                match step {
                    8 => fast::scores::<8>(kv, base, len, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), sc),
                    4 => fast::scores::<4>(kv, base, len, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), sc),
                    2 => fast::scores::<2>(kv, base, len, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), sc),
                    _ => fast::scores::<1>(kv, base, len, hd, self.scale, [qrow(s0)], sc),
                }
                s0 += step;
            }
            let mut a = if fast { len } else { 0 };
            while a < len {
                let tn = TP.min(len - a);
                neon::f32_rows(kv, false, base + a, tn, hd, &mut tf);
                let mut s0 = 0;
                while s0 < slot_count {
                    let step = if slot_count - s0 >= 4 { 4 } else { 1 };
                    // SAFETY: the tile holds tn rows of hd, the slots' query rows hold hd, and st holds slot_count rows
                    // of PAGE
                    unsafe {
                        let sp = st.as_mut_ptr().add(s0 * PAGE + a);
                        if step == 4 {
                            let q4: [*const f32; 4] = std::array::from_fn(|k| qrow(s0 + k).as_ptr());
                            neon::scores_rows2::<4, f32>(tf.as_ptr(), &ks[a..], tn, hd, self.scale, q4, sp);
                        } else {
                            neon::scores_rows::<1, f32>(
                                tf.as_ptr(),
                                &ks[a..],
                                tn,
                                hd,
                                self.scale,
                                [qrow(s0).as_ptr()],
                                sp,
                            );
                        }
                    }
                    s0 += step;
                }
                a += tn;
            }
            for s in 0..slot_count {
                post(s, &mut st[s * PAGE..(s + 1) * PAGE]);
            }
            let mut a = 0;
            while a < len {
                let tn = TP.min(len - a);
                neon::f32_rows(kv, true, base + a, tn, hd, &mut tf);
                let mut s0 = 0;
                while s0 < slot_count {
                    let step = if slot_count - s0 >= 8 { 8 } else { 1 };
                    // SAFETY: as above; each slot's partial holds hd values after its two numbers
                    unsafe {
                        let wp = st.as_ptr().add(s0 * PAGE + a);
                        if step == 8 {
                            let op: [*mut f32; 8] = std::array::from_fn(|k| {
                                let (j, g) = slot(s0 + k);
                                acc(j, g).as_mut_ptr()
                            });
                            neon::values_tile::<8>(tf.as_ptr(), tn, hd, wp, op, a == 0);
                        } else {
                            let (j, g) = slot(s0);
                            neon::values_tile::<1>(tf.as_ptr(), tn, hd, wp, [acc(j, g).as_mut_ptr()], a == 0);
                        }
                    }
                    s0 += step;
                }
                a += tn;
            }
            return;
        }
        // a decode token with a QSA selection (one token, below the f32-copy size): only the positions it can weigh
        // are read, in runs (the pools it picked, and everything from the end of its last complete pool), instead of
        // scoring the whole page and masking. The scores are per position, the weights run over the whole row as
        // before (the left-out positions are masked by post, as they would be), and the values chain on from run to
        // run in position order (values_from), so the bits are the same. TORC_QSA_GATHER=0 turns it off
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
        ))]
        if !f32c
            && j1 == first + 1
            && gather_on()
            && seen(first).is_some()
            && let Some(se) = sel.filter(|se| se.nv[first] != u32::MAX)
        {
            let (from, to) = seen(first).unwrap();
            let lim = se.nv[first] as usize * se.kp;
            let picked = &se.rows[first * se.stride..];
            let vis = |i: usize| {
                let t = page_lo + i;
                t >= lim || picked[t / se.kp] != 0
            };
            // the runs of visible positions in from..to
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut i = from;
            while i < to {
                if !vis(i) {
                    i += 1;
                    continue;
                }
                let a = i;
                while i < to && vis(i) {
                    i += 1;
                }
                runs.push((a, i));
            }
            // a page more than the slots need, so that the row offset a run starts at (below PAGE) still leaves the
            // kernels their NS x PAGE
            let mut sc = crate::scratch::buf::<f32>("fattn.scg", 9 * PAGE);
            // TORC_ATTN_FAST=1 scores here as it does over whole pages, so a token gets the same bits alone or in a
            // block
            let fast = fast_on() && fast_fits(kv, hd);
            let mut s0 = 0;
            while s0 < slot_count {
                let step = batch(slot_count - s0);
                for &(a, b) in &runs {
                    let (w, at, n) = (&mut sc[a..], base + a, b - a);
                    match (step, fast) {
                        (8, true) => {
                            fast::scores::<8>(kv, at, n, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), w)
                        }
                        (4, true) => {
                            fast::scores::<4>(kv, at, n, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), w)
                        }
                        (2, true) => {
                            fast::scores::<2>(kv, at, n, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), w)
                        }
                        (_, true) => fast::scores::<1>(kv, at, n, hd, self.scale, [qrow(s0)], w),
                        (8, false) => scores::<8>(kv, at, n, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), w),
                        (4, false) => scores::<4>(kv, at, n, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), w),
                        (2, false) => scores::<2>(kv, at, n, hd, self.scale, std::array::from_fn(|k| qrow(s0 + k)), w),
                        _ => scores::<1>(kv, at, n, hd, self.scale, [qrow(s0)], w),
                    }
                }
                for k in 0..step {
                    post(s0 + k, &mut sc[k * PAGE..(k + 1) * PAGE]);
                }
                for &(a, b) in &runs {
                    let w = &sc[a..];
                    match step {
                        8 => values_from::<8>(
                            kv,
                            base + a,
                            b - a,
                            hd,
                            w,
                            std::array::from_fn(|k| {
                                let (j, g) = slot(s0 + k);
                                acc(j, g)
                            }),
                        ),
                        4 => values_from::<4>(
                            kv,
                            base + a,
                            b - a,
                            hd,
                            w,
                            std::array::from_fn(|k| {
                                let (j, g) = slot(s0 + k);
                                acc(j, g)
                            }),
                        ),
                        2 => values_from::<2>(
                            kv,
                            base + a,
                            b - a,
                            hd,
                            w,
                            std::array::from_fn(|k| {
                                let (j, g) = slot(s0 + k);
                                acc(j, g)
                            }),
                        ),
                        _ => {
                            let (j, g) = slot(s0);
                            values_from::<1>(kv, base + a, b - a, hd, w, [acc(j, g)]);
                        }
                    }
                }
                s0 += step;
            }
            return;
        }
        let mut sc = crate::scratch::buf::<f32>("fattn.sc", 8 * PAGE);
        let mut kf = crate::scratch::buf::<f32>("fattn.kf", if f32c { len * hd } else { 0 });
        let mut vf = crate::scratch::buf::<f32>("fattn.vf", if f32c { len * hd } else { 0 });
        #[cfg(target_arch = "aarch64")]
        if f32c {
            neon::page_f32(kv, base, len, hd, &mut kf, &mut vf);
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
        if f32c {
            x86::page_f32(kv, base, len, hd, &mut kf, &mut vf);
        }
        let mut s0 = 0;
        while s0 < slot_count {
            let step = batch(slot_count - s0);
            {
                match step {
                    8 => scores_c::<8>(
                        f32c,
                        &kf,
                        kv,
                        base,
                        len,
                        hd,
                        self.scale,
                        std::array::from_fn(|k| qrow(s0 + k)),
                        &mut sc,
                    ),
                    4 => scores_c::<4>(
                        f32c,
                        &kf,
                        kv,
                        base,
                        len,
                        hd,
                        self.scale,
                        std::array::from_fn(|k| qrow(s0 + k)),
                        &mut sc,
                    ),
                    2 => scores_c::<2>(
                        f32c,
                        &kf,
                        kv,
                        base,
                        len,
                        hd,
                        self.scale,
                        std::array::from_fn(|k| qrow(s0 + k)),
                        &mut sc,
                    ),
                    _ => scores_c::<1>(f32c, &kf, kv, base, len, hd, self.scale, [qrow(s0)], &mut sc),
                }
            }
            for k in 0..step {
                post(s0 + k, &mut sc[k * PAGE..(k + 1) * PAGE]);
            }
            // the values go straight into the partials (a weight of 0 adds nothing: fma(v, 0, a) is a)
            match step {
                8 => values_c::<8>(
                    f32c,
                    &vf,
                    kv,
                    base,
                    len,
                    hd,
                    &sc,
                    std::array::from_fn(|k| {
                        let (j, g) = slot(s0 + k);
                        acc(j, g)
                    }),
                ),
                4 => values_c::<4>(
                    f32c,
                    &vf,
                    kv,
                    base,
                    len,
                    hd,
                    &sc,
                    std::array::from_fn(|k| {
                        let (j, g) = slot(s0 + k);
                        acc(j, g)
                    }),
                ),
                2 => values_c::<2>(
                    f32c,
                    &vf,
                    kv,
                    base,
                    len,
                    hd,
                    &sc,
                    std::array::from_fn(|k| {
                        let (j, g) = slot(s0 + k);
                        acc(j, g)
                    }),
                ),
                _ => {
                    let (j, g) = slot(s0);
                    values_c::<1>(f32c, &vf, kv, base, len, hd, &sc, [acc(j, g)]);
                }
            }
            s0 += step;
        }
    }
}

#[cfg(target_arch = "aarch64")]
use neon::{scores, values, values_from, weights};
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
)))]
use portable::{scores, values, weights};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
use x86::{scores, values, values_from, weights};

/// The kernels in plain Rust, for any target: the definition the NEON and AVX2 kernels match bit for bit.
#[cfg_attr(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2")), allow(dead_code))]
mod portable {
    use super::*;

    /// Eight numbers of a cache row (keys, or values when `v`) at `at`, as f32 (exact).
    #[inline(always)]
    pub(super) fn load8(kv: &KvCache, v: bool, at: usize) -> [f32; 8] {
        if kv.k16.is_empty() {
            let b = if v { &kv.v } else { &kv.k };
            std::array::from_fn(|l| b[at + l] as f32)
        } else {
            let b = if v { &kv.v16 } else { &kv.k16 };
            std::array::from_fn(|l| b[at + l] as f32)
        }
    }

    /// The scores of NS slots (queries qs) over rows base..base + nt of the cache, into sc[k PAGE + t]. As SPEC.md 1
    /// defines them: eight lanes of fused multiply-adds per slot, folded, then times scale, then times the key's scale.
    /// Each row's eight numbers are converted once for all NS slots, whose sums stay in registers.
    #[inline(always)]
    pub(super) fn scores<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [&[f32]; NS],
        sc: &mut [f32],
    ) {
        for t in 0..nt {
            let at = (base + t) * hd;
            let mut acc = [[0.0f32; 8]; NS];
            for i in (0..hd).step_by(8) {
                let k8 = load8(kv, false, at + i);
                for s in 0..NS {
                    let q = &qs[s][i..i + 8];
                    for l in 0..8 {
                        acc[s][l] = q[l].mul_add(k8[l], acc[s][l]);
                    }
                }
            }
            let ks = kv.k_scale[base + t];
            for s in 0..NS {
                sc[s * PAGE + t] = (fold8(&acc[s]) * scale) * ks;
            }
        }
    }

    /// The values of NS slots (with weights w[k PAGE + t]) over rows base..base + nt, into out[k]. As SPEC.md 1 defines
    /// them: for each number, a fused chain over the positions in order. Eight numbers of head_dim are done at a time,
    /// with all NS slots' sums in registers across every position.
    #[inline(always)]
    pub(super) fn values<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        w: &[f32],
        out: [&mut [f32]; NS],
    ) {
        for i in (0..hd).step_by(8) {
            let mut acc = [[0.0f32; 8]; NS];
            for t in 0..nt {
                let v8 = load8(kv, true, (base + t) * hd + i);
                for s in 0..NS {
                    let ws = w[s * PAGE + t];
                    for l in 0..8 {
                        acc[s][l] = v8[l].mul_add(ws, acc[s][l]);
                    }
                }
            }
            for s in 0..NS {
                out[s][i..i + 8].copy_from_slice(&acc[s]);
            }
        }
    }

    /// A slot's weights over its row (SPEC.md 1). m is the largest of the first `live` scores; a live position weighs
    /// expf(score - m), or 0 where score - m < EXP_MIN (e^d would be subnormal), and the positions from `live` to
    /// `count` weigh 0. The row is rewritten in place with each weight times its value scale (vs), while the weights
    /// themselves are summed in eight lanes, eight positions to a group and lane j taking the group's j-th, then
    /// folded. Returns (m, the sum).
    #[inline(always)]
    pub(super) fn weights(scores: &mut [f32], count: usize, live: usize, vs: &[f32]) -> (f32, f32) {
        let m = scores[..live].iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
        let mut lanes = [0.0f32; 8];
        for (group, scale) in scores[..live].chunks_mut(8).zip(vs.chunks(8)) {
            for ((s, &x), lane) in group.iter_mut().zip(scale).zip(lanes.iter_mut()) {
                let d = *s - m;
                let w = if d >= crate::act::EXP_MIN { expf(d) } else { 0.0 };
                *lane += w;
                *s = w * x;
            }
        }
        scores[live..count].fill(0.0);
        (m, fold8(&lanes))
    }
}

/// The kernels on ARM64 with NEON. Eight numbers of a row are held as two registers of four lanes (lanes 0-3 and 4-7 of
/// SPEC.md 1's eight), so each lane's chain of fused multiply-adds is the portable one and the bits are the same. A row
/// is converted once for all of a batch's slots, whose sums are held in registers. The exp is SPEC.md 1's, four lanes
/// at a time.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// A cache number type: eight of them at p, as two registers of four floats (exact).
    pub(super) trait Row: Copy {
        unsafe fn row8(p: *const Self) -> (float32x4_t, float32x4_t);
    }
    /// An f32 copy of an int8 or int16 cache (each number is its exact conversion), so there is nothing to convert. A
    /// trial (bench_parts): the same numbers, so the same bits.
    impl Row for f32 {
        #[inline(always)]
        unsafe fn row8(p: *const f32) -> (float32x4_t, float32x4_t) {
            unsafe { (vld1q_f32(p), vld1q_f32(p.add(4))) }
        }
    }

    /// TORC_ATTN_F32: the cache rows base..base + nt as f32, exactly (K into kf and V into vf, nt x hd each).
    pub(super) fn page_f32(kv: &KvCache, base: usize, nt: usize, hd: usize, kf: &mut [f32], vf: &mut [f32]) {
        unsafe fn conv<E: Row>(src: &[E], out: &mut [f32]) {
            assert!(out.len() >= src.len() && src.len() % 8 == 0);
            // SAFETY: src.len() numbers are read and as many written (checked)
            unsafe {
                for i in (0..src.len()).step_by(8) {
                    let (l, h) = E::row8(src.as_ptr().add(i));
                    vst1q_f32(out.as_mut_ptr().add(i), l);
                    vst1q_f32(out.as_mut_ptr().add(i + 4), h);
                }
            }
        }
        let (at, n) = (base * hd, nt * hd);
        // SAFETY: conv does its own bounds checks
        unsafe {
            if kv.k16.is_empty() {
                conv(&kv.k[at..at + n], kf);
                conv(&kv.v[at..at + n], vf);
            } else {
                conv(&kv.k16[at..at + n], kf);
                conv(&kv.v16[at..at + n], vf);
            }
        }
    }

    /// TORC_ATTN_TILE: nrows cache rows from row0 (keys, or values when v) as f32, exactly, into out (page_f32's
    /// conversion).
    pub(super) fn f32_rows(kv: &KvCache, v: bool, row0: usize, nrows: usize, hd: usize, out: &mut [f32]) {
        unsafe fn conv<E: Row>(src: &[E], out: &mut [f32]) {
            assert!(out.len() >= src.len() && src.len() % 8 == 0);
            // SAFETY: src.len() numbers are read and as many written (checked)
            unsafe {
                for i in (0..src.len()).step_by(8) {
                    let (l, h) = E::row8(src.as_ptr().add(i));
                    vst1q_f32(out.as_mut_ptr().add(i), l);
                    vst1q_f32(out.as_mut_ptr().add(i + 4), h);
                }
            }
        }
        let (at, n) = (row0 * hd, nrows * hd);
        // SAFETY: conv does its own bounds checks
        unsafe {
            match (kv.k16.is_empty(), v) {
                (true, false) => conv(&kv.k[at..at + n], out),
                (true, true) => conv(&kv.v[at..at + n], out),
                (false, false) => conv(&kv.k16[at..at + n], out),
                (false, true) => conv(&kv.v16[at..at + n], out),
            }
        }
    }

    /// TORC_ATTN_TILE: values_rows over a tile of tn f32 value rows (weights w, with slot s at w[s PAGE..]), into out.
    /// With `first` the chains start at 0 (as values_rows does); otherwise each carries on from out. A slot's chain is
    /// the same fused steps in position order however the page is tiled, so the bits are values_rows' bits.
    ///
    /// # Safety
    /// vf holds tn rows of hd, w holds NS rows of PAGE from the tile's first position, and each out holds hd numbers.
    #[inline(always)]
    pub(super) unsafe fn values_tile<const NS: usize>(
        vf: *const f32,
        tn: usize,
        hd: usize,
        w: *const f32,
        out: [*mut f32; NS],
        first: bool,
    ) {
        unsafe {
            if first {
                return values_rows::<NS, f32>(vf, tn, hd, w, out);
            }
            let mut i = 0;
            while i < hd {
                let mut a0: [float32x4_t; NS] = std::array::from_fn(|s| vld1q_f32(out[s].add(i)));
                let mut a1: [float32x4_t; NS] = std::array::from_fn(|s| vld1q_f32(out[s].add(i + 4)));
                let mut t = 0;
                while t + 4 <= tn {
                    let (l0, h0) = <f32 as Row>::row8(vf.add(t * hd + i));
                    let (l1, h1) = <f32 as Row>::row8(vf.add((t + 1) * hd + i));
                    let (l2, h2) = <f32 as Row>::row8(vf.add((t + 2) * hd + i));
                    let (l3, h3) = <f32 as Row>::row8(vf.add((t + 3) * hd + i));
                    for s in 0..NS {
                        let wt = vld1q_f32(w.add(s * PAGE + t));
                        a0[s] = vfmaq_laneq_f32::<0>(a0[s], l0, wt);
                        a1[s] = vfmaq_laneq_f32::<0>(a1[s], h0, wt);
                        a0[s] = vfmaq_laneq_f32::<1>(a0[s], l1, wt);
                        a1[s] = vfmaq_laneq_f32::<1>(a1[s], h1, wt);
                        a0[s] = vfmaq_laneq_f32::<2>(a0[s], l2, wt);
                        a1[s] = vfmaq_laneq_f32::<2>(a1[s], h2, wt);
                        a0[s] = vfmaq_laneq_f32::<3>(a0[s], l3, wt);
                        a1[s] = vfmaq_laneq_f32::<3>(a1[s], h3, wt);
                    }
                    t += 4;
                }
                while t < tn {
                    let (lo, hi) = <f32 as Row>::row8(vf.add(t * hd + i));
                    for s in 0..NS {
                        let ws = *w.add(s * PAGE + t);
                        a0[s] = vfmaq_n_f32(a0[s], lo, ws);
                        a1[s] = vfmaq_n_f32(a1[s], hi, ws);
                    }
                    t += 1;
                }
                for s in 0..NS {
                    vst1q_f32(out[s].add(i), a0[s]);
                    vst1q_f32(out[s].add(i + 4), a1[s]);
                }
                i += 8;
            }
        }
    }

    /// scores::<NS> over page_f32's keys (kf, with ks the page's key scales): 8 or 4 slots go two positions a pass
    /// (four slots at a time), and fewer go one position a pass. Each (position, slot) uses the same lanes and order,
    /// so the bits are the same.
    #[inline(always)]
    pub(super) fn scores_f32<const NS: usize>(
        kf: &[f32],
        ks: &[f32],
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [&[f32]; NS],
        sc: &mut [f32],
    ) {
        assert!(
            sc.len() >= NS * PAGE
                && nt <= PAGE
                && kf.len() >= nt * hd
                && ks.len() >= nt
                && qs.iter().all(|q| q.len() >= hd)
        );
        let qp = qs.map(|q| q.as_ptr());
        // SAFETY: the rows, the query rows and sc are in bounds (checked)
        unsafe {
            if NS % 4 == 0 {
                for h in 0..NS / 4 {
                    let q4: [*const f32; 4] = std::array::from_fn(|s| qp[4 * h + s]);
                    scores_rows2::<4, f32>(kf.as_ptr(), ks, nt, hd, scale, q4, sc.as_mut_ptr().add(4 * h * PAGE));
                }
            } else {
                scores_rows::<NS, f32>(kf.as_ptr(), ks, nt, hd, scale, qp, sc.as_mut_ptr());
            }
        }
    }

    /// values::<NS> over page_f32's values (vf), with the same bits.
    #[inline(always)]
    pub(super) fn values_f32<const NS: usize>(vf: &[f32], nt: usize, hd: usize, w: &[f32], out: [&mut [f32]; NS]) {
        assert!(w.len() >= NS * PAGE && nt <= PAGE && vf.len() >= nt * hd && out.iter().all(|o| o.len() >= hd));
        let op = out.map(|o| o.as_mut_ptr());
        // SAFETY: the rows, the weights and the output rows are in bounds (checked)
        unsafe { values_rows::<NS, f32>(vf.as_ptr(), nt, hd, w.as_ptr(), op) }
    }
    impl Row for i8 {
        #[inline(always)]
        unsafe fn row8(p: *const i8) -> (float32x4_t, float32x4_t) {
            unsafe {
                let x = vmovl_s8(vld1_s8(p));
                (vcvtq_f32_s32(vmovl_s16(vget_low_s16(x))), vcvtq_f32_s32(vmovl_high_s16(x)))
            }
        }
    }
    impl Row for i16 {
        #[inline(always)]
        unsafe fn row8(p: *const i16) -> (float32x4_t, float32x4_t) {
            unsafe {
                let x = vld1q_s16(p);
                (vcvtq_f32_s32(vmovl_s16(vget_low_s16(x))), vcvtq_f32_s32(vmovl_high_s16(x)))
            }
        }
    }

    /// Eight lanes in two registers (lanes 0-3 in a, 4-7 in b) summed as fold8's tree: two pairwise adds give
    /// (0 + 1) + (2 + 3) and (4 + 5) + (6 + 7), then the two are added.
    #[inline(always)]
    unsafe fn fold(a: float32x4_t, b: float32x4_t) -> f32 {
        unsafe {
            let pairs = vpaddq_f32(a, b);
            let quads = vpaddq_f32(pairs, pairs);
            vgetq_lane_f32::<0>(quads) + vgetq_lane_f32::<1>(quads)
        }
    }

    /// crate::act::expf on four lanes, step for step (the same fused operations in the same order), so every lane has
    /// the scalar function's bits.
    #[inline(always)]
    pub(super) unsafe fn exp4(x: float32x4_t) -> float32x4_t {
        use crate::act::{EXP_MAX, EXP_MIN, EXP_P, LN_2_LO, ROUND_SHIFT};
        use std::f32::consts::{LN_2, LOG2_E};
        unsafe {
            let x = vminnmq_f32(vmaxnmq_f32(x, vdupq_n_f32(EXP_MIN)), vdupq_n_f32(EXP_MAX));
            let t = vfmaq_f32(vdupq_n_f32(ROUND_SHIFT), x, vdupq_n_f32(LOG2_E));
            let k = vsubq_f32(t, vdupq_n_f32(ROUND_SHIFT));
            let r = vfmsq_f32(vfmsq_f32(x, k, vdupq_n_f32(LN_2)), k, vdupq_n_f32(LN_2_LO));
            let mut p = vdupq_n_f32(EXP_P[0]);
            for &c in &EXP_P[1..] {
                p = vfmaq_f32(vdupq_n_f32(c), p, r);
            }
            let er = vaddq_f32(vfmaq_f32(r, vmulq_f32(r, r), p), vdupq_n_f32(1.0));
            let two_k = vaddq_u32(vshlq_n_u32::<23>(vreinterpretq_u32_f32(t)), vdupq_n_u32(127 << 23));
            vmulq_f32(er, vreinterpretq_f32_u32(two_k))
        }
    }

    #[inline(always)]
    pub(super) unsafe fn scores_rows<const NS: usize, E: Row>(
        k: *const E,
        ks: &[f32],
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [*const f32; NS],
        sc: *mut f32,
    ) {
        unsafe {
            for t in 0..nt {
                let r = k.add(t * hd);
                let mut a0 = [vdupq_n_f32(0.0); NS];
                let mut a1 = [vdupq_n_f32(0.0); NS];
                let mut i = 0;
                while i < hd {
                    let (lo, hi) = E::row8(r.add(i));
                    for s in 0..NS {
                        a0[s] = vfmaq_f32(a0[s], vld1q_f32(qs[s].add(i)), lo);
                        a1[s] = vfmaq_f32(a1[s], vld1q_f32(qs[s].add(i + 4)), hi);
                    }
                    i += 8;
                }
                let kst = ks[t];
                for s in 0..NS {
                    *sc.add(s * PAGE + t) = (fold(a0[s], a1[s]) * scale) * kst;
                }
            }
        }
    }

    /// scores_rows over two positions a pass: each K row is converted once for both, and the NS slots' query halves are
    /// loaded once for both positions. Each (position, slot) uses the same lanes and order, so the bits are the same. A
    /// trial (bench_parts).
    #[inline(always)]
    pub(super) unsafe fn scores_rows2<const NS: usize, E: Row>(
        k: *const E,
        ks: &[f32],
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [*const f32; NS],
        sc: *mut f32,
    ) {
        unsafe {
            let mut t = 0;
            while t + 2 <= nt {
                let (r0, r1) = (k.add(t * hd), k.add((t + 1) * hd));
                let mut a0 = [[vdupq_n_f32(0.0); NS]; 2];
                let mut a1 = [[vdupq_n_f32(0.0); NS]; 2];
                let mut i = 0;
                while i < hd {
                    let (l0, h0) = E::row8(r0.add(i));
                    let (l1, h1) = E::row8(r1.add(i));
                    for s in 0..NS {
                        let (ql, qh) = (vld1q_f32(qs[s].add(i)), vld1q_f32(qs[s].add(i + 4)));
                        a0[0][s] = vfmaq_f32(a0[0][s], ql, l0);
                        a1[0][s] = vfmaq_f32(a1[0][s], qh, h0);
                        a0[1][s] = vfmaq_f32(a0[1][s], ql, l1);
                        a1[1][s] = vfmaq_f32(a1[1][s], qh, h1);
                    }
                    i += 8;
                }
                for p in 0..2 {
                    let kst = ks[t + p];
                    for s in 0..NS {
                        *sc.add(s * PAGE + t + p) = (fold(a0[p][s], a1[p][s]) * scale) * kst;
                    }
                }
                t += 2;
            }
            if t < nt {
                scores_rows::<NS, E>(k.add(t * hd), &ks[t..], nt - t, hd, scale, qs, sc.add(t));
            }
        }
    }

    /// portable::scores on NEON (the same arguments and the same bits).
    #[inline(always)]
    pub(super) fn scores<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [&[f32]; NS],
        sc: &mut [f32],
    ) {
        assert!(sc.len() >= NS * PAGE && nt <= PAGE && qs.iter().all(|q| q.len() >= hd));
        let (at, ks) = (base * hd, &kv.k_scale[base..base + nt]);
        let qp = qs.map(|q| q.as_ptr());
        // SAFETY: rows base..base + nt of the cache, hd numbers each, and the query rows are in bounds (checked)
        // TORC_ATTN_ROWS2=1: two positions a pass through scores_rows2, with the slots four at a time
        // (8 slots as two halves of 4). Each (position, slot) uses the same lanes and order, so the bits are the same
        static ROWS2: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_ATTN_ROWS2"));
        unsafe {
            if kv.k16.is_empty() {
                assert!(kv.k.len() >= at + nt * hd);
                let k = kv.k.as_ptr().add(at);
                if *ROWS2 && NS % 4 == 0 {
                    for h in 0..NS / 4 {
                        let q4: [*const f32; 4] = std::array::from_fn(|s| qp[4 * h + s]);
                        scores_rows2::<4, i8>(k, ks, nt, hd, scale, q4, sc.as_mut_ptr().add(4 * h * PAGE));
                    }
                } else {
                    scores_rows::<NS, i8>(k, ks, nt, hd, scale, qp, sc.as_mut_ptr());
                }
            } else {
                assert!(kv.k16.len() >= at + nt * hd);
                scores_rows::<NS, i16>(kv.k16.as_ptr().add(at), ks, nt, hd, scale, qp, sc.as_mut_ptr());
            }
        }
    }

    #[inline(always)]
    pub(super) unsafe fn values_rows<const NS: usize, E: Row>(
        v: *const E,
        nt: usize,
        hd: usize,
        w: *const f32,
        out: [*mut f32; NS],
    ) {
        unsafe { values_rows_from::<NS, E, false>(v, nt, hd, w, out) }
    }

    /// values_rows, with FROM: each chain starts from what out already holds instead of 0, so a page's positions can
    /// go through in several runs (in position order) with the same bits as one pass: the chains are the same fmas in
    /// the same order, and a position left out has weight 0, whose fma would have left the sum as it was.
    #[inline(always)]
    pub(super) unsafe fn values_rows_from<const NS: usize, E: Row, const FROM: bool>(
        v: *const E,
        nt: usize,
        hd: usize,
        w: *const f32,
        out: [*mut f32; NS],
    ) {
        unsafe {
            let mut i = 0;
            while i < hd {
                let mut a0 = [vdupq_n_f32(0.0); NS];
                let mut a1 = [vdupq_n_f32(0.0); NS];
                if FROM {
                    for s in 0..NS {
                        a0[s] = vld1q_f32(out[s].add(i));
                        a1[s] = vld1q_f32(out[s].add(i + 4));
                    }
                }
                let mut t = 0;
                while t + 4 <= nt {
                    let (l0, h0) = E::row8(v.add(t * hd + i));
                    let (l1, h1) = E::row8(v.add((t + 1) * hd + i));
                    let (l2, h2) = E::row8(v.add((t + 2) * hd + i));
                    let (l3, h3) = E::row8(v.add((t + 3) * hd + i));
                    for s in 0..NS {
                        let wt = vld1q_f32(w.add(s * PAGE + t));
                        a0[s] = vfmaq_laneq_f32::<0>(a0[s], l0, wt);
                        a1[s] = vfmaq_laneq_f32::<0>(a1[s], h0, wt);
                        a0[s] = vfmaq_laneq_f32::<1>(a0[s], l1, wt);
                        a1[s] = vfmaq_laneq_f32::<1>(a1[s], h1, wt);
                        a0[s] = vfmaq_laneq_f32::<2>(a0[s], l2, wt);
                        a1[s] = vfmaq_laneq_f32::<2>(a1[s], h2, wt);
                        a0[s] = vfmaq_laneq_f32::<3>(a0[s], l3, wt);
                        a1[s] = vfmaq_laneq_f32::<3>(a1[s], h3, wt);
                    }
                    t += 4;
                }
                while t < nt {
                    let (lo, hi) = E::row8(v.add(t * hd + i));
                    for s in 0..NS {
                        let ws = *w.add(s * PAGE + t);
                        a0[s] = vfmaq_n_f32(a0[s], lo, ws);
                        a1[s] = vfmaq_n_f32(a1[s], hi, ws);
                    }
                    t += 1;
                }
                for s in 0..NS {
                    vst1q_f32(out[s].add(i), a0[s]);
                    vst1q_f32(out[s].add(i + 4), a1[s]);
                }
                i += 8;
            }
        }
    }

    /// values over positions base..base + nt with each output's chain carried on from what it holds (values_rows_from):
    /// a run of the page, for the decode gather (Fattn::item). w holds the run's weights at the start of each row.
    #[inline(always)]
    pub(super) fn values_from<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        w: &[f32],
        out: [&mut [f32]; NS],
    ) {
        assert!(w.len() >= (NS - 1) * PAGE + nt && nt <= PAGE && out.iter().all(|o| o.len() >= hd));
        let at = base * hd;
        let op = out.map(|o| o.as_mut_ptr());
        // SAFETY: rows base..base + nt, the weights and the output rows are in bounds (checked)
        unsafe {
            if kv.v16.is_empty() {
                assert!(kv.v.len() >= at + nt * hd);
                values_rows_from::<NS, i8, true>(kv.v.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            } else {
                assert!(kv.v16.len() >= at + nt * hd);
                values_rows_from::<NS, i16, true>(kv.v16.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            }
        }
    }

    /// portable::values on NEON (the same arguments and the same bits).
    #[inline(always)]
    pub(super) fn values<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        w: &[f32],
        out: [&mut [f32]; NS],
    ) {
        assert!(w.len() >= NS * PAGE && nt <= PAGE && out.iter().all(|o| o.len() >= hd));
        let at = base * hd;
        let op = out.map(|o| o.as_mut_ptr());
        // SAFETY: rows base..base + nt, the weights and the output rows are in bounds (checked)
        unsafe {
            if kv.v16.is_empty() {
                assert!(kv.v.len() >= at + nt * hd);
                values_rows::<NS, i8>(kv.v.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            } else {
                assert!(kv.v16.len() >= at + nt * hd);
                values_rows::<NS, i16>(kv.v16.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            }
        }
    }

    /// KEEP[8 - r..16 - r]: all ones in the first r lanes, zero in the rest.
    static KEEP: [u32; 16] = [!0, !0, !0, !0, !0, !0, !0, !0, 0, 0, 0, 0, 0, 0, 0, 0];

    /// Four positions' weights: expf(s - top) where s - top >= floor, else +0.
    #[inline(always)]
    unsafe fn soft4(s: float32x4_t, top: float32x4_t, floor: float32x4_t) -> float32x4_t {
        unsafe {
            let d = vsubq_f32(s, top);
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(exp4(d)), vcgeq_f32(d, floor)))
        }
    }

    /// portable::weights on NEON, a group of eight positions as two registers (the row has room for PAGE). The largest
    /// score is found by FMAXNM, sixteen at a time in four registers. That gives the same number as the scalar scan
    /// (both skip a NaN), except for the sign of a largest score of zero, and every use of it (exp(score - m) here,
    /// exp(m - M) in the merge) turns either sign into the same bits. Groups below live run unmasked, the group that
    /// holds live is cut by KEEP, and the rest of the row is zeroed; adding nothing for them leaves the lane sums as
    /// adding +0 would.
    #[inline(always)]
    pub(super) fn weights(scores: &mut [f32], count: usize, live: usize, vs: &[f32]) -> (f32, f32) {
        assert!(scores.len() >= count.div_ceil(8) * 8 && live <= count && vs.len() >= count);
        let v16 = live / 16 * 16;
        // SAFETY: the positions below live are in the scores (checked)
        let mut m = unsafe {
            let p = scores.as_ptr();
            let mut a = [vdupq_n_f32(f32::NEG_INFINITY); 4];
            let mut t = 0;
            while t < v16 {
                for (k, ak) in a.iter_mut().enumerate() {
                    *ak = vmaxnmq_f32(*ak, vld1q_f32(p.add(t + 4 * k)));
                }
                t += 16;
            }
            vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(a[0], a[1]), vmaxnmq_f32(a[2], a[3])))
        };
        for &v in &scores[v16..live] {
            if v > m {
                m = v;
            }
        }
        let (p, x) = (scores.as_mut_ptr(), vs.as_ptr());
        let whole = live / 8 * 8;
        // SAFETY: positions 0..count rounded up to 8 are in the scores, and 0..count in vs (checked)
        unsafe {
            let (top, floor) = (vdupq_n_f32(m), vdupq_n_f32(crate::act::EXP_MIN));
            let mut sum = [vdupq_n_f32(0.0); 2];
            let mut t = 0;
            // groups wholly below live: weigh, add up, scale in place
            while t < whole {
                for h in 0..2 {
                    let w = soft4(vld1q_f32(p.add(t + 4 * h)), top, floor);
                    sum[h] = vaddq_f32(sum[h], w);
                    vst1q_f32(p.add(t + 4 * h), vmulq_f32(w, vld1q_f32(x.add(t + 4 * h))));
                }
                t += 8;
            }
            // the group live falls inside: its first live - whole lanes count, the rest are cut to +0
            if whole < live {
                let cut = KEEP.as_ptr().add(8 - (live - whole));
                for h in 0..2 {
                    let w = soft4(vld1q_f32(p.add(t + 4 * h)), top, floor);
                    let w = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(w), vld1q_u32(cut.add(4 * h))));
                    sum[h] = vaddq_f32(sum[h], w);
                    vst1q_f32(p.add(t + 4 * h), w);
                }
                for i in whole..live {
                    *p.add(i) *= vs[i];
                }
                t += 8;
            }
            // nothing past live weighs anything
            while t < count {
                vst1q_f32(p.add(t), vdupq_n_f32(0.0));
                vst1q_f32(p.add(t + 4), vdupq_n_f32(0.0));
                t += 8;
            }
            (m, fold(sum[0], sum[1]))
        }
    }
}

/// The kernels on x86-64 with AVX2 and FMA (the build's target-cpu, see .cargo/config.toml). SPEC.md 1's eight lanes
/// are one register, so each lane's chain of fused multiply-adds is the portable one and the bits are the same, and the
/// eight are summed as fold8 (the high half onto the low, then the pairs, then the two). A row is converted once for
/// all of a batch's slots, whose sums are held in registers, and the exp runs eight lanes at a time with the same
/// operations.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// A cache number type: eight of them at p, as one register of eight floats (exact).
    trait Row: Copy {
        unsafe fn row8(p: *const Self) -> __m256;
    }
    impl Row for i8 {
        #[inline(always)]
        unsafe fn row8(p: *const i8) -> __m256 {
            unsafe { _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i))) }
        }
    }
    impl Row for i16 {
        #[inline(always)]
        unsafe fn row8(p: *const i16) -> __m256 {
            unsafe { _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(p as *const __m128i))) }
        }
    }
    /// TORC_ATTN_F32's f32 copy of the cache rows (exact), so there is nothing to convert.
    impl Row for f32 {
        #[inline(always)]
        unsafe fn row8(p: *const f32) -> __m256 {
            unsafe { _mm256_loadu_ps(p) }
        }
    }

    /// TORC_ATTN_F32: the cache rows base..base + nt as f32, exactly (K into kf and V into vf, nt x hd each).
    pub(super) fn page_f32(kv: &KvCache, base: usize, nt: usize, hd: usize, kf: &mut [f32], vf: &mut [f32]) {
        unsafe fn conv<E: Row>(src: &[E], out: &mut [f32]) {
            assert!(out.len() >= src.len() && src.len().is_multiple_of(8));
            // SAFETY: src.len() numbers are read and as many written (checked)
            unsafe {
                for i in (0..src.len()).step_by(8) {
                    _mm256_storeu_ps(out.as_mut_ptr().add(i), E::row8(src.as_ptr().add(i)));
                }
            }
        }
        let (at, n) = (base * hd, nt * hd);
        // SAFETY: conv does its own bounds checks
        unsafe {
            if kv.k16.is_empty() {
                conv(&kv.k[at..at + n], kf);
                conv(&kv.v[at..at + n], vf);
            } else {
                conv(&kv.k16[at..at + n], kf);
                conv(&kv.v16[at..at + n], vf);
            }
        }
    }

    /// scores_rows over two positions a pass (each query register is loaded once for both). Each (position, slot) uses
    /// the same lanes and order, so the bits are the same.
    #[inline(always)]
    unsafe fn scores_rows2<const NS: usize, E: Row>(
        k: *const E,
        ks: &[f32],
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [*const f32; NS],
        sc: *mut f32,
    ) {
        unsafe {
            let mut t = 0;
            while t + 2 <= nt {
                let (r0, r1) = (k.add(t * hd), k.add((t + 1) * hd));
                let mut a = [[_mm256_setzero_ps(); NS]; 2];
                let mut i = 0;
                while i < hd {
                    let (k0, k1) = (E::row8(r0.add(i)), E::row8(r1.add(i)));
                    for s in 0..NS {
                        let q = _mm256_loadu_ps(qs[s].add(i));
                        a[0][s] = _mm256_fmadd_ps(q, k0, a[0][s]);
                        a[1][s] = _mm256_fmadd_ps(q, k1, a[1][s]);
                    }
                    i += 8;
                }
                for p in 0..2 {
                    let kst = ks[t + p];
                    for s in 0..NS {
                        *sc.add(s * PAGE + t + p) = (fold(a[p][s]) * scale) * kst;
                    }
                }
                t += 2;
            }
            if t < nt {
                scores_rows::<NS, E>(k.add(t * hd), &ks[t..], nt - t, hd, scale, qs, sc.add(t));
            }
        }
    }

    /// scores::<NS> over page_f32's keys: 8 or 4 slots go two positions a pass (four slots at a time), and fewer go
    /// one position a pass. The same bits.
    #[inline(always)]
    pub(super) fn scores_f32<const NS: usize>(
        kf: &[f32],
        ks: &[f32],
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [&[f32]; NS],
        sc: &mut [f32],
    ) {
        assert!(
            sc.len() >= NS * PAGE
                && nt <= PAGE
                && kf.len() >= nt * hd
                && ks.len() >= nt
                && qs.iter().all(|q| q.len() >= hd)
        );
        let qp = qs.map(|q| q.as_ptr());
        // SAFETY: the rows, the query rows and sc are in bounds (checked)
        unsafe {
            if NS.is_multiple_of(4) {
                for h in 0..NS / 4 {
                    let q4: [*const f32; 4] = std::array::from_fn(|s| qp[4 * h + s]);
                    scores_rows2::<4, f32>(kf.as_ptr(), ks, nt, hd, scale, q4, sc.as_mut_ptr().add(4 * h * PAGE));
                }
            } else {
                scores_rows::<NS, f32>(kf.as_ptr(), ks, nt, hd, scale, qp, sc.as_mut_ptr());
            }
        }
    }

    /// values::<NS> over page_f32's values, with the same bits.
    #[inline(always)]
    pub(super) fn values_f32<const NS: usize>(vf: &[f32], nt: usize, hd: usize, w: &[f32], out: [&mut [f32]; NS]) {
        assert!(w.len() >= NS * PAGE && nt <= PAGE && vf.len() >= nt * hd && out.iter().all(|o| o.len() >= hd));
        let op = out.map(|o| o.as_mut_ptr());
        // SAFETY: the rows, the weights and the output rows are in bounds (checked)
        unsafe { values_rows::<NS, f32>(vf.as_ptr(), nt, hd, w.as_ptr(), op) }
    }

    /// fold8's tree on one register: horizontal adds within each 128-bit half give (0 + 1) + (2 + 3) in lane 0 and
    /// (4 + 5) + (6 + 7) in lane 4, which are then added.
    #[inline(always)]
    unsafe fn fold(a: __m256) -> f32 {
        unsafe {
            let pairs = _mm256_hadd_ps(a, a);
            let quads = _mm256_hadd_ps(pairs, pairs);
            _mm256_cvtss_f32(quads) + _mm_cvtss_f32(_mm256_extractf128_ps::<1>(quads))
        }
    }

    /// crate::act::expf on eight lanes, step for step, so every lane has the scalar function's bits.
    #[inline(always)]
    pub(super) unsafe fn exp8(x: __m256) -> __m256 {
        use crate::act::{EXP_MAX, EXP_MIN, EXP_P, LN_2_LO, ROUND_SHIFT};
        use std::f32::consts::{LN_2, LOG2_E};
        unsafe {
            // max_ps returns its second operand when the first is a NaN: a NaN becomes EXP_MIN, as in expf
            let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(EXP_MIN)), _mm256_set1_ps(EXP_MAX));
            let t = _mm256_fmadd_ps(x, _mm256_set1_ps(LOG2_E), _mm256_set1_ps(ROUND_SHIFT));
            let k = _mm256_sub_ps(t, _mm256_set1_ps(ROUND_SHIFT));
            let r = _mm256_fnmadd_ps(k, _mm256_set1_ps(LN_2_LO), _mm256_fnmadd_ps(k, _mm256_set1_ps(LN_2), x));
            let mut p = _mm256_set1_ps(EXP_P[0]);
            for &c in &EXP_P[1..] {
                p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
            }
            let er = _mm256_add_ps(_mm256_fmadd_ps(_mm256_mul_ps(r, r), p, r), _mm256_set1_ps(1.0));
            let two_k = _mm256_add_epi32(_mm256_slli_epi32::<23>(_mm256_castps_si256(t)), _mm256_set1_epi32(127 << 23));
            _mm256_mul_ps(er, _mm256_castsi256_ps(two_k))
        }
    }

    #[inline(always)]
    unsafe fn scores_rows<const NS: usize, E: Row>(
        k: *const E,
        ks: &[f32],
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [*const f32; NS],
        sc: *mut f32,
    ) {
        unsafe {
            for t in 0..nt {
                let r = k.add(t * hd);
                let mut a = [_mm256_setzero_ps(); NS];
                let mut i = 0;
                while i < hd {
                    let k8 = E::row8(r.add(i));
                    for s in 0..NS {
                        a[s] = _mm256_fmadd_ps(_mm256_loadu_ps(qs[s].add(i)), k8, a[s]);
                    }
                    i += 8;
                }
                let kst = ks[t];
                for s in 0..NS {
                    *sc.add(s * PAGE + t) = (fold(a[s]) * scale) * kst;
                }
            }
        }
    }

    /// portable::scores with AVX2 (the same arguments and the same bits).
    #[inline(always)]
    pub(super) fn scores<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        scale: f32,
        qs: [&[f32]; NS],
        sc: &mut [f32],
    ) {
        assert!(sc.len() >= NS * PAGE && nt <= PAGE && qs.iter().all(|q| q.len() >= hd));
        let (at, ks) = (base * hd, &kv.k_scale[base..base + nt]);
        let qp = qs.map(|q| q.as_ptr());
        // SAFETY: rows base..base + nt of the cache, hd numbers each, and the query rows are in bounds (checked)
        unsafe {
            if kv.k16.is_empty() {
                assert!(kv.k.len() >= at + nt * hd);
                scores_rows::<NS, i8>(kv.k.as_ptr().add(at), ks, nt, hd, scale, qp, sc.as_mut_ptr());
            } else {
                assert!(kv.k16.len() >= at + nt * hd);
                scores_rows::<NS, i16>(kv.k16.as_ptr().add(at), ks, nt, hd, scale, qp, sc.as_mut_ptr());
            }
        }
    }

    #[inline(always)]
    unsafe fn values_rows<const NS: usize, E: Row>(
        v: *const E,
        nt: usize,
        hd: usize,
        w: *const f32,
        out: [*mut f32; NS],
    ) {
        unsafe { values_rows_from::<NS, E, false>(v, nt, hd, w, out) }
    }

    /// values_rows, with FROM: each chain starts from what out already holds instead of 0, so a page's positions can
    /// go through in several runs (in position order) with the same bits as one pass, as neon::values_rows_from.
    #[inline(always)]
    unsafe fn values_rows_from<const NS: usize, E: Row, const FROM: bool>(
        v: *const E,
        nt: usize,
        hd: usize,
        w: *const f32,
        out: [*mut f32; NS],
    ) {
        unsafe {
            let mut i = 0;
            while i < hd {
                let mut a = [_mm256_setzero_ps(); NS];
                if FROM {
                    for s in 0..NS {
                        a[s] = _mm256_loadu_ps(out[s].add(i));
                    }
                }
                for t in 0..nt {
                    let v8 = E::row8(v.add(t * hd + i));
                    for s in 0..NS {
                        a[s] = _mm256_fmadd_ps(v8, _mm256_set1_ps(*w.add(s * PAGE + t)), a[s]);
                    }
                }
                for s in 0..NS {
                    _mm256_storeu_ps(out[s].add(i), a[s]);
                }
                i += 8;
            }
        }
    }

    /// values over positions base..base + nt with each output's chain carried on from what it holds (values_rows_from):
    /// a run of the page, for the decode gather (Fattn::item), as neon::values_from.
    #[inline(always)]
    pub(super) fn values_from<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        w: &[f32],
        out: [&mut [f32]; NS],
    ) {
        assert!(w.len() >= (NS - 1) * PAGE + nt && nt <= PAGE && out.iter().all(|o| o.len() >= hd));
        let at = base * hd;
        let op = out.map(|o| o.as_mut_ptr());
        // SAFETY: rows base..base + nt, the weights and the output rows are in bounds (checked)
        unsafe {
            if kv.v16.is_empty() {
                assert!(kv.v.len() >= at + nt * hd);
                values_rows_from::<NS, i8, true>(kv.v.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            } else {
                assert!(kv.v16.len() >= at + nt * hd);
                values_rows_from::<NS, i16, true>(kv.v16.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            }
        }
    }

    /// portable::values with AVX2 (the same arguments and the same bits).
    #[inline(always)]
    pub(super) fn values<const NS: usize>(
        kv: &KvCache,
        base: usize,
        nt: usize,
        hd: usize,
        w: &[f32],
        out: [&mut [f32]; NS],
    ) {
        assert!(w.len() >= NS * PAGE && nt <= PAGE && out.iter().all(|o| o.len() >= hd));
        let at = base * hd;
        let op = out.map(|o| o.as_mut_ptr());
        // SAFETY: rows base..base + nt, the weights and the output rows are in bounds (checked)
        unsafe {
            if kv.v16.is_empty() {
                assert!(kv.v.len() >= at + nt * hd);
                values_rows::<NS, i8>(kv.v.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            } else {
                assert!(kv.v16.len() >= at + nt * hd);
                values_rows::<NS, i16>(kv.v16.as_ptr().add(at), nt, hd, w.as_ptr(), op);
            }
        }
    }

    /// KEEP[8 - r..16 - r]: all ones in the first r lanes, zero in the rest.
    static KEEP: [i32; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

    /// Eight positions' weights: expf(s - top) where s - top >= floor, else +0.
    #[inline(always)]
    unsafe fn soft8(s: __m256, top: __m256, floor: __m256) -> __m256 {
        unsafe {
            let d = _mm256_sub_ps(s, top);
            _mm256_and_ps(exp8(d), _mm256_cmp_ps::<_CMP_GE_OQ>(d, floor))
        }
    }

    /// portable::weights with AVX2, a group of eight positions to a register (the row has room for PAGE). The largest
    /// score is kept sixteen at a time in two registers by max(v, a), which gives v where v > a and a otherwise (the
    /// scalar scan's step, skipping a NaN), and their eight lanes are then scanned as the scalar code does. That gives
    /// the same number except for the sign of a largest score of zero, and every use of it turns either sign into the
    /// same bits. Groups below live run unmasked, the group that holds live is cut by KEEP, and the rest of the row is
    /// zeroed; adding nothing for them leaves the lane sums as adding +0 would.
    #[inline(always)]
    pub(super) fn weights(scores: &mut [f32], count: usize, live: usize, vs: &[f32]) -> (f32, f32) {
        assert!(scores.len() >= count.div_ceil(8) * 8 && live <= count && vs.len() >= count);
        let v16 = live / 16 * 16;
        // SAFETY: the positions below live are in the scores (checked)
        let mut m = unsafe {
            let p = scores.as_ptr();
            let (mut a0, mut a1) = (_mm256_set1_ps(f32::NEG_INFINITY), _mm256_set1_ps(f32::NEG_INFINITY));
            let mut t = 0;
            while t < v16 {
                a0 = _mm256_max_ps(_mm256_loadu_ps(p.add(t)), a0);
                a1 = _mm256_max_ps(_mm256_loadu_ps(p.add(t + 8)), a1);
                t += 16;
            }
            let mut lanes = [f32::NEG_INFINITY; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_max_ps(a0, a1));
            lanes.iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m })
        };
        for &v in &scores[v16..live] {
            if v > m {
                m = v;
            }
        }
        let (p, x) = (scores.as_mut_ptr(), vs.as_ptr());
        let whole = live / 8 * 8;
        // SAFETY: positions 0..count rounded up to 8 are in the scores, and 0..count in vs (checked)
        unsafe {
            let (top, floor) = (_mm256_set1_ps(m), _mm256_set1_ps(crate::act::EXP_MIN));
            let mut sum = _mm256_setzero_ps();
            let mut t = 0;
            // groups wholly below live: weigh, add up, scale in place
            while t < whole {
                let w = soft8(_mm256_loadu_ps(p.add(t)), top, floor);
                sum = _mm256_add_ps(sum, w);
                _mm256_storeu_ps(p.add(t), _mm256_mul_ps(w, _mm256_loadu_ps(x.add(t))));
                t += 8;
            }
            // the group live falls inside: its first live - whole lanes count, the rest are cut to +0
            if whole < live {
                let cut = _mm256_loadu_si256(KEEP.as_ptr().add(8 - (live - whole)) as *const __m256i);
                let w = _mm256_and_ps(soft8(_mm256_loadu_ps(p.add(t)), top, floor), _mm256_castsi256_ps(cut));
                sum = _mm256_add_ps(sum, w);
                _mm256_storeu_ps(p.add(t), w);
                for i in whole..live {
                    *p.add(i) *= vs[i];
                }
                t += 8;
            }
            // nothing past live weighs anything
            while t < count {
                _mm256_storeu_ps(p.add(t), _mm256_setzero_ps());
                t += 8;
            }
            (m, fold(sum))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, in f64, on the cache's dequantised rows.
    fn reference(f: &Fattn, kv: &KvCache, q: &[f32], n: usize, pos0: usize) -> Vec<f32> {
        reference_sel(f, kv, q, n, pos0, None)
    }

    /// The definition with a selection: token t sees position j only where `sel` keeps it.
    fn reference_sel(f: &Fattn, kv: &KvCache, q: &[f32], n: usize, pos0: usize, sel: Option<&Sel>) -> Vec<f32> {
        let keep = |t: usize, j: usize| match sel {
            Some(se) if se.nv[t] != u32::MAX => {
                j >= se.nv[t] as usize * se.kp || se.rows[t * se.stride + j / se.kp] != 0
            }
            _ => true,
        };
        let (hd, g, qd) = (f.head_dim, f.n_head / f.n_head_kv, f.n_head * f.head_dim);
        let mut out = vec![0.0f32; n * qd];
        let mut row = vec![0.0f32; hd];
        let lo = |p: usize| if f.window > 0 { (p + 1).saturating_sub(f.window) } else { 0 };
        for t in 0..n {
            let p = pos0 + t;
            for h in 0..f.n_head {
                let kh = h / g;
                let qv = &q[t * qd + h * hd..t * qd + (h + 1) * hd];
                let js: Vec<usize> = (lo(p)..=p).filter(|&j| keep(t, j)).collect();
                if js.is_empty() {
                    continue;
                }
                let sc: Vec<f64> = js
                    .iter()
                    .map(|&j| {
                        row_f32(kv, false, (kh * kv.cap + j) * hd, hd, &mut row);
                        qv.iter().zip(row.iter()).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>()
                            * f.scale as f64
                            * kv.k_scale[kh * kv.cap + j] as f64
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = w.iter().sum();
                let o = &mut out[t * qd + h * hd..t * qd + (h + 1) * hd];
                for (k, &j) in js.iter().enumerate() {
                    row_f32(kv, true, (kh * kv.cap + j) * hd, hd, &mut row);
                    let vs = kv.v_scale[kh * kv.cap + j] as f64;
                    for d in 0..hd {
                        o[d] += (w[k] / z * row[d] as f64 * vs) as f32;
                    }
                }
            }
        }
        out
    }

    fn rnd(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn cache(s: &mut u64, nkv: usize, hd: usize, cap: usize, kv_f16: bool) -> KvCache {
        let mut kv = KvCache::with(nkv, hd, cap, kv_f16);
        // keys and values of magnitude around 1-3, as in a model
        let sc = if kv_f16 { 1e-4 } else { 0.02 };
        for i in 0..kv.k_scale.len() {
            kv.k_scale[i] = sc * (1.0 + 0.2 * rnd(s).abs());
            kv.v_scale[i] = sc * (1.0 + 0.2 * rnd(s).abs());
        }
        if kv_f16 {
            for x in kv.k16.iter_mut().chain(kv.v16.iter_mut()) {
                *x = (rnd(s) * 30000.0) as i16;
            }
        } else {
            for x in kv.k.iter_mut().chain(kv.v.iter_mut()) {
                *x = (rnd(s) * 120.0) as i8;
            }
        }
        kv
    }

    const CASES: [(usize, usize, usize, usize, bool, usize, usize); 10] = [
        (16, 8, 64, 0, true, 1, 300),
        (16, 2, 128, 0, true, 5, 700),
        (16, 8, 64, 100, true, 3, 450),
        (16, 8, 64, 1024, true, 4, 1400),
        (8, 2, 64, 0, false, 9, 40),
        (4, 4, 32, 7, false, 20, 0),
        // the Qwen 35B's attention shape at MTP verify sizes, with an int8 cache (16-63 slots an item: TORC_ATTN_TILE's
        // path)
        (16, 2, 256, 0, false, 4, 2000),
        (16, 2, 256, 0, false, 5, 3001),
        (16, 2, 256, 300, false, 4, 2105),
        (16, 2, 256, 0, false, 7, 5000),
    ];

    use crate::pool::one_caller;

    /// TORC_ATTN_FAST's scores: this processor's kernel gives the portable definition's bits, for every batch size,
    /// and the scores are the exact path's to within the query's rounding.
    #[test]
    fn fast_scores_are_the_integer_definition() {
        let mut s = 0x5eed_1234_abcd_0001u64;
        for &(hd, nt) in &[(256usize, 256usize), (256, 37), (128, 200), (64, 1), (32, 255)] {
            let kv = cache(&mut s, 2, hd, 600, false);
            let q: Vec<f32> = (0..8 * hd).map(|_| rnd(&mut s)).collect();
            let rows: [&[f32]; 8] = std::array::from_fn(|i| &q[i * hd..(i + 1) * hd]);
            let base = 300 + 5;
            let (mut a, mut b, mut e) = (vec![0f32; 8 * PAGE], vec![0f32; 8 * PAGE], vec![0f32; 8 * PAGE]);
            fast::scores::<8>(&kv, base, nt, hd, 0.05, rows, &mut a);
            fast::scores_portable::<8>(&kv, base, nt, hd, 0.05, rows, &mut b);
            portable::scores::<8>(&kv, base, nt, hd, 0.05, rows, &mut e);
            let top = e.iter().fold(0f32, |m, v| m.max(v.abs()));
            for sl in 0..8 {
                for t in 0..nt {
                    let i = sl * PAGE + t;
                    assert_eq!(a[i].to_bits(), b[i].to_bits(), "hd {hd} nt {nt}: slot {sl} position {t}");
                    assert!((a[i] - e[i]).abs() <= 2e-4 * top, "hd {hd}: {} against {} (top {top})", a[i], e[i]);
                }
            }
            // a slot's scores are the same in a batch of 8, 4, 2 or 1
            let mut one = vec![0f32; PAGE];
            fast::scores::<1>(&kv, base, nt, hd, 0.05, [rows[5]], &mut one);
            assert!((0..nt).all(|t| one[t].to_bits() == a[5 * PAGE + t].to_bits()), "hd {hd}: a slot alone differs");
            let mut two = vec![0f32; 2 * PAGE];
            fast::scores::<2>(&kv, base, nt, hd, 0.05, [rows[4], rows[5]], &mut two);
            assert!((0..nt).all(|t| two[PAGE + t].to_bits() == a[5 * PAGE + t].to_bits()), "hd {hd}: batch of 2");
        }
    }

    /// The score kernels alone on one thread: a page of 256 int8 keys of 256 against 8 queries, the exact kernel and
    /// TORC_ATTN_FAST's, over 40 pages (2.6 MB of keys, as an item meets them); and the values kernel for scale.
    #[test]
    #[ignore]
    fn bench_fast_scores() {
        let mut s = 0x0f0f_1234_5678_9abcu64;
        let (hd, pages) = (256usize, 40usize);
        let kv = cache(&mut s, 1, hd, pages * PAGE, false);
        let q: Vec<f32> = (0..8 * hd).map(|_| rnd(&mut s)).collect();
        let rows: [&[f32]; 8] = std::array::from_fn(|i| &q[i * hd..(i + 1) * hd]);
        let mut sc = vec![0f32; 8 * PAGE];
        let mut outs = vec![0f32; 8 * hd];
        let reps = 50;
        let mac = (reps * pages * PAGE * hd * 8) as f64;
        let time = |f: &mut dyn FnMut(usize)| {
            for p in 0..pages {
                f(p);
            }
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for p in 0..pages {
                    f(p);
                }
            }
            mac / t.elapsed().as_secs_f64() / 1e9
        };
        let exact = time(&mut |p| scores::<8>(&kv, p * PAGE, PAGE, hd, 0.05, rows, &mut sc));
        let fast = time(&mut |p| fast::scores::<8>(&kv, p * PAGE, PAGE, hd, 0.05, rows, &mut sc));
        let vals = time(&mut |p| {
            let mut it = outs.chunks_exact_mut(hd);
            let o: [&mut [f32]; 8] = std::array::from_fn(|_| it.next().unwrap());
            values::<8>(&kv, p * PAGE, PAGE, hd, &sc, o)
        });
        let exact2 = time(&mut |p| scores::<8>(&kv, p * PAGE, PAGE, hd, 0.05, rows, &mut sc));
        println!(
            "scores, 8 slots, one thread: exact {exact:.1} / {exact2:.1} GMAC/s, fast {fast:.1} GMAC/s ({:.2}x); values {vals:.1} GMAC/s",
            fast / exact.max(exact2)
        );
    }

    #[test]
    fn matches_the_definition() {
        let _one = one_caller();
        let mut s = 0x1234_5678_9abc_def1u64;
        for &(nh, nkv, hd, win, kv_f16, n, pos0) in &CASES {
            let kv = cache(&mut s, nkv, hd, pos0 + n + 3, kv_f16);
            let qd = nh * hd;
            let q: Vec<f32> = (0..n * qd).map(|_| rnd(&mut s)).collect();
            let f = Fattn::new(nh, nkv, hd, Some(0.05), win);
            let mut out = vec![0.0f32; n * qd];
            f.run_block(&kv, &q, qd, n, pos0 as u32, &mut out, None);
            let r = reference(&f, &kv, &q, n, pos0);
            let worst = out.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let scale = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(worst <= 1e-5 * scale.max(1e-3), "nh {nh} nkv {nkv} hd {hd} win {win} n {n}: {worst} of {scale}");
        }
    }

    /// SPEC.md 1's block invariance: each token of a block has the bits it has alone.
    #[test]
    fn blocks_are_tokens() {
        let _one = one_caller();
        let mut s = 0x0bad_cafe_1234_5678u64;
        for &(nh, nkv, hd, win, kv_f16, n, pos0) in &CASES {
            let kv = cache(&mut s, nkv, hd, pos0 + n + 3, kv_f16);
            let qd = nh * hd;
            let q: Vec<f32> = (0..n * qd).map(|_| rnd(&mut s)).collect();
            let f = Fattn::new(nh, nkv, hd, Some(0.05), win);
            let mut blk = vec![0.0f32; n * qd];
            f.run_block(&kv, &q, qd, n, pos0 as u32, &mut blk, None);
            for t in 0..n {
                let mut one = vec![0.0f32; qd];
                f.run_block(&kv, &q[t * qd..(t + 1) * qd], qd, 1, (pos0 + t) as u32, &mut one, None);
                let same = one.iter().zip(&blk[t * qd..(t + 1) * qd]).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "nh {nh} nkv {nkv} hd {hd} win {win} n {n}: token {t} differs alone");
            }
        }
    }

    /// QSA: random picks of pools of 64 positions (with some tokens picking none), checked against the definition, and
    /// each token alone with its own row of picks giving the bits it has in the block.
    #[test]
    fn selection() {
        let _one = one_caller();
        let mut s = 0x005e_ed0f_9a5a_u64;
        for &(nh, nkv, hd, kv_f16, n, pos0) in
            &[(16, 2, 128, true, 6, 700), (8, 2, 64, true, 3, 300), (8, 8, 64, false, 5, 1000)]
        {
            let kv = cache(&mut s, nkv, hd, pos0 + n + 3, kv_f16);
            let (qd, kp) = (nh * hd, 64);
            let stride = (pos0 + n).div_ceil(kp);
            let q: Vec<f32> = (0..n * qd).map(|_| rnd(&mut s)).collect();
            let rows: Vec<u8> = (0..n * stride).map(|_| (rnd(&mut s) > 0.3) as u8).collect();
            // the pools each token's picks cover (the rest is its tail); the last token has no selection at all
            let nv: Vec<u32> =
                (0..n).map(|t| if t + 1 == n { u32::MAX } else { ((pos0 + t) / kp) as u32 - (t % 2) as u32 }).collect();
            let se = Sel { rows: &rows, nv: &nv, stride, kp };
            let f = Fattn::new(nh, nkv, hd, Some(0.05), 0);
            let mut out = vec![0.0f32; n * qd];
            f.run_block(&kv, &q, qd, n, pos0 as u32, &mut out, Some(&se));
            let r = reference_sel(&f, &kv, &q, n, pos0, Some(&se));
            let worst = out.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let scale = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(worst <= 1e-5 * scale.max(1e-3), "nh {nh} hd {hd} n {n}: {worst} of {scale}");
            for t in 0..n {
                let one_sel = Sel { rows: &rows[t * stride..], nv: &nv[t..], stride, kp };
                let mut one = vec![0.0f32; qd];
                f.run_block(&kv, &q[t * qd..(t + 1) * qd], qd, 1, (pos0 + t) as u32, &mut one, Some(&one_sel));
                assert!(
                    one.iter().zip(&out[t * qd..(t + 1) * qd]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "token {t}"
                );
            }
        }
    }

    /// TORC_ATTN_F32's kernels (the page as f32, the scores two positions a pass) give the bits of scores and values:
    /// int8 and int16 caches, every slot count, and whole and partial pages (odd position counts).
    #[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    #[test]
    fn f32_page_same_bits() {
        #[cfg(target_arch = "aarch64")]
        use neon as arch;
        #[cfg(target_arch = "x86_64")]
        use x86 as arch;
        let mut s = 0x5151_2026_1103_0001u64;
        let hd = 256;
        for kv_f16 in [false, true] {
            let kv = cache(&mut s, 2, hd, 2 * PAGE, kv_f16);
            let q: Vec<Vec<f32>> = (0..8).map(|_| (0..hd).map(|_| rnd(&mut s)).collect()).collect();
            for (base, nt) in [(0usize, PAGE), (2 * PAGE - 77, 77), (PAGE, 1), (5, 2), (2 * PAGE - 255, 255)] {
                let mut kf = vec![0.0f32; nt * hd];
                let mut vf = vec![0.0f32; nt * hd];
                arch::page_f32(&kv, base, nt, hd, &mut kf, &mut vf);
                let ks = &kv.k_scale[base..base + nt];
                macro_rules! check {
                    ($ns:literal) => {{
                        let qs: [&[f32]; $ns] = std::array::from_fn(|k| &q[k][..]);
                        let (mut a, mut b) = (vec![0.0f32; 8 * PAGE], vec![0.0f32; 8 * PAGE]);
                        scores::<$ns>(&kv, base, nt, hd, 0.0625, qs, &mut a);
                        arch::scores_f32::<$ns>(&kf, ks, nt, hd, 0.0625, qs, &mut b);
                        for k in 0..$ns {
                            for p in 0..nt {
                                assert_eq!(
                                    a[k * PAGE + p].to_bits(),
                                    b[k * PAGE + p].to_bits(),
                                    "scores ns {} nt {nt} kv_f16 {kv_f16}",
                                    $ns
                                );
                            }
                        }
                        let w: Vec<f32> = (0..8 * PAGE).map(|_| rnd(&mut s).abs()).collect();
                        let mut oa: Vec<Vec<f32>> = (0..$ns).map(|_| vec![0.0f32; hd]).collect();
                        let mut ob = oa.clone();
                        {
                            let mut it = oa.iter_mut();
                            values::<$ns>(&kv, base, nt, hd, &w, std::array::from_fn(|_| &mut it.next().unwrap()[..]));
                            let mut it = ob.iter_mut();
                            arch::values_f32::<$ns>(
                                &vf,
                                nt,
                                hd,
                                &w,
                                std::array::from_fn(|_| &mut it.next().unwrap()[..]),
                            );
                        }
                        for k in 0..$ns {
                            for d in 0..hd {
                                assert_eq!(
                                    oa[k][d].to_bits(),
                                    ob[k][d].to_bits(),
                                    "values ns {} nt {nt} kv_f16 {kv_f16}",
                                    $ns
                                );
                            }
                        }
                    }};
                }
                check!(1);
                check!(2);
                check!(4);
                check!(8);
            }
        }
    }

    /// The core's three parts timed on one thread, for a 256-position page of an int8 cache (head_dim 256) against 8
    /// slots, as a long prompt's items run them: `cargo test --release -p torc-model --lib bench_parts -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn bench_parts() {
        let mut s = 0x7777_1234_abcd_0001u64;
        let (hd, nt, reps) = (256usize, PAGE, 4000usize);
        let kv = cache(&mut s, 1, hd, PAGE, false);
        let q: Vec<Vec<f32>> = (0..8).map(|_| (0..hd).map(|_| rnd(&mut s)).collect()).collect();
        let mut sc = vec![0.0f32; 8 * PAGE];
        let mut out: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; hd]).collect();
        let qs: [&[f32]; 8] = std::array::from_fn(|k| &q[k][..]);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            scores::<8>(&kv, 0, nt, hd, 0.05, qs, &mut sc);
        }
        let ts = t.elapsed().as_secs_f64();
        let saved = sc.clone();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            sc.copy_from_slice(&saved);
            for k in 0..8 {
                std::hint::black_box(weights(&mut sc[k * PAGE..(k + 1) * PAGE], nt, nt, &kv.v_scale[..nt]));
            }
        }
        let tw = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let o: [&mut [f32]; 8] = {
                let mut it = out.iter_mut();
                std::array::from_fn(|_| &mut it.next().unwrap()[..])
            };
            values::<8>(&kv, 0, nt, hd, &sc, o);
        }
        let tv = t.elapsed().as_secs_f64();
        // the trial: two positions a pass, with the 8 slots as two halves of 4; the same bits as scores::<8>
        #[cfg(target_arch = "aarch64")]
        {
            let mut sc2 = vec![0.0f32; 8 * PAGE];
            let ks = &kv.k_scale[..nt];
            let qa: [*const f32; 4] = std::array::from_fn(|k| q[k].as_ptr());
            let qb: [*const f32; 4] = std::array::from_fn(|k| q[k + 4].as_ptr());
            let t = std::time::Instant::now();
            for _ in 0..reps {
                // SAFETY: the cache rows 0..nt, the query rows and sc2 (8 x PAGE) are in bounds
                unsafe {
                    neon::scores_rows2::<4, i8>(kv.k.as_ptr(), ks, nt, hd, 0.05, qa, sc2.as_mut_ptr());
                    neon::scores_rows2::<4, i8>(kv.k.as_ptr(), ks, nt, hd, 0.05, qb, sc2.as_mut_ptr().add(4 * PAGE));
                }
            }
            let t2 = t.elapsed().as_secs_f64();
            let same = (0..8).all(|k| (0..nt).all(|i| sc2[k * PAGE + i].to_bits() == saved[k * PAGE + i].to_bits()));
            println!(
                "scores_rows2 {:.1} ms ({:.2} GMAC/s), same bits {same}",
                t2 * 1e3,
                (reps * nt * hd * 8) as f64 / t2 / 1e9
            );
            // the trial: an f32 copy of the cache (exact), so the loops need no conversion - scores for 8 slots a
            // position, 4 slots two positions a pass, and values for 8 slots
            let kf: Vec<f32> = kv.k[..nt * hd].iter().map(|&x| x as f32).collect();
            let vf: Vec<f32> = kv.v[..nt * hd].iter().map(|&x| x as f32).collect();
            let gm = |s: f64| (reps * nt * hd * 8) as f64 / s / 1e9;
            let same_sc =
                |a: &[f32]| (0..8).all(|k| (0..nt).all(|i| a[k * PAGE + i].to_bits() == saved[k * PAGE + i].to_bits()));
            let qp8: [*const f32; 8] = std::array::from_fn(|k| q[k].as_ptr());
            let t = std::time::Instant::now();
            for _ in 0..reps {
                // SAFETY: as above, on the f32 copy
                unsafe { neon::scores_rows::<8, f32>(kf.as_ptr(), ks, nt, hd, 0.05, qp8, sc2.as_mut_ptr()) };
            }
            let tf = t.elapsed().as_secs_f64();
            println!("scores f32 cache {:.1} ms ({:.2} GMAC/s), same bits {}", tf * 1e3, gm(tf), same_sc(&sc2));
            let t = std::time::Instant::now();
            for _ in 0..reps {
                // SAFETY: as above
                unsafe {
                    neon::scores_rows2::<4, f32>(kf.as_ptr(), ks, nt, hd, 0.05, qa, sc2.as_mut_ptr());
                    neon::scores_rows2::<4, f32>(kf.as_ptr(), ks, nt, hd, 0.05, qb, sc2.as_mut_ptr().add(4 * PAGE));
                }
            }
            let tf2 = t.elapsed().as_secs_f64();
            println!("scores_rows2 f32 cache {:.1} ms ({:.2} GMAC/s), same bits {}", tf2 * 1e3, gm(tf2), same_sc(&sc2));
            let mut outf: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; hd]).collect();
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let op: [*mut f32; 8] = {
                    let mut it = outf.iter_mut();
                    std::array::from_fn(|_| it.next().unwrap().as_mut_ptr())
                };
                // SAFETY: as above; the output rows hold hd each, and the weights 8 x PAGE
                unsafe { neon::values_rows::<8, f32>(vf.as_ptr(), nt, hd, saved.as_ptr(), op) };
            }
            let tvf = t.elapsed().as_secs_f64();
            println!("values f32 cache {:.1} ms ({:.2} GMAC/s)", tvf * 1e3, gm(tvf));
            std::hint::black_box(&outf);
        }
        let macs = (reps * nt * hd * 8) as f64;
        println!(
            "scores {:.1} ms ({:.2} GMAC/s), weights {:.1} ms, values {:.1} ms ({:.2} GMAC/s), one thread, {reps} reps",
            ts * 1e3,
            macs / ts / 1e9,
            tw * 1e3,
            tv * 1e3,
            macs / tv / 1e9
        );
        std::hint::black_box(&out);
    }

    /// f32 to bf16, rounding to nearest even (finite inputs).
    #[cfg(target_arch = "aarch64")]
    fn bf16_of(x: f32) -> u16 {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }

    /// BFMMLA (FEAT_BF16): acc.4s += A . B^T, where A and B are each 2 rows x 4 bf16, giving the lanes [a0.b0, a0.b1,
    /// a1.b0, a1.b1].
    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[target_feature(enable = "bf16")]
    unsafe fn bfmmla(
        mut acc: std::arch::aarch64::float32x4_t,
        a: std::arch::aarch64::uint16x8_t,
        b: std::arch::aarch64::uint16x8_t,
    ) -> std::arch::aarch64::float32x4_t {
        unsafe {
            std::arch::asm!("bfmmla {acc:v}.4s, {a:v}.8h, {b:v}.8h", acc = inout(vreg) acc, a = in(vreg) a,
                b = in(vreg) b, options(pure, nomem, nostack));
        }
        acc
    }

    /// Rows (n x hd, f32) packed in pairs for BFMMLA: pair p's dims 4d..4d + 3 go at [(p (hd / 4) + d) 8..][..8], four
    /// from each of the two rows (an odd last row is paired with itself).
    #[cfg(target_arch = "aarch64")]
    fn pack_pairs(rows: &[f32], n: usize, hd: usize) -> Vec<u16> {
        let np = n.div_ceil(2);
        let mut out = vec![0u16; np * hd * 2];
        for p in 0..np {
            let (r0, r1) = (2 * p, (2 * p + 1).min(n - 1));
            for d in 0..hd / 4 {
                for e in 0..4 {
                    out[(p * (hd / 4) + d) * 8 + e] = bf16_of(rows[r0 * hd + 4 * d + e]);
                    out[(p * (hd / 4) + d) * 8 + 4 + e] = bf16_of(rows[r1 * hd + 4 * d + e]);
                }
            }
        }
        out
    }

    /// The bf16 alternative's scores, timed and checked: one 256-position page of an int8 cache against 8 query slots
    /// by BFMMLA (the keys are exact in bf16, the queries are rounded to bf16), 2 slot pairs x 4 position pairs a pass,
    /// compared with the f32 kernel.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bench_bf16_scores() {
        use std::arch::aarch64::*;
        if !std::arch::is_aarch64_feature_detected!("bf16") {
            println!("no FEAT_BF16 on this CPU");
            return;
        }
        let mut s = 0x7777_1234_abcd_0001u64;
        let (hd, nt, reps) = (256usize, PAGE, 4000usize);
        let kv = cache(&mut s, 1, hd, PAGE, false);
        let q: Vec<f32> = (0..8 * hd).map(|_| rnd(&mut s)).collect();
        // the keys as f32 integers (their scale is applied afterwards), packed once
        let kf: Vec<f32> = kv.k[..nt * hd].iter().map(|&x| x as f32).collect();
        let t = std::time::Instant::now();
        let kp = pack_pairs(&kf, nt, hd);
        let tpack = t.elapsed().as_secs_f64();
        let qp = pack_pairs(&q, 8, hd);
        let mut sc = vec![0.0f32; 8 * PAGE];
        let d4 = hd / 4;
        /// The 8 slots' raw scores over nt positions (sc[s PAGE + p]), 2 slot pairs x 4 position pairs a pass.
        #[target_feature(enable = "bf16")]
        unsafe fn run(qp: &[u16], kp: &[u16], nt: usize, d4: usize, sc: &mut [f32]) {
            unsafe {
                for sp in (0..4).step_by(2) {
                    let mut pp = 0;
                    while pp < nt / 2 {
                        let mut acc = [[vdupq_n_f32(0.0); 4]; 2];
                        for d in 0..d4 {
                            let a0 = vld1q_u16(qp.as_ptr().add((sp * d4 + d) * 8));
                            let a1 = vld1q_u16(qp.as_ptr().add(((sp + 1) * d4 + d) * 8));
                            for j in 0..4 {
                                let b = vld1q_u16(kp.as_ptr().add(((pp + j) * d4 + d) * 8));
                                acc[0][j] = bfmmla(acc[0][j], a0, b);
                                acc[1][j] = bfmmla(acc[1][j], a1, b);
                            }
                        }
                        for (h, a) in acc.iter().enumerate() {
                            for (j, &v) in a.iter().enumerate() {
                                let mut o = [0.0f32; 4];
                                vst1q_f32(o.as_mut_ptr(), v);
                                let (s0, p0) = (2 * (sp + h), 2 * (pp + j));
                                *sc.get_unchecked_mut(s0 * PAGE + p0) = o[0];
                                *sc.get_unchecked_mut(s0 * PAGE + p0 + 1) = o[1];
                                *sc.get_unchecked_mut((s0 + 1) * PAGE + p0) = o[2];
                                *sc.get_unchecked_mut((s0 + 1) * PAGE + p0 + 1) = o[3];
                            }
                        }
                        pp += 4;
                    }
                }
            }
        }
        let t = std::time::Instant::now();
        for _ in 0..reps {
            // SAFETY: FEAT_BF16 was checked, and every index in run is inside qp (4 pairs), kp (nt / 2 pairs) and sc (8
            // x PAGE)
            unsafe { run(&qp, &kp, nt, d4, &mut sc) };
        }
        let tb = t.elapsed().as_secs_f64();
        // the f32 kernel's scores (with scale and the key scale applied) to compare these against (with the same
        // applied here)
        let mut want = vec![0.0f32; 8 * PAGE];
        let qs: [&[f32]; 8] = std::array::from_fn(|k| &q[k * hd..(k + 1) * hd]);
        scores::<8>(&kv, 0, nt, hd, 0.05, qs, &mut want);
        let (mut worst, mut mag, mut sq) = (0.0f32, 0.0f32, 0.0f64);
        for k in 0..8 {
            for i in 0..nt {
                let g = sc[k * PAGE + i] * 0.05 * kv.k_scale[i];
                worst = worst.max((g - want[k * PAGE + i]).abs());
                sq += ((g - want[k * PAGE + i]) as f64).powi(2);
                mag = mag.max(want[k * PAGE + i].abs());
            }
        }
        let macs = (reps * nt * hd * 8) as f64;
        println!(
            "bf16 scores {:.1} ms ({:.2} GMAC/s), packing the page's keys {:.3} ms once, worst error {worst:.2e} of {mag:.2e}, rms {:.2e}",
            tb * 1e3,
            macs / tb / 1e9,
            tpack * 1e3,
            (sq / (8 * nt) as f64).sqrt()
        );

        // the int8 route: the queries quantised per row (scale max / 127), the keys as they are, SMMLA over 2 slots x 2
        // positions x 8
        let mut q8 = vec![0i8; 8 * hd];
        let mut qsc = [0.0f32; 8];
        for k in 0..8 {
            let row = &q[k * hd..(k + 1) * hd];
            let m = row.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            qsc[k] = m / 127.0;
            for i in 0..hd {
                q8[k * hd + i] = (row[i] / qsc[k]).round() as i8;
            }
        }
        // pairs of rows, 8 bytes of each per step: pair p's step d (dims 8d..8d + 7) goes at [(p (hd / 8) + d)
        // 16..][..16]
        let pack8 = |rows: &[i8], n: usize| {
            let mut out = vec![0i8; n.div_ceil(2) * hd * 2];
            for p in 0..n.div_ceil(2) {
                let (r0, r1) = (2 * p, (2 * p + 1).min(n - 1));
                for d in 0..hd / 8 {
                    for e in 0..8 {
                        out[(p * (hd / 8) + d) * 16 + e] = rows[r0 * hd + 8 * d + e];
                        out[(p * (hd / 8) + d) * 16 + 8 + e] = rows[r1 * hd + 8 * d + e];
                    }
                }
            }
            out
        };
        let (qp8, kp8) = (pack8(&q8, 8), pack8(&kv.k[..nt * hd], nt));
        let mut si = vec![0i32; 8 * PAGE];
        #[target_feature(enable = "i8mm")]
        unsafe fn run8(qp: &[i8], kp: &[i8], nt: usize, d8: usize, si: &mut [i32]) {
            unsafe {
                for sp in (0..4).step_by(2) {
                    let mut pp = 0;
                    while pp < nt / 2 {
                        let mut acc = [[vdupq_n_s32(0); 4]; 2];
                        for d in 0..d8 {
                            let a0 = vld1q_s8(qp.as_ptr().add((sp * d8 + d) * 16));
                            let a1 = vld1q_s8(qp.as_ptr().add(((sp + 1) * d8 + d) * 16));
                            for j in 0..4 {
                                let b = vld1q_s8(kp.as_ptr().add(((pp + j) * d8 + d) * 16));
                                acc[0][j] = smmla8(acc[0][j], a0, b);
                                acc[1][j] = smmla8(acc[1][j], a1, b);
                            }
                        }
                        for (h, a) in acc.iter().enumerate() {
                            for (j, &v) in a.iter().enumerate() {
                                let mut o = [0i32; 4];
                                vst1q_s32(o.as_mut_ptr(), v);
                                let (s0, p0) = (2 * (sp + h), 2 * (pp + j));
                                *si.get_unchecked_mut(s0 * PAGE + p0) = o[0];
                                *si.get_unchecked_mut(s0 * PAGE + p0 + 1) = o[1];
                                *si.get_unchecked_mut((s0 + 1) * PAGE + p0) = o[2];
                                *si.get_unchecked_mut((s0 + 1) * PAGE + p0 + 1) = o[3];
                            }
                        }
                        pp += 4;
                    }
                }
            }
        }
        let t = std::time::Instant::now();
        for _ in 0..reps {
            // SAFETY: FEAT_I8MM (checked by the caller), and the indexes stay inside qp8, kp8 and si
            unsafe { run8(&qp8, &kp8, nt, hd / 8, &mut si) };
        }
        let t8 = t.elapsed().as_secs_f64();
        let (mut worst8, mut sum2, mut n2) = (0.0f32, 0.0f64, 0usize);
        for k in 0..8 {
            for i in 0..nt {
                let g = si[k * PAGE + i] as f32 * qsc[k] * 0.05 * kv.k_scale[i];
                let e = g - want[k * PAGE + i];
                worst8 = worst8.max(e.abs());
                sum2 += (e as f64) * (e as f64);
                n2 += 1;
            }
        }
        println!(
            "int8 scores {:.1} ms ({:.2} GMAC/s), worst error {worst8:.2e}, rms {:.2e}",
            t8 * 1e3,
            macs / t8 / 1e9,
            (sum2 / n2 as f64).sqrt()
        );
    }

    /// SMMLA for the int8 trial (as the kernels crate's smmla: acc.4s += A . B^T on 2 x 8 bytes each).
    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[target_feature(enable = "i8mm")]
    unsafe fn smmla8(
        mut acc: std::arch::aarch64::int32x4_t,
        a: std::arch::aarch64::int8x16_t,
        b: std::arch::aarch64::int8x16_t,
    ) -> std::arch::aarch64::int32x4_t {
        unsafe {
            std::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a,
                b = in(vreg) b, options(pure, nomem, nostack));
        }
        acc
    }

    /// The vector exp of each architecture against crate::act::expf, bit for bit, over the weights' range and past
    /// both ends of it.
    #[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    #[test]
    fn vector_exp_is_expf() {
        let xs: Vec<f32> = (-9200..=9000).map(|i| i as f32 / 100.0).chain([f32::NAN, -1e9, 1e9, -0.0]).collect();
        for c in xs.chunks(8) {
            let mut x = [0.0f32; 8];
            x[..c.len()].copy_from_slice(c);
            let mut y = [0.0f32; 8];
            // SAFETY: eight floats in and out
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                vst1q_f32(y.as_mut_ptr(), neon::exp4(vld1q_f32(x.as_ptr())));
                vst1q_f32(y.as_mut_ptr().add(4), neon::exp4(vld1q_f32(x.as_ptr().add(4))));
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                _mm256_storeu_ps(y.as_mut_ptr(), x86::exp8(_mm256_loadu_ps(x.as_ptr())));
            }
            for k in 0..8 {
                assert_eq!(y[k].to_bits(), expf(x[k]).to_bits(), "x {}", x[k]);
            }
        }
    }

    /// The NEON and AVX2 weights against the definition (portable::weights): the weights and their sum must match bit
    /// for bit and the largest score must be equal (its sign may differ when it is zero), for every visible length, for
    /// rows with zeros of both signs, masked (-inf) scores, and a largest score of zero.
    #[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    #[test]
    fn weights_match_portable() {
        let mut s = 0x7a3c_91e5_2d48_b607u64;
        for case in 0..400 {
            let nt = 1 + case % PAGE;
            let vis = 1 + (case * 37) % nt;
            let mut row = vec![0.0f32; PAGE];
            for (t, x) in row.iter_mut().enumerate().take(nt) {
                *x = match (case + t) % 11 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::NEG_INFINITY,
                    _ if case % 3 == 0 => -rnd(&mut s).abs() * 30.0,
                    _ => rnd(&mut s) * 30.0,
                };
            }
            let vs: Vec<f32> = (0..PAGE).map(|_| 0.5 + rnd(&mut s).abs()).collect();
            let (mut a, mut b) = (row.clone(), row.clone());
            #[cfg(target_arch = "aarch64")]
            let (ma, la) = neon::weights(&mut a, nt, vis, &vs);
            #[cfg(target_arch = "x86_64")]
            let (ma, la) = x86::weights(&mut a, nt, vis, &vs);
            let (mb, lb) = portable::weights(&mut b, nt, vis, &vs);
            assert!(ma == mb || (ma.is_infinite() && mb.is_infinite()), "case {case}: m {ma} vs {mb}");
            assert_eq!(la.to_bits(), lb.to_bits(), "case {case}: sum {la} vs {lb}");
            for t in 0..nt {
                assert_eq!(a[t].to_bits(), b[t].to_bits(), "case {case} nt {nt} vis {vis}: weight {t}");
            }
        }
    }

    /// The Qwen3.6-35B's attention at an MTP verify pass (16 query heads over 2 KV heads of 256, 8k positions, int8
    /// cache): run_block over BENCH_LAYERS caches in turn (default 10), as a pass meets its 10 attention layers (80 MB
    /// of keys and values, past the L2), on BENCH_THREADS threads (default 8), for blocks of 1 and 4 tokens. Then one
    /// thread's item of 32 slots (a page of 256 positions) on TORC_ATTN_TILE's path, with its parts timed separately.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bench_verify() {
        let _one = one_caller();
        let env = |k: &str, d: usize| crate::switches::num(k).unwrap_or(d);
        let (layers, threads) = (env("BENCH_LAYERS", 10), env("BENCH_THREADS", 8));
        let pool = crate::pool::init(threads);
        let mut s = 0x1234_5678_9abc_def0u64;
        let (nh, nkv, hd, end) = (16usize, 2usize, 256usize, 8000usize);
        let kvs: Vec<KvCache> = (0..layers).map(|_| cache(&mut s, nkv, hd, end + 8, false)).collect();
        let qd = nh * hd;
        let q: Vec<f32> = (0..4 * qd).map(|_| rnd(&mut s)).collect();
        let f = Fattn::new(nh, nkv, hd, Some(0.05), 0);
        let mut out = vec![0.0f32; 4 * qd];
        for n in [1usize, 4] {
            let pos0 = (end - n) as u32;
            for kv in &kvs {
                f.run_block(kv, &q, qd, n, pos0, &mut out, None);
            }
            let reps = 30;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for kv in &kvs {
                    f.run_block(kv, &q, qd, n, pos0, &mut out, None);
                }
            }
            let el = t.elapsed().as_secs_f64() / (reps * layers) as f64;
            let mac = (n * nh * end * hd * 2) as f64;
            println!(
                "run_block n {n}: {:.3} ms a layer, {:.1} GMAC/s, {:.1} a thread ({} threads, {layers} caches)",
                el * 1e3,
                mac / el / 1e9,
                mac / el / 1e9 / pool.threads as f64,
                pool.threads
            );
        }
        // one item on one thread: 4 tokens x the 8 query heads of KV head 0 = 32 slots, page 10 (256 positions)
        let (ns, tp, c) = (32usize, 32usize, 10usize);
        let mut st = vec![0.0f32; ns * PAGE];
        let mut tf = vec![0.0f32; tp * hd];
        let mut outs = vec![0.0f32; ns * hd];
        let mut tm = [0.0f64; 5];
        let reps = 200;
        for r in 0..reps {
            let kv = &kvs[r % layers];
            let base = c * PAGE;
            let ks = &kv.k_scale[base..base + PAGE];
            let qrow = |s: usize| &q[(s / 8) * qd + (s % 8) * hd..][..hd];
            let mut t = std::time::Instant::now();
            let mut lap = |i: usize, t: &mut std::time::Instant| {
                let now = std::time::Instant::now();
                tm[i] += (now - *t).as_secs_f64();
                *t = now;
            };
            let mut a = 0;
            while a < PAGE {
                neon::f32_rows(kv, false, base + a, tp, hd, &mut tf);
                lap(0, &mut t);
                for s0 in (0..ns).step_by(4) {
                    let q4: [*const f32; 4] = std::array::from_fn(|k| qrow(s0 + k).as_ptr());
                    // SAFETY: as in item
                    unsafe {
                        neon::scores_rows2::<4, f32>(
                            tf.as_ptr(),
                            &ks[a..],
                            tp,
                            hd,
                            0.05,
                            q4,
                            st.as_mut_ptr().add(s0 * PAGE + a),
                        )
                    };
                }
                lap(1, &mut t);
                a += tp;
            }
            for sl in 0..ns {
                std::hint::black_box(neon::weights(
                    &mut st[sl * PAGE..(sl + 1) * PAGE],
                    PAGE,
                    PAGE,
                    &kv.v_scale[base..base + PAGE],
                ));
            }
            lap(2, &mut t);
            let mut a = 0;
            while a < PAGE {
                neon::f32_rows(kv, true, base + a, tp, hd, &mut tf);
                lap(3, &mut t);
                for s0 in (0..ns).step_by(8) {
                    let op: [*mut f32; 8] = std::array::from_fn(|k| outs[(s0 + k) * hd..].as_mut_ptr());
                    // SAFETY: as in item
                    unsafe { neon::values_tile::<8>(tf.as_ptr(), tp, hd, st.as_ptr().add(s0 * PAGE + a), op, a == 0) };
                }
                lap(4, &mut t);
                a += tp;
            }
        }
        // the real item() over every (KV head, page) of a 4-token block on one thread, into a partial store laid out
        // as run_block lays it out
        {
            let n = 4usize;
            let pos0 = end - n;
            let span = (pos0 + n).div_ceil(PAGE);
            let group = nh / nkv;
            let (mut stat, mut acc) = (vec![0.0f32; span * n * nh * 2], vec![0.0f32; span * n * nh * hd]);
            let parts =
                Partials { stat: SyncPtr::new(&mut stat), acc: SyncPtr::new(&mut acc), n, n_head: nh, head_dim: hd };
            let qf = &q[..n * qd];
            let reps = 10;
            let t = std::time::Instant::now();
            for r in 0..reps {
                let kv = &kvs[r % layers];
                for kvh in 0..nkv {
                    for c in 0..span {
                        f.item(kv, qf, qd, 0, n, pos0, kvh, c, 0, group, &parts, c, None);
                    }
                }
            }
            let ti = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "item() x {}: {:.3} ms a layer on one thread ({:.1} us an item)",
                nkv * span,
                ti * 1e3,
                ti * 1e6 / (nkv * span) as f64
            );
        }
        let us = |x: f64| x * 1e6 / reps as f64;
        let mac = (ns * PAGE * hd) as f64 * reps as f64;
        println!(
            "item of {ns} slots, one thread: K to f32 {:.1} us, scores {:.1} us ({:.1} GMAC/s), weights {:.1} us, \
            V to f32 {:.1} us, values {:.1} us ({:.1} GMAC/s); total {:.1} us ({layers} caches)",
            us(tm[0]),
            us(tm[1]),
            mac / tm[1] / 1e9,
            us(tm[2]),
            us(tm[3]),
            us(tm[4]),
            mac / tm[4] / 1e9,
            us(tm.iter().sum())
        );
    }
}
