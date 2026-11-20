//! dots.rs - the inner product the portable quant kernels end with: a run of unpacked int8 weights against a run of
//! int8 activations, the widths a format's groups come in (16 for Q2_K and the IQ formats' half groups, 32 for the
//! group dots and Q2_0). Every portable kernel reads a block once, unpacks it, then asks for this once per vector,
//! which is why the widths live here rather than in each format's file.
//!
//! x86-64 leaves these loops scalar: the iq2_xs kernel of a `-C target-cpu=x86-64-v3` build is 27 `imull` and not one
//! vector instruction, where the same shape in an ISA path is two instructions per 32 weights. So on x86-64 the form
//! is written out (x86mm::dot_w16, dot_w32); elsewhere the loop below runs, unchanged from the per-format code it came
//! out of. The x86 arm is gated on the same three features `platform::cpu::x86mm` is a module behind, so a build
//! without them - a plain x86-64 target, or any other arch - takes the scalar loop instead of failing to name it.
//!
//! On aarch64 dot_w32 is written out too (neon::dot_w32). There the compiler does vectorise these loops, well enough
//! for the 16-wide ones (IQ2_S and IQ2_XS measured 3 to 7 % slower through NEON's form), but Q2_0 measured +12 to
//! +68 % at 2-64 vectors through it, and IQ3_XXS level (Snapdragon X, bench_portable_quants, three rounds each).

/// Sixteen weights against sixteen activations.
#[inline(always)]
pub(crate) fn dot_w16(w: &[i8], a: &[i8]) -> i32 {
    debug_assert!(w.len() >= 16 && a.len() >= 16);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    return unsafe { crate::x86mm::dot_w16(w.as_ptr(), a.as_ptr()) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c")))]
    {
        let mut s = 0i32;
        for j in 0..16 {
            s += w[j] as i32 * a[j] as i32;
        }
        s
    }
}

/// Sixteen plain codes (0..3, Q2_K) against sixteen activations: no sign fold.
#[inline(always)]
pub(crate) fn dot_w16u(w: &[i8], a: &[i8]) -> i32 {
    debug_assert!(w.len() >= 16 && a.len() >= 16);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    return unsafe { crate::x86mm::dot_w16u(w.as_ptr(), a.as_ptr()) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c")))]
    {
        let mut s = 0i32;
        for j in 0..16 {
            s += w[j] as i32 * a[j] as i32;
        }
        s
    }
}

/// Thirty-two weights against thirty-two activations.
#[inline(always)]
pub(crate) fn dot_w32(w: &[i8], a: &[i8]) -> i32 {
    debug_assert!(w.len() >= 32 && a.len() >= 32);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    return unsafe { crate::x86mm::dot_w32(w.as_ptr(), a.as_ptr()) };
    #[cfg(target_arch = "aarch64")]
    return unsafe { crate::neon::dot_w32(w.as_ptr(), a.as_ptr()) };
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"),
        target_arch = "aarch64"
    )))]
    {
        let mut s = 0i32;
        for j in 0..32 {
            s += w[j] as i32 * a[j] as i32;
        }
        s
    }
}
