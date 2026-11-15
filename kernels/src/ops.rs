//! ops.rs - the operations the model calls by name, each on the fastest kernel this crate has for it.
//!
//! The Q8_K quantisers, f32 dot and axpy, f16 conversion, block sizes, and dequantisation.

use crate::Q8m;

/// Q8_K activations (SPEC.md section 4).
pub fn q8k_quantize(x: &[f32], y: Q8m) {
    crate::q8k_quantize(x, y)
}

/// Q8_K with ggml's scales: qwen4exp's block inputs, hidden activations, and attention / DeltaNet outputs (SPEC.md
/// section 4).
pub fn q8k_quantize_gg(x: &[f32], y: Q8m) {
    assert_eq!(x.len(), y.len() * crate::QK);
    crate::q8k::quantize_gg(x, y);
}

/// a . b.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    crate::fvec::dot_f32(a, b)
}

/// y += a * x.
pub fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    crate::fvec::axpy_f32(y, a, x)
}

/// f32 to fp16, rounding to nearest even.
pub fn fp32_to_fp16(f: f32) -> u16 {
    crate::q8k::f32_to_f16(f)
}

/// How many bytes 256 weights of ggml type ty take, or None for a type no kernel in this crate multiplies.
pub fn block_size(ty: u32) -> Option<usize> {
    // this crate's table (dequant::block_geometry), as bytes per 256 numbers; the float types and Q5_0 (0, 1, 6, 30)
    // are handled elsewhere, so they give None, as does every type without a kernel here
    match ty {
        0 | 1 | 6 | 30 => None,
        _ => crate::dequant::block_geometry(ty).map(|(bs, qk)| bs * (crate::QK / qk)),
    }
}

/// Weights to f32 (by ggml's definitions).
pub fn dequantize(ty: u32, w: &[u8], y: &mut [f32]) {
    if crate::dequant::block_geometry(ty).is_some_and(|(_, qk)| y.len().is_multiple_of(qk))
        && crate::dequant::dequantize(ty, w, y)
    {
        return;
    }
    panic!("dequantize: no dequantiser for ggml type {ty}");
}
