//! q8_k.rs - the activations' formats: a vector is quantised once for each family of weight formats, into two arrays
//! held side by side (Q8), block i of each covering the same 256 numbers. `k` is ggml's block_q8_K (one scale for the
//! 256, the values, and sums of 16), which the K-quants multiply against; `g` is the same numbers as eight groups of
//! 32, each with its own scale as in ggml's block_q8_0, plus each group's sum, which the 32-weight formats (Q4_0, Q4_1,
//! Q5_0, Q5_1, Q8_0, IQ4_NL) multiply against. Kept apart, a matmul streams only the array its weights use. This file
//! holds the types and the scalar definition.

use std::slice::SliceIndex;

pub const QK: usize = 256;

/// A range of blocks that indexes both arrays (a.b, a.., ..b or ..).
pub trait Blocks:
    SliceIndex<[BlockQ8K], Output = [BlockQ8K]> + SliceIndex<[Groups32], Output = [Groups32]> + Clone
{
}
impl<R: SliceIndex<[BlockQ8K], Output = [BlockQ8K]> + SliceIndex<[Groups32], Output = [Groups32]> + Clone> Blocks
    for R
{
}

/// The eight 32-number groups of 256 activations: each group's scale, the 256 values group by group, and each group's
/// sum (at most 32 x 127 in size, so an i16). Arrays rather than eight records, so one vector load takes the scales or
/// the sums of four or eight groups.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Groups32 {
    pub d: [f32; 8],
    pub qs: [i8; QK],
    pub sums: [i16; 8],
}

/// ggml's block_q8_K: 256 activations under one scale, with the sum of each 16.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK],
    pub bsums: [i16; QK / 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 4 + QK + 32);
const _: () = assert!(std::mem::size_of::<Groups32>() == 32 + QK + 16);

impl Default for BlockQ8K {
    fn default() -> Self {
        BlockQ8K { d: 0.0, qs: [0; QK], bsums: [0; 16] }
    }
}

impl Default for Groups32 {
    fn default() -> Self {
        Groups32 { d: [0.0; 8], qs: [0; QK], sums: [0; 8] }
    }
}

/// A quantised vector (or several, block after block): `k` for the K-quants, `g` for the 32-weight formats.
#[derive(Clone, Debug, Default)]
pub struct Q8 {
    pub k: Vec<BlockQ8K>,
    pub g: Vec<Groups32>,
}

impl Q8 {
    /// n blocks of 256 in each array, zeroed.
    pub fn new(n: usize) -> Q8 {
        Q8 { k: vec![BlockQ8K::default(); n], g: vec![Groups32::default(); n] }
    }
    pub fn len(&self) -> usize {
        self.k.len()
    }
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }
    pub fn view(&self) -> Q8s<'_> {
        Q8s { k: &self.k, g: &self.g }
    }
    pub fn view_mut(&mut self) -> Q8m<'_> {
        Q8m { k: &mut self.k, g: &mut self.g }
    }
    /// Blocks r of both arrays.
    pub fn sub(&self, r: impl Blocks) -> Q8s<'_> {
        self.view().sub(r)
    }
    pub fn sub_mut(&mut self, r: impl Blocks) -> Q8m<'_> {
        Q8m { k: &mut self.k[r.clone()], g: &mut self.g[r] }
    }
}

/// A borrowed quantised vector: the two arrays, the same length.
#[derive(Clone, Copy, Debug)]
pub struct Q8s<'a> {
    pub k: &'a [BlockQ8K],
    pub g: &'a [Groups32],
}

impl<'a> Q8s<'a> {
    pub fn len(&self) -> usize {
        self.k.len()
    }
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }
    #[allow(clippy::should_implement_trait)] // a range of blocks, not subtraction
    pub fn sub(self, r: impl Blocks) -> Q8s<'a> {
        Q8s { k: &self.k[r.clone()], g: &self.g[r] }
    }
    /// The two arrays as their bytes (both types are plain data: repr(C) f32, i8 and i16, no padding).
    pub fn bytes(self) -> (&'a [u8], &'a [u8]) {
        // SAFETY: as above; the lengths are the arrays' own sizes
        unsafe {
            (
                std::slice::from_raw_parts(self.k.as_ptr() as *const u8, std::mem::size_of_val(self.k)),
                std::slice::from_raw_parts(self.g.as_ptr() as *const u8, std::mem::size_of_val(self.g)),
            )
        }
    }
}

/// A mutably borrowed quantised vector.
#[derive(Debug)]
pub struct Q8m<'a> {
    pub k: &'a mut [BlockQ8K],
    pub g: &'a mut [Groups32],
}

impl<'a> Q8m<'a> {
    pub fn len(&self) -> usize {
        self.k.len()
    }
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }
    pub fn sub(&mut self, r: impl Blocks) -> Q8m<'_> {
        Q8m { k: &mut self.k[r.clone()], g: &mut self.g[r] }
    }
    /// The two arrays as their bytes, to be filled (any bytes are a valid value of either type).
    pub fn bytes_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        // SAFETY: both types are plain data (f32, i8, i16) with no padding, so any bytes are a valid value
        unsafe {
            (
                std::slice::from_raw_parts_mut(self.k.as_mut_ptr() as *mut u8, std::mem::size_of_val(self.k)),
                std::slice::from_raw_parts_mut(self.g.as_mut_ptr() as *mut u8, std::mem::size_of_val(self.g)),
            )
        }
    }
    pub fn view(&self) -> Q8s<'_> {
        Q8s { k: self.k, g: self.g }
    }
    /// Copies both arrays from src, the same length.
    pub fn copy_from(&mut self, src: Q8s) {
        self.k.copy_from_slice(src.k);
        self.g.copy_from_slice(src.g);
    }
}

/// NEON FMAX semantics: NaN if either argument is NaN.
#[inline]
fn fmax_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}

fn amax(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, &v| fmax_nan(a, v.abs()))
}

/// q = round-to-nearest-even(x * id), saturated to i32 and then clamped to +-127 (as vcvtnq_s32_f32 and vmin/vmax do).
#[inline]
fn q8(x: f32, id: f32) -> i8 {
    ((x * id).round_ties_even() as i32).clamp(-127, 127) as i8
}

fn groups(x: &[f32], qs: &mut [i8], bsums: &mut [i16], id: f32) {
    for (g, xg) in x.chunks_exact(16).enumerate() {
        let mut s = 0i32;
        for (i, &v) in xg.iter().enumerate() {
            let q = q8(v, id);
            qs[g * 16 + i] = q;
            s += q as i32;
        }
        bsums[g] = s as i16;
    }
}

/// Q8 activations (SPEC.md section 4): super::q8k::quantize where there are vector kernels, else the scalar form.
pub fn q8k_quantize(x: &[f32], y: Q8m) {
    super::q8k::quantize(x, y)
}

/// Q8 activations, one number at a time (the definition); n = x.len() must be a multiple of 256.
pub fn q8k_quantize_scalar(x: &[f32], y: Q8m) {
    assert!(x.len().is_multiple_of(QK) && y.len() >= x.len() / QK && y.g.len() == y.k.len());
    for ((xb, yb), gb) in x.chunks_exact(QK).zip(y.k.iter_mut()).zip(y.g.iter_mut()) {
        let a = amax(xb);
        let d = a / 127.0;
        yb.d = d;
        groups(xb, &mut yb.qs, &mut yb.bsums, if d > 0.0 { 1.0 / d } else { 0.0 });
        for k in 0..8 {
            let xk = &xb[32 * k..32 * k + 32];
            let a = amax(xk);
            let d = a / 127.0;
            gb.d[k] = d;
            let mut halves = [0i16; 2];
            groups(xk, &mut gb.qs[32 * k..32 * k + 32], &mut halves, if d > 0.0 { 1.0 / d } else { 0.0 });
            gb.sums[k] = halves[0] + halves[1];
        }
    }
}
