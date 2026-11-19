//! iq2_xxs.rs - IQ2_XXS (ggml): two bits a weight with no per-group scale, at 2.0 bpw - the cheapest quant that
//! still keeps its accuracy by picking eight-weight vectors out of a table instead of coding weights on their own.
//! A block is a 16-bit block scale and, per 32 weights, four table indices and one 7-bit sign code each, with one
//! 4-bit exponent shared by all four.

use crate::quant::iq2xxs_grid::IQ2XXS_GRID;
use crate::quant::ksigns::{KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::{BlockQ8K, f16_to_f32};

/// IQ2_XXS's block: fp16 d, then 32 u16 - four eight-weight sub-blocks per 32 weights.
pub const IQ2_XXS_BLOCK: usize = 66;

/// How many of the nv vectors are carried at a time: the grid lookups and the sign bits of a block are read once
/// and applied to all of them.
const VECTORS: usize = 16;

/// Thirty-two values against the same thirty-two activations.
///
/// This format keeps its own loop, measured rather than argued: through `super::dots` - the form the other six
/// formats call, and faster there - IQ2_XXS is about 6 % slower at nv >= 16 on a 7735HS, in both rounds of the
/// bench (15.85 and 15.94 GMAC/s as written, 14.93 and 15.09 through the shared dot). Its unpack is a table lookup
/// per eight weights followed by a short straight loop, and that shape the compiler handles on its own.
#[inline(always)]
fn dot_group(w: &[i8], a: &[i8]) -> i32 {
    let mut s = 0i32;
    for j in 0..32 {
        s += w[j] as i32 * a[j] as i32;
    }
    s
}

/// One row against one vector: a block is read out of its packed form and dotted in the same pass, with nothing kept
/// between the two. Kept apart from the multi-vector path below so that each of them gets its own registers.
#[inline(never)]
fn row_one(wr: &[u8], nb: usize, x: &[BlockQ8K]) -> f32 {
    let mut acc = 0f32;
    for b in 0..nb {
        let (blk, xa) = (&wr[b * IQ2_XXS_BLOCK..(b + 1) * IQ2_XXS_BLOCK], &x[b]);
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) * xa.d;
        let (mut bsum, mut q8) = (0i32, 0usize);
        for ib in 0..8 {
            let at = 2 + 8 * ib;
            let (a0, a1) = (
                u32::from_le_bytes(blk[at..at + 4].try_into().unwrap()),
                u32::from_le_bytes(blk[at + 4..at + 8].try_into().unwrap()),
            );
            let ls = (2 * (a1 >> 28) + 1) as i32;
            let mut sumi = 0i32;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[a0.to_le_bytes()[l] as usize];
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let g = ((grid >> (8 * j)) & 0xff) as i32;
                    sumi += g * xa.qs[q8 + j] as i32 * if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        acc = d.mul_add(bsum as f32, acc);
    }
    acc
}

/// Rows of IQ2_XXS against nv vectors, portably (ggml's ggml_vec_dot_iq2_xxs_q8_K): every sub-block's magnitudes
/// come from the grid, the sign code flips the ones it says, and the shared exponent scales the group. The whole
/// dot ends up scaled by 1/8, which is how ggml folds the sub-block's offset into the answer.
pub fn iq2xxs_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * IQ2_XXS_BLOCK && w.len() >= rows * row_bytes);
    // One vector, which is what a decoded token is.
    if nv == 1 {
        for r in 0..rows {
            y[r] = row_one(&w[r * row_bytes..], nb, &x[..nb]) * 0.125;
        }
        return;
    }
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v0 in (0..nv).step_by(VECTORS) {
            let k = VECTORS.min(nv - v0);
            let mut acc = [0f32; VECTORS];
            for b in 0..nb {
                let blk = &wr[b * IQ2_XXS_BLOCK..(b + 1) * IQ2_XXS_BLOCK];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                // Signed grid vectors and the exponent of each group of 32, for the whole chunk at once.
                let (mut qv, mut ls) = ([0i8; 256], [0i32; 8]);
                for ib in 0..8 {
                    let at = 2 + 8 * ib;
                    let (a0, a1) = (
                        u32::from_le_bytes(blk[at..at + 4].try_into().unwrap()),
                        u32::from_le_bytes(blk[at + 4..at + 8].try_into().unwrap()),
                    );
                    ls[ib] = (2 * (a1 >> 28) + 1) as i32;
                    for l in 0..4 {
                        let grid = IQ2XXS_GRID[a0.to_le_bytes()[l] as usize];
                        let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                        for j in 0..8 {
                            let g = ((grid >> (8 * j)) & 0xff) as i32;
                            qv[32 * ib + 8 * l + j] = (g * (1 - 2 * ((signs >> j) & 1) as i32)) as i8;
                        }
                    }
                }
                for j in 0..k {
                    let xa = &x[(v0 + j) * nb + b];
                    let mut bsum = 0i32;
                    for ib in 0..8 {
                        bsum += ls[ib] * dot_group(&qv[32 * ib..], &xa.qs[32 * ib..]);
                    }
                    acc[j] = (d * xa.d).mul_add(bsum as f32, acc[j]);
                }
            }
            for j in 0..k {
                y[(v0 + j) * y_stride + r] = acc[j] * 0.125;
            }
        }
    }
}
