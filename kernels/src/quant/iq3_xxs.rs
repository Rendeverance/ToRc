//! iq3_xxs.rs - IQ3_XXS (ggml): 3.06 bpw. A 3-bit index into a 256-entry grid of four magnitudes per weight pair,
//! and per 32 weights one 32-bit word carrying a 4-bit exponent plus a 7-bit sign code for each of the four
//! sub-blocks. Signs are shared between the pair, which is what keeps it under 3.25 bpw.

use crate::quant::iq3xxs_grid::IQ3XXS_GRID;
use crate::quant::ksigns::{KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::{BlockQ8K, f16_to_f32};

/// IQ3_XXS's block: fp16 d, then 96 bytes - 64 grid indices and 32 words of exponents and sign codes.
pub const IQ3_XXS_BLOCK: usize = 98;

/// How many of the nv vectors are carried at a time: the grid lookups and the sign bits of a block are read once
/// and applied to all of them.
const VECTORS: usize = 16;

/// Thirty-two values against the same thirty-two activations.
#[inline(always)]
fn dot_group(w: &[i8], a: &[i8]) -> i32 {
    let mut s = 0i32;
    for j in 0..32 {
        s += w[j] as i32 * a[j] as i32;
    }
    s
}

/// Rows of IQ3_XXS against nv vectors, portably (ggml's ggml_vec_dot_iq3_xxs_q8_K): two grid entries make a
/// sub-block of eight weights, the first four taking the low four sign bits and the second the high four. The
/// result is scaled by 1/4, the grid holding magnitudes of 1..15 rather than the weights themselves.
pub fn iq3xxs_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * IQ3_XXS_BLOCK && w.len() >= rows * row_bytes);
    // One vector, which is what a decoded token is: the grid vectors are read out and dotted in the same pass, so
    // nothing is materialised for a block that only one vector uses.
    if nv == 1 {
        for r in 0..rows {
            let wr = &w[r * row_bytes..];
            let mut acc = 0f32;
            for b in 0..nb {
                let (blk, xa) = (&wr[b * IQ3_XXS_BLOCK..(b + 1) * IQ3_XXS_BLOCK], &x[b]);
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) * xa.d;
                let (q3, gas) = (&blk[2..66], &blk[66..98]);
                let (mut bsum, mut q8) = (0i32, 0usize);
                for ib in 0..8 {
                    let aux = u32::from_le_bytes(gas[4 * ib..4 * ib + 4].try_into().unwrap());
                    let ls = (2 * (aux >> 28) + 1) as i32;
                    let mut sumi = 0i32;
                    for l in 0..4 {
                        let (g1, g2) =
                            (IQ3XXS_GRID[q3[8 * ib + 2 * l] as usize], IQ3XXS_GRID[q3[8 * ib + 2 * l + 1] as usize]);
                        let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        for j in 0..4 {
                            let a = ((g1 >> (8 * j)) & 0xff) as i32;
                            let c = ((g2 >> (8 * j)) & 0xff) as i32;
                            sumi += a * xa.qs[q8 + j] as i32 * if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 }
                                + c * xa.qs[q8 + j + 4] as i32 * if signs & KMASK_IQ2XS[j + 4] != 0 { -1 } else { 1 };
                        }
                        q8 += 8;
                    }
                    bsum += sumi * ls;
                }
                acc = d.mul_add(bsum as f32, acc);
            }
            y[r] = acc * 0.25;
        }
        return;
    }
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v0 in (0..nv).step_by(VECTORS) {
            let k = VECTORS.min(nv - v0);
            let mut acc = [0f32; VECTORS];
            for b in 0..nb {
                let blk = &wr[b * IQ3_XXS_BLOCK..(b + 1) * IQ3_XXS_BLOCK];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let (q3, gas) = (&blk[2..66], &blk[66..98]);
                // Signed grid vectors and the exponent of each group of 32, for the whole chunk at once.
                let (mut qv, mut ls) = ([0i8; 256], [0i32; 8]);
                for ib in 0..8 {
                    let aux = u32::from_le_bytes(gas[4 * ib..4 * ib + 4].try_into().unwrap());
                    ls[ib] = (2 * (aux >> 28) + 1) as i32;
                    for l in 0..4 {
                        let (g1, g2) =
                            (IQ3XXS_GRID[q3[8 * ib + 2 * l] as usize], IQ3XXS_GRID[q3[8 * ib + 2 * l + 1] as usize]);
                        let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        for j in 0..4 {
                            let a = ((g1 >> (8 * j)) & 0xff) as i32;
                            let c = ((g2 >> (8 * j)) & 0xff) as i32;
                            let at = 32 * ib + 8 * l + j;
                            qv[at] = (a * (1 - 2 * ((signs >> j) & 1) as i32)) as i8;
                            qv[at + 4] = (c * (1 - 2 * ((signs >> (j + 4)) & 1) as i32)) as i8;
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
                y[(v0 + j) * y_stride + r] = acc[j] * 0.25;
            }
        }
    }
}
