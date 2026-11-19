//! iq2_xs.rs - IQ2_XS (ggml): 2.31 bpw. Same table of eight-weight vectors as IQ2_XXS, but with a 9-bit index
//! (512 vectors instead of 256) and two 4-bit exponents per 32 weights instead of one, which is what buys back the
//! accuracy IQ2_XXS gives up.

use crate::quant::iq2xs_grid::IQ2XS_GRID;
use crate::quant::ksigns::{KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::{BlockQ8K, f16_to_f32};

/// IQ2_XS's block: fp16 d, 32 u16 sub-blocks (9 index bits + a 7-bit sign code each), and 8 scale bytes.
pub const IQ2_XS_BLOCK: usize = 74;

/// How many of the nv vectors are carried at a time: the grid lookups and the sign bits of a block are read once
/// and applied to all of them.
const VECTORS: usize = 16;

/// Sixteen values against the same sixteen activations - the width one of the group's two exponents covers.
#[inline(always)]
fn dot_half(w: &[i8], a: &[i8]) -> i32 {
    super::dots::dot_w16(w, a)
}

/// Rows of IQ2_XS against nv vectors, portably (ggml's ggml_vec_dot_iq2_xs_q8_K): two exponents a 32-weight group,
/// the first for its first two sub-blocks and the second for its last two. Scaled by 1/8 as IQ2_XXS is.
pub fn iq2xs_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * IQ2_XS_BLOCK && w.len() >= rows * row_bytes);
    // One vector, which is what a decoded token is: the grid vectors are read out and dotted in the same pass, so
    // nothing is materialised for a block that only one vector uses.
    if nv == 1 {
        for r in 0..rows {
            let wr = &w[r * row_bytes..];
            let mut acc = 0f32;
            for b in 0..nb {
                let (blk, xa) = (&wr[b * IQ2_XS_BLOCK..(b + 1) * IQ2_XS_BLOCK], &x[b]);
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) * xa.d;
                let (mut bsum, mut q8) = (0i32, 0usize);
                for ib in 0..8 {
                    let (ls1, ls2) = ((2 * (blk[66 + ib] & 0xf) + 1) as i32, (2 * (blk[66 + ib] >> 4) + 1) as i32);
                    let mut sumi = 0i32;
                    for l in 0..4 {
                        let at = 2 + 2 * (4 * ib + l);
                        let q2 = u16::from_le_bytes([blk[at], blk[at + 1]]) as u32;
                        let grid = IQ2XS_GRID[(q2 & 511) as usize];
                        let signs = KSIGNS_IQ2XS[(q2 >> 9) as usize];
                        for j in 0..8 {
                            let g = ((grid >> (8 * j)) & 0xff) as i32;
                            sumi += g * xa.qs[q8 + j] as i32 * if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                        }
                        q8 += 8;
                        if l == 1 {
                            bsum += sumi * ls1;
                            sumi = 0;
                        }
                    }
                    bsum += sumi * ls2;
                }
                acc = d.mul_add(bsum as f32, acc);
            }
            y[r] = acc * 0.125;
        }
        return;
    }
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v0 in (0..nv).step_by(VECTORS) {
            let k = VECTORS.min(nv - v0);
            let mut acc = [0f32; VECTORS];
            for b in 0..nb {
                let blk = &wr[b * IQ2_XS_BLOCK..(b + 1) * IQ2_XS_BLOCK];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                // Signed grid vectors, and the two exponents of each group of 32, for the whole chunk at once.
                let (mut qv, mut ls) = ([0i8; 256], [0i32; 16]);
                for ib in 0..8 {
                    ls[2 * ib] = (2 * (blk[66 + ib] & 0xf) + 1) as i32;
                    ls[2 * ib + 1] = (2 * (blk[66 + ib] >> 4) + 1) as i32;
                    for l in 0..4 {
                        let at = 2 + 2 * (4 * ib + l);
                        let q2 = u16::from_le_bytes([blk[at], blk[at + 1]]) as u32;
                        let grid = IQ2XS_GRID[(q2 & 511) as usize];
                        let signs = KSIGNS_IQ2XS[(q2 >> 9) as usize];
                        for j in 0..8 {
                            let g = ((grid >> (8 * j)) & 0xff) as i32;
                            qv[32 * ib + 8 * l + j] = (g * (1 - 2 * ((signs >> j) & 1) as i32)) as i8;
                        }
                    }
                }
                for j in 0..k {
                    let xa = &x[(v0 + j) * nb + b];
                    let mut bsum = 0i32;
                    for h in 0..16 {
                        bsum += ls[h] * dot_half(&qv[16 * h..], &xa.qs[16 * h..]);
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
