//! iq2_s.rs - IQ2_S (ggml): 2.56 bpw. No grid of signs: the signs of an eight-weight sub-block are their own bits,
//! and the index into a 1024-entry grid of magnitudes is split between a byte of the block and two high bits in a
//! separate byte, so a whole group of four sub-blocks stays byte aligned.

use crate::quant::iq2s_grid::IQ2S_GRID;
use crate::quant::ksigns::KMASK_IQ2XS;
use crate::{BlockQ8K, f16_to_f32};

/// IQ2_S's block: fp16 d, 32 grid indices followed by their 32 sign bytes, 8 bytes holding the two top index bits
/// of each group, and 8 scale bytes.
pub const IQ2_S_BLOCK: usize = 82;

/// How many of the nv vectors are carried at a time: the grid lookups and the sign bits of a block are read once
/// and applied to all of them.
const VECTORS: usize = 16;

/// Sixteen values against the same sixteen activations - the width one of the group's two exponents covers.
#[inline(always)]
fn dot_half(w: &[i8], a: &[i8]) -> i32 {
    let mut s = 0i32;
    for j in 0..16 {
        s += w[j] as i32 * a[j] as i32;
    }
    s
}

/// Rows of IQ2_S against nv vectors, portably (ggml's ggml_vec_dot_iq2_s_q8_K): two exponents a 32-weight group as
/// IQ2_XS has, but the signs come from the block itself, one bit per weight, so no sign table is read.
pub fn iq2s_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * IQ2_S_BLOCK && w.len() >= rows * row_bytes);
    // One vector, which is what a decoded token is: the grid vectors are read out and dotted in the same pass, so
    // nothing is materialised for a block that only one vector uses.
    if nv == 1 {
        for r in 0..rows {
            let wr = &w[r * row_bytes..];
            let mut acc = 0f32;
            for b in 0..nb {
                let (blk, xa) = (&wr[b * IQ2_S_BLOCK..(b + 1) * IQ2_S_BLOCK], &x[b]);
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) * xa.d;
                let (qs, qh, sc, signs) = (&blk[2..66], &blk[66..74], &blk[74..82], &blk[34..66]);
                let (mut bsum, mut q8) = (0i32, 0usize);
                for ib in 0..8 {
                    let (ls1, ls2) = ((1 + 2 * (sc[ib] & 0xf)) as i32, (1 + 2 * (sc[ib] >> 4)) as i32);
                    let (mut sumi1, mut sumi2) = (0i32, 0i32);
                    for l in 0..4 {
                        let idx = (qs[4 * ib + l] as u32 | ((qh[ib] as u32) << (8 - 2 * l)) & 0x300) as usize;
                        let grid = IQ2S_GRID[idx];
                        for j in 0..8 {
                            let g = ((grid >> (8 * j)) & 0xff) as i32;
                            let s = if signs[4 * ib + l] & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                            if l < 2 {
                                sumi1 += g * xa.qs[q8 + j] as i32 * s;
                            } else {
                                sumi2 += g * xa.qs[q8 + j] as i32 * s;
                            }
                        }
                        q8 += 8;
                    }
                    bsum += ls1 * sumi1 + ls2 * sumi2;
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
                let blk = &wr[b * IQ2_S_BLOCK..(b + 1) * IQ2_S_BLOCK];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let (qs, qh, sc, signs) = (&blk[2..66], &blk[66..74], &blk[74..82], &blk[34..66]);
                // Signed grid vectors, and the two exponents of each group of 32, for the whole chunk at once.
                let (mut qv, mut ls) = ([0i8; 256], [0i32; 16]);
                for ib in 0..8 {
                    ls[2 * ib] = (1 + 2 * (sc[ib] & 0xf)) as i32;
                    ls[2 * ib + 1] = (1 + 2 * (sc[ib] >> 4)) as i32;
                    for l in 0..4 {
                        let idx = (qs[4 * ib + l] as u32 | ((qh[ib] as u32) << (8 - 2 * l)) & 0x300) as usize;
                        let grid = IQ2S_GRID[idx];
                        for j in 0..8 {
                            let g = ((grid >> (8 * j)) & 0xff) as i32;
                            let s = (signs[4 * ib + l] >> j) & 1;
                            qv[32 * ib + 8 * l + j] = (g * (1 - 2 * s as i32)) as i8;
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
