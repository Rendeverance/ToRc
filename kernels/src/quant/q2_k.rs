//! q2_k.rs - Q2_K (ggml): 256 weights a block, two bits each, with a 4-bit scale and a 4-bit minimum shared by
//! each group of 16 and two fp16 block scales. The two-bit values are unsigned (0..3), so the block's minimum is
//! taken off once at the end against the activations' own group sums - which is why the dot has a second term.

use crate::{BlockQ8K, f16_to_f32};

/// Q2_K's block: 16 scale bytes, 64 bytes of two-bit values, and the fp16 d and dmin.
pub const Q2_K_BLOCK: usize = 84;

/// How many of the nv vectors are carried at a time. The block's values come out of their packed form once and are
/// then used for all of them, which is the point of the loop order below.
const VECTORS: usize = 16;

/// The sixteen values of one group against the same sixteen activations.
#[inline(always)]
fn dot_group(w: &[i8], a: &[i8]) -> i32 {
    super::dots::dot_w16u(w, a)
}

/// Rows of Q2_K against nv vectors, portably (ggml's ggml_vec_dot_q2_K_q8_K): per group of 16, the group's scale
/// times the integer dot of its values with the activations, summed; minus the block minimum times the activations'
/// sums over the same groups. y[v * y_stride + r] for v < nv and r < rows.
///
/// Group g's sixteen two-bit values sit in the g / 8th 32-byte half of qs, at byte g % 2 * 16 of it, shifted
/// (g % 8) / 2 * 2 bits - four shifts a half, which is how ggml packs a block without wasting bits.
///
/// The vectors are taken in chunks so that unpacking a block is paid once per chunk rather than once per vector:
/// over a prompt that is the difference between reading a block a thousand times and reading it sixty.
pub fn q2k_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * Q2_K_BLOCK && w.len() >= rows * row_bytes);
    // One vector, which is what a decoded token is: reading the block and dotting it in one pass needs nothing kept
    // aside, and the chunk below would pay for materialising a block a single vector uses once.
    if nv == 1 {
        for r in 0..rows {
            let wr = &w[r * row_bytes..];
            let mut acc = 0f32;
            for b in 0..nb {
                let (blk, xa) = (&wr[b * Q2_K_BLOCK..(b + 1) * Q2_K_BLOCK], &x[b]);
                let (sc, q2) = (&blk[0..16], &blk[16..80]);
                let (mut sumi, mut summs) = (0i32, 0i32);
                for g in 0..16 {
                    let (chunk, half, shift) = (32 * (g / 8), 16 * (g % 2), 2 * ((g % 8) / 2));
                    let mut isuml = 0i32;
                    for l in 0..16 {
                        isuml += xa.qs[16 * g + l] as i32 * (((q2[chunk + half + l] >> shift) & 3) as i32);
                    }
                    sumi += (sc[g] & 0xF) as i32 * isuml;
                    summs += xa.bsums[g] as i32 * (sc[g] >> 4) as i32;
                }
                let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
                let dm = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
                acc = (xa.d * d).mul_add(sumi as f32, acc);
                acc -= (xa.d * dm) * summs as f32;
            }
            y[r] = acc;
        }
        return;
    }
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v0 in (0..nv).step_by(VECTORS) {
            let k = VECTORS.min(nv - v0);
            let mut acc = [0f32; VECTORS];
            for b in 0..nb {
                let blk = &wr[b * Q2_K_BLOCK..(b + 1) * Q2_K_BLOCK];
                let (sc, q2) = (&blk[0..16], &blk[16..80]);
                // The block's values in the order the activations are in, once for every vector in this chunk.
                let mut qv = [0i8; 256];
                for g in 0..16 {
                    let (chunk, half, shift) = (32 * (g / 8), 16 * (g % 2), 2 * ((g % 8) / 2));
                    let (src, dst) = (&q2[chunk + half..], &mut qv[16 * g..]);
                    for l in 0..16 {
                        dst[l] = ((src[l] >> shift) & 3) as i8;
                    }
                }
                let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
                let dm = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
                for j in 0..k {
                    let xa = &x[(v0 + j) * nb + b];
                    let (mut sumi, mut summs) = (0i32, 0i32);
                    for g in 0..16 {
                        sumi += (sc[g] & 0xF) as i32 * dot_group(&qv[16 * g..], &xa.qs[16 * g..]);
                        summs += xa.bsums[g] as i32 * (sc[g] >> 4) as i32;
                    }
                    acc[j] = (xa.d * d).mul_add(sumi as f32, acc[j]);
                    acc[j] -= (xa.d * dm) * summs as f32;
                }
            }
            for j in 0..k {
                y[(v0 + j) * y_stride + r] = acc[j];
            }
        }
    }
}
