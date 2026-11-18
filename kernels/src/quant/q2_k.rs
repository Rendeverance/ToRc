//! q2_k.rs - Q2_K (ggml): 256 weights a block, two bits each, with a 4-bit scale and a 4-bit minimum shared by
//! each group of 16 and two fp16 block scales. The two-bit values are unsigned (0..3), so the block's minimum is
//! taken off once at the end against the activations' own group sums - which is why the dot has a second term.

use crate::{BlockQ8K, f16_to_f32};

/// Q2_K's block: 16 scale bytes, 64 bytes of two-bit values, and the fp16 d and dmin.
pub const Q2_K_BLOCK: usize = 84;

/// Rows of Q2_K against nv vectors, portably (ggml's ggml_vec_dot_q2_K_q8_K): per group of 16, the group's scale
/// times the integer dot of its values with the activations, summed; minus the block minimum times the activations'
/// sums over the same groups. y[v * y_stride + r] for v < nv and r < rows.
///
/// Group g's sixteen two-bit values sit in the g / 8th 32-byte half of qs, at byte g % 2 * 16 of it, shifted
/// (g % 8) / 2 * 2 bits - four shifts a half, which is how ggml packs a block without wasting bits.
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
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v in 0..nv {
            let mut acc = 0f32;
            for b in 0..nb {
                let blk = &wr[b * Q2_K_BLOCK..(b + 1) * Q2_K_BLOCK];
                let xa = &x[v * nb + b];
                let (sc, q2) = (&blk[0..16], &blk[16..80]);
                let (mut sumi, mut summs) = (0i32, 0i32);
                for g in 0..16 {
                    let (chunk, half, shift) = (32 * (g / 8), 16 * (g % 2), 2 * ((g % 8) / 2));
                    let s = (sc[g] & 0xF) as i32;
                    let mut isuml = 0i32;
                    for l in 0..16 {
                        isuml += xa.qs[16 * g + l] as i32 * (((q2[chunk + half + l] >> shift) & 3) as i32);
                    }
                    sumi += s * isuml;
                    summs += xa.bsums[g] as i32 * (sc[g] >> 4) as i32;
                }
                let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
                let dm = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
                acc = (xa.d * d).mul_add(sumi as f32, acc);
                acc -= (xa.d * dm) * summs as f32;
            }
            y[v * y_stride + r] = acc;
        }
    }
}
