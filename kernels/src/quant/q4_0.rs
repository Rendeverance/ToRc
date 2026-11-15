//! q4_0.rs - Q4_0 (ggml): an fp16 scale and 32 nibbles for each block of 32.

use super::{Groups32, f16_to_f32};

/// GGUF Q4_0: an fp16 scale and 32 nibbles (weights 0-15 in the low nibbles, 16-31 in the high), with the weight d *
/// (q - 8).
pub const Q4_0_BLOCK: usize = 18;

pub const GGML_Q4_0: u32 = 2;

/// Eight lanes added pairwise: ((0+4) + (2+6)) + ((1+5) + (3+7)).
#[inline]
fn sum8(a: &[f32; 8]) -> f32 {
    let s = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
    (s[0] + s[2]) + (s[1] + s[3])
}

/// One row of nb * 256 Q4_0 weights (nb * 8 blocks) against nb Q8_K blocks. This is the portable reference every
/// platform kernel matches, with one lane per block position k: acc[k] = fma(d_w * g.d[k], dot_k, acc[k]).
pub fn q4_0_dot_scalar(w: &[u8], x: &[Groups32]) -> f32 {
    let nb = x.len();
    assert!(w.len() >= nb * 8 * Q4_0_BLOCK);
    let mut acc = [0.0f32; 8];
    for (b, xb) in x.iter().enumerate() {
        for k in 0..8 {
            let blk = &w[(b * 8 + k) * Q4_0_BLOCK..(b * 8 + k + 1) * Q4_0_BLOCK];
            let a = &xb.qs[32 * k..32 * k + 32];
            let mut dot = 0i32;
            for l in 0..16 {
                let q = blk[2 + l];
                dot += (q & 15) as i32 * a[l] as i32 + (q >> 4) as i32 * a[16 + l] as i32;
            }
            // the nibbles less 8: 8 x the block's activation sum, as an i16 pair sum widened (exact)
            let bias = (xb.sums[k] as i32) << 3;
            dot -= bias;
            let sc = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) * xb.d[k];
            acc[k] = sc.mul_add(dot as f32, acc[k]);
        }
    }
    sum8(&acc)
}
