//! mxfp4.rs - MXFP4 (OCP microscaling FP4, as ggml stores it): 32 weights a block, one E8M0 scale byte (a power of two)
//! and 16 bytes of 4-bit E2M1 values, low nibbles for weights 0-15 and high ones for 16-31, as IQ4_NL lays them out.
//! gpt-oss ships its experts this way.
//!
//! ggml keeps the E2M1 values as small integers, twice their size (0, 1, 2, 3, 4, 6, 8, 12 and their negatives), and
//! halves the scale to match, so a block multiplies against Q8 activations exactly as IQ4_NL's does: integer dot
//! products of the looked-up levels, then one scale per block.

/// A block's size in bytes: the scale, then 16 bytes of nibbles.
pub const MXFP4_BLOCK: usize = 17;

/// The 16 E2M1 values, doubled to integers (ggml's kvalues_mxfp4).
pub const MXFP4_LEVELS: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// An E8M0 scale halved (2^(e - 128)), as an f32 built from its bits: exact for every e, the smallest two as
/// subnormals (ggml's GGML_E8M0_TO_FP32_HALF).
#[inline]
pub fn e8m0_half(e: u8) -> f32 {
    let bits = if e < 2 { 0x0020_0000u32 << e } else { ((e as u32) - 1) << 23 };
    f32::from_bits(bits)
}

/// Rows of MXFP4 against nv vectors, portably (the CPUs without a vector kernel for it): for each 32-weight block,
/// the integer dot of its levels with the activations' group, times the block's halved scale and the group's scale,
/// summed over the blocks in order. y[v * y_stride + r] for v < nv and r < rows.
pub fn mxfp4_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[crate::Groups32],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 8 * MXFP4_BLOCK && w.len() >= rows * row_bytes);
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v in 0..nv {
            let mut acc = 0f32;
            for b in 0..nb {
                let g = &x[v * nb + b];
                for k in 0..8 {
                    let blk = &wr[(b * 8 + k) * MXFP4_BLOCK..(b * 8 + k + 1) * MXFP4_BLOCK];
                    let q = &g.qs[32 * k..32 * k + 32];
                    let mut s = 0i32;
                    for j in 0..16 {
                        s += MXFP4_LEVELS[(blk[1 + j] & 15) as usize] as i32 * q[j] as i32;
                        s += MXFP4_LEVELS[(blk[1 + j] >> 4) as usize] as i32 * q[j + 16] as i32;
                    }
                    acc = (e8m0_half(blk[0]) * g.d[k]).mul_add(s as f32, acc);
                }
            }
            y[v * y_stride + r] = acc;
        }
    }
}
