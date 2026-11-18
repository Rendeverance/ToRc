//! iq1_m.rs - IQ1_M (ggml): 1.75 bpw, and the reason a "1-bit" quant is still useful. A block is a 2048-entry
//! table of eight signed values, eight bits of index per sub-block with three more index bits and a sign carried in
//! a separate byte, and a scale assembled from four 16-bit words. The weights are not ±1: they are table vectors
//! offset by 1/8, and that offset is what the extra term in the dot below is for.

use crate::quant::iq1s_grid::IQ1S_GRID;
use crate::{BlockQ8K, f16_to_f32};

/// IQ1_M's block: 32 grid indices, 16 bytes holding three more index bits and a shift bit per sub-block, and eight
/// scale bytes. The block's scale is not stored as an fp16 - four of these words carry it.
pub const IQ1_M_BLOCK: usize = 56;

/// The offset every IQ1_S vector carries, which the dot has to account for separately (ggml's IQ1M_DELTA).
const IQ1M_DELTA: f32 = 0.125;

/// Rows of IQ1_M against nv vectors, portably (ggml's ggml_vec_dot_iq1_m_q8_K): the grid values dot the activations
/// directly, and because each of them sits 1/8 off zero, the same group's activations summed - signed by the shift
/// bit - give the second term.
pub fn iq1m_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * IQ1_M_BLOCK && w.len() >= rows * row_bytes);
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v in 0..nv {
            let mut acc = 0f32;
            for b in 0..nb {
                let blk = &wr[b * IQ1_M_BLOCK..(b + 1) * IQ1_M_BLOCK];
                let xa = &x[v * nb + b];
                let sc: [u16; 4] = std::array::from_fn(|i| u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]));
                let scale =
                    f16_to_f32((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
                let (qs, qh) = (&blk[0..32], &blk[32..48]);
                let (mut sumi1, mut sumi2, mut q8) = (0i32, 0i32, 0usize);
                for ib in 0..8 {
                    let delta = [
                        if qh[2 * ib] & 0x08 != 0 { -1 } else { 1 },
                        if qh[2 * ib] & 0x80 != 0 { -1 } else { 1 },
                        if qh[2 * ib + 1] & 0x08 != 0 { -1 } else { 1 },
                        if qh[2 * ib + 1] & 0x80 != 0 { -1 } else { 1 },
                    ];
                    let (mut sum1, mut sum2) = ([0i32; 2], [0i32; 2]);
                    for l in 0..4 {
                        let idx = qs[4 * ib + l] as usize
                            | (((qh[2 * ib + l / 2] as u32) << (8 - 4 * (l % 2))) & 0x700) as usize;
                        let grid = IQ1S_GRID[idx];
                        let (mut lsum1, mut lsum2) = (0i32, 0i32);
                        for j in 0..8 {
                            let g = (((grid >> (8 * j)) & 0xff) as u8) as i8 as i32;
                            lsum1 += xa.qs[q8 + j] as i32 * g;
                            lsum2 += xa.qs[q8 + j] as i32;
                        }
                        q8 += 8;
                        sum1[l / 2] += lsum1;
                        sum2[l / 2] += lsum2 * delta[l];
                    }
                    let (ls1, ls2) = (
                        (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1) as i32,
                        (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1) as i32,
                    );
                    sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
                    sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
                }
                acc = (scale * xa.d).mul_add(sumi1 as f32 + IQ1M_DELTA * (sumi2 as f32), acc);
            }
            y[v * y_stride + r] = acc;
        }
    }
}
