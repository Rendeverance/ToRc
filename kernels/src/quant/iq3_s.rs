//! iq3_s.rs - IQ3_S (ggml): grid-indexed 3-bit weights with signs and 4-bit group scales.

use super::iq3s_grid;

/// IQ3_S's block: fp16 d, 64 bytes of index low bits (qs), 8 of index ninth bits (qh), 32 of signs and 4 of scales.
pub const IQ3_S_BLOCK: usize = 110;

/// The integer scale of group g in an IQ3_S block, from its 4-bit field s: 2s + 1 (ggml's definition).
#[inline(always)]
pub fn iq3s_group_scale(blk: &[u8], g: usize) -> i32 {
    1 + 2 * ((blk[106 + g / 2] >> (4 * (g % 2))) & 15) as i32
}

/// Group g's 32 unsigned levels in an IQ3_S block (eight grid entries, with index = qs | the qh bit << 8), and its 32
/// sign bits (bit e set means weight e is negative).
#[inline(always)]
pub fn iq3s_group(blk: &[u8], g: usize) -> ([u8; 32], u32) {
    let qh = blk[66 + g];
    let mut l = [0u8; 32];
    for k in 0..8 {
        let idx = blk[2 + 8 * g + k] as usize | (((qh >> k) & 1) as usize) << 8;
        l[4 * k..4 * k + 4].copy_from_slice(&iq3s_grid::IQ3S_GRID[idx].to_le_bytes());
    }
    let s = u32::from_le_bytes([blk[74 + 4 * g], blk[75 + 4 * g], blk[76 + 4 * g], blk[77 + 4 * g]]);
    (l, s)
}
