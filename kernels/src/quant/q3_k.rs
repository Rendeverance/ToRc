//! q3_k.rs - Q3_K (ggml): 256 weights per block, in 16 groups with 6-bit scales.

/// Decodes a Q3_K block's sixteen 6-bit signed scales (-32..31) from their 12 packed bytes, in ggml's layout.
pub fn q3k_unpack_scales(q: &[u8; 12]) -> [i8; 16] {
    let w = |i: usize| u32::from_le_bytes([q[i], q[i + 1], q[i + 2], q[i + 3]]);
    let (a0, a1, a2) = (w(0), w(4), w(8));
    let (low4, top2) = (0x0f0f0f0fu32, 0x03030303u32);
    let o = [
        (a0 & low4) | ((a2 & top2) << 4),
        (a1 & low4) | (((a2 >> 2) & top2) << 4),
        ((a0 >> 4) & low4) | (((a2 >> 4) & top2) << 4),
        ((a1 >> 4) & low4) | (((a2 >> 6) & top2) << 4),
    ];
    std::array::from_fn(|i| (o[i / 4].to_le_bytes()[i % 4] as i8).wrapping_sub(32))
}

/// Q3_K's block: 32 bytes of high bits (hmask), 64 of two-bit values (qs), 12 of scales, and fp16 d.
pub const Q3_K_BLOCK: usize = 110;
