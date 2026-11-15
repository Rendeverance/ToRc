//! iq4.rs - IQ4_NL and IQ4_XS (ggml): 16 non-uniform levels, with IQ4_XS adding 6-bit group scales.

/// The 8 scales of an IQ4_XS block (136 bytes: fp16 d, a u16 scales_h, 4 bytes of scales_l and 128 of nibbles), in
/// -32..31: group g's low four bits come from scales_l[g / 2] and its top two from scales_h.
pub fn iq4xs_scales(blk: &[u8]) -> [i32; 8] {
    let sh = u16::from_le_bytes([blk[2], blk[3]]) as i32;
    std::array::from_fn(|g| (((blk[4 + g / 2] >> (4 * (g % 2))) & 15) as i32 | (((sh >> (2 * g)) & 3) << 4)) - 32)
}

/// IQ4_XS's block size in bytes (256 numbers).
pub const IQ4_XS_BLOCK: usize = 136;

/// IQ4_NL and IQ4_XS's 16 levels (ggml's kvalues_iq4nl).
pub const IQ4_LEVELS: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
