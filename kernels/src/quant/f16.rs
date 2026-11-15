//! f16.rs - IEEE half precision.

/// fp16 to fp32, exactly (subnormals, infinities and NaN included).
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match (e, m) {
        (0, 0) => s,
        (0, _) => {
            // subnormal: normalise it
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, 0) => s | 0x7f80_0000,
        (31, _) => s | 0x7fc0_0000 | (m << 13),
        _ => s | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}
