//! q4_k.rs - Q4_K and Q5_K (ggml): 256 weights per block, in 8 groups with 6-bit scales and minimums.

/// The 12 scale bytes of a Q4_K / Q5_K block as 8 scales and 8 mins (ggml's get_scale_min_k4, four at a time).
#[inline(always)]
pub fn q45k_scales(q: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let w = |i: usize| u32::from_le_bytes([q[i], q[i + 1], q[i + 2], q[i + 3]]);
    let (a, b, c) = (w(0), w(4), w(8));
    let (low6, low4, top2) = (0x3f3f3f3fu32, 0x0f0f0f0fu32, 0x03030303u32);
    let s0 = a & low6;
    let s1 = (c & low4) | (((a >> 6) & top2) << 4);
    let m0 = b & low6;
    let m1 = ((c >> 4) & low4) | (((b >> 6) & top2) << 4);
    let cat = |x: u32, y: u32| {
        let (x, y) = (x.to_le_bytes(), y.to_le_bytes());
        [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
    };
    (cat(s0, s1), cat(m0, m1))
}
