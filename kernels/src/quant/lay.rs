//! lay.rs - the experts' K-quant rows rearranged 4 at a time for the ARM SMMLA / SDOT kernels (neon.rs gemm_q45k_lay,
//! q45k_lay_matmul, gemm_q6_lay, q6_lay_matmul). This is plain Rust, so a converter on any machine can write laid-out
//! files. The integers and scales are the source's, only rearranged.

const Q4K: usize = 144;
const Q5K: usize = 176;
const Q6K: usize = 210;

/// The bytes of one block of 4 Q4_K (five = false) or Q5_K rows laid out by q45k_lay4.
pub const fn q45k_lay_bytes(five: bool) -> usize {
    if five { 16 + 64 + 512 + 128 } else { 16 + 64 + 512 }
}

/// Weight n (0..32) of sub-block k of a Q4_K (FIVE = false) or Q5_K block: 0..15 or 0..31.
fn q45k_weight<const FIVE: bool>(blk: &[u8], k: usize, n: usize) -> u8 {
    let qs = if FIVE { 48 } else { 16 };
    let byte = blk[qs + 32 * (k / 2) + n];
    let nib = if k.is_multiple_of(2) { byte & 15 } else { byte >> 4 };
    if FIVE { nib | (((blk[16 + n] >> k) & 1) << 4) } else { nib }
}

/// Lays out 4 Q4_K (five = false) or Q5_K rows (nb blocks each, row_bytes apart) for gemm_q45k_lay, block after block
/// (q45k_lay_bytes each, about 3% more than the source). Each block holds the 4 fp16 d values and the 4 fp16 dmin
/// values; then, for each sub-block k, the 4 rows' 6-bit scales as bytes [s0 s1 s2 s3], and the mins the same way;
/// then, for each sub-block k and tile t (rows 2t and 2t + 1), two registers whose low and high nibbles are SMMLA's A
/// operands as they stand: register 1's low nibbles are [row i w0-7, row j w0-7] and its high ones [w8-15, w8-15], and
/// register 2 is the same for w16-23 and w24-31 (w being the sub-block's 32 weights). For Q5_K there is then one
/// register of fifth bits per k: bit n of byte m is that of tile 0's operand n, byte m, and bit 4 + n is tile 1's. The
/// integers and scales are the source's, only rearranged, so products over it have the source's bits. out holds nb *
/// q45k_lay_bytes(five).
pub fn q45k_lay4(five: bool, src: &[u8], row_bytes: usize, nb: usize, out: &mut [u8]) {
    if five { q45k_lay4_t::<true>(src, row_bytes, nb, out) } else { q45k_lay4_t::<false>(src, row_bytes, nb, out) }
}

fn q45k_lay4_t<const FIVE: bool>(src: &[u8], row_bytes: usize, nb: usize, out: &mut [u8]) {
    let (bs, lb) = (if FIVE { Q5K } else { Q4K }, q45k_lay_bytes(FIVE));
    assert!(src.len() >= 3 * row_bytes + nb * bs && out.len() >= nb * lb);
    for b in 0..nb {
        let blk = |i: usize| &src[i * row_bytes + b * bs..i * row_bytes + (b + 1) * bs];
        let o = &mut out[b * lb..(b + 1) * lb];
        let scales: [([u8; 8], [u8; 8]); 4] =
            core::array::from_fn(|i| crate::q45k_scales(blk(i)[4..16].try_into().expect("12 scale bytes")));
        for i in 0..4 {
            o[2 * i..2 * i + 2].copy_from_slice(&blk(i)[..2]);
            o[8 + 2 * i..8 + 2 * i + 2].copy_from_slice(&blk(i)[2..4]);
        }
        if FIVE {
            // the fifth bits are ORed in, two tiles per byte
            o[592..720].fill(0);
        }
        for k in 0..8 {
            for i in 0..4 {
                o[16 + 4 * k + i] = scales[i].0[k];
                o[48 + 4 * k + i] = scales[i].1[k];
            }
            for t in 0..2 {
                let (ri, rj) = (blk(2 * t), blk(2 * t + 1));
                let w = |r: &[u8], n: usize| q45k_weight::<FIVE>(r, k, n);
                let base = 80 + (2 * k + t) * 32;
                for m in 0..8 {
                    o[base + m] = (w(ri, m) & 15) | ((w(ri, m + 8) & 15) << 4);
                    o[base + 8 + m] = (w(rj, m) & 15) | ((w(rj, m + 8) & 15) << 4);
                    o[base + 16 + m] = (w(ri, 16 + m) & 15) | ((w(ri, 24 + m) & 15) << 4);
                    o[base + 24 + m] = (w(rj, 16 + m) & 15) | ((w(rj, 24 + m) & 15) << 4);
                }
                if FIVE {
                    let hb = 592 + 16 * k;
                    for m in 0..16 {
                        let r = if m < 8 { ri } else { rj };
                        for n in 0..4 {
                            o[hb + m] |= (w(r, 8 * n + m % 8) >> 4) << (n + 4 * t);
                        }
                    }
                }
            }
        }
    }
}

/// The bytes of one block of 4 Q6_K rows laid out by q6_lay4: the source's 4 x 210, with nothing added.
pub const Q6_LAY_BYTES: usize = 8 + 64 + 512 + 256;

/// Weight n (0..256) of a Q6_K block as stored, 0..63 (q + 32), in q6k_unpack's order.
fn q6_weight(blk: &[u8], n: usize) -> u8 {
    let (c, l2) = (n / 128, n % 128);
    let (qd, l) = (l2 / 32, l2 % 32);
    let b = blk[64 * c + l + 32 * (qd & 1)];
    let lo = if qd < 2 { b & 15 } else { b >> 4 };
    lo | (((blk[128 + 32 * c + l] >> (2 * qd)) & 3) << 4)
}

/// Lays out 4 Q6_K rows (nb blocks each, row_bytes apart) for gemm_q6_lay / q6_lay_matmul, block after block
/// (Q6_LAY_BYTES each, the source's size). Each block holds the 4 fp16 d values; for each group g of 16 weights, the 4
/// rows' int8 scales [s0 s1 s2 s3]; for each group g and tile t (rows 2t and 2t + 1), one register whose low and high
/// nibbles are the low 4 bits of SMMLA's two A operands ([row i w0-7, row j w0-7] and [w8-15, w8-15] of the group); and
/// for each group one register of the high 2 bits, where byte m holds tile 0's operand 0 in bits 0-1, its operand 1 in
/// bits 2-3, and tile 1's in bits 4-5 and 6-7. The integers and scales are the source's, only rearranged, so products
/// over it have the source's bits. out holds nb * Q6_LAY_BYTES.
pub fn q6_lay4(src: &[u8], row_bytes: usize, nb: usize, out: &mut [u8]) {
    assert!(src.len() >= 3 * row_bytes + nb * Q6K && out.len() >= nb * Q6_LAY_BYTES);
    for b in 0..nb {
        let blk = |i: usize| &src[i * row_bytes + b * Q6K..i * row_bytes + (b + 1) * Q6K];
        let o = &mut out[b * Q6_LAY_BYTES..(b + 1) * Q6_LAY_BYTES];
        o[584..].fill(0);
        for i in 0..4 {
            o[2 * i..2 * i + 2].copy_from_slice(&blk(i)[208..210]);
        }
        for g in 0..16 {
            for i in 0..4 {
                o[8 + 4 * g + i] = blk(i)[192 + g];
            }
            for t in 0..2 {
                let (ri, rj) = (blk(2 * t), blk(2 * t + 1));
                for m in 0..16 {
                    let r = if m < 8 { ri } else { rj };
                    let (w0, w1) = (q6_weight(r, 16 * g + m % 8), q6_weight(r, 16 * g + 8 + m % 8));
                    o[72 + (2 * g + t) * 16 + m] = (w0 & 15) | ((w1 & 15) << 4);
                    o[584 + 16 * g + m] |= ((w0 >> 4) << (4 * t)) | ((w1 >> 4) << (4 * t + 2));
                }
            }
        }
    }
}

const Q80: usize = 34;

/// The bytes of one block position (32 weights) of 4 Q8_0 rows laid out by q8_0_lay4: the source's 4 x 34.
pub const Q8_LAY_GROUP: usize = 8 + 128;

/// Lays out 4 Q8_0 rows (nb * 8 blocks each, row_bytes apart) for neon::gemm_x4 / x4_matmul (Q8x4), block position
/// after block position (Q8_LAY_GROUP each, the source's size). Each holds the 4 fp16 scales, then, for each tile t
/// (rows 2t and 2t + 1), four registers that are SMMLA's A operands as they stand: [row i w0-7, row j w0-7], [w8-15,
/// w8-15], [w16-23, w16-23] and [w24-31, w24-31]. The integers and scales are the source's, only rearranged, so
/// products over it have the source's bits. out holds nb * 8 * Q8_LAY_GROUP.
pub fn q8_0_lay4(src: &[u8], row_bytes: usize, nb: usize, out: &mut [u8]) {
    assert!(src.len() >= 3 * row_bytes + nb * 8 * Q80 && out.len() >= nb * 8 * Q8_LAY_GROUP);
    for bk in 0..nb * 8 {
        let blk = |i: usize| &src[i * row_bytes + bk * Q80..i * row_bytes + (bk + 1) * Q80];
        let o = &mut out[bk * Q8_LAY_GROUP..(bk + 1) * Q8_LAY_GROUP];
        for i in 0..4 {
            o[2 * i..2 * i + 2].copy_from_slice(&blk(i)[..2]);
        }
        for t in 0..2 {
            for m in 0..4 {
                let base = 8 + 64 * t + 16 * m;
                o[base..base + 8].copy_from_slice(&blk(2 * t)[2 + 8 * m..10 + 8 * m]);
                o[base + 8..base + 16].copy_from_slice(&blk(2 * t + 1)[2 + 8 * m..10 + 8 * m]);
            }
        }
    }
}
