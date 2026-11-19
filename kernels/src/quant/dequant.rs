//! dequant.rs - weights back to f32 (SPEC.md section 4).
//!
//! ggml's block formats as ggml defines them (ggml-quants.c, MIT), with one multiply per number in the order written
//! there. The embeddings' rows and the requantised heads go through here.

use crate::f16_to_f32;
use crate::quant::iq1s_grid::IQ1S_GRID;
use crate::quant::iq2s_grid::IQ2S_GRID;
use crate::quant::iq2xs_grid::IQ2XS_GRID;
use crate::quant::iq2xxs_grid::IQ2XXS_GRID;
use crate::quant::iq3xxs_grid::IQ3XXS_GRID;
use crate::quant::ksigns::{KMASK_IQ2XS, KSIGNS_IQ2XS};

/// IQ4_NL's sixteen levels (ggml-common.h).
const IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

fn h(b: &[u8], o: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

/// The scale and minimum of sub-block j, from a Q4_K / Q5_K block's 12 scale bytes (six bits each).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// The bytes in a block and the numbers it holds, for the types here.
pub fn block_geometry(ty: u32) -> Option<(usize, usize)> {
    Some(match ty {
        0 => (4, 1),
        1 => (2, 1),
        30 => (2, 1),
        2 => (18, 32),
        42 => (18, 64),
        3 => (20, 32),
        6 => (22, 32),
        7 => (24, 32),
        8 => (34, 32),
        12 => (144, 256),
        13 => (176, 256),
        14 => (210, 256),
        10 => (84, 256),
        16 => (66, 256),
        17 => (74, 256),
        18 => (98, 256),
        22 => (82, 256),
        29 => (56, 256),
        11 => (110, 256),
        23 => (136, 256),
        21 => (110, 256),
        20 => (18, 32),
        39 => (17, 32),
        _ => return None,
    })
}

/// y.len() numbers of type `ty` from w; returns false where the type is not one of these (the caller then has no
/// values).
pub fn dequantize(ty: u32, w: &[u8], y: &mut [f32]) -> bool {
    let Some((bs, qk)) = block_geometry(ty) else { return false };
    assert!(y.len().is_multiple_of(qk) && w.len() >= y.len() / qk * bs);
    for (b, y) in w.chunks_exact(bs).zip(y.chunks_exact_mut(qk)) {
        match ty {
            0 => y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            1 => y[0] = h(b, 0),
            30 => y[0] = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
            2 => {
                let d = h(b, 0);
                for j in 0..16 {
                    y[j] = ((b[2 + j] & 15) as i32 - 8) as f32 * d;
                    y[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
            // Two bits, four codes, no zero-point stored: the codes are the values, offset by one.
            42 => {
                let d = h(b, 0);
                for j in 0..64 {
                    y[j] = (((b[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 - 1) as f32 * d;
                }
            }
            3 => {
                let (d, m) = (h(b, 0), h(b, 2));
                for j in 0..16 {
                    y[j] = (b[4 + j] & 15) as f32 * d + m;
                    y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
                }
            }
            6 | 7 => {
                let one = ty == 7;
                let d = h(b, 0);
                let (m, o) = if one { (h(b, 2), 4) } else { (0.0, 2) };
                let qh = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                let qs = &b[o + 4..o + 20];
                for j in 0..16 {
                    let h0 = ((qh >> j) << 4) as u8 & 0x10;
                    let h1 = (qh >> (j + 12)) as u8 & 0x10;
                    let (x0, x1) = ((qs[j] & 15) | h0, (qs[j] >> 4) | h1);
                    if one {
                        y[j] = x0 as f32 * d + m;
                        y[j + 16] = x1 as f32 * d + m;
                    } else {
                        y[j] = (x0 as i32 - 16) as f32 * d;
                        y[j + 16] = (x1 as i32 - 16) as f32 * d;
                    }
                }
            }
            8 => {
                let d = h(b, 0);
                for j in 0..32 {
                    y[j] = b[2 + j] as i8 as f32 * d;
                }
            }
            12 | 13 => {
                let five = ty == 13;
                let (d, mn) = (h(b, 0), h(b, 2));
                let sc = &b[4..16];
                let (qh, ql) = if five { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
                for g in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * g, sc);
                    let (s2, m2) = scale_min_k4(2 * g + 1, sc);
                    let (d1, m1) = (d * s1 as f32, mn * m1 as f32);
                    let (d2, m2) = (d * s2 as f32, mn * m2 as f32);
                    let q = &ql[32 * g..32 * g + 32];
                    for l in 0..32 {
                        let (mut lo, mut hi) = ((q[l] & 15) as u32, (q[l] >> 4) as u32);
                        if five {
                            lo += if qh[l] & (1 << (2 * g)) != 0 { 16 } else { 0 };
                            hi += if qh[l] & (2 << (2 * g)) != 0 { 16 } else { 0 };
                        }
                        y[64 * g + l] = d1 * lo as f32 - m1;
                        y[64 * g + 32 + l] = d2 * hi as f32 - m2;
                    }
                }
            }
            14 => {
                let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
                let d = h(b, 208);
                for n in 0..2 {
                    let (ql, qh, sc) = (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 15) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let y = &mut y[128 * n..];
                        y[l] = d * sc[is] as i8 as f32 * q1 as f32;
                        y[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                        y[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                        y[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                    }
                }
            }
            21 => {
                // (d x scale_g) x the signed level
                let d = h(b, 0);
                for g in 0..8 {
                    let db = d * crate::iq3s_group_scale(b, g) as f32;
                    let (l, s) = crate::iq3s_group(b, g);
                    for e in 0..32 {
                        let v = l[e] as i32;
                        y[32 * g + e] = db * (if (s >> e) & 1 == 1 { -v } else { v }) as f32;
                    }
                }
            }
            23 => {
                // (d x scale_g) x the level
                let sc = crate::iq4xs_scales(b);
                let d = h(b, 0);
                for g in 0..8 {
                    let dl = d * sc[g] as f32;
                    for l in 0..16 {
                        let q = b[8 + 16 * g + l];
                        y[32 * g + l] = dl * crate::IQ4_LEVELS[(q & 15) as usize] as f32;
                        y[32 * g + l + 16] = dl * crate::IQ4_LEVELS[(q >> 4) as usize] as f32;
                    }
                }
            }
            11 => {
                // d x scale x value, with the value (-4..3) from two qs bits and an hmask bit
                let sc = crate::q3k_unpack_scales(b[96..108].try_into().unwrap());
                let d = h(b, 108);
                for (e, ye) in y.iter_mut().enumerate() {
                    let (n, j, hh, l) = (e / 128, e % 128 / 32, e % 32 / 16, e % 16);
                    let q = (b[32 + 32 * n + 16 * hh + l] >> (2 * j)) & 3;
                    let hi = (b[16 * hh + l] >> (j + 4 * n)) & 1;
                    *ye = d * sc[e / 16] as f32 * (q as i32 - if hi == 1 { 0 } else { 4 }) as f32;
                }
            }
            20 => {
                let d = h(b, 0);
                for j in 0..16 {
                    y[j] = d * IQ4NL[(b[2 + j] & 15) as usize] as f32;
                    y[j + 16] = d * IQ4NL[(b[2 + j] >> 4) as usize] as f32;
                }
            }
            39 => {
                let d = crate::e8m0_half(b[0]);
                for j in 0..16 {
                    y[j] = d * crate::MXFP4_LEVELS[(b[1 + j] & 15) as usize] as f32;
                    y[j + 16] = d * crate::MXFP4_LEVELS[(b[1 + j] >> 4) as usize] as f32;
                }
            }
            10 => {
                // the value, its 4-bit group scale, and the block minimum off the group's own offset
                let (sc, q) = (&b[0..16], &b[16..80]);
                let (d, mn) = (h(b, 80), h(b, 82));
                for g in 0..16 {
                    let (chunk, half, shift) = (32 * (g / 8), 16 * (g % 2), 2 * ((g % 8) / 2));
                    let (dl, ml) = (d * (sc[g] & 0xF) as f32, mn * (sc[g] >> 4) as f32);
                    for l in 0..16 {
                        y[16 * g + l] = dl * ((q[chunk + half + l] >> shift) & 3) as f32 - ml;
                    }
                }
            }
            16 => {
                // eight magnitudes from the grid under one exponent the four sub-blocks of a group share
                let d = h(b, 0);
                let mut o = 0usize;
                for ib in 0..8 {
                    let a0 = u32::from_le_bytes(b[2 + 8 * ib..6 + 8 * ib].try_into().unwrap());
                    let a1 = u32::from_le_bytes(b[6 + 8 * ib..10 + 8 * ib].try_into().unwrap());
                    let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let grid = IQ2XXS_GRID[a0.to_le_bytes()[l] as usize];
                        let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                        for j in 0..8 {
                            let v = ((grid >> (8 * j)) & 0xff) as f32;
                            y[o + j] = db * v * if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        }
                        o += 8;
                    }
                }
            }
            17 => {
                // a 9-bit index into 512 vectors, the bits above it being where the signs live
                let d = h(b, 0);
                let mut o = 0usize;
                for ib in 0..8 {
                    let (db1, db2) =
                        (d * (0.5 + (b[66 + ib] & 0xf) as f32) * 0.25, d * (0.5 + (b[66 + ib] >> 4) as f32) * 0.25);
                    for l in 0..4 {
                        let q2 = u16::from_le_bytes([b[2 + 2 * (4 * ib + l)], b[3 + 2 * (4 * ib + l)]]) as usize;
                        let grid = IQ2XS_GRID[q2 & 511];
                        let signs = KSIGNS_IQ2XS[q2 >> 9];
                        let db = if l < 2 { db1 } else { db2 };
                        for j in 0..8 {
                            let v = ((grid >> (8 * j)) & 0xff) as f32;
                            y[o + j] = db * v * if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        }
                        o += 8;
                    }
                }
            }
            22 => {
                // the signs are their own bytes here, so the index only needs two bits from elsewhere
                let d = h(b, 0);
                let (qs, qh, sc, signs) = (&b[2..66], &b[66..74], &b[74..82], &b[34..66]);
                let mut o = 0usize;
                for ib in 0..8 {
                    let (db1, db2) =
                        (d * (0.5 + (sc[ib] & 0xf) as f32) * 0.25, d * (0.5 + (sc[ib] >> 4) as f32) * 0.25);
                    for l in 0..4 {
                        let db = if l < 2 { db1 } else { db2 };
                        let idx = (qs[4 * ib + l] as u32 | ((qh[ib] as u32) << (8 - 2 * l)) & 0x300) as usize;
                        let grid = IQ2S_GRID[idx];
                        for j in 0..8 {
                            let v = ((grid >> (8 * j)) & 0xff) as f32;
                            y[o + j] = db * v * if signs[4 * ib + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        }
                        o += 8;
                    }
                }
            }
            18 => {
                // two grid entries make a sub-block of eight, four sign bits each, under a shared exponent
                let d = h(b, 0);
                let (q3, gas) = (&b[2..66], &b[66..98]);
                let mut o = 0usize;
                for ib in 0..8 {
                    let aux = u32::from_le_bytes(gas[4 * ib..4 * ib + 4].try_into().unwrap());
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    for l in 0..4 {
                        let g1 = IQ3XXS_GRID[q3[8 * ib + 2 * l] as usize];
                        let g2 = IQ3XXS_GRID[q3[8 * ib + 2 * l + 1] as usize];
                        let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        for j in 0..4 {
                            let v1 = ((g1 >> (8 * j)) & 0xff) as f32;
                            let v2 = ((g2 >> (8 * j)) & 0xff) as f32;
                            y[o + j] = db * v1 * if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            y[o + j + 4] = db * v2 * if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                        }
                        o += 8;
                    }
                }
            }
            29 => {
                // a scale of four 4-bit fields, and every value 1/8 off the grid number the shift bit signs
                let sc: [u16; 4] = std::array::from_fn(|i| u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]));
                let d =
                    f16_to_f32((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
                let (qs, qh) = (&b[0..32], &b[32..48]);
                for ib in 0..8 {
                    let (db1, db2) = (
                        d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32 + 1.0),
                        d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0),
                    );
                    for l in 0..4 {
                        let db = if l < 2 { db1 } else { db2 };
                        let delta = match l {
                            0 | 1 => {
                                if qh[2 * ib] & (0x08 << (4 * l)) != 0 {
                                    -0.125
                                } else {
                                    0.125
                                }
                            }
                            _ => {
                                if qh[2 * ib + 1] & (0x08 << (4 * (l - 2))) != 0 {
                                    -0.125
                                } else {
                                    0.125
                                }
                            }
                        };
                        let idx = qs[4 * ib + l] as usize
                            | (((qh[2 * ib + l / 2] as u32) << (8 - 4 * (l % 2))) & 0x700) as usize;
                        let grid = IQ1S_GRID[idx];
                        for j in 0..8 {
                            let v = (((grid >> (8 * j)) & 0xff) as u8) as i8 as f32;
                            y[32 * ib + 8 * l + j] = db * (v + delta);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    true
}
