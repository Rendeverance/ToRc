//! fvec.rs - two float vector helpers (SPEC.md section 4), in an order fixed so that every platform gives the same
//! bits.

/// a . b: sixteen lanes of fused multiply-adds (number i in lane i mod 16) over the whole sixteens, then eight more
/// steps in lanes 0-7 if eight remain; lane l is added to lane l + 8, those eight are summed as
/// ((0+4)+(2+6))+((1+5)+(3+7)), and the last n mod 8 products are fused onto the sum in order.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0.0f32; 16];
    let mut i = 0;
    while i + 16 <= n {
        for l in 0..16 {
            acc[l] = a[i + l].mul_add(b[i + l], acc[l]);
        }
        i += 16;
    }
    if i + 8 <= n {
        for l in 0..8 {
            acc[l] = a[i + l].mul_add(b[i + l], acc[l]);
        }
        i += 8;
    }
    let s: [f32; 8] = std::array::from_fn(|l| acc[l] + acc[l + 8]);
    let mut r = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    while i < n {
        r = a[i].mul_add(b[i], r);
        i += 1;
    }
    r
}

/// y += a x, with each number one fused multiply-add.
pub fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len());
    for (v, &u) in y.iter_mut().zip(x) {
        *v = a.mul_add(u, *v);
    }
}
