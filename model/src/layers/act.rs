//! act.rs - the activation functions, in f32, with the same bits on every machine.
//!
//! They use only IEEE operations (add, multiply, fused multiply-add and division, each correctly rounded) and never
//! the platform's maths library, whose exp differs between systems. Each is plain scalar code with no branch, so the
//! compiler vectorises a loop over it.

use std::f32::consts::{LN_2, LOG2_E};

/// Adding this to an f32 below 2^22 in magnitude rounds it to a whole number (ties to even), and the sum then holds
/// that whole number in its low mantissa bits: 1.5 * 2^23.
pub(crate) const ROUND_SHIFT: f32 = 12_582_912.0;

/// What the f32 nearest ln 2 (LN_2) misses of the true value, so that x - k ln 2 loses nothing in two fused steps.
pub(crate) const LN_2_LO: f32 = -1.904_654_3e-9;

/// The Cephes expf polynomial (see THIRD-PARTY-NOTICES.md), highest power first: e^r = 1 + r + r^2 P(r) for |r| up
/// to about ln 2 / 2.
#[allow(clippy::excessive_precision)]
pub(crate) const EXP_P: [f32; 6] = [
    1.987_569_15e-4,
    1.398_199_950_7e-3,
    8.333_451_907_3e-3,
    4.166_579_589_4e-2,
    1.666_666_545_9e-1,
    5.000_000_120_1e-1,
];

/// The range expf holds its argument to: below EXP_MIN the result would leave f32's normal range, and above EXP_MAX
/// 2^k would overflow the exponent field.
pub(crate) const EXP_MIN: f32 = -87.0;
pub(crate) const EXP_MAX: f32 = 88.0;

/// e^x. The argument is held to [EXP_MIN, EXP_MAX] = [-87, 88], so the result is always a normal f32 (from about
/// 1.6e-38 to 1.65e38); a NaN argument gives e^-87.
pub fn expf(x: f32) -> f32 {
    let x = x.max(EXP_MIN).min(EXP_MAX);
    // x = k ln 2 + r, with k the nearest whole number to x / ln 2, so |r| <= ln 2 / 2 (a little over at a tie)
    let t = x.mul_add(LOG2_E, ROUND_SHIFT);
    let k = t - ROUND_SHIFT;
    let r = (-k).mul_add(LN_2_LO, (-k).mul_add(LN_2, x));
    let p = EXP_P.iter().fold(0.0f32, |acc, &c| acc.mul_add(r, c));
    let er = (r * r).mul_add(p, r) + 1.0;
    // 2^k: the low nine bits of t are k in two's complement; moved into the exponent field and biased by 127, which
    // keeps the field in 1..=254 because k is in -126..=127
    er * f32::from_bits((t.to_bits() << 23).wrapping_add(127 << 23))
}

/// The logistic function, 1 / (1 + e^-x).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf(-x))
}

/// SiLU (also called swish): x sigmoid(x), as one division.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + expf(-x))
}

/// GELU in its tanh form, x/2 (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). Because (1 + tanh z) / 2 equals sigmoid(2z),
/// it is x / (1 + e^(-2z)), with no tanh needed.
#[allow(clippy::excessive_precision)]
pub fn gelu_tanh(x: f32) -> f32 {
    const TWO_SQRT_2_OVER_PI: f32 = 1.595_769_121_6;
    let two_z = TWO_SQRT_2_OVER_PI * x * (0.044_715 * x).mul_add(x, 1.0);
    x / (1.0 + expf(-two_z))
}

/// A gated feed-forward unit's hidden value: the activation of the gate times the up projection. GELU gives GeGLU
/// (Gemma 4), SiLU gives SwiGLU (Qwen).
#[inline]
pub fn glu(gelu: bool, gate: f32, up: f32) -> f32 {
    (if gelu { gelu_tanh(gate) } else { silu(gate) }) * up
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The relative error of f against the f64 value g.
    fn rel(f: f32, g: f64) -> f64 {
        ((f as f64 - g) / g).abs()
    }

    #[test]
    fn expf_within_two_ulp() {
        let mut x = -87.0f32;
        while x <= 88.0 {
            assert!(rel(expf(x), (x as f64).exp()) < 2.5e-7, "x {x}");
            x += 0.013;
        }
    }

    #[test]
    fn expf_holds_the_range() {
        assert_eq!(expf(-1e6), expf(-87.0));
        assert_eq!(expf(1e6), expf(88.0));
        assert!(expf(f32::NAN) == expf(-87.0));
        assert!(expf(88.0).is_finite() && expf(-87.0).is_normal());
        assert_eq!(expf(0.0), 1.0);
    }

    #[test]
    fn activations_match_f64() {
        for i in -400..400 {
            let x = i as f32 * 0.05;
            let xd = x as f64;
            let sig = 1.0 / (1.0 + (-xd).exp());
            assert!(rel(sigmoid(x), sig) < 5e-7, "sigmoid {x}");
            if x != 0.0 {
                assert!(rel(silu(x), xd * sig) < 5e-7, "silu {x}");
                // the reference in its sigmoid form too: 1 + tanh z cancels to nothing in f64 for large negative x
                let z = (2.0 / std::f64::consts::PI).sqrt() * (xd + 0.044715 * xd.powi(3));
                // e^(-2z) passes an f32 rounding of 2z (about 3e-7 of it) on to the result, which matters for large
                // negative x; below about x = -10 the exp argument passes 88 and is held there, leaving under 1e-36
                let g = xd / (1.0 + (-2.0 * z).exp());
                let tol = 2e-6 + 4e-7 * (2.0 * z).abs();
                assert!(rel(gelu_tanh(x), g) < tol || (gelu_tanh(x) as f64 - g).abs() < 1e-36, "gelu {x}");
            }
        }
        assert!(silu(-1e4).abs() < 1e-30 && gelu_tanh(-1e4).abs() < 1e-30);
        assert_eq!(gelu_tanh(1e4), 1e4);
    }
    /// cargo test --release -p torc-model --lib act::tests::bench_glu -- --ignored --nocapture
    /// ns an element of the SwiGLU loop the experts run (rows of 640, gate values spread as a model's are), and of
    /// expf alone, on one thread.
    #[test]
    #[ignore]
    fn bench_glu() {
        let (rows, ff) = (4096usize, 640usize);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let g: Vec<f32> = (0..rows * ff).map(|_| 6.0 * rnd()).collect();
        let u: Vec<f32> = (0..rows * ff).map(|_| rnd()).collect();
        let mut h = vec![0.0f32; ff];
        // decided at run time, as in the engine (a constant would let the compiler drop the other branch)
        let gelu = std::hint::black_box(false);
        for round in 0..2 {
            let mut sink = 0.0f32;
            let t = std::time::Instant::now();
            for r in 0..rows {
                for i in 0..ff {
                    h[i] = glu(gelu, g[r * ff + i], u[r * ff + i]);
                }
                sink += h[r % ff];
            }
            let a = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            for r in 0..rows {
                for i in 0..ff {
                    h[i] = expf(-g[r * ff + i]);
                }
                sink += h[r % ff];
            }
            let b = t.elapsed().as_secs_f64();
            let n = (rows * ff) as f64;
            println!("round {round}: glu {:.2} ns, expf {:.2} ns an element ({sink})", a * 1e9 / n, b * 1e9 / n);
        }
    }
}
