//! ops.rs - element-wise operations, each in a fixed order of operations. Nothing here is fused unless it is written as
//! a fused step. expf, sqrtf, powf, cosf and sinf come from the C runtime (llvm-mingw's UCRT), so they round the same
//! as in any other build that uses it.

/// RMS norm. The squares are summed in f64, in order; then scale = 1 / sqrtf((float)(ss / n) + eps) and out = x *
/// scale * w.
pub fn rms_norm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x {
        ss += v as f64 * v as f64;
    }
    let scale = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

/// RMS norm in place: the same steps as rms_norm.
pub fn rms_norm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x.iter() {
        ss += v as f64 * v as f64;
    }
    let scale = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * scale * w[i];
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// TORC_TRACE: one line about a hidden state after a layer (its root mean square, largest magnitude and first three
/// values), enough to see where two runs part.
pub fn trace_state(layer: usize, v: &[f32]) {
    let ms = v.iter().map(|&a| a as f64 * a as f64).sum::<f64>() / v.len().max(1) as f64;
    let top = v.iter().fold(0.0f32, |m, a| m.max(a.abs()));
    let head: Vec<String> = v.iter().take(3).map(|a| format!("{a:+.6}")).collect();
    eprintln!("trace layer {layer}: rms {:.6} max {top:.4} head {}", ms.sqrt(), head.join(" "));
    // TORC_DUMP=dir: the state itself as dir/<layer>.f32 (the last call per layer wins), to compare with a reference
    if let Some(dir) = crate::switches::get("TORC_DUMP") {
        let b: Vec<u8> = v.iter().flat_map(|a| a.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/{layer}.f32"), b);
    }
}

/// NeoX-style rotary embedding on the first n dimensions, rotating the pairs (i, i + n/2).
pub fn rope_neox(v: &mut [f32], n: usize, cs: &[f32], sn: &[f32]) {
    let half = n / 2;
    for i in 0..half {
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * cs[i] - b * sn[i];
        v[i + half] = a * sn[i] + b * cs[i];
    }
}

/// Rotary embedding on adjacent pairs (2i, 2i + 1) of the first n dimensions (llama.cpp's "norm" type: ernie4_5-moe),
/// with the same table as rope_neox.
pub fn rope_pairs(v: &mut [f32], n: usize, cs: &[f32], sn: &[f32]) {
    for i in 0..n / 2 {
        let (a, b) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = a * cs[i] - b * sn[i];
        v[2 * i + 1] = a * sn[i] + b * cs[i];
    }
}

/// The RoPE table for one position: theta_i = pos * powf(base, -2i / n_rot).
pub fn rope_row(cs: &mut [f32], sn: &mut [f32], pos: u32, n_rot: u32, base: f32) {
    for i in 0..(n_rot / 2) as usize {
        let theta = pos as f32 * base.powf(-2.0 * i as f32 / n_rot as f32);
        cs[i] = theta.cos();
        sn[i] = theta.sin();
    }
}

/// The order a top-K is taken in: value descending (by total_cmp), and equal values by the lower id. It is a total
/// order on (id, value) pairs, so the first k form one set however they are found.
pub fn by_rank(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

/// Up to this k, top_k finds its set in one scan of the row; above it, it selects from every pair.
pub const TOP_K_SCAN: usize = 64;

/// The k highest-ranked values of l under by_rank, as (index, value) pairs in rank order. Up to TOP_K_SCAN this is a
/// single pass: a value is only kept if it ranks above the current k-th, and because indices rise as the scan goes, an
/// equal value never displaces one already kept. Almost every value costs one comparison. Above TOP_K_SCAN, every pair
/// is selected and sorted instead. Both paths give the same set in the same order (test top_k_is_the_select).
pub fn top_k(l: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(l.len());
    if k == 0 {
        return Vec::new();
    }
    if k > TOP_K_SCAN {
        let mut c: Vec<(u32, f32)> = l.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        if k < c.len() {
            c.select_nth_unstable_by(k - 1, by_rank);
            c.truncate(k);
        }
        c.sort_by(by_rank);
        return c;
    }
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, &v) in l.iter().enumerate() {
        if best.len() == k && v.total_cmp(&best[k - 1].1) != std::cmp::Ordering::Greater {
            continue;
        }
        let e = (i as u32, v);
        let at = best.partition_point(|x| by_rank(x, &e) == std::cmp::Ordering::Less);
        best.insert(at, e);
        best.truncate(k);
    }
    best
}

/// The index of the first largest value - exactly what a plain scan from f32::MIN gives when it keeps each value that
/// is greater: a NaN is never kept, ties go to the lower index, -0 and +0 count as equal, and a row with nothing above
/// f32::MIN gives 0. On ARM64 it finds the largest value first (FMAXNM, sixteen at a time) and then the first index
/// holding it, which is the same index.
pub fn argmax(l: &[f32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n16 = l.len() / 16 * 16;
        // SAFETY: every index read is below n16, which is at most l.len().
        let mut m = unsafe {
            let p = l.as_ptr();
            let mut a = [vdupq_n_f32(f32::MIN); 4];
            let mut i = 0;
            while i < n16 {
                for (k, ak) in a.iter_mut().enumerate() {
                    *ak = vmaxnmq_f32(*ak, vld1q_f32(p.add(i + 4 * k)));
                }
                i += 16;
            }
            vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(a[0], a[1]), vmaxnmq_f32(a[2], a[3])))
        };
        for &v in &l[n16..] {
            if v > m {
                m = v;
            }
        }
        // nothing above f32::MIN (or an empty row): the plain scan would return 0
        if m <= f32::MIN {
            return 0;
        }
        return l.iter().position(|&v| v == m).unwrap_or(0);
    }
    #[allow(unreachable_code)]
    l.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// top_k against selecting from every pair and sorting: the same pairs in the same order, with k on both sides of
    /// the scan's limit, rows full of ties and signed zeros, short rows and a full vocabulary.
    #[test]
    fn top_k_is_the_select() {
        let full = |l: &[f32], k: usize| {
            let mut c: Vec<(u32, f32)> = l.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
            let k = k.min(c.len());
            if k > 0 && k < c.len() {
                c.select_nth_unstable_by(k - 1, by_rank);
            }
            c.truncate(k);
            c.sort_by(by_rank);
            c
        };
        let mut s = 0x3c6e_f372_fe94_f82bu64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0
        };
        let rows: Vec<Vec<f32>> = vec![
            (0..10).map(|_| rnd()).collect(),
            (0..1000).map(|_| (rnd() / 3.0).round()).collect(),
            (0..1000).map(|i| if i % 2 == 0 { -0.0 } else { 0.0 }).collect(),
            (0..248_320).map(|_| rnd()).collect(),
            (0..248_320).map(|_| (rnd() * 2.0).round()).collect(),
        ];
        for (r, l) in rows.iter().enumerate() {
            for k in [0usize, 1, 2, 3, 20, 63, 64, 65, 300, 2000] {
                let (a, b) = (top_k(l, k), full(l, k));
                assert!(
                    a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.0 == y.0 && x.1.to_bits() == y.1.to_bits()),
                    "row {r} k {k}"
                );
            }
        }
    }

    /// argmax against the plain scan: random rows, rows with ties, signed zeros, -inf, NaN and nothing above f32::MIN,
    /// every length from 0 to 69 and a full vocabulary.
    #[test]
    fn argmax_is_the_scan() {
        let scan =
            |l: &[f32]| l.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        let mut s = 0x2b7e_1516_28ae_d2a6u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0
        };
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for n in 0..70usize {
            rows.push((0..n).map(|_| rnd()).collect());
            rows.push((0..n).map(|_| (rnd() / 4.0).round()).collect());
            rows.push(
                (0..n)
                    .map(|i| {
                        if i % 3 == 0 {
                            -0.0
                        } else if i % 3 == 1 {
                            0.0
                        } else {
                            -1.0 - rnd().abs()
                        }
                    })
                    .collect(),
            );
            rows.push((0..n).map(|i| if i % 5 == 0 { f32::NAN } else { rnd() }).collect());
            rows.push(vec![f32::NEG_INFINITY; n]);
            rows.push(vec![f32::MIN; n]);
        }
        rows.push((0..248_320).map(|_| rnd()).collect());
        let mut big: Vec<f32> = (0..248_320).map(|_| (rnd() / 2.0).round()).collect();
        big[200_000] = 1e9;
        big[1000] = 1e9;
        rows.push(big);
        for (r, l) in rows.iter().enumerate() {
            assert_eq!(argmax(l), scan(l), "row {r} (len {})", l.len());
        }
    }

    #[test]
    fn rms_norm_unit_weights_gives_unit_rms() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 7.0).collect();
        let mut y = vec![0.0; 256];
        rms_norm(&mut y, &x, &vec![1.0; 256], 1e-6);
        let rms = (y.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 256.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
    }
}
