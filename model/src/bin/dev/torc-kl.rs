//! torc-kl <tokens.int32> <ref.f32> <a.f32> [b.f32 ...]: how far each logits dump (from torc-run dump=, every position
//! of the same tokens) is from a reference dump. For each dump it prints the mean and largest KL(ref || dump) over the
//! positions (in nats), the top-1 agreement, and the perplexity of the tokens under it against the reference's. The
//! reference is the judge: for the mixer kernels, a TORC_MIX=exact run (f64 on the exact weights).

use std::io::Read;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: torc-kl <tokens.int32> <ref.f32> <a.f32> [b.f32 ...]");
        std::process::exit(2);
    }
    let tb = std::fs::read(&a[1]).expect("tokens");
    let tokens: Vec<usize> = tb.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
    let rlen = std::fs::metadata(&a[2]).expect("ref").len() as usize;
    let np = (2..a.len()).map(|i| std::fs::metadata(&a[i]).expect("dump").len() as usize).min().unwrap();
    // positions: as many as every file holds whole rows for; the vocabulary size comes from the reference
    let n = tokens.len().min(rlen / 4).max(1);
    let nv = rlen / 4 / n;
    let n = n.min(np / 4 / nv);
    assert!(nv > 0 && n > 0, "empty dumps");
    let mut files: Vec<std::fs::File> = (2..a.len()).map(|i| std::fs::File::open(&a[i]).unwrap()).collect();
    let nd = files.len() - 1;
    let mut rows = vec![vec![0.0f32; nv]; files.len()];
    let (mut kl, mut klmax, mut top, mut nll) = (vec![0f64; nd], vec![0f64; nd], vec![0usize; nd], vec![0f64; nd + 1]);
    let lsm = |l: &[f32]| -> Vec<f64> {
        let mx = l.iter().fold(f32::MIN, |a, &v| a.max(v)) as f64;
        let z: f64 = l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln() + mx;
        l.iter().map(|&v| v as f64 - z).collect()
    };
    // TORC_KL_BUCKET=b: also the mean KL over each b positions, and the first positions where KL passes 1 nat
    // TORC_KL_FROM=p: score positions p.. only (llama-perplexity scores the second half of each chunk)
    let from: usize = model::switches::num("TORC_KL_FROM").unwrap_or(0);
    let bucket: usize = model::switches::num("TORC_KL_BUCKET").unwrap_or(0);
    let mut bk = vec![vec![0f64; if bucket > 0 { n.div_ceil(bucket) } else { 0 }]; nd];
    let mut big: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nd];
    let am = |l: &[f32]| l.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
    for p in 0..n {
        for (f, r) in files.iter_mut().zip(rows.iter_mut()) {
            // SAFETY: the f32s viewed as bytes
            let b = unsafe { std::slice::from_raw_parts_mut(r.as_mut_ptr().cast::<u8>(), nv * 4) };
            f.read_exact(b).expect("short dump");
        }
        if p < from {
            continue;
        }
        let lr = lsm(&rows[0]);
        let tr = am(&rows[0]);
        if p + 1 < tokens.len() {
            nll[0] -= lr[tokens[p + 1]];
        }
        for d in 0..nd {
            let ld = lsm(&rows[d + 1]);
            let k: f64 = lr.iter().zip(&ld).map(|(&x, &y)| x.exp() * (x - y)).sum();
            kl[d] += k;
            if let Some(b) = p.checked_div(bucket) {
                bk[d][b] += k;
                if k > 1.0 && big[d].len() < 12 {
                    big[d].push((p, k));
                }
            }
            klmax[d] = klmax[d].max(k);
            top[d] += (am(&rows[d + 1]) == tr) as usize;
            if p + 1 < tokens.len() {
                nll[d + 1] -= ld[tokens[p + 1]];
            }
        }
    }
    let k = (n.min(tokens.len()) - 1).saturating_sub(from).max(1) as f64;
    let n = n - from.min(n);
    println!("{n} positions, vocabulary {nv}; reference {}: perplexity {:.5}", a[2], (nll[0] / k).exp());
    for d in 0..nd {
        println!(
            "{}: KL mean {:.3e} max {:.3e} nats, top-1 {}/{n}, perplexity {:.5} ({:+.4}%)",
            a[d + 3],
            kl[d] / n as f64,
            klmax[d],
            top[d],
            (nll[d + 1] / k).exp(),
            100.0 * ((nll[d + 1] - nll[0]) / k).exp_m1()
        );
        if bucket > 0 {
            let m: Vec<String> = bk[d]
                .iter()
                .enumerate()
                .map(|(i, x)| format!("{:.2e}", x / (n.min((i + 1) * bucket) - i * bucket) as f64))
                .collect();
            println!("  mean KL per {bucket} positions: {}", m.join(" "));
            let f: Vec<String> = big[d].iter().map(|(p, k)| format!("{p}:{k:.1}")).collect();
            println!("  first positions over 1 nat: {}", f.join(" "));
        }
    }
}
