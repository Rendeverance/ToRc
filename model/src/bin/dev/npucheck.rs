//! npucheck <model> [n] [layer] [threads]: the NPU smoke test, on the dense products the NPU tier takes over (a
//! DeltaNet layer's attn_qkv, attn_gate and ssm_out). Each matrix is uploaded to the NPU, run over n f32 input rows
//! (deterministic pseudo-random, unit scale), and compared with the CPU's product (Q8 inputs, as the forward pass
//! uses) and, for attn_qkv, with an f64 product of the dequantised weights. It prints the largest |difference| and the
//! relative RMS of each against the f64 reference or the CPU, and the median time of 5 runs on each side (the CPU using
//! the whole pool). Needs TORC_NPU=1, and TORC_NPU_DIR if the GenieX ggml build is not in its default folder.

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn main() {
    use kernels::npu::{Npu, NpuGroup, NpuMm, configure};
    use kernels::{Q8, QK, ops as kops};
    use model::moe::{Mat, apply_group_multi};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: npucheck <model> [n=128] [layer=0] [threads=8]");
        std::process::exit(2);
    }
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let l: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let nt: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    if !configure() {
        eprintln!("npucheck: set TORC_NPU=1");
        std::process::exit(2);
    }
    let threads = model::pool::init(nt).threads as i32;
    let _ = threads;
    // the weights only: Model::load would start its own NPU tier (a second session) and upload every product
    let w = model::weights::Weights::load(&args[0]).expect("model");
    let mut npu = match Npu::create() {
        Ok(Some(p)) => p,
        Ok(None) => {
            eprintln!("npucheck: no NPU session (TORC_NPU, TORC_NPU_DIR)");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("npucheck: NPU not usable: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("NPU up, min_n {}, CPU threads {threads}", npu.min_tokens());
    let names = ["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"];
    let ts: Vec<_> = names.iter().map(|s| w.layer(l, s).expect("tensor")).collect();
    let mut ids = Vec::new();
    for (t, name) in ts.iter().zip(names) {
        let (cols, rows) = t.shape();
        // SAFETY: the mmap outlives the upload below
        let id = unsafe { npu.weight(t.t.ty, t.data.as_ptr(), rows, cols) };
        eprintln!("{name}: type {} {rows} x {cols}, id {id:?}", t.t.ty);
        ids.push(id.expect("weight refused"));
    }
    let (bytes, secs) = npu.upload().expect("upload");
    eprintln!(
        "uploaded {:.1} MiB in {:.2} s; ok: {:?}",
        bytes as f64 / 1048576.0,
        secs,
        ids.iter().map(|&i| npu.ok(i)).collect::<Vec<_>>()
    );
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    for (k, (t, name)) in ts.iter().zip(names).enumerate() {
        let (cols, rows) = t.shape();
        let x: Vec<f32> = (0..n * cols).map(|_| rnd()).collect();
        // NPU: one group, one product, into y[t * rows + r]
        let mut yn = vec![0f32; n * rows];
        let mm = [NpuMm { w: ids[k], y: yn.as_mut_ptr(), y_stride: rows }];
        let g =
            [NpuGroup { x: x.as_ptr(), cols, n, mm: mm.as_ptr(), n_mm: 1, ffn: [-1; 3], ffn_y: std::ptr::null_mut() }];
        let mut tn = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            // SAFETY: x and yn live across the call
            assert!(unsafe { npu.run(&g) }, "npu run failed");
            tn.push(t0.elapsed().as_secs_f64());
        }
        // CPU: the forward pass's path (each row quantised to Q8, then the group product)
        let mut xq = Q8::new(n * cols / QK);
        kops::q8k_quantize(&x, xq.view_mut());
        let mut yc = vec![0f32; n * rows];
        let mut tc = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            apply_group_multi(&mut [(Mat::of(t), &mut yc[..])], xq.view(), n);
            tc.push(t0.elapsed().as_secs_f64());
        }
        let cmp = |a: &[f32], b: &[f64]| {
            let (mut mx, mut se, mut sr) = (0f64, 0f64, 0f64);
            for (&p, &q) in a.iter().zip(b) {
                mx = mx.max((p as f64 - q).abs());
                se += (p as f64 - q).powi(2);
                sr += q * q;
            }
            (mx, (se / sr.max(1e-300)).sqrt())
        };
        let ycd: Vec<f64> = yc.iter().map(|&v| v as f64).collect();
        let (mx, rr) = cmp(&yn, &ycd);
        eprintln!(
            "{name}: n {n}: NPU {:.2} ms vs CPU {:.2} ms ({:.2}x); NPU vs CPU max |d| {mx:.4}, rel_rms {rr:.2e}",
            med(&mut tn) * 1e3,
            med(&mut tc) * 1e3,
            med(&mut tc.clone()) / med(&mut tn.clone())
        );
        if k == 0 {
            // an f64 reference over the dequantised weights (the first 8 rows of x, to keep it quick)
            let mut wf = vec![0f32; rows * cols];
            kops::dequantize(t.t.ty, t.data, &mut wf);
            let nr = n.min(8);
            let mut yr = vec![0f64; nr * rows];
            for ti in 0..nr {
                for r in 0..rows {
                    let (wr, xr) = (&wf[r * cols..(r + 1) * cols], &x[ti * cols..(ti + 1) * cols]);
                    yr[ti * rows + r] = wr.iter().zip(xr).map(|(&a, &b)| a as f64 * b as f64).sum();
                }
            }
            let (m1, r1) = cmp(&yn[..nr * rows], &yr);
            let (m2, r2) = cmp(&yc[..nr * rows], &yr);
            eprintln!(
                "{name}: vs f64 (first {nr} rows): NPU max |d| {m1:.4} rel_rms {r1:.2e}; CPU max |d| {m2:.4} rel_rms {r2:.2e}"
            );
        }
    }
    let (w, g) = npu.stats();
    eprintln!("npu stats: {g} graphs, {:.1} ms waited", w * 1e3);
}

#[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux"))))]
fn main() {
    eprintln!("npucheck: ARM64 Windows only (the Snapdragon NPU)");
    std::process::exit(2);
}
