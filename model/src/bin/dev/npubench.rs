//! npubench [sessions] [reps]: does the NPU overlap work from two sessions? Each session gets K Q4_0 matrices (2816 x
//! 4096, Gemma's widths) and two graphs: "mv", K matrix-vector products (the bandwidth side), and "small", 500 chained
//! RMS norms of one row (the per-op latency side). Each graph is timed on session 0 alone, then on every session at
//! once, each from its own thread. With real overlap, the wall time of the concurrent runs stays close to the single
//! run's. Sessions past HTP0 need GGML_HEXAGON_NDEV (or _DEVICES) set to that many.

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
fn main() {
    use kernels::ggml::{self, Ggml, T};
    use std::time::Instant;
    let args: Vec<String> = std::env::args().collect();
    let ns: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20);
    const K: usize = 48;
    let (cols, rows) = (2816usize, 4096usize);
    // a Q4_0 row: for every 32 weights, an f16 scale (1.0) and 16 bytes of nibbles
    let row_bytes = cols / 32 * 18;
    let mut data = vec![0u8; rows * row_bytes];
    let mut s = 0x9e3779b97f4a7c15u64;
    for (i, b) in data.iter_mut().enumerate() {
        if i % 18 == 0 {
            *b = 0x00;
        } else if i % 18 == 1 {
            *b = 0x20; // f16 0x2000 = 2^-7
        } else {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = s as u8;
        }
    }
    struct Sess {
        g: Ggml,
        mv: (ggml::Graph, T),
        small: (ggml::Graph, T),
    }
    unsafe impl Send for Sess {}
    unsafe impl Sync for Sess {}
    let mut sess = Vec::new();
    for i in 0..ns {
        let g = Ggml::open_dev(&format!("HTP{i}")).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1)
        });
        let f = &g.f;
        // SAFETY: plain graph construction and upload on this session
        let (mv, small) = unsafe {
            let wctx = g.meta_ctx(K + 1, false);
            let ws: Vec<T> =
                (0..K).map(|_| (f.ggml_new_tensor_2d)(wctx, ggml::Q4_0, cols as i64, rows as i64)).collect();
            let buf = (f.ggml_backend_alloc_ctx_tensors_from_buft)(wctx, g.buft);
            (f.ggml_backend_buffer_set_usage)(buf, ggml::USAGE_WEIGHTS);
            for &w in &ws {
                (f.ggml_backend_tensor_set)(w, data.as_ptr().cast(), 0, data.len());
            }
            let mk = |small: bool| {
                let ctx = g.meta_ctx(2048, true);
                let gf = (f.ggml_new_graph_custom)(ctx, 2048, false);
                let x = (f.ggml_new_tensor_2d)(ctx, ggml::F32, cols as i64, 1);
                (f.ggml_set_input)(x);
                let mut last = x;
                if small {
                    for _ in 0..500 {
                        last = (f.ggml_rms_norm)(ctx, last, 1e-6);
                    }
                    (f.ggml_build_forward_expand)(gf, last);
                } else {
                    for &w in &ws {
                        last = (f.ggml_mul_mat)(ctx, w, x);
                        (f.ggml_build_forward_expand)(gf, last);
                    }
                }
                let al = (f.ggml_gallocr_new)(g.buft);
                assert!((f.ggml_gallocr_alloc_graph)(al, gf));
                let xs = vec![0.01f32; cols];
                (f.ggml_backend_tensor_set)(x, xs.as_ptr().cast(), 0, cols * 4);
                (gf, x)
            };
            (mk(false), mk(true))
        };
        sess.push(Sess { g, mv, small });
    }
    let bytes = (K * rows * row_bytes) as f64;
    for which in ["mv", "small"] {
        let run = |s: &Sess| {
            let gf = if which == "mv" { s.mv.0 } else { s.small.0 };
            // SAFETY: the graph and its buffers live in s
            unsafe { (s.g.f.ggml_backend_graph_compute)(s.g.be, gf) };
        };
        run(&sess[0]);
        let t = Instant::now();
        for _ in 0..reps {
            run(&sess[0]);
        }
        let one = t.elapsed().as_secs_f64() / reps as f64;
        // every session at once
        for s in &sess {
            run(s);
        }
        let t = Instant::now();
        std::thread::scope(|sc| {
            for s in &sess {
                sc.spawn(move || {
                    for _ in 0..reps {
                        run(s);
                    }
                });
            }
        });
        let all = t.elapsed().as_secs_f64() / reps as f64;
        let extra = if which == "mv" {
            format!(", {:.1} GB/s alone, {:.1} GB/s all", bytes / one / 1e9, ns as f64 * bytes / all / 1e9)
        } else {
            format!(", {:.1} us/op alone", one * 1e6 / 500.0)
        };
        println!(
            "{which}: one session {:.2} ms a graph; {ns} at once {:.2} ms wall for {ns} graphs (speed-up {:.2}x){extra}",
            one * 1e3,
            all * 1e3,
            ns as f64 * one / all
        );
    }
}

#[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux"))))]
fn main() {
    eprintln!("npubench: ARM64 Windows only");
}
