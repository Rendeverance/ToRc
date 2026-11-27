//! torc-gpubench: the GPU attention kernel against the CPU's on one attention shape, with made-up keys and queries and
//! no model loaded, so nothing else competes for the memory. Walks a prompt of `ctx` tokens in blocks of `block`, and
//! times every block twice: all of it on the CPU (fattn.rs), and all of it on the GPU (gpu.rs, TORC_GPU_SHARE=1 set
//! here). Prints both totals, their ratio, and the largest difference between the two outputs.
//!
//!     torc-gpubench <query heads> <KV heads> <head dim> <16-bit 0|1> <ctx> [block] [threads] [window]
//!
//! block, threads and window are plain numbers in that order (512, 8 and no sliding window if left out). With a
//! window, the walk also goes back once to the middle of the prompt and reads on from there, as a chat edited
//! half-way up does.
//!
//! Flash Next's shape is 24 2 256 1, the Qwen 35B's 16 2 256 0, Gemma 4's sliding layers 16 8 256 1 with window 1024.

fn main() {
    #[cfg(feature = "gpu")]
    run();
    #[cfg(not(feature = "gpu"))]
    eprintln!("torc-gpubench: built without the gpu feature");
}

#[cfg(feature = "gpu")]
fn run() {
    use model::attn::KvCache;
    use model::fattn::Fattn;
    use std::time::Instant;
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!(
            "usage: torc-gpubench <query heads> <KV heads> <head dim> <16-bit 0|1> <ctx> [block] [threads] [window]"
        );
        std::process::exit(2);
    }
    let num = |i: usize, d: usize| a.get(i).and_then(|v| v.parse::<usize>().ok()).unwrap_or(d);
    let (nh, nkv, hd, k16, ctx) = (num(1, 0), num(2, 0), num(3, 0), num(4, 0) == 1, num(5, 0));
    let (block, threads, window) = (num(6, 512), num(7, 8), num(8, 0));
    // set before the GPU and the pool start below and read them
    model::switches::set("TORC_GPU_ATTN", "1");
    model::switches::set("TORC_GPU_SHARE", "1");
    model::switches::set("TORC_GPU_FROM", "0");
    model::switches::set("TORC_GPU_MIN", "1");
    model::pool::set_width(threads);
    let n = nkv * ctx * hd;
    let kv = KvCache {
        k: if k16 { vec![] } else { (0..n).map(|i| (i % 251) as i8).collect() },
        v: if k16 { vec![] } else { (0..n).map(|i| (i % 241) as i8).collect() },
        k16: if k16 { (0..n).map(|i| (i % 30011) as i16 - 15000).collect() } else { vec![] },
        v16: if k16 { (0..n).map(|i| (i % 30013) as i16 - 15000).collect() } else { vec![] },
        k32: vec![],
        v32: vec![],
        ring: false,
        k_scale: (0..nkv * ctx).map(|i| 1e-4 + (i % 7) as f32 * 1e-5).collect(),
        v_scale: (0..nkv * ctx).map(|i| 1e-4 + (i % 5) as f32 * 1e-5).collect(),
        cap: ctx,
        ix_raw: vec![],
        ix_pk: vec![],
        ix_kp: 0,
        ix_npool: 0,
        sel: vec![],
        nv: vec![],
        qsa: false,
        ix_sc: vec![],
        gpu_lo: 0,
    };
    let mut kv = kv;
    let f = Fattn::new(nh, nkv, hd, None, window);
    let qd = nh * hd;
    let q: Vec<f32> = (0..block * qd).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let (mut out, mut cpu) = (vec![0f32; block * qd], vec![0f32; block * qd]);
    let (mut tc, mut tg) = (0f64, 0f64);
    // the largest difference between the two outputs, and the largest output
    let (mut dmax, mut amax) = (0f32, 0f32);
    // with a window: once the end is reached, back to the middle (the block there starts off a block boundary)
    let mut back = window > 0 && ctx > 2 * block;
    let mut pos0 = 0;
    while pos0 < ctx {
        let m = block.min(ctx - pos0);
        let t = Instant::now();
        f.run_block(&kv, &q[..m * qd], qd, m, pos0 as u32, &mut cpu[..m * qd], None);
        tc += t.elapsed().as_secs_f64();
        let t = Instant::now();
        match model::gpu::run_block(&kv, &f, &q[..m * qd], qd, m, pos0, &mut out[..m * qd]) {
            Some(hi) => kv.gpu_lo = hi,
            None => {
                eprintln!("torc-gpubench: the GPU tier did not take the block at {pos0}");
                std::process::exit(1);
            }
        }
        tg += t.elapsed().as_secs_f64();
        for (x, y) in out[..m * qd].iter().zip(&cpu[..m * qd]) {
            dmax = dmax.max((x - y).abs());
            amax = amax.max(y.abs());
        }
        pos0 += m;
        if pos0 >= ctx && back {
            back = false;
            pos0 = ctx / 2 + 7;
            kv.gpu_lo = kv.gpu_lo.min(pos0);
        }
    }
    println!(
        "shape {nh}/{nkv} hd {hd} {}{} ctx {ctx} block {block}: CPU {tc:.2} s, GPU {tg:.2} s, GPU/CPU {:.1}x; largest \
        difference {dmax:.3e} of outputs up to {amax:.3e}",
        if k16 { "16-bit" } else { "8-bit" },
        if window > 0 { format!(" window {window}") } else { String::new() },
        tg / tc
    );
}
