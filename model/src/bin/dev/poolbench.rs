//! poolbench: the cost of one empty run of the compute pool (ToRc's own, or TORC_POOL=rayon) - the overhead every
//! parallel step of a token pays.
use std::time::Instant;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    model::pool::init(n);
    let sink = std::sync::atomic::AtomicUsize::new(0);
    for items in [8usize, 64, 512] {
        let runs = 20000;
        let t0 = Instant::now();
        for _ in 0..runs {
            model::pool::par_for(items, &|i| {
                sink.fetch_add(i, std::sync::atomic::Ordering::Relaxed);
            });
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;
        println!("{n} threads, {items:3} items: {us:6.2} us per run");
    }
}
