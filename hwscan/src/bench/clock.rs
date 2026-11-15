//! clock.rs - real clock under load: a dependent add chain retires one add per cycle, so adds / second = cycles /
//! second. Catches power caps the OS does not report.

use super::{Mode, median};
use crate::inventory::cpu::CpuInfo;
use crate::sys::{self, LogicalCpu};
use serde::Serialize;
use std::sync::Barrier;
use std::time::{Duration, Instant};

const ADDS_PER_ITER: u64 = 8;

#[derive(Debug, Clone, Serialize)]
pub struct ClockResult {
    pub method: &'static str,
    /// Each physical core alone, in core order: best of several short visits made round-robin (so every core is sampled
    /// early and late). Shows preferred cores.
    pub single_per_core: Vec<f64>,
    /// Sustained single-core run on the fastest core: ~50 ms samples.
    pub single_core_logical_id: u32,
    pub single_ghz_first: f64,
    pub single_ghz_median: f64,
    pub single_ghz_last: f64,
    pub single_seconds: f64,
    /// All physical cores at once.
    pub all_threads: usize,
    /// Median across cores of each core's first / median / last sample.
    pub all_ghz_first: f64,
    pub all_ghz_median: f64,
    pub all_ghz_last: f64,
    pub all_ghz_min_core: f64,
    pub all_ghz_max_core: f64,
    pub all_per_core: Vec<f64>,
    pub all_seconds: f64,
    pub seconds: f64,
}

/// `iters` loop iterations of 8 dependent adds. Returns the chain value so it stays live. The addend is a register:
/// some cores fold chains of `add reg, imm` in the renamer, which would overstate the clock.
#[cfg(target_arch = "x86_64")]
fn add_chain(iters: u64) -> u64 {
    let mut x: u64 = 0;
    let mut n = iters.max(1);
    let one: u64 = std::hint::black_box(1);
    // SAFETY: register-only arithmetic, no memory or stack access.
    unsafe {
        std::arch::asm!(
            "2:",
            "add {x}, {one}", "add {x}, {one}", "add {x}, {one}", "add {x}, {one}",
            "add {x}, {one}", "add {x}, {one}", "add {x}, {one}", "add {x}, {one}",
            "dec {n}",
            "jnz 2b",
            x = inout(reg) x,
            n = inout(reg) n,
            one = in(reg) one,
            options(nomem, nostack),
        );
    }
    let _ = n;
    x
}

#[cfg(target_arch = "aarch64")]
fn add_chain(iters: u64) -> u64 {
    let mut x: u64 = 0;
    let mut n = iters.max(1);
    // SAFETY: register-only arithmetic, no memory or stack access.
    unsafe {
        std::arch::asm!(
            "2:",
            "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
            "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            x = inout(reg) x,
            n = inout(reg) n,
            options(nomem, nostack),
        );
    }
    let _ = n;
    x
}

/// Portable fallback: not a reliable clock (the compiler may fold it), reported as such.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn add_chain(iters: u64) -> u64 {
    let mut x = 0u64;
    for _ in 0..iters {
        x = std::hint::black_box(x + ADDS_PER_ITER);
    }
    x
}

fn method() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "inline asm: 8 dependent `add reg, reg` per iteration"
    } else if cfg!(target_arch = "aarch64") {
        "inline asm: 8 dependent `add xN, xN, #1` per iteration"
    } else {
        "portable loop (not trustworthy)"
    }
}

/// GHz of one timed chunk.
fn sample(iters: u64) -> f64 {
    let t = Instant::now();
    std::hint::black_box(add_chain(iters));
    (iters * ADDS_PER_ITER) as f64 / t.elapsed().as_nanos().max(1) as f64
}

/// Iterations that take about `target` on this core right now.
fn calibrate(target: Duration) -> u64 {
    let probe = 1_000_000u64;
    let ghz = sample(probe).max(0.1);
    ((target.as_nanos() as f64 * ghz / ADDS_PER_ITER as f64) as u64).max(1000)
}

/// Spin for `warm`, then take ~50 ms samples for `window`. Returns the samples.
fn spin_and_sample(warm: Duration, window: Duration) -> Vec<f64> {
    let iters = calibrate(Duration::from_millis(50));
    let t = Instant::now();
    while t.elapsed() < warm {
        sample(iters);
    }
    let t = Instant::now();
    let mut v = Vec::new();
    while t.elapsed() < window || v.is_empty() {
        v.push(sample(iters));
    }
    v
}

/// Spin pinned to `c` for `warm`, then sample for `window`.
fn on_core(c: LogicalCpu, warm: Duration, window: Duration) -> Vec<f64> {
    std::thread::scope(|s| {
        s.spawn(|| {
            sys::pin_current_thread(c);
            spin_and_sample(warm, window)
        })
        .join()
        .unwrap_or_default()
    })
}

fn first(v: &[f64]) -> f64 {
    v.first().copied().unwrap_or(0.0)
}

fn last(v: &[f64]) -> f64 {
    v.last().copied().unwrap_or(0.0)
}

pub fn run(cpu: &CpuInfo, mode: Mode, log: &dyn Fn(&str)) -> Option<ClockResult> {
    let t = Instant::now();
    let cores: Vec<LogicalCpu> = cpu.one_per_core();
    let first_core = *cores.first()?;

    // 1. Each core alone, round-robin short visits; keep each core's best visit.
    log("clock: each core alone (round-robin)");
    let rounds = mode.pick(3, 2);
    let visit = mode.pick(Duration::from_millis(100), Duration::from_millis(60));
    let mut single_per_core = vec![0.0f64; cores.len()];
    for _ in 0..rounds {
        for (i, &c) in cores.iter().enumerate() {
            let m = median(&on_core(c, Duration::from_millis(40), visit));
            single_per_core[i] = single_per_core[i].max(m);
        }
    }
    let fastest =
        single_per_core.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(first_core, |(i, _)| cores[i]);

    // 2. Sustained single core on the fastest core.
    let single_window = mode.pick(Duration::from_secs(6), Duration::from_millis(1500));
    log(&format!("clock: one core sustained ({:.1} s)", single_window.as_secs_f64()));
    let t1 = Instant::now();
    let single = on_core(fastest, Duration::ZERO, single_window);
    let single_seconds = t1.elapsed().as_secs_f64();

    // 3. All physical cores at once, sustained.
    let window = mode.pick(Duration::from_secs(8), Duration::from_millis(1500));
    log(&format!("clock: all {} physical cores sustained ({:.1} s)", cores.len(), window.as_secs_f64()));
    let t2 = Instant::now();
    let barrier = Barrier::new(cores.len());
    let per_core: Vec<Vec<f64>> = std::thread::scope(|s| {
        let hs: Vec<_> = cores
            .iter()
            .map(|&c| {
                let b = &barrier;
                s.spawn(move || {
                    sys::pin_current_thread(c);
                    b.wait();
                    spin_and_sample(Duration::ZERO, window)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    let all_seconds = t2.elapsed().as_secs_f64();
    let medians: Vec<f64> = per_core.iter().map(|v| median(v)).collect();
    let firsts: Vec<f64> = per_core.iter().map(|v| first(v)).collect();
    let lasts: Vec<f64> = per_core.iter().map(|v| last(v)).collect();

    Some(ClockResult {
        method: method(),
        single_per_core,
        single_core_logical_id: fastest.flat_id(),
        single_ghz_first: first(&single),
        single_ghz_median: median(&single),
        single_ghz_last: last(&single),
        single_seconds,
        all_threads: cores.len(),
        all_ghz_first: median(&firsts),
        all_ghz_median: median(&medians),
        all_ghz_last: median(&lasts),
        all_ghz_min_core: medians.iter().copied().fold(f64::INFINITY, f64::min),
        all_ghz_max_core: medians.iter().copied().fold(0.0, f64::max),
        all_per_core: medians,
        all_seconds,
        seconds: t.elapsed().as_secs_f64(),
    })
}
