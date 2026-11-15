//! ram.rs - RAM read bandwidth: the decode ceiling and the best thread count.
//!
//! One 1 GiB buffer, touched first. For each thread count, threads are pinned one per physical core (fastest first) and
//! sum 4 MiB chunks with wide loads, taking the next chunk from a shared counter, so one preempted thread cannot
//! stretch the pass (this measures the bus ceiling, not straggler sensitivity). Time per pass = latest finish -
//! earliest start across threads (worker-side timestamps, so the coordinating thread's wake-up latency is not counted).
//! Configurations are visited round-robin (a warm-up pass + 2 timed passes per visit), so a burst of background
//! activity cannot sink one configuration.

use super::{Mode, median};
use crate::inventory::cpu::CpuInfo;
use crate::sys::{self, AlignedBuf, LogicalCpu};
use serde::Serialize;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const PASSES_PER_VISIT: usize = 2;
const QUIT: usize = usize::MAX;
const CHUNK: usize = 4 << 20;

#[derive(Debug, Clone, Serialize)]
pub struct RamRow {
    pub threads: usize,
    pub placement: &'static str,
    pub gbs_median: f64,
    pub gbs_min: f64,
    pub gbs_max: f64,
    pub passes: usize,
    pub all_pinned: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RamResult {
    pub buffer_bytes: usize,
    pub kernel: &'static str,
    /// GB = 1e9 bytes.
    pub rows: Vec<RamRow>,
    pub best_threads: usize,
    pub best_gbs: f64,
    /// Fewest threads within 5% of the best. A pure read needs fewer cores to saturate the bus than a compute kernel
    /// (Q4_K matvec) does.
    pub saturates_at_threads: usize,
    pub seconds: f64,
}

#[derive(Clone, Copy)]
enum Kernel {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Portable,
}

impl Kernel {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return Kernel::Avx2;
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Kernel::Neon;
        }
        #[allow(unreachable_code)]
        Kernel::Portable
    }

    fn name(self) -> &'static str {
        match self {
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => "avx2 (4x256-bit loads/iter)",
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => "neon (4x128-bit loads/iter)",
            Kernel::Portable => "portable u64",
        }
    }

    fn sum(self, buf: &[u8]) -> u64 {
        match self {
            // SAFETY: Avx2 is only chosen after runtime detection.
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { sum_avx2(buf) },
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => sum_neon(buf),
            Kernel::Portable => sum_portable(buf),
        }
    }
}

fn sum_portable(buf: &[u8]) -> u64 {
    let mut acc = [0u64; 4];
    let chunks = buf.chunks_exact(32);
    let rem = chunks.remainder();
    for c in chunks {
        for (i, a) in acc.iter_mut().enumerate() {
            let w = u64::from_ne_bytes(c[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
            *a = a.wrapping_add(w);
        }
    }
    rem.iter().fold(acc.iter().fold(0u64, |s, &a| s.wrapping_add(a)), |s, &b| s.wrapping_add(b as u64))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn sum_avx2(buf: &[u8]) -> u64 {
    use std::arch::x86_64::*;
    let (mut a0, mut a1, mut a2, mut a3) =
        (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
    let chunks = buf.chunks_exact(128);
    let rem = chunks.remainder();
    for c in chunks {
        let p = c.as_ptr() as *const __m256i;
        // SAFETY: `c` is exactly 128 bytes = four unaligned 256-bit loads.
        unsafe {
            a0 = _mm256_add_epi64(a0, _mm256_loadu_si256(p));
            a1 = _mm256_add_epi64(a1, _mm256_loadu_si256(p.add(1)));
            a2 = _mm256_add_epi64(a2, _mm256_loadu_si256(p.add(2)));
            a3 = _mm256_add_epi64(a3, _mm256_loadu_si256(p.add(3)));
        }
    }
    let s = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
    let mut lanes = [0u64; 4];
    // SAFETY: lanes is 32 bytes.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, s) };
    lanes.iter().fold(sum_portable(rem), |x, &l| x.wrapping_add(l))
}

#[cfg(target_arch = "aarch64")]
fn sum_neon(buf: &[u8]) -> u64 {
    use std::arch::aarch64::*;
    // SAFETY: NEON is mandatory on aarch64; every load reads 16 bytes inside a 64-byte chunk.
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0));
        let chunks = buf.chunks_exact(64);
        let rem = chunks.remainder();
        for c in chunks {
            let p = c.as_ptr() as *const u64;
            a0 = vaddq_u64(a0, vld1q_u64(p));
            a1 = vaddq_u64(a1, vld1q_u64(p.add(2)));
            a2 = vaddq_u64(a2, vld1q_u64(p.add(4)));
            a3 = vaddq_u64(a3, vld1q_u64(p.add(6)));
        }
        let s = vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3));
        sum_portable(rem).wrapping_add(vgetq_lane_u64::<0>(s)).wrapping_add(vgetq_lane_u64::<1>(s))
    }
}

/// Fill the buffer in parallel so every page is committed before timing.
fn touch(buf: &mut AlignedBuf, threads: usize) {
    let len = buf.len();
    let per = len.div_ceil(threads.max(1)).next_multiple_of(4096);
    std::thread::scope(|s| {
        for (i, chunk) in buf.as_mut_slice().chunks_mut(per).enumerate() {
            s.spawn(move || {
                for (j, w) in chunk.chunks_mut(8).enumerate() {
                    let v = ((i as u64) << 40 | j as u64).to_ne_bytes();
                    w.copy_from_slice(&v[..w.len()]);
                }
            });
        }
    });
}

/// One visit: a warm-up pass then PASSES_PER_VISIT timed passes with `cpus.len()` pinned threads.
fn measure(buf: &AlignedBuf, cpus: &[LogicalCpu], kernel: Kernel) -> (Vec<f64>, bool) {
    let n = cpus.len();
    let chunks = buf.len() / CHUNK;
    let bytes = (chunks * CHUNK) as f64;
    let base = Instant::now();
    let next = AtomicUsize::new(0);
    let go = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let pinned = AtomicUsize::new(0);
    let starts: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let ends: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let mut gbs = Vec::new();

    std::thread::scope(|s| {
        for (i, &cpu) in cpus.iter().enumerate() {
            let (next, go, done, pinned, starts, ends) = (&next, &go, &done, &pinned, &starts, &ends);
            let data = buf.as_slice();
            s.spawn(move || {
                if sys::pin_current_thread(cpu) {
                    pinned.fetch_add(1, Ordering::Relaxed);
                }
                let mut seen = 0;
                loop {
                    let g = loop {
                        let g = go.load(Ordering::Acquire);
                        if g != seen {
                            break g;
                        }
                        std::hint::spin_loop();
                    };
                    if g == QUIT {
                        break;
                    }
                    seen = g;
                    let t0 = base.elapsed().as_nanos() as u64;
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= chunks {
                            break;
                        }
                        black_box(kernel.sum(black_box(&data[c * CHUNK..(c + 1) * CHUNK])));
                    }
                    let t1 = base.elapsed().as_nanos() as u64;
                    starts[i].store(t0, Ordering::Relaxed);
                    ends[i].store(t1, Ordering::Relaxed);
                    done.fetch_add(1, Ordering::AcqRel);
                }
            });
        }
        for pass in 1..=PASSES_PER_VISIT + 1 {
            done.store(0, Ordering::Release);
            next.store(0, Ordering::Release);
            go.store(pass, Ordering::Release);
            while done.load(Ordering::Acquire) < n {
                std::thread::sleep(Duration::from_micros(500));
            }
            let t0 = starts.iter().map(|a| a.load(Ordering::Relaxed)).min().unwrap_or(0);
            let t1 = ends.iter().map(|a| a.load(Ordering::Relaxed)).max().unwrap_or(0);
            if pass > 1 && t1 > t0 {
                gbs.push(bytes / (t1 - t0) as f64); // bytes per ns = GB/s
            }
        }
        go.store(QUIT, Ordering::Release);
    });
    (gbs, pinned.load(Ordering::Relaxed) == n)
}

pub fn run(cpu: &CpuInfo, mode: Mode, log: &dyn Fn(&str)) -> Option<RamResult> {
    let t = Instant::now();
    let kernel = Kernel::detect();
    let mut buf = AlignedBuf::new(1 << 30, 1 << 21).or_else(|| AlignedBuf::new(512 << 20, 1 << 21))?;
    log(&format!("RAM: touching {} MiB buffer", buf.len() >> 20));
    touch(&mut buf, cpu.logical_cpus.max(1));

    let per_core = cpu.one_per_core();
    let mut counts: Vec<usize> =
        std::iter::successors(Some(1usize), |&c| Some(c * 2)).take_while(|&c| c <= per_core.len()).collect();
    if counts.last() != Some(&per_core.len()) && !per_core.is_empty() {
        counts.push(per_core.len());
    }
    let mut configs: Vec<(Vec<LogicalCpu>, &'static str)> =
        counts.iter().map(|&c| (per_core[..c].to_vec(), "one per physical core")).collect();
    let all = cpu.all_logical();
    if all.len() > per_core.len() {
        configs.push((all, "all logical CPUs (SMT)"));
    }
    let rounds = mode.pick(5, 3);
    let cap = mode.pick(Duration::from_secs(15), Duration::from_secs(6));
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
    let mut pinned = vec![true; configs.len()];
    log(&format!(
        "RAM: {} configurations x {rounds} rounds (threads {})",
        configs.len(),
        configs.iter().map(|c| c.0.len().to_string()).collect::<Vec<_>>().join("/")
    ));
    'rounds: for _ in 0..rounds {
        for (i, (cpus, _)) in configs.iter().enumerate() {
            let (g, p) = measure(&buf, cpus, kernel);
            samples[i].extend(g);
            pinned[i] &= p;
            if t.elapsed() > cap {
                break 'rounds;
            }
        }
    }
    let rows: Vec<RamRow> = configs
        .iter()
        .zip(samples.iter().zip(&pinned))
        .map(|((cpus, placement), (g, &all_pinned))| RamRow {
            threads: cpus.len(),
            placement,
            gbs_median: median(g),
            gbs_min: g.iter().copied().fold(f64::INFINITY, f64::min),
            gbs_max: g.iter().copied().fold(0.0, f64::max),
            passes: g.len(),
            all_pinned,
        })
        .collect();
    let best = rows.iter().max_by(|a, b| a.gbs_median.total_cmp(&b.gbs_median))?;
    let saturates =
        rows.iter().filter(|r| r.gbs_median >= 0.95 * best.gbs_median).map(|r| r.threads).min().unwrap_or(best.threads);
    Some(RamResult {
        buffer_bytes: buf.len(),
        kernel: kernel.name(),
        best_threads: best.threads,
        best_gbs: best.gbs_median,
        saturates_at_threads: saturates,
        rows,
        seconds: t.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_agree() {
        let data: Vec<u8> = (0..100_003u32).map(|i| (i * 7 + 3) as u8).collect();
        let want = sum_portable(&data);
        assert_eq!(Kernel::detect().sum(&data), want);
        // Simple reference on aligned u64 words + tail bytes.
        let words = data.len() / 32 * 32;
        let mut r = 0u64;
        for c in data[..words].chunks_exact(8) {
            r = r.wrapping_add(u64::from_ne_bytes(c.try_into().unwrap()));
        }
        for &b in &data[words..] {
            r = r.wrapping_add(b as u64);
        }
        assert_eq!(want, r);
    }
}
