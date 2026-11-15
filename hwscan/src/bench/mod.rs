//! bench - section 3: speed tests. Each answers one question about the machine and has a time cap.

pub mod clock;
pub mod disk;
pub mod ram;

use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Full,
    Quick,
}

impl Mode {
    /// Pick the full or quick variant of a budget.
    pub fn pick<T>(self, full: T, quick: T) -> T {
        match self {
            Mode::Full => full,
            Mode::Quick => quick,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchResults {
    pub mode: &'static str,
    pub ram: Option<ram::RamResult>,
    pub clock: Option<clock::ClockResult>,
    pub drives: Vec<disk::DriveResult>,
    pub gpu: &'static str,
    pub total_seconds: f64,
}

pub fn median(v: &[f64]) -> f64 {
    percentile(v, 50.0)
}

/// Nearest-rank percentile; 0.0 for an empty slice.
pub fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * s.len() as f64).ceil() as usize;
    s[rank.clamp(1, s.len()) - 1]
}

pub fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

/// Small deterministic PRNG (xorshift64*), enough for picking offsets.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 100.0), 4.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }
}
