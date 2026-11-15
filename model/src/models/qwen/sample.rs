//! sample.rs - how the Qwen engine picks the next token: greedy at temperature 0, or sampled at temperature t from
//! the top_k / top_p set, with a seeded generator so a seed repeats a reply. Presence and frequency penalties (as
//! OpenAI defines them) first lower the logits of tokens the reply has already picked, so every pick (decode's, and
//! each position of a draft check, in order) sees exactly the tokens picked before it.

/// How the next token is picked: greedy (temperature 0), or sampled at temperature t from the top_k / top_p set.
#[derive(Clone)]
pub struct Sampling {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub rng: u64,
    presence: f32,
    frequency: f32,
    /// how many times each token has been picked (kept only with a penalty)
    counts: std::collections::HashMap<u32, u32>,
    /// the penalised copy of a row
    row: Vec<f32>,
}

impl Sampling {
    pub fn new(temp: f32, top_p: f32, top_k: usize, rng: u64) -> Sampling {
        let (presence, frequency, counts, row) = (0.0, 0.0, Default::default(), Vec::new());
        Sampling { temp, top_p, top_k, rng, presence, frequency, counts, row }
    }

    /// Presence and frequency penalties (0 for none).
    pub fn penalties(&mut self, presence: f32, frequency: f32) {
        self.presence = presence;
        self.frequency = frequency;
    }

    pub fn penalised(&self) -> bool {
        self.presence != 0.0 || self.frequency != 0.0
    }

    /// The next token from logits l: with a penalty, from l less each picked token's penalty, and then counted.
    pub fn pick(&mut self, l: &[f32]) -> u32 {
        if !self.penalised() {
            return self.pick_from(l);
        }
        let mut row = std::mem::take(&mut self.row);
        row.clear();
        row.extend_from_slice(l);
        for (&t, &n) in &self.counts {
            if let Some(v) = row.get_mut(t as usize) {
                *v -= self.presence + self.frequency * n as f32;
            }
        }
        let t = self.pick_from(&row);
        self.row = row;
        *self.counts.entry(t).or_insert(0) += 1;
        t
    }

    fn pick_from(&mut self, l: &[f32]) -> u32 {
        if self.temp <= 0.0 {
            return crate::ops::argmax(l) as u32;
        }
        // top_k 0 means no top-k cut (the OpenAI-style "off"): every token, then top_p
        let k = if self.top_k == 0 { l.len() } else { self.top_k.min(l.len()) };
        // value descending, equal values by the lower id (ops::by_rank, the same order as pipe::topk), so the piped
        // top-K row and the full row pick the same set even on an exact tie at the k-th value
        let c = crate::ops::top_k(l, k);
        let mx = c[0].1;
        let mut ps: Vec<f64> = c.iter().map(|x| (((x.1 - mx) / self.temp) as f64).exp()).collect();
        let z: f64 = ps.iter().sum();
        let (mut cum, mut n) = (0.0f64, 0usize);
        for p in ps.iter_mut() {
            *p /= z;
            cum += *p;
            n += 1;
            if cum >= self.top_p as f64 {
                break;
            }
        }
        // xorshift64*: a uniform value in [0, cum)
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let u = (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * cum;
        let mut acc = 0.0;
        for (i, p) in ps.iter().take(n).enumerate() {
            acc += p;
            if u < acc {
                return c[i].0;
            }
        }
        c[n - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pick by the plain method (every logit copied into a vector, selected, sorted): the reference.
    fn pick_full(s: &mut Sampling, l: &[f32]) -> u32 {
        let mut c: Vec<(u32, f32)> = l.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        let k = if s.top_k == 0 { c.len() } else { s.top_k.min(c.len()) };
        c.select_nth_unstable_by(k - 1, crate::ops::by_rank);
        c.truncate(k);
        c.sort_by(crate::ops::by_rank);
        let mx = c[0].1;
        let mut ps: Vec<f64> = c.iter().map(|x| (((x.1 - mx) / s.temp) as f64).exp()).collect();
        let z: f64 = ps.iter().sum();
        let (mut cum, mut n) = (0.0f64, 0usize);
        for p in ps.iter_mut() {
            *p /= z;
            cum += *p;
            n += 1;
            if cum >= s.top_p as f64 {
                break;
            }
        }
        s.rng ^= s.rng >> 12;
        s.rng ^= s.rng << 25;
        s.rng ^= s.rng >> 27;
        let u = (s.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * cum;
        let mut acc = 0.0;
        for (i, p) in ps.iter().take(n).enumerate() {
            acc += p;
            if u < acc {
                return c[i].0;
            }
        }
        c[n - 1].0
    }

    fn logits(seed: &mut u64, n: usize, coarse: bool) -> Vec<f32> {
        (0..n)
            .map(|i| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                let x = ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 30.0 - 15.0;
                // coarse: many exact ties (and zeros of both signs) at every rank
                if coarse { if i % 97 == 0 { -0.0 } else { (x * 2.0).round() / 2.0 } } else { x }
            })
            .collect()
    }

    /// top_k's pick against the full select-and-sort: the same token on every draw, for top_k 1..64 (the scan) and
    /// above, ties and signed zeros included.
    #[test]
    fn topk_pick_same() {
        let mut seed = 0x51ab_c0de_2468_ace1u64;
        for (n, coarse) in [(1000usize, false), (1000, true), (248_320, false), (248_320, true)] {
            let l = logits(&mut seed, n, coarse);
            for &k in &[1usize, 2, 5, 20, 40, 64, 65, 200] {
                for &(temp, top_p) in &[(0.7f32, 0.8f32), (1.0, 1.0), (0.2, 0.95), (1.5, 0.3)] {
                    let mut a = Sampling::new(temp, top_p, k, 0x1234_5678 ^ k as u64);
                    let mut b = a.clone();
                    for d in 0..(if n > 10_000 { 8 } else { 40 }) {
                        assert_eq!(
                            a.pick(&l),
                            pick_full(&mut b, &l),
                            "n {n} coarse {coarse} k {k} t {temp} p {top_p} draw {d}"
                        );
                    }
                }
            }
        }
    }

    /// Ignored by default: one sampled pick over the 35B's 248,320 logits (top_k 20), the full
    /// select against top_k.
    #[test]
    #[ignore]
    fn bench_pick() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let l = logits(&mut seed, 248_320, false);
        let reps = 200;
        let mut s = Sampling::new(0.7, 0.8, 20, 7);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(pick_full(&mut s, &l));
        }
        let full = t.elapsed().as_secs_f64() / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(s.pick(&l));
        }
        let scan = t.elapsed().as_secs_f64() / reps as f64;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(crate::ops::argmax(&l));
        }
        let am = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "pick over 248320 logits, top_k 20: full select {:.3} ms, top_k scan {:.3} ms; argmax {:.3} ms",
            full * 1e3,
            scan * 1e3,
            am * 1e3
        );
    }
}
