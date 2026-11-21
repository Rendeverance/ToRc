//! plan.rs - the MTP draft planner and the draft confidence, shared by torc-run and torc-server.

/// DraftPlan: how many of the drafts so far to verify in the next pass, chosen for the fewest expected seconds per
/// token.
///
/// A pass verifies the token and k drafts in one block of 1 + k. It yields 1 token plus every draft up to the first
/// rejected one, so its expected yield is 1 + p_1 + .. + p_k, where p_i is the chance that drafts 1..i all survive. Its
/// time is the verify block's, t(1 + k), plus the chain that made drafts 2..k, (k - 1) x chain (the first draft comes
/// with the previous pass's last MTP step). plan() prices every k as time / yield and takes the cheapest.
/// - chain: a forgetting mean of the timed draft steps (weight 0.97^age, as for t(n)).
/// - p_i: the drafts' raw survival s_i (the product of their confidences, model::plan::conf), put through a logistic
///   calibration p = 1 / (1 + exp(-(a logit(s) + b))) that is fitted online to the outcomes (Platt scaling), starting
///   at the identity (a = 1, b = 0);
/// - t(n), for n >= 2: a straight line alpha + beta (n - 1), fitted by weighted least squares to the last passes'
///   verify times for blocks of 2 or more (with weight 0.97^age), so a size never run is priced from the line. With
///   fewer than two such sizes seen, beta = 0.15 alpha. t(1), a pass with no drafts, runs the decode path, whose time
///   is not on that line, so it gets its own weighted mean, using the line's value until one is seen. Before any
///   pass, alpha = 1 (the units cancel in the ratio).
/// - t(1) is never above t(2) once a block of 2 or more has been timed: a pass of one token does less work than a
///   block of two. Without that, one slow plain pass (the first after a load can take a second) makes every plain
///   pass look dear, the planner then drafts on every pass, and no new plain pass ever comes to correct it.
/// - A pass slower than STALL times what the planner expects for its size is recorded at STALL times that: a pass
///   held up by the system says nothing about the price of a block, and one such time among a few blocks would tilt
///   the line for the next 64 passes.
pub struct DraftPlan {
    /// (n, verify seconds) for the last passes, oldest first
    samples: std::collections::VecDeque<(usize, f64)>,
    /// the draft steps timed so far, as forgetting sums: (sum of 0.97^age x seconds, sum of 0.97^age)
    chain_sums: (f64, f64),
    /// passes in a row recorded without drafts
    plain_run: u32,
    /// the calibration's slope and offset on logit(s)
    pub a: f64,
    pub b: f64,
    /// the drafts whose outcomes the calibration has seen
    pub seen: u64,
}

const SAMPLES: usize = 64;
const FORGET: f64 = 0.97;
/// After this many passes in a row without drafts, pick() verifies one anyway, so the price of drafting is measured
/// again rather than trusted for good.
const EXPLORE: u32 = 24;
/// A pass this many times slower than expected counts as held up (see DraftPlan).
const STALL: f64 = 3.0;

fn logit(s: f64) -> f64 {
    let s = s.clamp(1e-6, 1.0 - 1e-6);
    (s / (1.0 - s)).ln()
}

impl DraftPlan {
    pub fn new() -> DraftPlan {
        DraftPlan { samples: Default::default(), chain_sums: (0.0, 0.0), plain_run: 0, a: 1.0, b: 0.0, seen: 0 }
    }

    /// One draft step's time: the forgetting mean of the timed steps (0 before any).
    pub fn chain(&self) -> f64 {
        let (s, w) = self.chain_sums;
        if w > 0.0 { s / w } else { 0.0 }
    }

    /// Adds one timed draft step.
    pub fn time_chain(&mut self, secs: f64) {
        let (s, w) = self.chain_sums;
        self.chain_sums = (s * FORGET + secs, w * FORGET + 1.0);
    }

    /// The calibrated chance for a raw survival s.
    pub fn p(&self, s: f64) -> f64 {
        1.0 / (1.0 + (-(self.a * logit(s) + self.b)).exp())
    }

    /// (alpha, beta) of the verify-time line.
    pub fn line(&self) -> (f64, f64) {
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut sizes = 0u32;
        let k = self.samples.len();
        for (i, &(n, t)) in self.samples.iter().enumerate() {
            if n < 2 {
                continue;
            }
            let w = FORGET.powi((k - 1 - i) as i32);
            let x = (n - 1) as f64;
            sw += w;
            sx += w * x;
            sy += w * t;
            sxx += w * x * x;
            sxy += w * x * t;
            sizes |= 1 << n.min(31);
        }
        if sw == 0.0 {
            return (1.0, 0.15);
        }
        let den = sw * sxx - sx * sx;
        if sizes.count_ones() >= 2 && den > 1e-12 {
            let beta = ((sw * sxy - sx * sy) / den).max(0.0);
            let alpha = ((sy - beta * sx) / sw).max(1e-6);
            return (alpha, beta);
        }
        // only one size seen: its mean, with the slope assumed
        let mean_t = sy / sw;
        let mean_x = sx / sw;
        let alpha = mean_t / (1.0 + 0.15 * mean_x);
        (alpha, 0.15 * alpha)
    }

    /// The verify time of a block of n.
    pub fn cost(&self, n: usize) -> f64 {
        let (a, b) = self.line();
        if n == 1 {
            let k = self.samples.len();
            let (mut sw, mut st) = (0.0, 0.0);
            for (i, &(m, t)) in self.samples.iter().enumerate() {
                if m == 1 {
                    let w = FORGET.powi((k - 1 - i) as i32);
                    sw += w;
                    st += w * t;
                }
            }
            if sw > 0.0 {
                let t1 = st / sw;
                return if self.has(|m| m >= 2) { t1.min(a + b) } else { t1 };
            }
        }
        a + b * (n - 1) as f64
    }

    /// Whether a pass of a size that passes f has been recorded.
    fn has(&self, f: impl Fn(usize) -> bool) -> bool {
        self.samples.iter().any(|&(m, _)| f(m))
    }

    /// How many of the drafts (raw survivals surv, in order) are worth verifying: 0..=surv.len(), the cheapest in
    /// seconds per expected token (the smaller count on a tie).
    pub fn plan(&self, surv: &[f64]) -> usize {
        let chain = self.chain();
        // expected tokens from verifying the first k drafts, k = 0..=surv.len()
        let tokens = std::iter::once(1.0).chain(surv.iter().scan(1.0, |y, &s| {
            *y += self.p(s);
            Some(*y)
        }));
        let price = tokens.enumerate().map(|(k, y)| (self.cost(1 + k) + chain * k.saturating_sub(1) as f64) / y);
        price.enumerate().fold((0, f64::INFINITY), |best, (k, c)| if c < best.1 { (k, c) } else { best }).0
    }

    /// plan(), except after EXPLORE plain passes in a row, when it takes one draft if there is one.
    pub fn pick(&self, surv: &[f64]) -> usize {
        match self.plan(surv) {
            0 if !surv.is_empty() && self.plain_run >= EXPLORE => 1,
            k => k,
        }
    }

    /// Records a pass's outcome: the raw survivals of the drafts it verified, how many were accepted, its block size
    /// and the verify's seconds.
    pub fn record(&mut self, surv: &[f64], acc: usize, n: usize, secs: f64) {
        // what the planner expects for a block of n, once it has timed one of that kind (plain, or 2 or more)
        let secs = if self.has(|m| (m >= 2) == (n >= 2)) { secs.min(STALL * self.cost(n)) } else { secs };
        if self.samples.len() == SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back((n, secs));
        self.plain_run = if n > 1 { 0 } else { self.plain_run + 1 };
        // one gradient step of the log loss per draft, where draft i survived when i < acc
        let lr = 0.05 / (1.0 + self.seen as f64 / 2000.0);
        for (i, &s) in surv.iter().enumerate() {
            let y = if i < acc { 1.0 } else { 0.0 };
            let x = logit(s);
            let e = y - self.p(s);
            self.a = (self.a + lr * e * x).clamp(0.2, 5.0);
            self.b = (self.b + lr * e).clamp(-6.0, 6.0);
            self.seen += 1;
        }
    }

    /// Saves the calibration to keep it between runs (TORC_DRAFTS): "TORCDPL1", a, b, seen.
    pub fn save(&self, path: &str) {
        let mut b = b"TORCDPL1".to_vec();
        b.extend(self.a.to_le_bytes());
        b.extend(self.b.to_le_bytes());
        b.extend(self.seen.to_le_bytes());
        let _ = std::fs::write(path, b);
    }

    pub fn load(&mut self, path: &str) {
        let Ok(b) = std::fs::read(path) else { return };
        if b.len() < 32 || &b[..8] != b"TORCDPL1" {
            return;
        }
        let f = |o: usize| f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        self.a = f(8).clamp(0.2, 5.0);
        self.b = f(16).clamp(-6.0, 6.0);
        self.seen = u64::from_le_bytes(b[24..32].try_into().unwrap());
    }
}

impl Default for DraftPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// The softmax probability of the first maximum of l, used as the draft confidence: an f32 exp sum eight lanes at a
/// time, written so that the compiler vectorises it.
pub fn conf(l: &[f32]) -> f64 {
    let ch = l.chunks_exact(8);
    let mut mx = [f32::MIN; 8];
    for c in ch.clone() {
        for i in 0..8 {
            mx[i] = if c[i] > mx[i] { c[i] } else { mx[i] };
        }
    }
    let m = ch.remainder().iter().chain(&mx).fold(f32::MIN, |a, &v| if v > a { v } else { a });
    let mut acc = [0.0f32; 8];
    for c in ch.clone() {
        for i in 0..8 {
            acc[i] += crate::act::expf(c[i] - m);
        }
    }
    let s = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
    1.0 / ch.remainder().iter().fold(s as f64, |a, &v| a + ((v - m) as f64).exp())
}

/// A draft planner, as the servers and torc-run drive one.
pub trait Plans: Send {
    /// While drafting: how many of the drafts so far are worth verifying (drafting goes on while that is all of them).
    fn plan(&self, surv: &[f64]) -> usize;
    /// The count to verify, once drafting has stopped.
    fn pick(&self, surv: &[f64]) -> usize;
    /// Records a pass's outcome: the drafts' raw survivals, how many were accepted, the block size and the verify's
    /// seconds.
    fn record(&mut self, surv: &[f64], acc: usize, n: usize, verify_secs: f64);
    fn chain(&self) -> f64;
    /// Adds one timed draft step.
    fn time_chain(&mut self, secs: f64);
    /// The predicted time of a block of n.
    fn cost(&self, n: usize) -> f64;
    /// One line on the planner's state, for the logs.
    fn report(&self, max_n: usize) -> String;
    fn save(&self, path: &str);
    fn load(&mut self, path: &str);
}

impl Plans for DraftPlan {
    fn plan(&self, surv: &[f64]) -> usize {
        DraftPlan::plan(self, surv)
    }
    fn pick(&self, surv: &[f64]) -> usize {
        DraftPlan::pick(self, surv)
    }
    fn record(&mut self, surv: &[f64], acc: usize, n: usize, verify_secs: f64) {
        DraftPlan::record(self, surv, acc, n, verify_secs)
    }
    fn chain(&self) -> f64 {
        DraftPlan::chain(self)
    }
    fn time_chain(&mut self, secs: f64) {
        DraftPlan::time_chain(self, secs)
    }
    fn cost(&self, n: usize) -> f64 {
        DraftPlan::cost(self, n)
    }
    fn report(&self, max_n: usize) -> String {
        let (a, b) = self.line();
        let t: Vec<String> = (1..=max_n).map(|n| format!("{n}:{:.1}", self.cost(n) * 1e3)).collect();
        format!(
            "planner DraftPlan t(n) ms {} (line {:.2} + {:.3} (n-1)) chain {:.2} ms, calibration a {:.2} b {:.2} over {} drafts",
            t.join(" "),
            a * 1e3,
            b * 1e3,
            DraftPlan::chain(self) * 1e3,
            self.a,
            self.b,
            self.seen
        )
    }
    fn save(&self, path: &str) {
        DraftPlan::save(self, path)
    }
    fn load(&mut self, path: &str) {
        DraftPlan::load(self, path)
    }
}

/// The draft planner.
pub fn planner() -> Box<dyn Plans> {
    Box::new(DraftPlan::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Passes as a laptop CPU times them: plain 40 ms, blocks 50 ms + 9 ms a row.
    fn steady(p: &mut DraftPlan, rounds: usize) {
        for _ in 0..rounds {
            p.record(&[], 0, 1, 0.040);
            p.record(&[0.9], 1, 2, 0.050);
            p.record(&[0.9, 0.9], 2, 3, 0.059);
        }
    }

    #[test]
    fn slow_first_plain_pass_is_not_trusted() {
        let mut p = DraftPlan::new();
        p.record(&[], 0, 1, 1.03);
        for _ in 0..8 {
            p.record(&[0.9], 1, 2, 0.050);
            p.record(&[0.9, 0.9], 2, 3, 0.059);
        }
        assert!(p.cost(1) <= p.cost(2) + 1e-12, "t(1) {} t(2) {}", p.cost(1), p.cost(2));
    }

    #[test]
    fn a_held_up_block_does_not_tilt_the_line() {
        let mut p = DraftPlan::new();
        steady(&mut p, 10);
        p.record(&[0.9, 0.9], 2, 3, 1.0);
        let (a, b) = p.line();
        // recorded as it came, the second's pass would make the slope about 0.14 s a row
        assert!((a + b - 0.050).abs() < 1e-3 && b < 0.040, "line {a} + {b} (n - 1)");
        // weak drafts still not worth a pass
        assert_eq!(p.plan(&[0.05]), 0);
    }

    #[test]
    fn steady_times_unchanged() {
        let mut p = DraftPlan::new();
        steady(&mut p, 10);
        assert!((p.cost(1) - 0.040).abs() < 1e-9);
        assert!((p.cost(2) - 0.050).abs() < 1e-6 && (p.cost(3) - 0.059).abs() < 1e-6);
        assert_eq!(p.plan(&[0.05]), 0);
        assert_eq!(p.plan(&[0.9, 0.9]), 2);
    }
}
