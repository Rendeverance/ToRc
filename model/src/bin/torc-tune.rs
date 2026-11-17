//! torc-tune <model.torc> [threads|auto] [ctx]: measures which prompt block size is best on this machine for this model
//! and layout (AUTOTUNE.md), then whether guided scheduling helps, whether to pin the threads and how many threads a
//! reply should use, and saves each where torc-server finds it (tune.rs). Run it with the server's environment:
//! TORC_NPU, and TORC_PIPE with the worker up for two machines. TORC_BLOCK still overrides the result.
//!
//! There are three fixed texts (prose, code and a story), each read as prompts of 600 and 1800 tokens (LENS) in each
//! candidate block size, and each prompt is followed by one decoded token. The candidates climb from the rule: the
//! rule, half and double it first, then one more doubling (or halving) at a time while each step is faster than the
//! last, up to the longest prompt rounded up to a power of two. In each stage a new candidate runs once to warm up
//! (discarded), then three passes in alternating order (A B C, C B A, A B C). A run's speed is the geometric mean of
//! its reading speeds, and a candidate's score is the median of its runs. A candidate only replaces the built-in rule's
//! block when it is better by more than the rule's own spread and by at least 2.5%, and no slower than the rule on any
//! text. Every candidate's decoded logits must match the rule's bit for bit on the CPU - the block size changes no
//! arithmetic there, so this catches a bug. With the NPU, each prompt must pick the same next token (see below). With
//! two machines the split is held where it was for every run, and the saved balance is put back afterwards.
//! AUTOTUNE.md's planned checks (a quiet machine first, the heat check at the end) are not built yet.

use serde_json::json;

/// The first prompt: plain prose with no markup, the same in every build (a changed text would make old scores
/// incomparable). It is repeated until it covers the longest prompt and its next token.
const TEXT: &str = "The lighthouse on the northern headland was built in the eighteen-forties, after a winter in which three \
ships were lost on the reef within sight of the harbour. The keepers who lived there kept a log of every night: the \
weather, the state of the lamp, the ships that passed and the ones that did not. Read today, the log is less a record \
of storms than of patience. Most nights nothing happened. The wick was trimmed, the lens was cleaned, the clockwork \
that turned the light was wound every few hours, and the keeper wrote a line that said so. The entries that stand out \
are the ones where something broke. A pane cracked in a gale and was covered with a board until the glazier could \
come out by boat. The clockwork jammed and the keeper turned the lens by hand until dawn, writing afterwards that his \
arms were no use for a week. A fog came down so thick that the light could not be seen from the gallery, and the fog \
bell was rung every half minute for two days. What the log shows, more than anything, is that a light which works \
every night is not a single achievement but thousands of small ones, each of them easy to skip and each of them \
noticed only when it is skipped. When the station was automated in the nineteen-eighties, the engineers who \
designed the new equipment read the old logs to learn how the light had failed in the past. They found that almost \
every failure had a warning in the days before it: a sound from the gears, a smell from the lamp, a draught that had \
not been there the week before. The keepers had noticed these things and written them down. The machines that \
replaced them were given sensors for the same signs, and an alarm that rang in an office on the mainland when one of \
them crossed a limit. The building still stands. The lamp room is open to visitors in the summer, and the last \
keeper's log lies open on a table by the window, at the page where he wrote that the light was now in other hands \
and that he hoped they would watch it as closely as he had.";

/// The second prompt: plain code (a chat about code routes to different experts than prose does), the same in every
/// build.
const CODE: &str = "use std::collections::HashMap;
use std::io::{self, BufRead, Write};

/// A word count over standard input: each line split on whitespace, lowercased, punctuation trimmed.
fn count(input: impl BufRead) -> io::Result<HashMap<String, usize>> {
    let mut counts = HashMap::new();
    for line in input.lines() {
        let line = line?;
        for word in line.split_whitespace() {
            let w: String = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if !w.is_empty() {
                *counts.entry(w).or_insert(0) += 1;
            }
        }
    }
    Ok(counts)
}

/// The `n` most frequent words, most frequent first; equal counts in alphabetical order.
fn top(counts: &HashMap<String, usize>, n: usize) -> Vec<(&str, usize)> {
    let mut v: Vec<(&str, usize)> = counts.iter().map(|(w, &c)| (w.as_str(), c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    v.truncate(n);
    v
}

struct Histogram {
    width: usize,
}

impl Histogram {
    fn render(&self, rows: &[(&str, usize)], out: &mut impl Write) -> io::Result<()> {
        let max = rows.iter().map(|r| r.1).max().unwrap_or(1).max(1);
        let pad = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
        for (word, c) in rows {
            let bar = (c * self.width + max - 1) / max;
            writeln!(out, \"{word:>pad$} {:<w$} {c}\", \"#\".repeat(bar), w = self.width)?;
        }
        Ok(())
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
    let counts = count(io::stdin().lock())?;
    let rows = top(&counts, n);
    let total: usize = counts.values().sum();
    let mut out = io::stdout().lock();
    writeln!(out, \"{} words, {} distinct\", total, counts.len())?;
    Histogram { width: 40 }.render(&rows, &mut out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_words_ignoring_case_and_punctuation() {
        let c = count(\"The cat, the HAT.\\nthe end\".as_bytes()).unwrap();
        assert_eq!(c[\"the\"], 3);
        assert_eq!(c[\"cat\"], 1);
        assert_eq!(top(&c, 1), vec![(\"the\", 3)]);
    }
}
";

/// The third prompt: a story with dialogue - the opening of Alice's Adventures in Wonderland (public domain), the same
/// text as the benchmark client's prompt.
const STORY: &str = include_str!("tune_story.txt");

/// The prompt lengths each candidate is read at: a chat's first message, and a longer one.
const LENS: [usize; 2] = [600, 1800];

fn fail(e: impl std::fmt::Display) -> ! {
    eprintln!("torc-tune: {e}");
    std::process::exit(1)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

fn main() {
    model::set_npu_env();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: torc-tune <model.torc> [threads|auto] [ctx]");
        std::process::exit(2);
    }
    let asked: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(0);
    let threads = model::pool::init(asked).threads as i32;
    let ctx: u32 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(8192);
    let tok = model::tok::Tokenizer::open(&a[1]).unwrap_or_else(|e| fail(e));
    let mut m = model::Model::load(&a[1], ctx).unwrap_or_else(|e| fail(e));
    let nv = m.n_vocab().unwrap_or_else(|e| fail(e));
    // two prompts, neither a multiple of any candidate: a prompt that is an exact multiple of a block has no tail,
    // which flatters that block size. TORC_TUNE_LENS=575,4678: other lengths
    let lens: Vec<usize> = model::switches::get("TORC_TUNE_LENS")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| LENS.to_vec())
        .into_iter()
        .filter(|&l| l > 0 && l < ctx as usize)
        .collect();
    if lens.is_empty() {
        fail(format!("context {ctx} too small for the {LENS:?}-token prompts"));
    }
    // the texts: a close call can depend on what is being read, so a candidate must be at least as fast as the rule on
    // each text. TORC_TUNE_TEXT=<file> uses that file's text alone
    let texts: Vec<(String, String)> = match model::switches::get("TORC_TUNE_TEXT") {
        Some(p) => vec![(p.clone(), std::fs::read_to_string(&p).unwrap_or_else(|e| fail(format!("{p}: {e}"))))],
        None => vec![("prose".into(), TEXT.into()), ("code".into(), CODE.into()), ("story".into(), STORY.into())],
    };
    let ids: Vec<Vec<u32>> = texts
        .iter()
        .map(|(_, t)| {
            let mut v = Vec::new();
            // long enough for the standard prompts and for the longer one a top-out race may add (up to the context)
            while v.len() <= (*lens.iter().max().unwrap()).max((ctx as usize).saturating_sub(1).min(16384)) {
                v.extend(tok.encode(t, false).unwrap_or_else(|e| fail(e)));
            }
            v
        })
        .collect();
    let nt = texts.len();
    let nl = m.h.n_layer as usize;
    let saved = m.piped().then(|| model::pipe::saved_split(nl));
    if m.piped() && !model::switches::given("TORC_PIPE_SPLIT") {
        // every candidate runs at the split the server would start at
        let s = model::pipe::split(nl);
        model::switches::set("TORC_PIPE_SPLIT", &s.to_string());
    }
    let rule = model::tune::block_rule();
    // The search climbs from the rule until it starts getting slower: the rule and its two neighbours (half and double)
    // first, then one more doubling (or halving) at a time while each step is faster than the one before. No fixed top:
    // a list that stops at 512 has models choosing 512, its top, when they read faster in bigger blocks. The ceiling is
    // the longest test prompt rounded up to a power of two (a larger block reads it the same way, so it cannot be
    // ranked) and the context; TORC_TUNE_LENS with a longer prompt lets it climb higher. The floor is 32 (the NPU's
    // smallest block).
    let longest = *lens.iter().max().unwrap();
    let ceil = longest.next_power_of_two().min(ctx as usize);
    let ok = |b: usize| b >= 32 && b <= ceil;
    eprintln!(
        "torc-tune: layout {}, threads {threads}, climbing from the rule {rule} (blocks 32..={ceil}), texts {:?}, \
        prompts of {lens:?}",
        model::tune::layout(),
        texts.iter().map(|t| t.0.as_str()).collect::<Vec<_>>()
    );
    // one run: each text's prompts read in blocks of b, each followed by its next token decoded. It returns the
    // geometric mean of the reading speeds for each text and over all of them (each prompt counts the same), and every
    // decoded row of logits (ls, n: the prompt lengths and how many of the texts; every stage but the top-out race uses
    // LENS and all texts)
    let mut run = |b: usize, ls: &[usize], n: usize| -> (Vec<f64>, f64, Vec<f32>) {
        let (mut per, mut lgs) = (Vec::with_capacity(n), Vec::with_capacity(nv * n * ls.len()));
        for tid in &ids[..n] {
            let mut lsum = 0.0f64;
            for &l in ls {
                m.clear_states();
                let t = std::time::Instant::now();
                m.prefill_blocks(&tid[..l], b, false, &mut |_, _| {}).unwrap_or_else(|e| fail(e));
                lsum += (l as f64 / t.elapsed().as_secs_f64()).ln();
                let mut lg = vec![0.0f32; nv];
                m.decode(tid[l], l as u32, &mut lg).unwrap_or_else(|e| fail(e));
                lgs.extend(lg);
            }
            per.push(lsum / ls.len() as f64);
        }
        let all = (per.iter().sum::<f64>() / n as f64).exp();
        (per.into_iter().map(f64::exp).collect(), all, lgs)
    };
    // the CPU path gives the same bits whatever the block. With the NPU, a block's rows only go to it from 32 rows up,
    // so a different block size can move a prompt's short tail between the NPU and the CPU and change the last bits.
    // There, each prompt must pick the same next token, and the largest logit difference is kept
    let npu = model::switches::on("TORC_NPU");
    let argmax = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap_or(0);
    // per block measured: its speeds, per-text speeds, bits identical to the rule's, same next tokens, largest
    // difference
    struct Seen {
        speeds: Vec<f64>,
        t: Vec<Vec<f64>>,
        exact: bool,
        token: bool,
        dmax: f32,
    }
    let mut seen: std::collections::BTreeMap<usize, Seen> = std::collections::BTreeMap::new();
    // what it is doing, a line a run, so the wait shows its progress
    let t0 = std::time::Instant::now();
    let at = move || format!("{:.0}s", t0.elapsed().as_secs_f64());
    eprintln!("  block {rule}: warming up (the rule)");
    let reference = run(rule, &lens, nt).2; // the rule's warm-up run, its logits the reference
    // one stage: each block not measured before runs once to warm up (discarded), then three passes in alternating
    // order (A B C, C B A, A B C) over the stage's blocks
    let mut stage = |set: &[usize], seen: &mut std::collections::BTreeMap<usize, Seen>| {
        for &b in set {
            if b != rule && !seen.contains_key(&b) {
                eprintln!("  block {b}: warming up ({})", at());
                run(b, &lens, nt);
            }
        }
        for pass in 0..3 {
            let order: Vec<usize> = if pass % 2 == 0 { set.to_vec() } else { set.iter().rev().copied().collect() };
            for b in order {
                let (per, tps, lg) = run(b, &lens, nt);
                eprintln!("  block {b}: pass {} of 3, {tps:.1} tokens/s ({})", pass + 1, at());
                let s = seen.entry(b).or_insert_with(|| Seen {
                    speeds: Vec::new(),
                    t: vec![Vec::new(); nt],
                    exact: true,
                    token: true,
                    dmax: 0.0,
                });
                s.speeds.push(tps);
                for (t, v) in per.into_iter().enumerate() {
                    s.t[t].push(v);
                }
                for (x, y) in lg.chunks(nv).zip(reference.chunks(nv)) {
                    if x.iter().zip(y).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        s.exact = false;
                        s.dmax = x.iter().zip(y).fold(s.dmax, |d, (a, b)| d.max((a - b).abs()));
                    }
                    if argmax(x) != argmax(y) {
                        s.token = false;
                    }
                }
            }
        }
    };
    let sc = |seen: &std::collections::BTreeMap<usize, Seen>, b: usize| median(&seen[&b].speeds);
    let first: Vec<usize> = [rule / 2, rule, rule * 2].into_iter().filter(|&b| b == rule || ok(b)).collect();
    stage(&first, &mut seen);
    // the margin to move, from the rule's own runs: at least 2.5% (a wider margin keeps settings that are measurably
    // slower), or the rule's spread if that is larger
    let spread = {
        let r = &seen[&rule].speeds;
        (r.iter().cloned().fold(f64::MIN, f64::max) - r.iter().cloned().fold(f64::MAX, f64::min)) / median(r)
    };
    let margin = spread.max(0.025);
    // a block may be used when it gives the rule's bits (or, with the NPU, its next tokens) and is no slower than the
    // rule on any text
    let tmed = |seen: &std::collections::BTreeMap<usize, Seen>, b: usize| -> Vec<f64> {
        seen[&b].t.iter().map(|v| median(v)).collect()
    };
    let usable = |seen: &std::collections::BTreeMap<usize, Seen>, b: usize| {
        (seen[&b].exact || (npu && seen[&b].token)) && tmed(seen, b).iter().zip(tmed(seen, rule)).all(|(x, r)| *x >= r)
    };
    // then each direction, one step at a time, while each step beats the one before by the margin (without it a climb
    // takes steps that are only noise); each direction ends where it stopped
    let mut ends = vec![rule];
    for up in [true, false] {
        let mut cur = rule;
        loop {
            let next = if up { cur * 2 } else { cur / 2 };
            if !seen.contains_key(&next) || !usable(&seen, next) || sc(&seen, next) <= sc(&seen, cur) * (1.0 + margin) {
                break;
            }
            cur = next;
            let beyond = if up { cur * 2 } else { cur / 2 };
            if !ok(beyond) {
                break;
            }
            stage(&[cur, beyond], &mut seen);
        }
        ends.push(cur);
    }
    drop(stage);
    let cands: Vec<usize> = seen.keys().copied().collect();
    let exact: Vec<bool> = cands.iter().map(|b| seen[b].exact).collect();
    let token: Vec<bool> = cands.iter().map(|b| seen[b].token).collect();
    let dmax: Vec<f32> = cands.iter().map(|b| seen[b].dmax).collect();
    let speeds: Vec<Vec<f64>> = cands.iter().map(|b| seen[b].speeds.clone()).collect();
    let tspeeds: Vec<Vec<Vec<f64>>> = cands.iter().map(|b| seen[b].t.clone()).collect();
    let same: Vec<bool> = (0..cands.len()).map(|i| exact[i] || (npu && token[i])).collect();
    if let Some(v) = saved {
        model::pipe::restore_split(nl, v);
    }
    let ri = cands.iter().position(|&b| b == rule).unwrap();
    let score: Vec<f64> = speeds.iter().map(|v| median(v)).collect();
    let tscore: Vec<Vec<f64>> = tspeeds.iter().map(|c| c.iter().map(|v| median(v)).collect()).collect();
    // no slower than the rule on any text
    let even = |i: usize| (0..nt).all(|t| tscore[i][t] >= tscore[ri][t]);
    // the faster end of the two climbs, if it still beats the rule by the margin with all its runs counted
    let pick = ends
        .iter()
        .map(|b| cands.iter().position(|c| c == b).unwrap())
        .filter(|&i| same[i] && even(i))
        .max_by(|&x, &y| score[x].total_cmp(&score[y]))
        .unwrap_or(ri);
    let chosen = if pick != ri && score[pick] > score[ri] * (1.0 + margin) { pick } else { ri };
    for (i, &b) in cands.iter().enumerate() {
        let each = if nt > 1 {
            let v: Vec<String> = texts.iter().zip(&tscore[i]).map(|(t, s)| format!("{} {s:.1}", t.0)).collect();
            format!(" [{}]", v.join(", "))
        } else {
            String::new()
        };
        let slow = if !same[i] || even(i) { "" } else { ", slower than the rule on a text: not used" };
        eprint!("  block {b:4}: {:6.1} tokens/s{each}{slow}", score[i]);
        let check = if exact[i] {
            String::new()
        } else if same[i] {
            format!(", same next tokens (NPU tails; logits up to {:.3} apart)", dmax[i])
        } else {
            format!(
                ", OUTPUT DIFFERS (logits up to {:.3} apart{}): not used",
                dmax[i],
                if token[i] { "" } else { ", another next token" }
            )
        };
        eprintln!(
            " (runs {}){}{}{}",
            speeds[i].iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(" "),
            if i == ri { ", the rule" } else { "" },
            check,
            if i == chosen { "  <- chosen" } else { "" }
        );
    }
    eprintln!("  the rule's spread {:.1}%, margin to move {:.1}%", 100.0 * spread, 100.0 * margin);
    // Top-out race (a machine may want 4096): when the climb chose the ceiling, the standard prompts cannot rank
    // anything bigger, so race the ceiling against double it on one longer prompt (the first text, one block of the
    // bigger size and a third, not a multiple of either), and keep doubling while the bigger block wins by the margin
    // with the same next token. Only machines that top out pay for the long prompt.
    let mut value = cands[chosen];
    let mut race: Vec<serde_json::Value> = Vec::new();
    while value >= ceil && cands[chosen] == ceil {
        let big = value * 2;
        let plen = big + big / 3 + 1;
        if plen + 1 >= ctx as usize || plen >= ids[0].len() {
            eprintln!(
                "  top-out race: {big} needs a {plen}-token prompt, past the context {ctx} (raise the ctx argument)"
            );
            break;
        }
        eprintln!("  top-out race: block {value} against {big} on a {plen}-token prompt ({})", at());
        let (mut sv, mut sb) = (Vec::new(), Vec::new());
        let (_, _, lv0) = run(value, &[plen], 1); // warm-ups, and the reference next token
        let (_, _, lb0) = run(big, &[plen], 1);
        let same_tok = argmax(&lv0) == argmax(&lb0);
        for pass in 0..3 {
            let order = if pass % 2 == 0 { [(value, false), (big, true)] } else { [(big, true), (value, false)] };
            for &(bb, is_big) in &order {
                let tps = run(bb, &[plen], 1).1;
                if is_big { sb.push(tps) } else { sv.push(tps) }
            }
        }
        let (mv, mb) = (median(&sv), median(&sb));
        let win = same_tok && mb > mv * (1.0 + margin);
        eprintln!(
            "  top-out race at {plen} tokens: block {value} {mv:.1}, block {big} {mb:.1} tokens/s{}  <- {}",
            if same_tok { "" } else { " (another next token: not used)" },
            if win { big } else { value }
        );
        race.push(json!({"prompt": plen, "block": value, "speeds": sv, "bigger": big, "bigger_speeds": sb}));
        if !win {
            break;
        }
        value = big;
    }
    let when = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let runs: serde_json::Map<String, serde_json::Value> =
        cands.iter().zip(&speeds).map(|(b, v)| (b.to_string(), json!(v))).collect();
    let entry = json!({"value": value, "top_out_race": race, "score": score[chosen], "rule": rule, "rule_score": score[ri],
        "rule_spread": spread, "runs": runs, "differ": cands.iter().zip(&same).filter(|(_, s)| !**s).map(|(b, _)| *b)
        .collect::<Vec<_>>(), "max_logit_diff": dmax, "text_scores": tscore,
        "texts": texts.iter().map(|t| t.0.clone()).collect::<Vec<_>>(), "prompt_tokens": lens, "threads": threads,
        "when": when});
    match model::tune::put("block", entry) {
        Ok(p) => eprintln!("torc-tune: block {value} kept in {}", p.display()),
        Err(e) => fail(format!("writing the result: {e}")),
    }
    // the MoE work split at the chosen block: even (the default) or guided, three passes in alternating order; guided
    // is kept only when it is faster by more than the margin. The split changes no arithmetic, so the logits must match
    let b = value;
    eprintln!("  work split at block {b}: measuring even and guided, three passes ({})", at());
    let mut gs = [Vec::new(), Vec::new()];
    let mut g_same = true;
    for pass in 0..3 {
        let order = if pass % 2 == 0 { [0usize, 1] } else { [1, 0] };
        for g in order {
            model::moe::set_guided(g == 1);
            let (_, tps, lg) = run(b, &lens, nt);
            gs[g].push(tps);
            if g == 1 && pass == 0 {
                let (_, _, lg0) = {
                    model::moe::set_guided(false);
                    run(b, &lens, nt)
                };
                g_same = lg.iter().zip(&lg0).all(|(a, c)| a.to_bits() == c.to_bits()) || npu;
            }
        }
    }
    let (s0, s1) = (median(&gs[0]), median(&gs[1]));
    let use_guided = g_same && s1 > s0 * (1.0 + margin);
    eprintln!(
        "  work split at block {b}: even {s0:.1}, guided {s1:.1} tokens/s{}  <- {}",
        if g_same { "" } else { " (OUTPUT DIFFERS: not used)" },
        if use_guided { "guided" } else { "even" }
    );
    let gentry = json!({"value": use_guided as u64, "even": gs[0], "guided": gs[1], "block": b, "threads": threads,
        "when": when});
    if let Err(e) = model::tune::put("guided", gentry) {
        fail(format!("writing the result: {e}"));
    }
    model::moe::set_guided(use_guided);
    // pinning (TORC_PIN): each thread held to one CPU, or left to the OS. Which is faster depends on the machine and on
    // what else runs on it, so both are measured, three passes in alternating order: the prompts read, then 64 tokens
    // written. The other setting replaces the platform's default only when reading and writing together are faster by
    // the margin and neither is slower by it. Pinning changes no arithmetic, so the logits must match. Not measured
    // when TORC_PIN chose, or where the threads can't be pinned. Not measured on Linux yet either: a thread started
    // from a pinned one keeps its CPU there, so the helper threads would share one core (pool::apply_tuned_pin)
    let pin0 = model::pool::pinned();
    if !cfg!(target_os = "linux") && !model::switches::given("TORC_PIN") && model::pool::set_pin(pin0) {
        eprintln!("  pinning: measuring pinned and not pinned, three passes ({})", at());
        // index 0 is the default, 1 the other
        let pins = [pin0, !pin0];
        let (mut rs, mut ws) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
        let mut p_same = true;
        let mut lg0: Vec<f32> = Vec::new();
        for pass in 0..3 {
            let order = if pass % 2 == 0 { [0usize, 1] } else { [1, 0] };
            for k in order {
                model::pool::set_pin(pins[k]);
                let (_, tps, lg) = run(b, &lens, nt);
                rs[k].push(tps);
                if pass == 0 && k == 0 {
                    lg0 = lg;
                } else if pass == 0 {
                    p_same = lg.iter().zip(&lg0).all(|(a, c)| a.to_bits() == c.to_bits()) || npu;
                }
            }
        }
        let (tid, l) = (&ids[0], lens[0]);
        model::pool::force_decode_width(0);
        for pass in 0..3 {
            let order = if pass % 2 == 0 { [0usize, 1] } else { [1, 0] };
            for k in order {
                model::pool::set_pin(pins[k]);
                m.clear_states();
                m.prefill_blocks(&tid[..l], b, false, &mut |_, _| {}).unwrap_or_else(|e| fail(e));
                let mut lg = vec![0.0f32; nv];
                let t = std::time::Instant::now();
                for i in 0..64 {
                    m.decode(tid[l + i], (l + i) as u32, &mut lg).unwrap_or_else(|e| fail(e));
                }
                ws[k].push(64.0 / t.elapsed().as_secs_f64());
            }
        }
        model::pool::force_decode_width(usize::MAX);
        let (r0, r1, w0, w1) = (median(&rs[0]), median(&rs[1]), median(&ws[0]), median(&ws[1]));
        let (gr, gw) = (r1 / r0, w1 / w0);
        let switch = p_same && (gr * gw).sqrt() > 1.0 + margin && gr >= 1.0 - margin && gw >= 1.0 - margin;
        let pin = if switch { !pin0 } else { pin0 };
        model::pool::set_pin(pin);
        let name = |p: bool| if p { "pinned" } else { "not pinned" };
        eprintln!(
            "  pinning: {} reads {r0:.1}, writes {w0:.1}; {} reads {r1:.1}, writes {w1:.1} tokens/s{}  <- {}",
            name(pin0),
            name(!pin0),
            if p_same { "" } else { " (OUTPUT DIFFERS: not used)" },
            name(pin)
        );
        let pentry = json!({"value": pin as u64, "default": pin0 as u64, "read": rs[0], "write": ws[0],
            "other_read": rs[1], "other_write": ws[1], "block": b, "threads": threads, "when": when});
        if let Err(e) = model::tune::put("pin", pentry) {
            fail(format!("writing the result: {e}"));
        }
    }
    // the decode threads (pool::Width): on a part with two threads a core the reply can write faster on one thread a
    // core while the prompt reads faster on all of them. Both widths decode the same 64 tokens after the same prompt,
    // three passes in alternating order; the narrower is kept only when faster by the margin. The width changes no
    // arithmetic, so the logits must match
    if let Some(cores) = model::pool::fast_core_count().filter(|&c| c < threads as usize) {
        let widths = [threads as usize, cores];
        eprintln!("  decode threads: measuring {} and {cores}, three passes ({})", widths[0], at());
        let (tid, l) = (&ids[0], lens[0]);
        let mut ds = [Vec::new(), Vec::new()];
        let mut lgs: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for pass in 0..3 {
            let order = if pass % 2 == 0 { [0usize, 1] } else { [1, 0] };
            for k in order {
                model::pool::force_decode_width(0);
                m.clear_states();
                m.prefill_blocks(&tid[..l], b, false, &mut |_, _| {}).unwrap_or_else(|e| fail(e));
                model::pool::force_decode_width(widths[k]);
                let mut lg = vec![0.0f32; nv];
                let t = std::time::Instant::now();
                for i in 0..64 {
                    m.decode(tid[l + i], (l + i) as u32, &mut lg).unwrap_or_else(|e| fail(e));
                }
                ds[k].push(64.0 / t.elapsed().as_secs_f64());
                if pass == 0 {
                    lgs[k] = lg;
                }
            }
        }
        model::pool::force_decode_width(usize::MAX);
        let same = lgs[0].iter().zip(&lgs[1]).all(|(a, c)| a.to_bits() == c.to_bits());
        let (d0, d1) = (median(&ds[0]), median(&ds[1]));
        let narrow = same && d1 > d0 * (1.0 + margin);
        let value = if narrow { cores } else { widths[0] };
        eprintln!(
            "  decode threads: {} {d0:.1}, {cores} (one a core) {d1:.1} tokens/s{}  <- {value}",
            widths[0],
            if same { "" } else { " (OUTPUT DIFFERS: not used)" }
        );
        let dentry = json!({"value": value as u64, "all": ds[0], "one_a_core": ds[1], "threads": threads,
            "cores": cores, "block": b, "when": when});
        if let Err(e) = model::tune::put("decode_threads", dentry) {
            fail(format!("writing the result: {e}"));
        }
    }
}
