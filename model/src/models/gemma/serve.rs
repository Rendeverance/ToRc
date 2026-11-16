//! serve.rs - Gemma 4 for torc-server (chat/serve.rs), one request at a time. Replies are greedy at temperature 0, or
//! sampled the way the Qwen engine samples (temperature, top_p, top_k, seed, penalties). A prompt that extends the last
//! one reads only its new tokens (the cache is positional). The tiers are the same as torc-gemma's: TORC_NPU=1 (the NPU
//! beside the CPU), TORC_NPU=pure (the whole model on the NPU), or neither (the CPU). Chats are saved to disk and
//! loaded back as the Qwen family's are (saved.rs), except on the whole-NPU tier, whose caches are on the NPU. A
//! request with tools gets them declared in the system turn, in Gemma 4's own format, and a call the model makes is
//! read out of the reply and kept out of its text (both in chat.rs). TORC_DRAFT=<assistant model> turns on speculative
//! decoding with Gemma 4's drafter (gemma_mtp.rs): up to TORC_DRAFT_N drafts a round (default 3), each verified by the
//! target's own pick, so the reply is the same tokens as without it. How many drafts a round gets: TORC_DRAFT_P=p
//! (default 0.9) ends a round at the first draft the drafter gives less than p (0 takes all of them), and
//! TORC_DRAFT_P=plan lets the draft planner decide (plan.rs DraftPlan, as torc-server's MTP does: pass times on a
//! fitted line, and the drafter's calibrated confidence).

use super::chat::{Calls, MARKERS, NO_THOUGHT, filled, parse_call, prompt_of, prompt_with_tools};
use crate::api::Chat;
use crate::models::qwen::sample::Sampling;
use crate::serve::{Call, Reply, Serve, defused, defused_tools};
use serde_json::Value;

struct Engine {
    m: crate::gemma::Gemma,
    tok: crate::tok::Tokenizer,
    nv: usize,
    ctx: usize,
    seen: Vec<u32>,
    /// the last prompt's final logits and hidden row. A prompt that repeats it exactly starts from these (with no
    /// pass), so it gets the same reply as the first time it was read; the NPU tier's block pass and a one-token pass
    /// differ in bits
    first: Option<(Vec<u32>, Vec<f32>, Vec<f32>)>,
    end_turn: Option<u32>,
    /// the thinking block's markers, <|channel> and <channel|>
    channel: Option<(u32, u32)>,
    name: String,
    draft: Option<crate::gemma_mtp::GemmaDraft>,
    n_draft: usize,
    /// TORC_DRAFT_P as a fixed threshold, or None for the planner
    p_min: Option<f32>,
    pl: Box<dyn crate::plan::Plans>,
    /// TORC_DRAFT_LOG: print the drafter's calibration and step times after each request
    log: bool,
    /// for each judged draft: (its probability, whether it was taken), over the server's whole life (TORC_DRAFT_LOG
    /// prints the bins after each request)
    dlog: Vec<(f32, bool)>,
    /// the prompt is read in blocks of this many tokens (TORC_BLOCK, default 512)
    block: usize,
    /// the last prompt's tokens that were not read again
    kept: usize,
    /// the last reply's timings (serve.rs)
    timed: (usize, f64, f64, usize, f64),
    replies: u64,
    saved: crate::chat::saved::Saved,
    /// NO_THOUGHT's tokens
    no_thought: Vec<u32>,
}

impl Engine {
    /// The request's prompt in tokens, the beginning-of-sequence token first.
    fn ids_of(&self, c: &Chat) -> Vec<u32> {
        let mut ids = vec![2u32];
        // a request that has nothing to do with tools gets the plain turn format what the user, the tools and the tool
        // servers wrote, made safe to put in the prompt
        let d: &dyn Fn(&str) -> std::borrow::Cow<str> = &|t| self.tok.defuse(t, MARKERS);
        let (msgs, tools) = (defused(&c.msgs, d), defused_tools(&c.tools, d));
        let plain = tools.is_empty() && !msgs.iter().any(|m| m["role"] == "tool" || filled(&m["tool_calls"]));
        let text = if plain { prompt_of(&msgs) } else { prompt_with_tools(&msgs, &tools) };
        ids.extend(self.tok.encode(&text, true).unwrap_or_default());
        ids
    }

    /// The next token from one row of logits. The model leaves the soft cap off (greedy doesn't need it: tanh never
    /// changes which logit is largest), so a sampled pick or a penalised one puts it back first, into `capped`.
    fn pick(&self, sp: &mut Sampling, row: &[f32], capped: &mut Vec<f32>) -> u32 {
        let c = self.m.h.softcap;
        if (sp.temp <= 0.0 && !sp.penalised()) || c <= 0.0 {
            return sp.pick(row);
        }
        capped.clear();
        capped.extend(row.iter().map(|&l| c * (l / c).tanh()));
        sp.pick(capped)
    }
}

impl Serve for Engine {
    fn name(&self) -> &str {
        &self.name
    }

    fn count(&self, c: &Chat) -> usize {
        self.ids_of(c).len()
    }

    fn reply(&mut self, c: &Chat, emit: &mut dyn FnMut(bool, &str) -> bool) -> format::Result<Reply> {
        let ids = self.ids_of(c);
        // a reply runs to max_tokens or to the end of the context, whichever comes first
        if ids.len() >= self.ctx {
            return Err(crate::chat::serve::too_long(ids.len(), self.ctx));
        }
        let max = c.max.min(self.ctx - ids.len());
        // the longest prefix already read (kept one token short, so the last prompt token runs for its logits), unless
        // the prompt repeats the last one exactly, in which case its first logits are the ones kept
        let mut last = vec![0.0f32; self.nv];
        // the engine's own timing (the `timings` sent with the usage): tokens read and seconds to the first token
        let tp = std::time::Instant::now();
        let mut read = 0usize;
        let again = self
            .first
            .as_ref()
            .filter(|(p, _, _)| *p == ids && self.seen.starts_with(&ids))
            .map(|(_, lg, h)| (lg.clone(), h.clone()));
        match again {
            Some((lg, hid)) => {
                last.copy_from_slice(&lg);
                self.m.hid = hid;
            }
            None => {
                let keep = self.seen.iter().zip(&ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1);
                // or a chat saved on disk that it begins with, when that is further on
                let keep = self.saved.begin(&mut self.m, &ids, keep);
                self.seen = ids[..keep].to_vec();
                // read in blocks, as torc-gemma's pp= does: one pass over a long prompt keeps an attention partial for
                // every row against every cache page (about 129 GB at 32k tokens)
                let mut p0 = keep;
                while p0 < ids.len() {
                    let p1 = (p0 + self.block).min(ids.len());
                    self.m.forward(&ids[p0..p1], p0 as u32, &mut last, false)?;
                    p0 = p1;
                }
                self.first = Some((ids.clone(), last.clone(), self.m.hid.clone()));
                // what a next turn begins with: the prompt without the empty thinking block
                let upto = if ids.ends_with(&self.no_thought) { ids.len() - self.no_thought.len() } else { ids.len() };
                self.saved.read(&ids[..upto]);
                read = ids.len() - keep;
            }
        }
        let prompt_s = tp.elapsed().as_secs_f64();
        let tg = std::time::Instant::now();
        self.seen = ids.clone();
        let (mut pend, mut n_out) = (Vec::<u8>::new(), 0usize);
        let mut finish = "length";
        let mut pos = ids.len();
        // the next token and, when drafting, the target's hidden state before it
        let mut sp = crate::chat::serve::sampling(c);
        let mut capped = Vec::new();
        let mut t = self.pick(&mut sp, &last[..self.nv], &mut capped);
        let mut h = self.m.hid.clone();
        let (mut rounds, mut acc) = (0usize, 0usize);
        let (mut t_draft, mut t_ver, mut t_one, mut n_one) = (0f64, 0f64, 0f64, 0usize);
        let mut probs_total = 0usize;
        // tokens decided but not yet sent (accepted drafts, then the target's own pick)
        let mut queue = std::collections::VecDeque::from([t]);
        // with tools in the request, the reply's text goes through a reader that takes the calls out of it
        let mut reader = (!c.tools.is_empty()).then(Calls::default);
        // inside a thinking block, and still on its first line (the channel's name); its tokens
        let (mut thinking, mut head, mut n_think) = (false, false, 0usize);
        'out: loop {
            while let Some(t) = queue.pop_front() {
                n_out += 1;
                if Some(t) == self.end_turn || t == 1 {
                    finish = "stop";
                    break 'out;
                }
                // a thinking block (<|channel>thought\n ... <channel|>; after a tool's result the model writes an empty
                // one itself) goes out as reasoning, without its markers and the channel's name
                let open = self.channel.is_some_and(|(o, _)| o == t);
                let close = self.channel.is_some_and(|(_, c)| c == t);
                if open || close {
                    (thinking, head) = (open, open);
                } else {
                    pend.extend(self.tok.decode(t));
                    // send whole UTF-8 characters only; a character split across tokens is held for the next one
                    let cut = match std::str::from_utf8(&pend) {
                        Ok(_) => pend.len(),
                        Err(e) if e.error_len().is_none() => e.valid_up_to(),
                        Err(_) => pend.len(),
                    };
                    let piece = String::from_utf8_lossy(&pend[..cut]).into_owned();
                    pend.drain(..cut);
                    let there = if thinking {
                        n_think += 1;
                        let mut p = piece.as_str();
                        if head {
                            match p.find('\n') {
                                Some(i) => (p, head) = (&p[i + 1..], false),
                                None => p = "",
                            }
                        }
                        p.is_empty() || emit(true, p)
                    } else {
                        match &mut reader {
                            Some(r) => r.push(&piece, emit),
                            None => piece.is_empty() || emit(false, &piece),
                        }
                    };
                    // the client has gone: the reply ends here
                    if !there {
                        break 'out;
                    }
                }
                // the model has made its calls and waits for their results
                if reader.as_ref().is_some_and(|r| r.waiting) {
                    finish = "stop";
                    break 'out;
                }
                if n_out == max {
                    break 'out;
                }
            }
            // t (the last token sent) is not in the cache yet: run it, with drafts after it when there is a drafter
            let left = max - n_out;
            let mut block = vec![t];
            let mut probs = Vec::new();
            // the drafts' chance of surviving: the product of their confidences so far
            let mut surv: Vec<f64> = Vec::new();
            let t0 = std::time::Instant::now();
            if let Some(d) = &self.draft {
                let (mut dt, mut dh) = (t, h.clone());
                while probs.len() < self.n_draft.min(left) {
                    // the planner: keep drafting while it still wants every draft made so far
                    if self.p_min.is_none() && !surv.is_empty() && self.pl.plan(&surv) != surv.len() {
                        break;
                    }
                    let tc = std::time::Instant::now();
                    match d.step(&self.m, dt, &dh, pos as u32, true) {
                        Ok((nt, p, nh)) => {
                            if self.p_min.is_some_and(|m| p < m) {
                                break;
                            }
                            surv.push(surv.last().copied().unwrap_or(1.0) * p as f64);
                            block.push(nt);
                            probs.push(p);
                            (dt, dh) = (nt, nh);
                        }
                        Err(e) => {
                            eprintln!("draft: {}", e.0);
                            break;
                        }
                    }
                    if surv.len() > 1 {
                        self.pl.time_chain(tc.elapsed().as_secs_f64());
                    }
                }
                if self.p_min.is_none() {
                    let nd = self.pl.pick(&surv);
                    block.truncate(1 + nd);
                    probs.truncate(nd);
                    surv.truncate(nd);
                }
            }
            let nb = block.len();
            let t1 = std::time::Instant::now();
            t_draft += (t1 - t0).as_secs_f64();
            if last.len() < nb * self.nv {
                last.resize(nb * self.nv, 0.0);
            }
            self.m.forward(&block, pos as u32, &mut last, nb > 1)?;
            let t_pass = t1.elapsed().as_secs_f64();
            if nb > 1 {
                t_ver += t1.elapsed().as_secs_f64();
            } else {
                t_one += t1.elapsed().as_secs_f64();
                n_one += 1;
            }
            // the drafts the target agrees with, then its own pick after them: one pick a position, in order, so the
            // reply is the same tokens as without drafts
            let mut j = 0;
            let pick = loop {
                let p = self.pick(&mut sp, &last[j * self.nv..(j + 1) * self.nv], &mut capped);
                if j + 1 == nb || p != block[j + 1] {
                    break p;
                }
                j += 1;
            };
            if self.draft.is_some() {
                // the pass's forward time (t(n)), including passes with zero drafts, and which drafts were taken
                self.pl.record(&surv, j, nb, t_pass);
            }
            probs_total += probs.len();
            // a draft is only judged when every draft before it was taken
            for (i, &p) in probs.iter().enumerate().take(j + 1) {
                self.dlog.push((p, i < j));
            }
            self.seen.extend_from_slice(&block[..j + 1]);
            pos += j + 1;
            queue.extend(&block[1..j + 1]);
            t = pick;
            queue.push_back(t);
            let dm = self.m.h.d_model;
            h = self.m.hid[j * dm..(j + 1) * dm].to_vec();
            if nb > 1 {
                rounds += 1;
                acc += j;
            }
        }
        let reply_s = tg.elapsed().as_secs_f64();
        if rounds > 0 {
            eprintln!(
                "draft: {rounds} rounds, {acc} of {} drafts taken ({:.2} tokens a round), drafting {:.1} ms a round, verifying {:.1} ms",
                rounds * self.n_draft,
                1.0 + acc as f64 / rounds as f64,
                t_draft * 1e3 / rounds as f64,
                t_ver * 1e3 / rounds as f64
            );
            if self.log {
                eprintln!("{}", self.pl.report(self.n_draft + 1));
                // taken / drafted, by the drafter's probability in tenths
                let mut bins = [(0usize, 0usize); 10];
                for &(p, ok) in &self.dlog {
                    let b = ((p * 10.0) as usize).min(9);
                    bins[b].0 += ok as usize;
                    bins[b].1 += 1;
                }
                let line: Vec<String> = bins.iter().enumerate().map(|(i, (a, n))| format!("{}x:{a}/{n}", i)).collect();
                eprintln!("draft p bins (all requests so far): {}", line.join(" "));
                let mut line = String::from("draft step, ms:");
                let steps = probs_total.max(1) as f64;
                for (i, p) in crate::gemma_mtp::PROF.iter().enumerate() {
                    let ns = p.swap(0, std::sync::atomic::Ordering::Relaxed);
                    line += &format!(" {} {:.2}", crate::gemma_mtp::PARTS[i], ns as f64 / 1e6 / steps);
                }
                eprintln!("{line}");
            }
        } else if n_one > 0 {
            eprintln!("decode: {n_one} steps, {:.1} ms each", t_one * 1e3 / n_one as f64);
        }
        self.kept = ids.len() - read;
        self.timed = (read, prompt_s, prompt_s, n_out, reply_s);
        self.replies += 1;
        let mut calls = Vec::new();
        if let Some(r) = &mut reader {
            r.flush(emit);
            for (name, args) in r.bodies.iter().filter_map(|b| parse_call(b)) {
                calls.push(Call {
                    name,
                    args: match args {
                        Value::Object(m) => m,
                        _ => Default::default(),
                    },
                });
            }
        }
        if !calls.is_empty() && finish == "stop" {
            finish = "tool_calls";
        }
        Ok(Reply { prompt: ids.len(), output: n_out, reasoning: n_think, finish, calls })
    }

    fn kept(&self) -> usize {
        self.kept
    }

    fn timed(&self) -> (usize, f64, f64, usize, f64) {
        self.timed
    }

    fn replies(&self) -> u64 {
        self.replies
    }

    fn idle(&mut self) {
        self.saved.idle(&mut self.m);
    }
}

/// Loads a Gemma 4 file for torc-server: the worker pool is up already, `ctx` is the context length and `name` the
/// name it is served under (the file's own when None).
pub fn open(path: &str, ctx: usize, name: Option<String>) -> Box<dyn Serve> {
    let t0 = std::time::Instant::now();
    let m = crate::gemma::Gemma::load(path, ctx as u32).unwrap_or_else(|e| {
        eprintln!("{path}: {}", e.0);
        std::process::exit(1)
    });
    let tok = crate::tok::Tokenizer::open(path).unwrap_or_else(|e| {
        eprintln!("tokenizer: {}", e.0);
        std::process::exit(1)
    });
    let nv = m.n_vocab().unwrap();
    let one = |s: &str| tok.encode(s, true).ok().filter(|v| v.len() == 1).map(|v| v[0]);
    let end_turn = one("<turn|>");
    let channel = one("<|channel>").zip(one("<channel|>"));
    let name = name.unwrap_or_else(|| {
        std::path::Path::new(path).file_stem().map_or("gemma".into(), |s| s.to_string_lossy().into_owned())
    });
    let draft = crate::switches::get("TORC_DRAFT").map(|p| {
        crate::gemma_mtp::GemmaDraft::load(&p, &m).unwrap_or_else(|e| {
            eprintln!("{p}: {}", e.0);
            std::process::exit(1)
        })
    });
    let n_draft = crate::switches::num("TORC_DRAFT_N").unwrap_or(3);
    let p_min = match crate::switches::get("TORC_DRAFT_P").as_deref() {
        Some("plan") => None,
        Some(v) => Some(v.parse().unwrap_or(0.9)),
        None => Some(0.9),
    };
    // the products the NPU takes change the numbers in the caches (TORC_NPU itself is in every file's identity)
    let settings = format!("gemma4 {}", crate::switches::get("TORC_GEMMA_NPU").unwrap_or_default());
    let saved = crate::chat::saved::Saved::new(!crate::gemma::pure_npu(), &m.w.file.header.model_id, &settings);
    let no_thought = tok.encode(NO_THOUGHT, true).unwrap_or_default();
    let mut e = Engine {
        m,
        tok,
        nv,
        ctx,
        seen: Vec::new(),
        first: None,
        end_turn,
        channel,
        name,
        draft,
        n_draft,
        p_min,
        saved,
        no_thought,
        pl: crate::plan::planner(),
        dlog: Vec::new(),
        block: crate::switches::num::<usize>("TORC_BLOCK").filter(|&b| b > 0).unwrap_or(512),
        log: crate::switches::given("TORC_DRAFT_LOG"),
        kept: 0,
        timed: (0, 0.0, 0.0, 0, 0.0),
        replies: 0,
    };
    // greedy: the soft cap (tanh, which is monotone) never changes which logit is largest
    e.m.cap_logits = false;
    // on the whole-NPU tier, the drafter reads the last layers' caches copied back to the CPU
    e.m.mirror_kv = e.draft.is_some() && crate::gemma::pure_npu();
    // a warm-up turn, so the first request does not pay the first-use costs (graphs, mappings)
    {
        let mut last = vec![0.0f32; nv];
        let ids = vec![2u32; 16];
        let _ = e.m.forward(&ids, 0, &mut last, false);
        let _ = e.m.forward(&[2], 16, &mut last, false);
    }
    eprintln!(
        "torc-server: Gemma 4, ctx {ctx}, loaded in {:.1} s (end of turn {:?})",
        t0.elapsed().as_secs_f64(),
        e.end_turn
    );
    Box::new(e)
}

/// What saving a chat needs from the model (chat/saved.rs): its whole state is its attention cache.
impl crate::chat::saved::Positional for crate::gemma::Gemma {
    fn state_bytes(&self, npos: usize) -> Vec<u8> {
        crate::gemma::Gemma::state_bytes(self, npos)
    }
    fn set_state_bytes(&mut self, npos: usize, b: &[u8]) -> format::Result<()> {
        crate::gemma::Gemma::set_state_bytes(self, npos, b)
    }
    fn n_layer(&self) -> usize {
        self.h.n_layer
    }
}
