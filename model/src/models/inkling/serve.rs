//! serve.rs - Inkling 2 Mini for torc-server (chat/serve.rs), one request at a time. Replies are greedy at
//! temperature 0, or sampled the way the Qwen engine samples (temperature, top_p, top_k, seed, penalties).
//!
//! The prompt is the model's own chat format (chat.rs), and the reply's messages go to the client as what they are:
//! thinking as reasoning, text as the text, a call as a tool call. It ends at <|content_model_end_sampling|>.
//!
//! A prompt that extends the last one reads only its new tokens. The keys and values are kept by position, but the
//! convolutions carry their last inputs along, so going on from a position needs their state as it was there. That
//! state is kept at the end of each prompt read (a mark): a next request that keeps the conversation up to there (the
//! usual case, the reply's reasoning dropped) goes on from the mark; one that differs earlier is read from the
//! nearest mark before, or from the start.

use super::chat::{Reader, prompt};
use super::inkling::{BLOCK_MAX, Ink};
use crate::api::Chat;
use crate::serve::{Call, Reply, Serve};
use serde_json::Value;

/// The marks kept: the ends of the last prompts read (on two machines, each in a worker mark slot of its own).
const MARKS: usize = 4;
const _: () = assert!(MARKS <= crate::pipe::MARK_SLOTS);

struct Engine {
    m: Ink,
    tok: crate::tok::Tokenizer,
    nv: usize,
    ctx: usize,
    /// the tokens the model has read, in order (its state is the one after them)
    seen: Vec<u32>,
    /// positions the convolution state is kept for, oldest first: (tokens read, state, the pipeline worker's mark
    /// slot holding its half)
    marks: Vec<(usize, Vec<f32>, usize)>,
    /// TORC_PIPE: the later layer passes on a second machine (pipe.rs)
    pipe: Option<super::pipe::Pipe>,
    stops: Vec<u32>,
    name: String,
    /// the prompt is read in blocks of this many tokens (TORC_BLOCK, at most BLOCK_MAX)
    block: usize,
    kept: usize,
    timed: (usize, f64, f64, usize, f64),
    replies: u64,
}

impl Engine {
    /// The model's state made the one after ids[..p] for the largest p it can be (at most `keep`, which ids and the
    /// tokens read share): where it stands, a mark, or the start. Returns p.
    fn rewind(&mut self, keep: usize) -> format::Result<usize> {
        if keep == self.seen.len() {
            return Ok(keep);
        }
        self.marks.retain(|(p, _, _)| *p <= keep);
        match self.marks.last() {
            Some((p, state, slot)) => {
                self.m.set_conv_state(state)?;
                if let Some(pp) = self.pipe.as_mut() {
                    pp.load(*slot)?;
                }
                Ok(*p)
            }
            None => Ok(0),
        }
    }

    /// After an error on the link to the pipeline worker: a new connection (the worker's states start from zero),
    /// this machine's state from zero and no marks kept, then `toks` read again from position 0 with the last one's
    /// logits into `last`. Up to 3 tries; on one machine nothing is retried (its errors are not the link's). The
    /// bits do not depend on the blocks a sequence is read in, so the state is the one the error interrupted.
    fn recover(&mut self, toks: &[u32], last: &mut [f32], e: format::Error) -> format::Result<()> {
        if self.pipe.is_none() || toks.is_empty() {
            return Err(e);
        }
        let mut err = e;
        for tries in 1..=3 {
            eprintln!(
                "torc-server: {}; reconnecting, the {} tokens so far read again (try {tries} of 3)",
                err.0,
                toks.len()
            );
            self.marks.clear();
            self.m.reset();
            if let Some(pp) = self.pipe.as_mut() {
                pp.forget_pending();
            }
            let r = crate::pipe::reconnect().and_then(|_| {
                let mut p0 = 0;
                while p0 < toks.len() {
                    let p1 = (p0 + self.block).min(toks.len());
                    let b = toks[p0..p1].to_vec();
                    self.fwd(&b, p0 as u32, if p1 == toks.len() { Some(&mut *last) } else { None })?;
                    p0 = p1;
                }
                Ok(())
            });
            match r {
                Ok(()) => return Ok(()),
                Err(e2) => err = e2,
            }
        }
        Err(err)
    }

    /// A block through the model (both machines when piped), with the last row's logits into `last` when given; a
    /// piped block without them does not wait for the worker.
    fn fwd(&mut self, toks: &[u32], pos: u32, last: Option<&mut [f32]>) -> format::Result<()> {
        // a mark whose window the sliding-window passes' ring is about to overwrite can't be gone back to any more
        // (on two machines the worker's ring is the same length, so the same marks go there too)
        let upto = pos as usize + toks.len() - 1;
        let m = &self.m;
        self.marks.retain(|(p, _, _)| !m.ring_lost(*p, upto));
        match (self.pipe.as_mut(), last) {
            (Some(pp), l) => pp.forward(&mut self.m, toks, pos, l),
            (None, Some(l)) => self.m.forward(toks, pos, l, false),
            (None, None) => {
                let mut scratch = vec![0.0f32; self.nv];
                self.m.forward(toks, pos, &mut scratch, false)
            }
        }
    }
}

impl Serve for Engine {
    fn name(&self) -> &str {
        &self.name
    }

    fn count(&self, c: &Chat) -> usize {
        self.tok.encode(&prompt(c, &self.tok), true).map_or(0, |v| v.len())
    }

    fn reply(&mut self, c: &Chat, emit: &mut dyn FnMut(bool, &str) -> bool) -> format::Result<Reply> {
        let ids = self.tok.encode(&prompt(c, &self.tok), true).unwrap_or_default();
        if ids.is_empty() {
            return Err(format::Error("the prompt could not be tokenized".into()));
        }
        if ids.len() >= self.ctx {
            return Err(crate::chat::serve::too_long(ids.len(), self.ctx));
        }
        let max = c.max.min(self.ctx - ids.len());
        let tp = std::time::Instant::now();
        // the longest prefix already read, kept one token short so the last prompt token runs for its logits
        let keep = self.seen.iter().zip(&ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1);
        let mut last = vec![0.0f32; self.nv];
        // a mark gone back to through a broken link: the whole prompt is read again from a new connection
        let (keep, mut recovered) = match self.rewind(keep) {
            Ok(k) => (k, false),
            Err(e) => {
                self.recover(&ids, &mut last, e)?;
                (0, true)
            }
        };
        self.seen = ids[..keep].to_vec();
        let mut p0 = keep;
        while p0 < ids.len() && !recovered {
            let p1 = (p0 + self.block).min(ids.len());
            // only the last block's logits are needed (a piped earlier block does not wait for the worker)
            let ids_b = ids[p0..p1].to_vec();
            if let Err(e) = self.fwd(&ids_b, p0 as u32, if p1 == ids.len() { Some(&mut last) } else { None }) {
                self.recover(&ids, &mut last, e)?;
                recovered = true;
            }
            p0 = p1;
        }
        // after a recovery the whole prompt was read again
        let keep = if recovered { 0 } else { keep };
        let read = ids.len() - keep;
        let prompt_s = tp.elapsed().as_secs_f64();
        self.seen = ids.clone();
        if self.marks.len() == MARKS {
            self.marks.remove(0);
        }
        // the worker keeps its half of the mark in a slot no other mark kept here uses (a mark the broken link could
        // not take is not kept: the state itself is read again, so the reply goes on)
        let slot = (0..MARKS).find(|k| self.marks.iter().all(|m| m.2 != *k)).unwrap_or(0);
        let saved = match self.pipe.as_mut().map(|pp| pp.save(slot)) {
            Some(Err(e)) => {
                self.recover(&ids, &mut last, e)?;
                false
            }
            _ => true,
        };
        if saved {
            self.marks.push((ids.len(), self.m.conv_state(), slot));
        }
        let tg = std::time::Instant::now();
        let (mut pend, mut n_out, mut n_think) = (Vec::<u8>::new(), 0usize, 0usize);
        let mut finish = "length";
        let mut reader = Reader::new();
        let mut pos = ids.len();
        let mut sp = crate::chat::serve::sampling(c);
        loop {
            let t = sp.pick(&last);
            n_out += 1;
            if self.stops.contains(&t) {
                finish = "stop";
                break;
            }
            pend.extend(self.tok.decode(t));
            // send whole UTF-8 characters only; a character split across tokens is held for the next one
            let cut = match std::str::from_utf8(&pend) {
                Ok(_) => pend.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => pend.len(),
            };
            let piece = String::from_utf8_lossy(&pend[..cut]).into_owned();
            pend.drain(..cut);
            n_think += reader.thinking() as usize;
            // the client has gone: the reply ends here
            if !reader.push(&piece, emit) {
                break;
            }
            if n_out == max || pos + 1 >= self.ctx {
                break;
            }
            if let Err(e) = self.fwd(&[t], pos as u32, Some(&mut last)) {
                // the reply so far and t read again from a new connection; the reply goes on from there
                let mut toks = self.seen.clone();
                toks.push(t);
                self.recover(&toks, &mut last, e)?;
            }
            self.seen.push(t);
            pos += 1;
        }
        self.kept = keep;
        self.timed = (read, prompt_s, prompt_s, n_out, tg.elapsed().as_secs_f64());
        self.replies += 1;
        reader.finish(emit);
        let calls: Vec<Call> = reader
            .calls
            .iter()
            .map(|(name, args)| Call {
                name: name.clone(),
                args: match serde_json::from_str(args) {
                    Ok(Value::Object(m)) => m,
                    _ => Default::default(),
                },
            })
            .collect();
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

    fn idle(&mut self) {}
}

/// Loads an Inkling 2 file for torc-server: the worker pool is up already, `ctx` is the context length and `name`
/// the name it is served under (the file's own when None).
pub fn open(path: &str, ctx: usize, name: Option<String>) -> Box<dyn Serve> {
    let t0 = std::time::Instant::now();
    let m = Ink::load(path, ctx as u32).unwrap_or_else(|e| {
        eprintln!("{path}: {}", e.0);
        std::process::exit(1)
    });
    let tok = crate::tok::Tokenizer::open(path).unwrap_or_else(|e| {
        eprintln!("tokenizer: {}", e.0);
        std::process::exit(1)
    });
    let nv = m.n_vocab();
    // the end of the model's turn, and the file's own end token
    let mut stops: Vec<u32> = tok.find("<|content_model_end_sampling|>").into_iter().collect();
    if let Some(eos) =
        format::meta::Meta::parse(&m.w.file.metadata).ok().and_then(|x| x.u32("tokenizer.ggml.eos_token_id"))
        && !stops.contains(&eos)
    {
        stops.push(eos);
    }
    let name = name.unwrap_or_else(|| {
        std::path::Path::new(path).file_stem().map_or("inkling-2-mini".into(), |s| s.to_string_lossy().into_owned())
    });
    let mut e = Engine {
        m,
        tok,
        nv,
        ctx,
        seen: Vec::new(),
        marks: Vec::new(),
        stops,
        name,
        block: crate::switches::num::<usize>("TORC_BLOCK").filter(|&b| b > 0).unwrap_or(512).min(BLOCK_MAX),
        pipe: None,
        kept: 0,
        timed: (0, 0.0, 0.0, 0, 0.0),
        replies: 0,
    };
    // a warm-up turn, so the first request does not pay the first-use costs
    {
        e.pipe = super::pipe::Pipe::from_env(&e.m).unwrap_or_else(|er| {
            eprintln!("{}", er.0);
            std::process::exit(1)
        });
        // two machines read in blocks of 256 unless TORC_BLOCK says otherwise: twice the blocks to overlap, and on
        // the boxes it read faster at every length (27 Nov: 256 v 512 short 101.5 v 78.4, 8k 111.5 v 105.2)
        if e.pipe.is_some() && !crate::switches::given("TORC_BLOCK") {
            e.block = 256;
        }
        let mut last = vec![0.0f32; nv];
        let _ = e.fwd(&[0u32; 16], 0, Some(&mut last));
        let _ = e.fwd(&[0], 16, Some(&mut last));
        e.m.reset();
    }
    eprintln!("torc-server: Inkling 2, ctx {ctx}, loaded in {:.1} s (stops {:?})", t0.elapsed().as_secs_f64(), e.stops);
    Box::new(e)
}
