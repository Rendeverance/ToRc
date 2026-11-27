//! pipe.rs - Inkling 2 over two machines through the shared layer pipeline (runtime/pipe.rs), as the qwen engine's
//! models run: the main machine runs a block's layer passes 0..S (Ink::forward_range) and sends what they carry
//! (Carry: the stream, the earlier blocks' outputs and the loop's carried inputs) as the request's rows; the worker
//! (`torc-run <model> serve-pipe=<port>`, or `torc-ink <model> serve=<port>`) runs S..end and the head and sends back
//! the logits asked for. A prompt goes block by block with the main machine already on the next block while the
//! worker does the last one; a token waits for its reply (one hop each way). Every pass runs the same code on the same
//! inputs on whichever machine runs it, so the bits are the same as on one machine. Each machine keeps the keys,
//! values and convolution state of its own passes; the worker keeps its convolution state for the server's marks in
//! the shared mark slots (OP_SAVE / OP_LOAD).
//!
//! Main: TORC_PIPE=host:port, TORC_PIPE_SPLIT=S (here a layer pass of n_calls; default 13 of every 24),
//! TORC_PIPE_TOPK=K (a token's K largest logits only; default the whole row).

use super::inkling::{Carry, Ink};
use crate::pipe::{self as sp, Worker};
use format::{Result, err};
use std::io::Write;

/// The main machine's end: the shared connection (pipe::client) and the split.
pub struct Pipe {
    pub split: usize,
    /// prompt blocks sent without waiting (no logits asked), whose empty replies are still to be read
    pending: usize,
    /// TORC_PIPE_TOPK=K: a token's logits come back as their K largest only, the rest -infinity (the full row is 0.8
    /// MB a token over the link); 0 (the default) brings the whole row
    topk: usize,
}

impl Pipe {
    /// TORC_PIPE (host:port) when set: connects (the shared hello checks the worker has this model file).
    pub fn from_env(m: &Ink) -> Result<Option<Pipe>> {
        if !crate::switches::given("TORC_PIPE") {
            return Ok(None);
        }
        let total = m.n_calls();
        // by default the main machine takes 13 of every 24 passes: the later half holds the heavier layers (the
        // last layers' experts are Q8_0)
        let split =
            crate::switches::num::<usize>("TORC_PIPE_SPLIT").filter(|&s| s > 0 && s < total).unwrap_or(total * 13 / 24);
        if sp::client().is_none() {
            return err("two boxes (pipeline): TORC_PIPE is set but there is no worker");
        }
        let topk = crate::switches::num::<usize>("TORC_PIPE_TOPK").unwrap_or(0);
        eprintln!("two boxes (pipeline): Inkling passes 0..{split} here, {split}..{total} there");
        Ok(Some(Pipe { split, pending: 0, topk }))
    }

    /// After a new connection (pipe::reconnect): the old one's unread replies are gone with it.
    pub fn forget_pending(&mut self) {
        self.pending = 0;
    }

    fn drain(&mut self, s: &mut std::net::TcpStream) -> Result<()> {
        while self.pending > 0 {
            sp::recv(s, &mut [])?;
            self.pending -= 1;
        }
        Ok(())
    }

    /// A block through both machines: passes 0..split here, the rest there. With `logits`, waits for the last row's;
    /// without, returns once the carry is sent (the worker's reply is read later).
    pub fn forward(&mut self, m: &mut Ink, tokens: &[u32], pos0: u32, logits: Option<&mut [f32]>) -> Result<()> {
        let carry = m.forward_range(tokens, pos0, None, 0, self.split, None)?.expect("a carry short of the end");
        let res = carry.to_f32();
        if res.len() != m.carry_len(self.split, tokens.len()) {
            return err("two boxes (pipeline): the carry's length is not the one the worker expects");
        }
        // mode 0: no logits, 1: the last row's (bits 16..: only its K largest)
        let want = match (&logits, self.topk) {
            (None, _) => 0,
            (Some(_), 0) => 1,
            (Some(_), k) => 1 | (k as u32) << sp::TOPK_SHIFT,
        };
        let cl = sp::client().ok_or_else(|| format::Error("two boxes (pipeline): no worker".into()))?;
        let mut s = cl.lock().unwrap_or_else(|p| p.into_inner());
        s.write_all(&sp::request(self.split, pos0, tokens, want, &res))
            .map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
        match logits {
            Some(out) => {
                self.drain(&mut s)?;
                if self.topk == 0 {
                    sp::recv(&mut s, out)?;
                } else {
                    // K values, highest first, then their K ids (pipe::topk)
                    let mut v = vec![0.0f32; 2 * self.topk];
                    sp::recv(&mut s, &mut v)?;
                    out.fill(f32::NEG_INFINITY);
                    let (vals, ids) = v.split_at(self.topk);
                    for (&x, id) in vals.iter().zip(ids) {
                        if let Some(o) = out.get_mut(id.to_bits() as usize) {
                            *o = x;
                        }
                    }
                }
            }
            None => self.pending += 1,
        }
        Ok(())
    }

    /// A control request (no tokens) once every earlier reply is in.
    fn control(&mut self, op: u32) -> Result<()> {
        let cl = sp::client().ok_or_else(|| format::Error("two boxes (pipeline): no worker".into()))?;
        let mut s = cl.lock().unwrap_or_else(|p| p.into_inner());
        self.drain(&mut s)?;
        s.write_all(&sp::request(self.split, 0, &[], op, &[]))
            .map_err(|e| format::Error(format!("two boxes (pipeline): {e}")))?;
        sp::recv(&mut s, &mut [])
    }

    /// The worker keeps its convolution state in mark slot `slot`.
    pub fn save(&mut self, slot: usize) -> Result<()> {
        self.control(sp::OP_SAVE | sp::slot_bits(slot))
    }

    /// The worker goes back to the state kept in slot `slot`.
    pub fn load(&mut self, slot: usize) -> Result<()> {
        self.control(sp::OP_LOAD | sp::slot_bits(slot))
    }
}

/// Inkling as a pipeline worker: the model, its marks' convolution states by slot, and the logits buffer.
struct InkWorker<'a> {
    m: &'a mut Ink,
    slots: [Option<Vec<f32>>; sp::MARK_SLOTS],
    logits: Vec<f32>,
}

impl Worker for InkWorker<'_> {
    fn n_ctx(&self) -> usize {
        self.m.n_ctx as usize
    }

    fn res_len(&self, l0: usize, n: usize) -> usize {
        self.m.carry_len(l0, n)
    }

    fn clear_states(&mut self) {
        self.m.reset();
    }

    fn clear_marks(&mut self) {
        self.slots = Default::default();
    }

    fn worker_mark(&mut self, _op: u32, _key: u64, _l0: usize, _npos: usize) -> Result<bool> {
        // states saved on the worker's disk are not kept for Inkling: the main machine reads a prompt again instead
        Ok(false)
    }

    fn head_rows(&self, _xq: kernels::Q8s, _r0: usize, _r1: usize) -> Result<Vec<f32>> {
        err("two boxes (pipeline): Inkling's head is not split between the machines")
    }

    fn hc_back(
        &mut self,
        tokens: &[u32],
        pos0: u32,
        l0: usize,
        res: Vec<f32>,
        want: u32,
        _early: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        let total = self.m.n_calls();
        if l0 == 0 || l0 >= total || pos0 > self.m.n_ctx {
            return err(format!("bad pipeline request: split {l0}, position {pos0} ({total} passes)"));
        }
        if n == 0 {
            // control requests: a new sequence, or a mark kept or gone back to
            if want & sp::OP_RESET != 0 {
                self.m.reset();
            } else if want & sp::OP_SAVE != 0 {
                self.slots[sp::slot_of(want)] = Some(self.m.conv_state());
            } else if want & sp::OP_LOAD != 0 {
                let k = sp::slot_of(want);
                let Some(st) = &self.slots[k] else { return err(format!("no mark in slot {k}")) };
                self.m.set_conv_state(st)?;
            } else {
                return err(format!("bad pipeline request: control {want:#x} is not one Inkling serves"));
            }
            return Ok(Vec::new());
        }
        let mode = want & 3;
        let k = (want >> sp::TOPK_SHIFT) as usize;
        if want & (sp::WANT_HIDDEN | sp::WANT_STATES) != 0
            || (want >> sp::HEAD_SPLIT_SHIFT) & sp::HEAD_SPLIT_MASK != 0
            || mode > 1
            || (k > 0 && mode != 1)
        {
            return err(format!("bad pipeline request: want {want:#x} is not one Inkling serves"));
        }
        let carry = Carry::from_f32(&res, n * self.m.h.d_model)?;
        if mode == 0 {
            self.m.forward_range(tokens, pos0, Some(carry), l0, total, None)?;
            return Ok(Vec::new());
        }
        self.m.forward_range(tokens, pos0, Some(carry), l0, total, Some((&mut self.logits, false)))?;
        Ok(if k > 0 { sp::topk(&self.logits, k.min(self.logits.len())) } else { self.logits.clone() })
    }
}

/// The worker: serves the main machine's requests through the shared pipeline (pipe::serve) until it is stopped.
pub fn serve(m: &mut Ink, port: u16) -> Result<()> {
    let nv = m.n_vocab();
    sp::serve(&mut InkWorker { m, slots: Default::default(), logits: vec![0.0; nv] }, port)
}
