//! pipe.rs - Inkling 2 over two machines as a layer pipeline, as runtime/pipe.rs does for the Qwen family: the main
//! machine runs a block's layer passes 0..S (Ink::forward_range) and sends what they carry (Carry: the stream, the
//! earlier blocks' outputs and the loop's carried inputs) to the worker, which runs S..end and the head and sends the
//! logits asked for. A prompt goes block by block with the main machine already on the next block while the worker
//! does the last one; a token waits for its reply (one hop each way). Every pass runs the same code on the same inputs
//! on whichever machine runs it, so the bits are the same as on one machine. Each machine keeps the keys, values and
//! convolution state of its own passes.
//!
//! Worker: `torc-ink <model> serve=<port>` (ctx=). Main: TORC_INK_PIPE=host:port, TORC_INK_SPLIT=S (default 13 of every 24
//! passes), TORC_INK_PIPE_TOPK=K (a token's K largest logits only). A request is MAGIC, op, then the op's words: RUN
//! pos0, n, S, want (0: no logits, 1: the last row's, 2 + K: its K largest as (index, value) pairs), the
//! carry's length and its f32s; STATE_GET; STATE_SET with the length and the f32s; HELLO with the main machine's
//! n_vocab, d_model and pass count. Every request is answered with a count and that many f32s (HELLO: one, 1 if the
//! worker's model has the same shape).

use super::inkling::{Carry, Ink};
use format::{Result, err};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};

const MAGIC: u32 = 0x3150_4B49;
const RUN: u32 = 1;
const STATE_GET: u32 = 2;
const STATE_SET: u32 = 3;
const HELLO: u32 = 4;

fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| format::Error(format!("inkling pipe: {e}")))
}

fn put_u32(w: &mut impl Write, v: &[u32]) -> Result<()> {
    for x in v {
        io(w.write_all(&x.to_le_bytes()))?;
    }
    Ok(())
}

fn put_f32(w: &mut impl Write, v: &[f32]) -> Result<()> {
    put_u32(w, &[v.len() as u32])?;
    // SAFETY: f32 has no padding; the bytes are little-endian on both machines (ARM64 and x86-64)
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    io(w.write_all(b))
}

fn get_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    io(r.read_exact(&mut b))?;
    Ok(u32::from_le_bytes(b))
}

fn get_f32(r: &mut impl Read, into: Option<&mut [f32]>) -> Result<Vec<f32>> {
    let n = get_u32(r)? as usize;
    let mut v = vec![0.0f32; n];
    // SAFETY: as put_f32
    let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
    io(r.read_exact(b))?;
    if let Some(o) = into {
        let k = o.len().min(n);
        o[..k].copy_from_slice(&v[..k]);
    }
    Ok(v)
}

/// The main machine's end of the pipe.
pub struct Pipe {
    r: BufReader<TcpStream>,
    w: BufWriter<TcpStream>,
    pub split: usize,
    /// RUN requests sent without waiting (no logits asked), whose empty replies are still to be read
    pending: usize,
    /// TORC_INK_PIPE_TOPK=K: a token's logits come back as their K largest (index, value) only, the rest -infinity
    /// (the full row is 0.8 MB a token over the link); 0 (the default) brings the whole row
    topk: usize,
}

impl Pipe {
    /// TORC_INK_PIPE (host:port) when set: connects and checks the worker's model has m's shape.
    pub fn from_env(m: &Ink) -> Result<Option<Pipe>> {
        let Some(addr) = crate::switches::get("TORC_INK_PIPE") else { return Ok(None) };
        let total = m.n_calls();
        // by default the main machine takes 13 of every 24 passes: the later half holds the heavier layers (the
        // last layers' experts are Q8_0), and 26 of 48 read fastest on the boxes (27 Nov: 4.3k 121.7 tok/s, 24: 118.5,
        // 28: 113.5)
        let split = crate::switches::num::<usize>("TORC_INK_SPLIT").filter(|&s| s > 0 && s < total).unwrap_or(total * 13 / 24);
        let st = io(TcpStream::connect(&addr))?;
        io(st.set_nodelay(true))?;
        let topk = crate::switches::num::<usize>("TORC_INK_PIPE_TOPK").unwrap_or(0);
        let mut p = Pipe {
            r: BufReader::with_capacity(1 << 20, io(st.try_clone())?),
            w: BufWriter::with_capacity(1 << 20, st),
            split,
            pending: 0,
            topk,
        };
        put_u32(&mut p.w, &[MAGIC, HELLO, m.n_vocab() as u32, m.h.d_model as u32, total as u32])?;
        io(p.w.flush())?;
        let ok = get_f32(&mut p.r, None)?;
        if ok.first() != Some(&1.0) {
            return err("inkling pipe: the worker has another model");
        }
        eprintln!("inkling pipe: worker {addr}, passes 0..{split} here, {split}..{total} there");
        Ok(Some(p))
    }

    fn drain(&mut self) -> Result<()> {
        while self.pending > 0 {
            get_f32(&mut self.r, None)?;
            self.pending -= 1;
        }
        Ok(())
    }

    /// A block through both machines: passes 0..split here, the rest there. With `logits`, waits for the last row's;
    /// without, returns once the carry is sent (the worker's reply is read later).
    pub fn forward(&mut self, m: &mut Ink, tokens: &[u32], pos0: u32, logits: Option<&mut [f32]>) -> Result<()> {
        let carry = m.forward_range(tokens, pos0, None, 0, self.split, None)?.expect("a carry short of the end");
        // 0: none, 1: the whole row, 2 + K: the K largest
        let want = match (&logits, self.topk) {
            (None, _) => 0,
            (Some(_), 0) => 1,
            (Some(_), k) => 2 + k as u32,
        };
        put_u32(&mut self.w, &[MAGIC, RUN, pos0, tokens.len() as u32, self.split as u32, want])?;
        put_f32(&mut self.w, &carry.to_f32())?;
        io(self.w.flush())?;
        match logits {
            Some(out) => {
                self.drain()?;
                if self.topk == 0 {
                    get_f32(&mut self.r, Some(out))?;
                } else {
                    let v = get_f32(&mut self.r, None)?;
                    out.fill(f32::NEG_INFINITY);
                    for pair in v.chunks_exact(2) {
                        if let Some(o) = out.get_mut(pair[0].to_bits() as usize) {
                            *o = pair[1];
                        }
                    }
                }
            }
            None => self.pending += 1,
        }
        Ok(())
    }

    /// The worker's convolution state (its passes'), for a mark.
    pub fn state(&mut self) -> Result<Vec<f32>> {
        self.drain()?;
        put_u32(&mut self.w, &[MAGIC, STATE_GET])?;
        io(self.w.flush())?;
        get_f32(&mut self.r, None)
    }

    /// Puts the worker's convolution state back (a mark's).
    pub fn set_state(&mut self, s: &[f32]) -> Result<()> {
        self.drain()?;
        put_u32(&mut self.w, &[MAGIC, STATE_SET])?;
        put_f32(&mut self.w, s)?;
        io(self.w.flush())?;
        get_f32(&mut self.r, None).map(|_| ())
    }
}

/// The worker: serves one main machine at a time on `port`, forever.
pub fn serve(m: &mut Ink, port: u16) -> Result<()> {
    let l = io(TcpListener::bind(("0.0.0.0", port)))?;
    eprintln!("inkling pipe worker on port {port}");
    let dm = m.h.d_model;
    let nv = m.n_vocab();
    let total = m.n_calls();
    for st in l.incoming() {
        let Ok(st) = st else { continue };
        let _ = st.set_nodelay(true);
        let (Ok(rs), ws) = (st.try_clone(), st) else { continue };
        let (mut r, mut w) = (BufReader::with_capacity(1 << 20, rs), BufWriter::with_capacity(1 << 20, ws));
        eprintln!("inkling pipe: main machine connected");
        let mut logits = vec![0.0f32; nv];
        let res: Result<()> = (|| loop {
            if get_u32(&mut r)? != MAGIC {
                return err("bad request");
            }
            match get_u32(&mut r)? {
                HELLO => {
                    let (a, b, c) = (get_u32(&mut r)?, get_u32(&mut r)?, get_u32(&mut r)?);
                    let same = a as usize == nv && b as usize == dm && c as usize == total;
                    put_f32(&mut w, &[same as u32 as f32])?;
                }
                RUN => {
                    let (pos0, n, from, want) = (get_u32(&mut r)?, get_u32(&mut r)? as usize, get_u32(&mut r)?, get_u32(&mut r)?);
                    let v = get_f32(&mut r, None)?;
                    let carry = Carry::from_f32(&v, n * dm)?;
                    let toks = vec![0u32; n];
                    m.forward_range(&toks, pos0, Some(carry), from as usize, total, Some((&mut logits, false)))?;
                    match want {
                        0 => put_f32(&mut w, &[])?,
                        1 => put_f32(&mut w, &logits)?,
                        k => {
                            // the K largest, as (index bits, value) pairs, largest first (an equal one: the lower index)
                            let k = (k as usize - 2).min(nv);
                            let mut idx: Vec<u32> = (0..nv as u32).collect();
                            idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
                                logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b))
                            });
                            let mut top: Vec<u32> = idx[..k].to_vec();
                            top.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b)));
                            let v: Vec<f32> = top.iter().flat_map(|&i| [f32::from_bits(i), logits[i as usize]]).collect();
                            put_f32(&mut w, &v)?;
                        }
                    }
                }
                STATE_GET => put_f32(&mut w, &m.conv_state())?,
                STATE_SET => {
                    let s = get_f32(&mut r, None)?;
                    m.set_conv_state(&s)?;
                    put_f32(&mut w, &[])?;
                }
                op => return err(&format!("unknown op {op}")),
            }
            io(w.flush())?;
        })();
        if let Err(e) = res {
            eprintln!("inkling pipe: connection ended ({})", e.0);
        }
    }
    Ok(())
}
