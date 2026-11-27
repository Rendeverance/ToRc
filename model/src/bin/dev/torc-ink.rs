//! torc-ink <model> [text=PROMPT | toks=tokens.int32] [n=N] [gen=G] [block=B] [first=B] [ref=logits.f32] [dump=logits.f32]
//! [ctx=N]: runs Inkling 2 Mini (inkling.rs). The prompt's tokens (text= goes through the model's tokeniser as it
//! stands, special tokens read; toks= is a file of token ids; n= takes the first N) are read one at a time, and the
//! token each position would pick is printed. ref= compares the logits of every position against a float32 file of
//! n x n_vocab (from llama.cpp, say); dump= writes ours. block=B reads the same tokens again B at a time and checks
//! that against the first pass: any bit that differs exits 1. gen=G then writes G tokens greedily.

use model::models::inkling::inkling::Ink;
use model::ops::argmax;
use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    // read in pieces: the whole file as bytes beside its floats doubled an 8k prompt's 6.9 GB
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(2)
    });
    let mut left = f.metadata().map(|m| m.len() as usize).unwrap_or(0) / 4 * 4;
    let mut out = Vec::with_capacity(left / 4);
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let k = left.min(buf.len());
        f.read_exact(&mut buf[..k]).unwrap();
        out.extend(buf[..k].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
        left -= k;
    }
    out
}

/// Two runs' logits compared position by position: how many positions differ in any bit, how many pick another
/// greedy token, and the biggest gap. True when no bit differs.
fn compare(a: &[f32], b: &[f32], nv: usize, what: &str) -> bool {
    let n = a.len().min(b.len()) / nv;
    let differ = (0..n)
        .filter(|&p| {
            a[p * nv..(p + 1) * nv].iter().zip(&b[p * nv..(p + 1) * nv]).any(|(x, y)| x.to_bits() != y.to_bits())
        })
        .count();
    let picks = (0..n).filter(|&p| argmax(&a[p * nv..(p + 1) * nv]) != argmax(&b[p * nv..(p + 1) * nv])).count();
    let gap = a[..n * nv].iter().zip(&b[..n * nv]).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("{what}: {differ} of {n} positions differ in some bit, {picks} pick another token, biggest gap {gap:.3e}");
    differ == 0
}

/// kbench=1: the one-token products alone, as GB/s of weights read: the experts' gate/up rows (IQ4_NL, 8 experts of a
/// layer, as one token's decode asks for them) and the layers' attn_q (Q8_0), each over the whole pool with one
/// quantised vector, over every layer.
/// kbench=2: 24 rounds, each asking for 8 other experts a layer (all 128 touched by round 16), as real decode does,
/// instead of the same 8; from round 16 on they are warm but not the last round's.
fn kbench(m: &Ink, spread: bool) {
    use format::BlockKind;
    use model::moe::{Mat, apply_group};
    let dm = m.h.d_model;
    let x: Vec<f32> = (0..dm).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
    let mut xq = kernels::Q8::new(dm / kernels::QK);
    kernels::ops::q8k_quantize(&x, xq.view_mut());
    let ff = m.h.d_ff;
    let rounds = if spread { 24 } else { 4 };
    for round in 0..rounds {
        // experts: 8 a layer (ids spread over the 128), gate and up of each, the real ff rows
        let (mut bytes, t) = (0usize, Instant::now());
        for l in 0..m.h.n_layer as u32 {
            let g = m.w.file.blocks.iter().find(|b| b.layer == l && b.kind == BlockKind::Moe).and_then(|b| b.experts);
            let Some(g) = g else { continue };
            let ids: Vec<u32> = (0..8u32).map(|i| i * 16 + if spread { (round as u32 * 5) % 16 } else { 3 }).collect();
            let packs = m.w.experts_of(l, g, &ids).unwrap();
            let mut outs = vec![vec![0.0f32; ff]; 16];
            let mut jobs: Vec<(Mat, &mut [f32])> = Vec::new();
            for (i, o) in outs.iter_mut().enumerate() {
                let (pk, pack) = packs[i / 2];
                let p = if i % 2 == 0 { &pack.gate } else { &pack.up };
                let rb = p.bytes as usize / p.rows as usize;
                let data = &pk[p.at as usize..p.at as usize + ff * rb];
                bytes += data.len();
                jobs.push((Mat { ty: p.ty, rows: ff, cols: p.cols as usize, data }, &mut o[..]));
            }
            apply_group(&mut jobs, xq.view());
        }
        let s = t.elapsed().as_secs_f64();
        println!("experts gate/up: {:.0} MB in {:.2} ms = {:.1} GB/s", bytes as f64 / 1e6, s * 1e3, bytes as f64 / s / 1e9);
        // attn_q of every layer (Q8_0)
        let (mut bytes, t) = (0usize, Instant::now());
        for l in 0..m.h.n_layer as u32 {
            let tq = m.w.layer(l, "attn_q.weight").unwrap();
            let mm = Mat::of(&tq);
            bytes += mm.data.len();
            let mut y = vec![0.0f32; mm.rows];
            apply_group(&mut [(mm, &mut y[..])], xq.view());
        }
        let s = t.elapsed().as_secs_f64();
        println!("attn_q (Q8_0): {:.0} MB in {:.2} ms = {:.1} GB/s", bytes as f64 / 1e6, s * 1e3, bytes as f64 / s / 1e9);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt =
        |k: &str| args.iter().find_map(|a| a.strip_prefix(k).and_then(|r| r.strip_prefix('=')).map(str::to_string));
    let Some(path) = args.first().filter(|a| !a.contains('=')) else {
        eprintln!(
            "usage: torc-ink <model> [text=PROMPT | toks=tokens.int32] [n=N] [gen=G] [block=B] [ref=F] [dump=F] [ctx=N]"
        );
        std::process::exit(2)
    };
    let fail = |e: format::Error| -> ! {
        eprintln!("torc-ink: {}", e.0);
        std::process::exit(1)
    };
    let tok = model::tok::Tokenizer::open(path).unwrap_or_else(|e| fail(e));
    // textfile=F: the prompt's text from a file (an 8k prompt is too long for a command line)
    let text = opt("text").or_else(|| opt("textfile").map(|f| {
        std::fs::read_to_string(&f).unwrap_or_else(|e| {
            eprintln!("{f}: {e}");
            std::process::exit(2)
        })
    }));
    let mut ids: Vec<u32> = match (text, opt("toks")) {
        (Some(t), _) => tok.encode(&t, true).unwrap_or_else(|e| fail(e)),
        (None, Some(f)) => std::fs::read(&f)
            .unwrap_or_else(|e| {
                eprintln!("{f}: {e}");
                std::process::exit(2)
            })
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        _ => tok.encode("The capital of France is", true).unwrap_or_else(|e| fail(e)),
    };
    if let Some(n) = opt("n").and_then(|v| v.parse::<usize>().ok()) {
        ids.truncate(n);
    }
    let gen_n: usize = opt("gen").and_then(|v| v.parse().ok()).unwrap_or(0);
    let ctx: u32 = opt("ctx").and_then(|v| v.parse().ok()).unwrap_or((ids.len() + gen_n + 16).max(512) as u32);
    let t0 = Instant::now();
    let mut m = Ink::load(path, ctx).unwrap_or_else(|e| fail(e));
    let nv = m.n_vocab();
    if opt("kbench").is_some() {
        kbench(&m, opt("kbench").as_deref() == Some("2"));
        return;
    }
    println!(
        "loaded in {:.1} s: {} layer passes, {} tokens to read: {:?}",
        t0.elapsed().as_secs_f64(),
        m.h.n_pass,
        ids.len(),
        ids
    );
    let show = |id: u32| String::from_utf8_lossy(&tok.decode(id)).into_owned();
    // one token at a time
    let mut all = vec![0.0f32; ids.len() * nv];
    let t1 = Instant::now();
    // first=B: read the prompt in blocks of B only (no one-at-a-time pass, no check), for writing at depth
    let first: usize = opt("first").and_then(|v| v.parse().ok()).unwrap_or(0);
    if first > 0 {
        let mut p = 0;
        for ch in ids.chunks(first) {
            m.forward(ch, p as u32, &mut all[p * nv..(p + ch.len()) * nv], true).unwrap_or_else(|e| fail(e));
            p += ch.len();
        }
    } else {
        for (p, &id) in ids.iter().enumerate() {
            m.forward(&[id], p as u32, &mut all[p * nv..(p + 1) * nv], false).unwrap_or_else(|e| fail(e));
            let top = argmax(&all[p * nv..(p + 1) * nv]) as u32;
            println!("  {p:4}  {id:6} {:?} -> {top:6} {:?}  ({:.3})", show(id), show(top), all[p * nv + top as usize]);
        }
    }
    let s = t1.elapsed().as_secs_f64();
    let how = if first > 0 { format!("in blocks of {first}") } else { "one at a time".to_string() };
    println!("read {} tokens {how} in {s:.2} s ({:.2} tok/s)", ids.len(), ids.len() as f64 / s);
    if let Some(f) = opt("dump") {
        // streamed: a byte copy of an 8k prompt's logits (6.9 GB) beside them ran a 32 GB box out of memory
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&f).unwrap());
        for v in &all {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        w.flush().unwrap();
        println!("logits written to {f}");
    }
    if let Some(f) = opt("ref") {
        compare(&all, &read_f32(&f), nv, "against the reference");
    }
    let mut ok = true;
    if let Some(b) = opt("block").and_then(|v| v.parse::<usize>().ok()).filter(|&b| b > 0) {
        let mut again = vec![0.0f32; ids.len() * nv];
        let t2 = Instant::now();
        let mut p = 0;
        for ch in ids.chunks(b) {
            m.forward(ch, p as u32, &mut again[p * nv..(p + ch.len()) * nv], true).unwrap_or_else(|e| fail(e));
            p += ch.len();
        }
        let s = t2.elapsed().as_secs_f64();
        println!("read again in blocks of {b} in {s:.2} s ({:.2} tok/s)", ids.len() as f64 / s);
        ok = compare(&all, &again, nv, "blocks against one at a time");
        // the state is the second pass's: the same sequence, so writing goes on from it
    }
    if gen_n > 0 && !ids.is_empty() {
        let mut next = argmax(&all[(ids.len() - 1) * nv..]) as u32;
        let mut lg = vec![0.0f32; nv];
        // the profile counts the writing only
        for c in model::models::inkling::inkling::PROF.iter() {
            c.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let t3 = Instant::now();
        let mut text = Vec::new();
        for g in 0..gen_n {
            text.extend(tok.decode(next));
            m.forward(&[next], (ids.len() + g) as u32, &mut lg, false).unwrap_or_else(|e| fail(e));
            next = argmax(&lg) as u32;
        }
        let s = t3.elapsed().as_secs_f64();
        println!(
            "wrote {gen_n} tokens in {s:.2} s ({:.2} tok/s):\n{}",
            gen_n as f64 / s,
            String::from_utf8_lossy(&text)
        );
    }
    // TORC_NPU_PROFILE_OUT=file: the experts this run used, as a usage file for TORC_NPU_PROFILE (the NPU's hot set)
    model::npu::write_profile(m.h.n_layer as u32, m.h.n_expert as u32);
    // vbench=1,2,4,..: what checking drafts would cost. After the prompt (and any writing), a forward of N tokens at
    // the next positions, all their logits out, 16 times for each N; ms a call against N = 1 (timing only: the state
    // after it is not the text's)
    if let Some(list) = opt("vbench") {
        let p0 = (ids.len() + gen_n) as u32;
        let mut base = 0.0f64;
        for nv_s in list.split(',') {
            let Ok(k) = nv_s.parse::<usize>() else { continue };
            let toks: Vec<u32> = ids.iter().copied().cycle().take(k).collect();
            let mut lg = vec![0.0f32; k * nv];
            m.forward(&toks, p0, &mut lg, k > 1).unwrap_or_else(|e| fail(e));
            let t = Instant::now();
            for _ in 0..16 {
                m.forward(&toks, p0, &mut lg, k > 1).unwrap_or_else(|e| fail(e));
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / 16.0;
            if k == 1 {
                base = ms;
            }
            println!("vbench {k} tokens at position {p0}: {ms:.1} ms a call ({:.2}x one token)", ms / base.max(1e-9));
        }
    }
    if model::switches::get("TORC_PROF").is_some_and(|v| v != "0") {
        use model::models::inkling::inkling::{PHASES, PROF};
        let ms = |i: usize| PROF[i].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
        let total: f64 = (0..PHASES.len()).map(ms).sum();
        for (i, name) in PHASES.iter().enumerate() {
            println!("  {:8.1} ms {:5.1}%  {name}", ms(i), 100.0 * ms(i) / total.max(1e-9));
        }
    }
    if !ok {
        std::process::exit(1);
    }
}
