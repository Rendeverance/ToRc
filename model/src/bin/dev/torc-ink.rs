//! torc-ink <model> [text=PROMPT | toks=tokens.int32] [n=N] [gen=G] [block=B] [ref=logits.f32] [dump=logits.f32]
//! [ctx=N]: runs Inkling 2 Mini (inkling.rs). The prompt's tokens (text= goes through the model's tokeniser as it
//! stands, special tokens read; toks= is a file of token ids; n= takes the first N) are read one at a time, and the
//! token each position would pick is printed. ref= compares the logits of every position against a float32 file of
//! n x n_vocab (from llama.cpp, say); dump= writes ours. block=B reads the same tokens again B at a time and checks
//! that against the first pass: any bit that differs exits 1. gen=G then writes G tokens greedily.

use model::models::inkling::inkling::Ink;
use model::ops::argmax;
use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(2)
    });
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
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
    let mut ids: Vec<u32> = match (opt("text"), opt("toks")) {
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
    for (p, &id) in ids.iter().enumerate() {
        m.forward(&[id], p as u32, &mut all[p * nv..(p + 1) * nv], false).unwrap_or_else(|e| fail(e));
        let top = argmax(&all[p * nv..(p + 1) * nv]) as u32;
        println!("  {p:4}  {id:6} {:?} -> {top:6} {:?}  ({:.3})", show(id), show(top), all[p * nv + top as usize]);
    }
    let s = t1.elapsed().as_secs_f64();
    println!("read {} tokens one at a time in {s:.2} s ({:.2} tok/s)", ids.len(), ids.len() as f64 / s);
    if let Some(f) = opt("dump") {
        let b: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&f, b).unwrap();
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
    if !ok {
        std::process::exit(1);
    }
}
