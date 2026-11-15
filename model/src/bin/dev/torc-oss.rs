//! torc-oss <model> <tokens.int32> [n] [threads] [ref=logits.f32] [dump=logits.f32] [block=B] [ctx=N]: runs gpt-oss
//! (gptoss.rs) one token at a time, as torc-gemma does Gemma 4. ref= compares against a float32 file of n x n_vocab
//! logits (from llama.cpp, say); block=B adds a pass in blocks of B and checks it against the first. ask=QUESTION
//! answers one chat turn in the harmony format, greedily (gen=G tokens), printed raw with its channels; pp=B reads the
//! tokens as a prompt in blocks of B and times it, then gen=G tokens.

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
/// greedy token, and the biggest gap.
fn compare(a: &[f32], b: &[f32], nv: usize, what: &str) {
    let n = a.len().min(b.len()) / nv;
    let differ = (0..n)
        .filter(|&p| {
            a[p * nv..(p + 1) * nv].iter().zip(&b[p * nv..(p + 1) * nv]).any(|(x, y)| x.to_bits() != y.to_bits())
        })
        .count();
    let picks = (0..n).filter(|&p| argmax(&a[p * nv..(p + 1) * nv]) != argmax(&b[p * nv..(p + 1) * nv])).count();
    let gap = a[..n * nv].iter().zip(&b[..n * nv]).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("{what}: {differ} of {n} positions differ in some bit, {picks} pick another token, biggest gap {gap:.3e}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| args.iter().find_map(|a| a.strip_prefix(k)).map(|s| s.to_string());
    let pos: Vec<&String> = args.iter().filter(|a| !a.contains('=')).collect();
    if pos.is_empty() || (pos.len() < 2 && opt("ask=").is_none()) {
        eprintln!(
            "usage: torc-oss <model.torc> <tokens.int32> [n] [threads] [ref=F] [dump=F] [block=B] [ctx=N] | ask=Q"
        );
        std::process::exit(2);
    }
    let asked: usize = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let threads = model::pool::init(asked).threads;
    let n_ctx: u32 = opt("ctx=").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let t0 = Instant::now();
    let mut m = model::gptoss::Oss::load(pos[0], n_ctx).unwrap_or_else(|e| {
        eprintln!("{}: {}", pos[0], e.0);
        std::process::exit(1)
    });
    let nv = m.n_vocab().unwrap();
    eprintln!(
        "loaded in {:.1} s: {} layers, d_model {}, vocab {nv}, threads {threads}",
        t0.elapsed().as_secs_f64(),
        m.h.n_layer,
        m.h.d_model
    );
    if let Some(q) = opt("ask=") {
        let tok = model::tok::Tokenizer::open(pos[0]).unwrap();
        let ids = tok.encode(&format!("<|start|>user<|message|>{q}<|end|><|start|>assistant"), true).unwrap();
        let mut last = vec![0.0f32; nv];
        let t = Instant::now();
        m.forward(&ids, 0, &mut last, false).unwrap();
        let tp = t.elapsed().as_secs_f64();
        let g: usize = opt("gen=").and_then(|v| v.parse().ok()).unwrap_or(96);
        let mut out = Vec::new();
        let mut p = ids.len();
        let t = Instant::now();
        let mut made = 0;
        for _ in 0..g {
            let id = argmax(&last) as u32;
            let piece = tok.decode(id);
            out.extend(&piece);
            made += 1;
            if piece == b"<|return|>" {
                break;
            }
            m.forward(&[id], p as u32, &mut last, false).unwrap();
            p += 1;
        }
        let tg = t.elapsed().as_secs_f64();
        println!(
            "prompt {} tokens in {tp:.2} s; {made} tokens in {tg:.2} s ({:.1} token/s)\n{}",
            ids.len(),
            made as f64 / tg,
            String::from_utf8_lossy(&out)
        );
        return;
    }
    let raw = std::fs::read(pos[1]).unwrap_or_else(|e| {
        eprintln!("{}: {e}", pos[1]);
        std::process::exit(2)
    });
    let mut tokens: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if let Some(n) = pos.get(2).and_then(|s| s.parse::<usize>().ok()) {
        tokens.truncate(n);
    }
    let n = tokens.len();
    if let Some(b) = opt("pp=").and_then(|v| v.parse::<usize>().ok()).filter(|&b| b > 0) {
        let mut last = vec![0.0f32; nv];
        let t = Instant::now();
        let mut p0 = 0;
        while p0 < n {
            let len = b.min(n - p0);
            m.forward(&tokens[p0..p0 + len], p0 as u32, &mut last, false).unwrap();
            p0 += len;
        }
        let dt = t.elapsed().as_secs_f64();
        println!("prompt {n} tokens in blocks of {b}: {dt:.2} s ({:.1} token/s)", n as f64 / dt);
        let g: usize = opt("gen=").and_then(|v| v.parse().ok()).unwrap_or(0);
        if g > 0 {
            let t = Instant::now();
            let mut id = argmax(&last) as u32;
            for i in 0..g {
                m.forward(&[id], (n + i) as u32, &mut last, false).unwrap();
                id = argmax(&last) as u32;
            }
            let dt = t.elapsed().as_secs_f64();
            println!("generated {g} tokens: {dt:.2} s ({:.1} token/s)", g as f64 / dt);
        }
        if model::switches::get("TORC_PROF").is_some_and(|v| v != "0") {
            let cells: Vec<String> = model::gptoss::PHASES
                .iter()
                .zip(model::gptoss::PROF.iter())
                .map(|(name, c)| format!("{name}={:.2}s", c.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9))
                .collect();
            println!("phases: {}", cells.join(" "));
        }
        return;
    }
    let mut all = vec![0.0f32; n * nv];
    let t = Instant::now();
    for (p, &id) in tokens.iter().enumerate() {
        m.forward(&[id], p as u32, &mut all[p * nv..(p + 1) * nv], false).unwrap_or_else(|e| {
            eprintln!("pos {p}: {}", e.0);
            std::process::exit(1)
        });
    }
    let dt = t.elapsed().as_secs_f64();
    println!("token by token: {n} in {dt:.1} s ({:.1} token/s)", n as f64 / dt);
    if let Some(d) = opt("dump=") {
        let b: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&d, b).expect("dump");
    }
    if let Some(r) = opt("ref=") {
        compare(&all, &read_f32(&r), nv, "vs reference");
    }
    if n > 1 {
        let mut hit = 0;
        let mut loss = 0.0f64;
        for p in 0..n - 1 {
            let l = &all[p * nv..(p + 1) * nv];
            hit += (argmax(l) == tokens[p + 1] as usize) as usize;
            let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z: f64 = l.iter().map(|&v| (v as f64 - mx).exp()).sum();
            loss += -(l[tokens[p + 1] as usize] as f64 - mx - z.ln());
        }
        let mean = loss / (n - 1) as f64;
        println!("text: top-1 {hit}/{}, loss {mean:.3}, ppl {:.3}", n - 1, mean.exp());
    }
    if let Some(b) = opt("block=").and_then(|v| v.parse::<usize>().ok()).filter(|&b| b > 0) {
        let mut blk = vec![0.0f32; n * nv];
        let t2 = Instant::now();
        let mut p0 = 0;
        while p0 < n {
            let len = b.min(n - p0);
            m.forward(&tokens[p0..p0 + len], p0 as u32, &mut blk[p0 * nv..(p0 + len) * nv], true).unwrap();
            p0 += len;
        }
        let dt = t2.elapsed().as_secs_f64();
        println!("blocks of {b}: {n} tokens in {dt:.2} s ({:.1} token/s)", n as f64 / dt);
        compare(&blk, &all, nv, "blocks vs tokens");
    }
}
