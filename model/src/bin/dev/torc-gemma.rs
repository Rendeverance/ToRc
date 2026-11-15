//! torc-gemma <model> <tokens.int32> [n] [threads] [ref=logits.f32] [dump=logits.f32] [block=B] [ctx=N]: runs Gemma 4
//! (gemma.rs) one token at a time. Given ref=, a float32 file of n x n_vocab logits, it reports for each position
//! whether the bits match, whether the top token agrees, and the worst difference. block=B adds a second pass that
//! feeds the same tokens B at a time, as prefill does, and checks it against the first: any bit that differs exits 1.

use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(2)
    });
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

use model::ops::argmax;

/// Compares two runs' logits position by position and prints one line: which positions differ in any bit, where the
/// first one is, how many pick a different greedy token, and the biggest gap seen. True when every bit matches.
fn compare(a: &[f32], b: &[f32], nv: usize, what: &str) -> bool {
    let n = a.len().min(b.len()) / nv;
    let ra: Vec<&[f32]> = a.chunks_exact(nv).take(n).collect();
    let rb: Vec<&[f32]> = b.chunks_exact(nv).take(n).collect();
    let differ: Vec<usize> =
        (0..n).filter(|&p| ra[p].iter().zip(rb[p]).any(|(x, y)| x.to_bits() != y.to_bits())).collect();
    let picks = (0..n).filter(|&p| argmax(ra[p]) != argmax(rb[p])).count();
    let gap =
        ra.iter().zip(&rb).flat_map(|(x, y)| x.iter().zip(y.iter())).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    match differ.first() {
        None => println!("{what}: all {n} positions bit-identical"),
        Some(first) => println!(
            "{what}: {} of {n} positions differ (first at {first}), {picks} pick another token, biggest gap {gap:.3e}",
            differ.len()
        ),
    }
    differ.is_empty()
}

/// The mean negative log-likelihood of each next token of the sequence under the logits (perplexity = e^nll).
fn nll(all: &[f32], tokens: &[u32], nv: usize) -> f64 {
    let n = (all.len() / nv).min(tokens.len());
    let mut sum = 0.0f64;
    for p in 0..n - 1 {
        let l = &all[p * nv..(p + 1) * nv];
        let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z: f64 = l.iter().map(|&v| (v as f64 - mx).exp()).sum();
        sum += -(l[tokens[p + 1] as usize] as f64 - mx - z.ln());
    }
    sum / (n - 1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| args.iter().find_map(|a| a.strip_prefix(k)).map(|s| s.to_string());
    let pos: Vec<&String> = args.iter().filter(|a| !a.contains('=')).collect();
    if pos.len() < 2 {
        eprintln!("usage: torc-gemma <model.torc> <tokens.int32> [n] [threads] [ref=F] [dump=F] [block=B] [ctx=N]");
        std::process::exit(2);
    }
    let raw = std::fs::read(pos[1]).unwrap_or_else(|e| {
        eprintln!("{}: {e}", pos[1]);
        std::process::exit(2)
    });
    let mut tokens: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if let Some(n) = pos.get(2).and_then(|s| s.parse::<usize>().ok()) {
        tokens.truncate(n);
    }
    let asked: usize = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let threads = model::pool::init(asked).threads as i32;
    // serve-ep=PORT: the other machine of a two-machine prompt (ep.rs), serving its half of the routed experts until it
    // is stopped. It loads the weights alone (no forward pass); each request's flags say GELU
    if let Some(port) = opt("serve-ep=").and_then(|v| v.parse::<u16>().ok()) {
        let w = model::weights::Weights::load(pos[0]).unwrap_or_else(|e| {
            eprintln!("{}: {}", pos[0], e.0);
            std::process::exit(1)
        });
        eprintln!("serving experts of {} on port {port}", pos[0]);
        if let Err(e) = model::ep::serve(&w, port) {
            eprintln!("serve-ep: {}", e.0);
            std::process::exit(1);
        }
        return;
    }
    let n_ctx: u32 = opt("ctx=").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let t0 = Instant::now();
    let mut m = model::gemma::Gemma::load(pos[0], n_ctx).unwrap_or_else(|e| {
        eprintln!("{}: {}", pos[0], e.0);
        std::process::exit(1)
    });
    let nv = m.n_vocab().unwrap();
    eprintln!(
        "loaded in {:.1} s: {} layers, d_model {}, vocab {nv}, threads {threads}, hp {:?}",
        t0.elapsed().as_secs_f64(),
        m.h.n_layer,
        m.h.d_model,
        (&m.h.full, &m.h.swa, m.h.window, m.h.softcap)
    );
    // ask=QUESTION: one chat turn (Gemma 4's <|turn> format), answered greedily (gen=G tokens, 64 by default) and
    // printed - the eyeball check that the model talks sense
    if let Some(q) = opt("ask=") {
        let tok = model::tok::Tokenizer::open(pos[0]).unwrap();
        let mut ids = vec![2u32];
        ids.extend(tok.encode(&format!("<|turn>user\n{q}<turn|>\n<|turn>model\n"), true).unwrap());
        let mut last = vec![0.0f32; nv];
        m.forward(&ids, 0, &mut last, false).unwrap();
        let g: usize = opt("gen=").and_then(|v| v.parse().ok()).unwrap_or(64);
        let mut out = Vec::new();
        let mut p = ids.len();
        for _ in 0..g {
            let t = argmax(&last) as u32;
            out.extend(tok.decode(t));
            if tok.decode(t) == b"<turn|>" {
                break;
            }
            m.forward(&[t], p as u32, &mut last, false).unwrap();
            p += 1;
        }
        println!("prompt {} tokens; answer: {}", ids.len(), String::from_utf8_lossy(&out));
        return;
    }
    let n = tokens.len();
    // pp=B: a prompt read as it really is, in blocks of B, with the logits of its last token only (as llama-bench and
    // geniex-bench time prefill), then gen=G tokens generated greedily from it; nothing else
    if let Some(b) = opt("pp=").and_then(|v| v.parse::<usize>().ok()).filter(|&b| b > 0) {
        // runs=R: the whole thing R times over the same positions (as geniex-bench -r does), each run reported
        let runs: usize = opt("runs=").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
        for run in 0..runs {
            if runs > 1 {
                println!("run {}:", run + 1);
            }
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
            // TORC_PROF=1: where the time went, as milliseconds per token for each counter (the counters reset as read)
            let prof = |what: &str, per: usize| {
                if !model::switches::get("TORC_PROF").is_some_and(|v| v != "0") {
                    return;
                }
                let ms = |c: &std::sync::atomic::AtomicU64| {
                    c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / per as f64
                };
                let (_, moe_names, moe_ns) = model::phase_tables()[0];
                let cells: Vec<String> = model::gemma::PHASES
                    .iter()
                    .zip(model::gemma::PROF.iter())
                    .chain(moe_names.iter().zip(moe_ns.iter()))
                    .map(|(name, c)| format!("{name}={:.2}", ms(c)))
                    .collect();
                println!("[{what}] ms per token: {}", cells.join(" "));
            };
            prof("prompt", n);
            let g: usize = opt("gen=").and_then(|v| v.parse().ok()).unwrap_or(0);
            if g > 0 {
                let t = Instant::now();
                let mut tok = argmax(&last) as u32;
                for i in 0..g {
                    m.forward(&[tok], (n + i) as u32, &mut last, false).unwrap();
                    tok = argmax(&last) as u32;
                }
                let dt = t.elapsed().as_secs_f64();
                println!("generated {g} tokens: {dt:.2} s ({:.1} token/s)", g as f64 / dt);
                prof("generate", g);
            }
        }
        if let Some((wait, graphs)) = model::npu::stats() {
            println!("npu: {graphs} graphs, {wait:.2} s waited for");
        }
        // TORC_NPU_PROFILE_OUT=file: write the counted expert use, for the NPU's hot experts (TORC_NPU_PROFILE)
        model::npu::write_profile(m.h.n_layer as u32, m.n_expert());
        return;
    }
    // every token on its own, each one timed: the median is the steady rate, without the first tokens' warm-up
    let mut all = vec![0.0f32; n * nv];
    let mut each_ms = Vec::with_capacity(n);
    for (p, &tok) in tokens.iter().enumerate() {
        let t = Instant::now();
        m.forward(&[tok], p as u32, &mut all[p * nv..(p + 1) * nv], false).unwrap_or_else(|e| {
            eprintln!("pos {p}: {}", e.0);
            std::process::exit(1)
        });
        each_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let total: f64 = each_ms.iter().sum::<f64>() / 1e3;
    let mut sorted = each_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    println!("token by token: {n} in {total:.1} s, median {median:.1} ms ({:.1} token/s)", 1e3 / median);
    if let Some(d) = opt("dump=") {
        let b: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&d, b).expect("dump");
    }
    if let Some(r) = opt("ref=") {
        compare(&all, &read_f32(&r), nv, "vs reference");
    }
    {
        // how the model does on the text itself: next-token top-1 accuracy, and the mean and median loss
        let mut losses = Vec::new();
        let mut hit = 0;
        for p in 0..n - 1 {
            let l = &all[p * nv..(p + 1) * nv];
            hit += (argmax(l) == tokens[p + 1] as usize) as usize;
            let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z: f64 = l.iter().map(|&v| (v as f64 - mx).exp()).sum();
            losses.push(-(l[tokens[p + 1] as usize] as f64 - mx - z.ln()));
        }
        // llama-perplexity's view of a 512-token context (BOS + 511 tokens): the predictions of positions 256..511.
        // pplctx=C scores its first chunk, BOS + C - 1 tokens, from position C/2 on, as llama-perplexity does
        let c: usize = opt("pplctx=").and_then(|v| v.parse().ok()).unwrap_or(512);
        if losses.len() >= c - 1 {
            let w = &losses[c / 2 - 1..c - 1];
            let m = w.iter().sum::<f64>() / w.len() as f64;
            println!("llama-perplexity chunk 1 (ctx {c}, second half): loss {m:.4}, ppl {:.3}", m.exp());
        }
        losses.sort_by(|a, b| a.total_cmp(b));
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        println!(
            "text: top-1 {hit}/{} ({:.1}%), loss mean {mean:.3} median {:.3}, worst tenth from {:.2}",
            n - 1,
            100.0 * hit as f64 / (n - 1) as f64,
            losses[losses.len() / 2],
            losses[losses.len() * 9 / 10]
        );
    }
    if let Some(b) = opt("block=").and_then(|v| v.parse::<usize>().ok()).filter(|&b| b > 0) {
        let mut blk = vec![0.0f32; n * nv];
        let t2 = Instant::now();
        let mut p0 = 0;
        while p0 < n {
            let len = b.min(n - p0);
            m.forward(&tokens[p0..p0 + len], p0 as u32, &mut blk[p0 * nv..(p0 + len) * nv], true).unwrap_or_else(|e| {
                eprintln!("block at {p0}: {}", e.0);
                std::process::exit(1)
            });
            p0 += len;
        }
        let dt = t2.elapsed().as_secs_f64();
        println!("blocks of {b}: {n} tokens in {dt:.2} s ({:.1} token/s)", n as f64 / dt);
        let same = compare(&blk, &all, nv, "blocks vs tokens");
        let (a, b) = (nll(&all, &tokens, nv), nll(&blk, &tokens, nv));
        println!("next-token NLL: tokens {a:.4} (ppl {:.3}), blocks {b:.4} (ppl {:.3})", a.exp(), b.exp());
        // reading in blocks must give the same bits as token by token: a difference is a failure, so torc-golden's
        // block entry fails with it instead of passing on the token-by-token dump alone
        if !same {
            std::process::exit(1);
        }
    }
}
