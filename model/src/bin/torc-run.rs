//! torc-run <model.torc> <tokens.int32> [n] [threads|auto] [dump=logits.f32]: decodes the tokens one by one and writes
//! each position's logits (float32 x n x n_vocab). This is the reference format that every other path is compared
//! against.

use model::plan::{Plans, conf};
use std::io::Write;
use std::time::Instant;

fn main() {
    model::pool::timer_1ms_if_asked();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: torc-run <model.torc> <tokens.int32> [n] [threads|auto] [dump=logits.f32]");
        std::process::exit(2);
    }
    let dump = args.iter().find_map(|a| a.strip_prefix("dump="));
    // block=N: read the tokens in blocks of N (Model::prefill, with every position's logits) rather than one by one
    let block: usize = args.iter().find_map(|a| a.strip_prefix("block=")).and_then(|v| v.parse().ok()).unwrap_or(0);
    // the positional arguments are what is left once the name=value options and the flags are taken out
    const OPTIONS: [&str; 10] =
        ["dump=", "block=", "spec=", "gen=", "mtp=", "plan=", "ref=", "serve-ep=", "serve-pipe=", "ctx="];
    const FLAGS: [&str; 5] = ["nologits", "warmup", "pipespec", "lastlogits", "klayall"];
    let pos: Vec<&String> =
        args.iter().filter(|a| !OPTIONS.iter().any(|o| a.starts_with(o)) && !FLAGS.contains(&a.as_str())).collect();
    // a pipeline or expert worker reads no tokens, so the tokens file may be left out
    let serving = args.iter().any(|a| a.starts_with("serve-pipe=") || a.starts_with("serve-ep="));
    let raw = match pos.get(1) {
        Some(p) => std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("cannot read the tokens file {p}: {e}");
            std::process::exit(2)
        }),
        None if serving => Vec::new(),
        None => {
            eprintln!("usage: torc-run <model.torc> <tokens.int32> [n] [threads] [options]");
            std::process::exit(2)
        }
    };
    let mut tokens: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if let Some(n) = pos.get(2).and_then(|s| s.parse::<usize>().ok()) {
        tokens.truncate(n);
    }
    // threads: a number, or auto (the default) for every logical CPU of the fastest core class
    let asked: usize = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let threads = model::pool::init(asked).threads as i32;
    eprintln!("threads: {threads}{}", if asked == 0 { " (auto: the fastest cores)" } else { "" });
    let t0 = Instant::now();
    // ctx=N: the context (4096 by default). A pipeline worker needs the main machine's context, or longer prompts fail
    // there
    let n_ctx = args.iter().find_map(|a| a.strip_prefix("ctx=")).and_then(|v| v.parse::<u32>().ok()).unwrap_or(4096);
    let mut m = match model::Model::load(pos[0], n_ctx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("torc-run: {e}");
            std::process::exit(1);
        }
    };
    let nv = m.n_vocab().expect("vocab");
    eprintln!("loaded {} ({} layers, vocab {nv}) in {:.1} s", pos[0], m.h.n_layer, t0.elapsed().as_secs_f64());
    // klayall: every layer's experts laid out now, as torc-server does at load, so a profile's first blocks don't pay
    // for the layout (each layer's first use otherwise lays it out, inside the MoE "jobs" phase)
    if args.iter().any(|a| a == "klayall") {
        match m.w.klay_all_now() {
            Ok(s) if s > 0.0 => eprintln!("klay: every layer laid out at load in {s:.1} s"),
            Ok(_) => {}
            Err(e) => {
                eprintln!("torc-run: {e}");
                std::process::exit(1);
            }
        }
    }
    // serve-ep=PORT: this machine is the other half of a two-machine prompt (ep.rs) and serves experts until it is
    // stopped
    if let Some(port) = args.iter().find_map(|a| a.strip_prefix("serve-ep=")).and_then(|v| v.parse::<u16>().ok()) {
        if let Err(e) = model::ep::serve(&m.w, port) {
            eprintln!("torc-run: {e}");
        }
        return;
    }
    // serve-pipe=PORT: this machine runs the second half of the layers of a two-machine pipeline (pipe.rs) until it is
    // stopped
    if let Some(port) = args.iter().find_map(|a| a.strip_prefix("serve-pipe=")).and_then(|v| v.parse::<u16>().ok()) {
        if let Err(e) = model::pipe::serve(&mut m, port) {
            eprintln!("torc-run: {e}");
        }
        return;
    }
    let mut out = dump.map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("dump file")));
    // gen=N [mtp=D]: greedy generation of N tokens after the prompt (the n tokens read), either plain (one decode per
    // token) or with D drafts a pass from the MTP block, each pass verified by one exact block. The ids must be the
    // same either way; they are printed as a hash and the first few.
    if let Some(ng) = args.iter().find_map(|a| a.strip_prefix("gen=")).and_then(|v| v.parse::<usize>().ok()) {
        // at most 16 drafts a pass (the size of the planner's table in model::plan, and torc-server's limit)
        let d =
            args.iter().find_map(|a| a.strip_prefix("mtp=")).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(16);
        // plan=1: d is a ceiling, and the draft planner (model::plan::DraftPlan) picks each pass's drafts (0..=d).
        let planned = args.iter().any(|a| a == "plan=1");
        // the chain steps' total time and count without the planner (TORC_PLANLOG)
        let (mut chain_secs, mut chain_steps) = (0.0f64, 0usize);
        let mut pj: Box<dyn Plans> = model::plan::planner();
        let bins = model::switches::get("TORC_DRAFTS");
        if let (true, Some(f)) = (planned, &bins) {
            pj.load(f);
        }
        // TORC_PLANLOG=1: one line a pass (drafts offered, taken and kept, its time, and the costs it was planned on)
        let planlog = model::switches::on("TORC_PLANLOG");
        let mut plog: Vec<String> = Vec::new();
        let argmax = model::ops::argmax;
        let np = tokens.len();
        // a hidden row: d_model wide, or the streams (hc x d_model) for qwen4exp's MTP block
        let dm = m.hidden_width();
        // warmup: an untimed plain generation of the same prompt and length, then a reset to the zero state. The expert
        // cache keeps what that pass loaded, so both timed runs start from the same, already warm, cache
        if args.iter().any(|a| a == "warmup") {
            let tw = Instant::now();
            let mut lw = vec![0.0f32; np * nv];
            m.prefill(&tokens, 0, &mut lw).expect("warm-up prefill");
            let mut tk = argmax(&lw[(np - 1) * nv..np * nv]) as u32;
            let mut l1 = vec![0.0f32; nv];
            for i in 0..ng {
                m.decode(tk, (np + i) as u32, &mut l1).expect("warm-up decode");
                tk = argmax(&l1) as u32;
            }
            m.clear_states();
            eprintln!("warm-up generation: {:.2} s", tw.elapsed().as_secs_f64());
        }
        // lastlogits: read the prompt as the server reads it, in blocks of block= with no logits, then the last token
        // alone for its row. Otherwise the whole prompt is one block with every row's logits, and an 8k prompt then
        // holds 8 GB of logits. The ids are the same either way on the CPU, since a token's bits do not depend on its
        // block
        let last = args.iter().any(|a| a == "lastlogits") && np > 1;
        let mut lg = vec![0.0f32; if last { nv } else { np * nv }];
        // with MTP, log the prompt's hidden rows as the server does (its MTP start reads them)
        if last && d > 0 {
            m.log_hidden();
        }
        let read = if last {
            m.prefill_blocks(&tokens[..np - 1], block.max(1), false, &mut |_, _| {})
                .and_then(|_| m.prefill(&tokens[np - 1..], (np - 1) as u32, &mut lg))
        } else {
            m.prefill(&tokens, 0, &mut lg)
        };
        if let Err(e) = read {
            eprintln!("prefill failed: {e}");
            std::process::exit(1);
        }
        let lg_last = if last { 0 } else { (np - 1) * nv };
        // TORC_PIPE_PULL=1: the prompt on two machines and the reply on this one, with the worker's layer states pulled
        // back
        if model::switches::on("TORC_PIPE_PULL")
            && let Err(e) = m.pull_states(np)
        {
            eprintln!("pull failed: {e}");
            std::process::exit(1);
        }
        let mut t = argmax(&lg[lg_last..lg_last + nv]) as u32;
        let mut ids: Vec<u32> = Vec::with_capacity(ng + 16);
        // from here on the profile tables hold the reply's work alone, without the prompt's
        m.prof = Default::default();
        for a in &model::pipe::TRIP {
            a.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let cache0 = m.w.cache.as_ref().map(|c| c.stats());
        for (_, _, table) in model::phase_tables() {
            for a in table {
                a.store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let mut t1 = Instant::now();
        let (mut passes, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
        // passes and seconds by pass size; kept / verified by position in the chain
        let mut sizes = vec![(0usize, 0.0f64); d + 2];
        let mut at = vec![(0usize, 0usize); d];
        // pipespec: two machines as a layer pipeline, speculating. While the worker runs its half of token p, this
        // machine runs its half of the MTP block's draft for p + 1 inside a revertible step. When p's logits come back,
        // a right draft goes to the worker at once; a wrong one is dropped and the true token's half is run. The worker
        // only ever sees the true tokens, and the ids must equal those of plain gen.
        if args.iter().any(|a| a == "pipespec") {
            m.mtp_note(&tokens).expect("note");
            m.mtp_note(&[t]).expect("note");
            let mut rows: Vec<u32> = tokens[1..].to_vec();
            rows.push(t);
            let hid = m.hidden.clone();
            let mut dl = vec![0.0f32; m.draft_width()];
            let tm = Instant::now();
            m.mtp(&rows, &hid, 0, &mut dl).expect("mtp");
            eprintln!("mtp block over the prompt ({} rows): {:.3} s", rows.len(), tm.elapsed().as_secs_f64());
            t1 = Instant::now();
            let mut dft = m.draft_id(argmax(&dl));
            let mut p = np; // position of t
            let mut l1 = vec![0.0f32; nv];
            let st = m.half(t, p as u32).expect("half");
            m.half_send(t, p as u32, &st).expect("send");
            let (mut hit, mut tried) = (0usize, 0usize);
            // seconds spent: the draft's half, waiting for the reply, the true token's half after a miss, and the MTP
            // row
            let mut tm = [0.0f64; 4];
            loop {
                ids.push(t);
                let last = ids.len() >= ng;
                let tk = Instant::now();
                let sd = if last {
                    None
                } else {
                    m.spec_begin();
                    Some(m.half(dft, (p + 1) as u32).expect("draft half"))
                };
                tm[0] += tk.elapsed().as_secs_f64();
                let tk = Instant::now();
                m.half_recv(&mut l1).expect("recv");
                tm[1] += tk.elapsed().as_secs_f64();
                if last {
                    break;
                }
                let nt = argmax(&l1) as u32;
                let hp = m.hidden.clone();
                tried += 1;
                let tk = Instant::now();
                let st = if nt == dft {
                    hit += 1;
                    m.spec_end(true);
                    sd.unwrap()
                } else {
                    m.spec_end(false);
                    m.half(nt, (p + 1) as u32).expect("half")
                };
                m.half_send(nt, (p + 1) as u32, &st).expect("send");
                tm[2] += tk.elapsed().as_secs_f64();
                // the next draft: the MTP row pairing nt (at p + 1) with p's streams
                let tk = Instant::now();
                m.mtp_note(&[nt]).expect("note");
                dl.resize(m.draft_width(), 0.0);
                m.mtp(&[nt], &hp, p as u32, &mut dl).expect("mtp");
                dft = m.draft_id(argmax(&dl));
                tm[3] += tk.elapsed().as_secs_f64();
                p += 1;
                t = nt;
            }
            let k = 1e3 / ids.len() as f64;
            eprintln!(
                "pipespec: {hit} of {tried} drafts right ({:.0}%); ms a token: draft half {:.2}, wait {:.2},                 true half after a miss {:.2}, mtp {:.2}",
                100.0 * hit as f64 / tried.max(1) as f64,
                tm[0] * k,
                tm[1] * k,
                tm[2] * k,
                tm[3] * k
            );
        } else if d == 0 {
            let mut l1 = vec![0.0f32; nv];
            // every generated position's logits folded into one FNV-1a hash of their bits, so a change of path
            // (TORC_HEAD_SPLIT, say) can be checked bit for bit without a dump
            let mut lh = 0xcbf29ce484222325u64;
            for i in 0..ng {
                ids.push(t);
                m.decode(t, (np + i) as u32, &mut l1).expect("decode");
                for v in &l1 {
                    lh = (lh ^ v.to_bits() as u64).wrapping_mul(0x100000001b3);
                }
                t = argmax(&l1) as u32;
            }
            eprintln!("logits hash {lh:016x} ({ng} positions)");
        } else {
            // the draft head notes the ids it has seen (the prompt, then every token generated); its rows are
            // m.draft_width() wide, and index i is token m.draft_id(i)
            m.mtp_note(&tokens).expect("note");
            m.mtp_note(&[t]).expect("note");
            // the MTP rows over the prompt: row i pairs token i + 1 (the first new token for the last row) with hidden
            // row i
            let mut rows: Vec<u32> = tokens[1..].to_vec();
            rows.push(t);
            let hid = m.hidden.clone();
            let wd = m.draft_width();
            let mut dl = vec![0.0f32; wd];
            let tm = Instant::now();
            let log = m.take_hidden_log().filter(|(s, l)| *s == 0 && l.len() == np * dm);
            let mut mh_last = if let Some((_, log)) = log {
                // lastlogits: as the server starts, run only the keys and values of the prompt rows, then the last row
                // in full
                m.mtp_fill(&rows[..np - 1], &log[..(np - 1) * dm], 0).expect("mtp fill");
                m.mtp(&rows[np - 1..], &log[(np - 1) * dm..], (np - 1) as u32, &mut dl).expect("mtp")
            } else {
                let mut mh = m.mtp(&rows, &hid, 0, &mut dl).expect("mtp");
                mh.split_off((np - 1) * dm)
            };
            // the MTP block over the prompt is prompt work, so it is timed separately from generation
            eprintln!("mtp block over the prompt ({} rows): {:.3} s", rows.len(), tm.elapsed().as_secs_f64());
            t1 = Instant::now();
            let mut d0 = m.draft_id(argmax(&dl));
            let mut c0 = conf(&dl);
            let mut p = np; // position of t
            let mut vl: Vec<f32> = Vec::new();
            let mut d1: Vec<f32> = Vec::new();
            // seconds spent in: the drafting chain, the verify, the commit, and the MTP rows after the pass
            let mut tm = [0.0f64; 4];
            let mut tk = Instant::now();
            let mut lapt = |i: usize, tk: &mut Instant| {
                let now = Instant::now();
                tm[i] += (now - *tk).as_secs_f64();
                *tk = now;
            };
            // plan=1: the last pass's commit runs at the start of the next pass, inside its time, just as a lazy replay
            // is paid for in the next forward (t(n) is forward time, replay included)
            let mut pending: Option<usize> = None;
            while ids.len() < ng {
                ids.push(t);
                // the chain: draft g comes from draft g - 1 and the block's own last hidden state, at position p - 1 +
                // g
                let mut drafts = vec![d0];
                let mut surv = vec![c0];
                let mut offered = 1;
                if planned {
                    d1.resize(m.draft_width(), 0.0);
                    while drafts.len() < d && pj.plan(&surv) == drafts.len() {
                        let g = drafts.len();
                        let tc = Instant::now();
                        let h1 = m.mtp(&[drafts[g - 1]], &mh_last, (p - 1 + g) as u32, &mut d1).expect("mtp");
                        surv.push(surv[g - 1] * conf(&d1));
                        drafts.push(m.draft_id(argmax(&d1)));
                        mh_last = h1;
                        pj.time_chain(tc.elapsed().as_secs_f64());
                    }
                    offered = drafts.len();
                    let nd = pj.pick(&surv);
                    drafts.truncate(nd);
                    surv.truncate(nd);
                }
                let tc = Instant::now();
                let mut chained = 0usize;
                d1.resize(m.draft_width(), 0.0);
                while !planned && !drafts.is_empty() && drafts.len() < d {
                    let g = drafts.len();
                    let h1 = m.mtp(&[drafts[g - 1]], &mh_last, (p - 1 + g) as u32, &mut d1).expect("mtp");
                    chained += 1;
                    drafts.push(m.draft_id(argmax(&d1)));
                    mh_last = h1;
                }
                chain_secs += tc.elapsed().as_secs_f64();
                chain_steps += chained;
                lapt(0, &mut tk);
                let nd = drafts.len();
                if planlog {
                    let est: Vec<String> = (1..=d + 1).map(|n| format!("{:.1}", 1e3 * pj.cost(n))).collect();
                    let sv: Vec<String> = surv.iter().map(|s| format!("{s:.2}")).collect();
                    let mean = chain_secs / chain_steps.max(1) as f64;
                    let step = if planned { pj.chain() } else { mean };
                    plog.push(format!(
                        "pass {passes} p {p}: offered {} took {nd} surv [{}] t [{}] chain {:.2}",
                        if planned { offered } else { nd },
                        sv.join(" "),
                        est.join(" "),
                        1e3 * step
                    ));
                }
                let mut input = vec![t];
                input.extend_from_slice(&drafts);
                vl.resize(input.len() * nv, 0.0);
                let tv = Instant::now();
                if let Some(k) = pending.take() {
                    m.commit(k);
                }
                let tvv = Instant::now();
                m.verify(&input, p as u32, &mut vl).expect("verify");
                let vt = tvv.elapsed().as_secs_f64();
                let dt = tv.elapsed().as_secs_f64();
                let mut acc = 0;
                while acc < nd && argmax(&vl[acc * nv..(acc + 1) * nv]) as u32 == drafts[acc] {
                    acc += 1;
                }
                let bonus = argmax(&vl[acc * nv..(acc + 1) * nv]) as u32;
                lapt(1, &mut tk);
                if planned {
                    pending = Some(acc + 1);
                } else {
                    m.commit(acc + 1);
                }
                lapt(2, &mut tk);
                if planned {
                    pj.record(&surv, acc, 1 + nd, vt);
                }
                if planlog {
                    plog.last_mut().unwrap().push_str(&format!(" -> kept {acc}, {:.1} ms", dt * 1e3));
                }
                let vh = m.hidden[..(acc + 1) * dm].to_vec();
                ids.extend_from_slice(&drafts[..acc]);
                passes += 1;
                drafted += nd;
                accepted += acc;
                sizes[1 + nd].0 += 1;
                sizes[1 + nd].1 += dt;
                for (i, a) in at[..nd].iter_mut().enumerate() {
                    a.0 += (i < acc) as usize;
                    a.1 += 1;
                }
                // the MTP rows of the kept positions again, this time on the main model's true hidden states
                let mut rows: Vec<u32> = drafts[..acc].to_vec();
                rows.push(bonus);
                m.mtp_note(&rows).expect("note");
                let wd = m.draft_width();
                let mut dl = vec![0.0f32; wd];
                let mut mh = m.mtp(&rows, &vh, p as u32, &mut dl).expect("mtp");
                d0 = m.draft_id(argmax(&dl));
                c0 = conf(&dl);
                mh_last = mh.split_off(acc * dm);
                p += acc + 1;
                t = bonus;
                lapt(3, &mut tk);
            }
            ids.truncate(ng);
            eprintln!(
                "ms a pass: chain {:.2}, verify {:.2}, commit {:.2}, mtp rows {:.2}",
                tm[0] * 1e3 / passes as f64,
                tm[1] * 1e3 / passes as f64,
                tm[2] * 1e3 / passes as f64,
                tm[3] * 1e3 / passes as f64
            );
        }
        let s = t1.elapsed().as_secs_f64();
        let hash = ids.iter().fold(0xcbf29ce484222325u64, |h, &v| (h ^ v as u64).wrapping_mul(0x100000001b3));
        eprintln!(
            "generated {ng} tokens in {s:.2} s = {:.2} tok/s, ids hash {hash:016x}, first {:?}",
            ng as f64 / s,
            &ids[..ids.len().min(12)]
        );
        if d > 0 {
            if planned {
                eprintln!("{}", pj.report(d + 1));
                if let Some(f) = &bins {
                    pj.save(f);
                }
            }
            let sz: Vec<String> = (1..sizes.len())
                .filter(|&n| sizes[n].0 > 0)
                .map(|n| format!("{n}: {}, {:.1}", sizes[n].0, sizes[n].1 * 1e3 / sizes[n].0 as f64))
                .collect();
            let ps: Vec<String> = at
                .iter()
                .enumerate()
                .filter(|(_, a)| a.1 > 0)
                .map(|(i, a)| format!("{}: {}/{}", i + 1, a.0, a.1))
                .collect();
            eprintln!("pass, tokens: passes, ms: {}; chain, position: kept/verified: {}", sz.join(" "), ps.join(" "));
            if planlog {
                eprintln!("{}", plog.join("\n"));
            }
            eprintln!(
                "mtp={d}: {passes} passes, {accepted} of {drafted} drafts accepted ({:.0}%), {:.2} tokens a pass",
                100.0 * accepted as f64 / drafted.max(1) as f64,
                ng as f64 / passes.max(1) as f64
            );
        }
        let un: Vec<String> = (1..=16)
            .filter_map(|n| {
                let c = model::moe::MOE_UNION[n][0].load(std::sync::atomic::Ordering::Relaxed);
                let e = model::moe::MOE_UNION[n][1].load(std::sync::atomic::Ordering::Relaxed);
                (c > 0).then(|| format!("n={n}: {:.1}", e as f64 / c as f64))
            })
            .collect();
        if !un.is_empty() {
            eprintln!("routed experts a layer by block size: {}", un.join(", "));
        }
        let p = m.prof;
        eprintln!(
            "ms per generated token: deltanet {:.2} attention {:.2} moe {:.2} output {:.2} norms {:.2}",
            p.rec * 1e3 / ng as f64,
            p.attn * 1e3 / ng as f64,
            p.moe * 1e3 / ng as f64,
            p.out * 1e3 / ng as f64,
            p.norm * 1e3 / ng as f64
        );
        {
            use std::sync::atomic::Ordering::Relaxed;
            let tn = model::pipe::TRIP[2].load(Relaxed);
            if tn > 0 {
                let (w, c) = (model::pipe::TRIP[0].load(Relaxed) as f64, model::pipe::TRIP[1].load(Relaxed) as f64);
                eprintln!(
                    "two boxes: {tn} passes, round trip {:.2} ms a pass, worker compute {:.2}, transfer+wait {:.2}",
                    w / 1e3 / tn as f64,
                    c / 1e3 / tn as f64,
                    (w - c) / 1e3 / tn as f64
                );
            }
        }
        if let Some(c) = &m.w.cache {
            let s = c.stats();
            eprintln!(
                "expert cache: {} requests, {:.1}% hits, {:.0} MB read, waited {:.2} s",
                s.asked,
                100.0 * s.resident as f64 / s.asked.max(1) as f64,
                s.loaded_bytes as f64 / 1e6,
                s.blocked
            );
            if let Some(a) = cache0 {
                let (rq, h) = (s.asked - a.asked, s.resident - a.resident);
                eprintln!(
                    "expert cache, generation only: {rq} requests, {:.1}% hits, {:.0} MB read, waited {:.2} s",
                    100.0 * h as f64 / rq.max(1) as f64,
                    (s.loaded_bytes - a.loaded_bytes) as f64 / 1e6,
                    s.blocked - a.blocked
                );
            }
        }
        let ts = &model::moe::TAIL_STATS;
        let (sk, cn) =
            (ts[0].load(std::sync::atomic::Ordering::Relaxed), ts[1].load(std::sync::atomic::Ordering::Relaxed));
        if cn > 0 {
            eprintln!("fast mode tail skip: {sk} of {cn} routed pairs skipped ({:.2}%)", 100.0 * sk as f64 / cn as f64);
        }
        for (name, phases, table) in model::phase_tables() {
            let ns: Vec<u64> = table.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect();
            if ns.iter().any(|&v| v > 0) {
                let parts: Vec<String> = phases
                    .iter()
                    .zip(&ns)
                    .filter(|(n, _)| **n != "-")
                    .map(|(n, &v)| format!("{n} {:.2}", v as f64 / 1e6 / ng as f64))
                    .collect();
                eprintln!("{name} ms/generated token: {}", parts.join(", "));
            }
        }
        return;
    }
    // spec=SEED: the MTP verification path, teacher-forced. It runs passes of [token, d drafts] (d random in 0..=8),
    // where each draft is the true next token or, one time in five, a wrong one. A draft is accepted while it equals
    // the truth; the accepted rows' logits are written in position order and the rest of the block is rolled back. The
    // dump must equal plain decoding's bit for bit, whatever the drafts were.
    if let Some(seed) = args.iter().find_map(|a| a.strip_prefix("spec=")).and_then(|v| v.parse::<u64>().ok()) {
        let mut rng = seed | 1;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let (mut p, mut passes, mut kept_total) = (0usize, 0usize, 0usize);
        let mut lg: Vec<f32> = Vec::new();
        let t1 = Instant::now();
        while p < tokens.len() {
            let d = ((next() % 9) as usize).min(tokens.len() - p - 1);
            let mut input = vec![tokens[p]];
            for i in 0..d {
                let truth = tokens[p + 1 + i];
                input.push(if next() % 5 == 0 { (truth + 1 + (next() % 1000) as u32) % nv as u32 } else { truth });
            }
            lg.resize(input.len() * nv, 0.0);
            if let Err(e) = m.verify(&input, p as u32, &mut lg) {
                eprintln!("verify failed at {p}: {e}");
                std::process::exit(1);
            }
            let acc = (0..d).take_while(|&i| input[1 + i] == tokens[p + 1 + i]).count();
            if let Some(f) = out.as_mut() {
                let rows = &lg[..(acc + 1) * nv];
                let bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), rows.len() * 4) };
                f.write_all(bytes).unwrap();
            }
            m.commit(acc + 1);
            p += acc + 1;
            passes += 1;
            kept_total += acc;
        }
        let s = t1.elapsed().as_secs_f64();
        eprintln!(
            "{} tokens in {passes} verify passes ({kept_total} drafts kept): {s:.1} s = {:.1} tok/s",
            tokens.len(),
            tokens.len() as f64 / s
        );
        return;
    }
    if block > 0 {
        let t1 = Instant::now();
        // nologits: read the prompt only, as llama-bench's pp test does (no vocabulary product per position)
        let none = args.iter().any(|a| a == "nologits");
        let mut lg = vec![0.0f32; if none { 0 } else { block * nv }];
        // ref=FILE: compare every position's logits with a reference dump of the same tokens (from a CPU run, or
        // another engine): top-1 agreement, the mean and largest |difference|, and the perplexity of the tokens under
        // each. This is the tolerance test for the NPU tier; the CPU path itself is compared with cmp, byte for byte
        let mut rf =
            args.iter().find_map(|a| a.strip_prefix("ref=")).map(|p| std::fs::File::open(p).expect("ref file"));
        let mut rl = vec![0.0f32; nv];
        let (mut same, mut npos, mut sum_d, mut max_d, mut nll_a, mut nll_b) = (0usize, 0usize, 0f64, 0f32, 0f64, 0f64);
        let lsm = |l: &[f32], t: usize| {
            let mx = l.iter().fold(f32::MIN, |a, &v| a.max(v)) as f64;
            let z: f64 = l.iter().map(|&v| (v as f64 - mx).exp()).sum();
            l[t] as f64 - mx - z.ln()
        };
        // TORC_PIPE: a two-machine layer pipeline with the blocks overlapping (dump= only; ref= and the block log need
        // one machine)
        if model::pipe::client().is_some() {
            let mut last = Vec::new();
            let r = m.prefill_blocks(&tokens, block, !none, &mut |_, lg| {
                if let Some(f) = out.as_mut() {
                    // SAFETY: the f32s viewed as bytes
                    f.write_all(unsafe { std::slice::from_raw_parts(lg.as_ptr().cast::<u8>(), lg.len() * 4) }).unwrap();
                }
                if lg.len() >= nv {
                    last = lg[lg.len() - nv..].to_vec();
                }
            });
            if let Err(e) = r {
                eprintln!("prefill failed: {e}");
                std::process::exit(1);
            }
            if !none {
                lg[(tokens.len() - 1) % block * nv..][..nv].copy_from_slice(&last);
            }
        }
        for (b, chunk) in tokens.chunks(block).enumerate().filter(|_| model::pipe::client().is_none()) {
            let lg = &mut lg[..if none { 0 } else { chunk.len() * nv }];
            if let Err(e) = m.prefill(chunk, (b * block) as u32, lg) {
                eprintln!("prefill failed at block {b}: {e}");
                std::process::exit(1);
            }
            if let Some(f) = out.as_mut() {
                let bytes = unsafe { std::slice::from_raw_parts(lg.as_ptr().cast::<u8>(), lg.len() * 4) };
                f.write_all(bytes).unwrap();
            }
            if let (Some(f), false) = (rf.as_mut(), none) {
                use std::io::Read;
                for (j, row) in lg.chunks_exact(nv).enumerate() {
                    // SAFETY: rl holds nv f32s, viewed as bytes to read into
                    let bytes = unsafe { std::slice::from_raw_parts_mut(rl.as_mut_ptr().cast::<u8>(), nv * 4) };
                    f.read_exact(bytes).expect("ref file too short");
                    let am = |l: &[f32]| {
                        l.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0
                    };
                    same += (am(row) == am(&rl)) as usize;
                    npos += 1;
                    for (a, r) in row.iter().zip(&rl) {
                        let d = (a - r).abs();
                        sum_d += d as f64;
                        max_d = max_d.max(d);
                    }
                    let p = b * block + j;
                    if p + 1 < tokens.len() {
                        let nt = tokens[p + 1] as usize;
                        nll_a -= lsm(row, nt);
                        nll_b -= lsm(&rl, nt);
                    }
                }
            }
        }
        let s = t1.elapsed().as_secs_f64();
        if npos > 0 {
            let k = (npos - 1).max(1) as f64;
            eprintln!(
                "vs ref: top-1 {same}/{npos}, mean |d| {:.4}, max |d| {max_d:.3}; perplexity {:.4} vs ref {:.4} ({:+.3}%)",
                sum_d / (npos * nv) as f64,
                (nll_a / k).exp(),
                (nll_b / k).exp(),
                100.0 * ((nll_a - nll_b) / k).exp_m1()
            );
        }
        if let Some((w, g)) = model::npu::stats() {
            eprintln!("npu: {g} graphs, {:.1} ms waited", w * 1e3);
        }
        let hp = model::moe::MOE_HOT.each_ref().map(|a| a.load(std::sync::atomic::Ordering::Relaxed));
        if hp[0] > 0 {
            eprintln!(
                "npu: hot experts took {} of {} routed pairs ({:.1}%)",
                hp[1],
                hp[0],
                100.0 * hp[1] as f64 / hp[0] as f64
            );
        }
        model::npu::write_profile(m.h.n_layer, m.h.n_expert);
        if !none {
            let top = lg[(tokens.len() - 1) % block * nv..][..nv]
                .iter()
                .enumerate()
                .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
            eprintln!("last: top {} ({:.4})", top.0, top.1);
        }
        eprintln!("{} tokens in blocks of {block}: {s:.1} s = {:.1} tok/s", tokens.len(), tokens.len() as f64 / s);
        if model::ep::client().is_some() {
            let (se, wa) = model::ep::stats();
            eprintln!("two boxes: sending {se:.2} s, waiting for the worker {wa:.2} s");
        }
        let p = m.prof;
        eprintln!(
            "ms/token: deltanet {:.2} attention {:.2} moe {:.2} output {:.2} norms {:.2}",
            p.rec * 1e3 / tokens.len() as f64,
            p.attn * 1e3 / tokens.len() as f64,
            p.moe * 1e3 / tokens.len() as f64,
            p.out * 1e3 / tokens.len() as f64,
            p.norm * 1e3 / tokens.len() as f64
        );
        let a = &model::attn::ATTN_NS;
        let ld = |i: usize| a[i].load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-6 / tokens.len() as f64;
        eprintln!(
            "attention ms/token: q/k/v {:.2}, core (scores, weights, values, merge) {:.2}, output {:.2} (its quantization {:.2})",
            ld(0),
            ld(1),
            ld(2),
            ld(3)
        );
        eprintln!("attention ms/token: preparation {:.2}, gate {:.2}", ld(4), ld(5));
        #[cfg(feature = "gpu")]
        if let Some((calls, [up, cpu, wait, back])) = model::gpu::stats() {
            eprintln!(
                "gpu attention: {calls} blocks; s writing {up:.2}, CPU's tokens {cpu:.2}, waiting {wait:.2}, reading back {back:.2}; share now {:.2}",
                model::gpu::share().unwrap_or(0.0)
            );
        }
        if let Some(c) = &m.w.cache {
            let s = c.stats();
            eprintln!(
                "expert cache: {} requests, {:.1}% hits, {:.0} MB read, waited {:.2} s",
                s.asked,
                100.0 * s.resident as f64 / s.asked.max(1) as f64,
                s.loaded_bytes as f64 / 1e6,
                s.blocked
            );
        }
        let ts = &model::moe::TAIL_STATS;
        let (sk, cn) =
            (ts[0].load(std::sync::atomic::Ordering::Relaxed), ts[1].load(std::sync::atomic::Ordering::Relaxed));
        if cn > 0 {
            eprintln!("fast mode tail skip: {sk} of {cn} routed pairs skipped ({:.2}%)", 100.0 * sk as f64 / cn as f64);
        }
        for (name, phases, table) in model::phase_tables() {
            let ns: Vec<u64> = table.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect();
            if ns.iter().any(|&v| v > 0) {
                let parts: Vec<String> = phases
                    .iter()
                    .zip(&ns)
                    .filter(|(n, _)| **n != "-")
                    .map(|(n, &v)| format!("{n} {:.2}", v as f64 / 1e6 / tokens.len() as f64))
                    .collect();
                eprintln!("{name} ms/token: {}", parts.join(", "));
            }
        }
        return;
    }
    let mut logits = vec![0.0f32; nv];
    // the first tokens warm the caches and the expert cache; the steady rate is timed from token `warm` on
    let warm = (tokens.len() / 4).min(32);
    let t1 = Instant::now();
    let mut t_steady = t1;
    let mut prof_warm = m.prof;
    for (p, &t) in tokens.iter().enumerate() {
        if p == warm {
            t_steady = Instant::now();
            prof_warm = m.prof;
        }
        if let Err(e) = m.decode(t, p as u32, &mut logits) {
            eprintln!("decode failed at {p}: {e}");
            std::process::exit(1);
        }
        if p < 4 || p + 1 == tokens.len() {
            let top = logits.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
            eprintln!("pos {p}: token {t} -> top {} ({:.4})", top.0, top.1);
        }
        if let Some(f) = out.as_mut() {
            // one write per position (little-endian f32s are the platform's own bytes on aarch64 and x86)
            let bytes = unsafe { std::slice::from_raw_parts(logits.as_ptr().cast::<u8>(), logits.len() * 4) };
            f.write_all(bytes).unwrap();
        }
    }
    let s = t1.elapsed().as_secs_f64();
    eprintln!("{} tokens in {s:.1} s: {:.2} tok/s", tokens.len(), tokens.len() as f64 / s);
    let steady = tokens.len() - warm;
    let ss = t_steady.elapsed().as_secs_f64();
    eprintln!(
        "steady ({steady} tokens after {warm} to warm up): {:.2} tok/s, {:.1} ms each",
        steady as f64 / ss,
        ss * 1e3 / steady as f64
    );
    let p = m.prof;
    let p = model::forward::Prof {
        rec: p.rec - prof_warm.rec,
        attn: p.attn - prof_warm.attn,
        moe: p.moe - prof_warm.moe,
        out: p.out - prof_warm.out,
        norm: p.norm - prof_warm.norm,
    };
    let ms = |v: f64| v * 1e3 / steady as f64;
    eprintln!(
        "steady, ms per token by part: deltanet {:.2}, attention {:.2}, moe {:.2}, output {:.2}, norms {:.2}",
        ms(p.rec),
        ms(p.attn),
        ms(p.moe),
        ms(p.out),
        ms(p.norm)
    );
    // what each part reads per token, and that over its time: a part near the machine's RAM read speed (hwscan's
    // figure) is bound by memory, one well under it by compute or by waiting
    let rd = token_bytes(&m, (warm + tokens.len() - 1) as f64 / 2.0);
    let gbs = |bytes: f64, t: f64| if t > 0.0 { bytes * steady as f64 / t / 1e9 } else { 0.0 };
    let all = rd.iter().sum::<f64>();
    eprintln!(
        "steady, MB read per token by part: deltanet {:.1}, attention {:.1}, moe {:.1}, output {:.1}; {:.1} in all",
        rd[0] / 1e6,
        rd[1] / 1e6,
        rd[2] / 1e6,
        rd[3] / 1e6,
        all / 1e6
    );
    eprintln!(
        "steady, GB/s by part: deltanet {:.1}, attention {:.1}, moe {:.1}, output {:.1}; {:.1} over the whole token",
        gbs(rd[0], p.rec),
        gbs(rd[1], p.attn),
        gbs(rd[2], p.moe),
        gbs(rd[3], p.out),
        all * steady as f64 / ss / 1e9
    );
    if let Some(c) = &m.w.cache {
        let s = c.stats();
        eprintln!(
            "expert cache: {} requests, {:.1}% hits, {:.0} MB read, waited {:.2} s",
            s.asked,
            100.0 * s.resident as f64 / s.asked.max(1) as f64,
            s.loaded_bytes as f64 / 1e6,
            s.blocked
        );
    }
}

/// The bytes one decoded token reads, by part (deltanet, attention, moe, output), at position pos: every tensor of the
/// part's layers, the top-k experts of each MoE layer, the attention cache up to pos (int8 keys and values with a scale
/// per head and position, the default) and the DeltaNet state, read and written back (fp16, the default). Norm
/// weights and the conv state are left out (a few KB a layer), as is the MTP block. A model's own layout
/// (qwen4exp's mixers) may read things this does not count.
fn token_bytes(m: &model::Model, pos: f64) -> [f64; 4] {
    let h = &m.h;
    let mut b = [0f64; 4];
    let attn = |l: u32| (l + 1).is_multiple_of(h.full_attention_interval);
    for t in &m.w.file.tensors {
        let bytes = t.bytes as f64;
        if let Some(rest) = t.name.strip_prefix("blk.") {
            let (Some(l), Some(what)) =
                rest.split_once('.').map_or((None, None), |(l, w)| (l.parse::<u32>().ok(), Some(w)))
            else {
                continue;
            };
            if l >= h.n_layer || what.contains("norm") {
                continue;
            }
            let part = if what.starts_with("ffn") {
                2
            } else if attn(l) {
                1
            } else {
                0
            };
            b[part] += bytes;
        } else if t.name == "output.weight" {
            b[3] += bytes;
        } else if t.name == "token_embd.weight" {
            b[3] += bytes / h.n_vocab.max(1) as f64;
        }
    }
    for e in &m.w.file.experts {
        if e.layer < h.n_layer {
            b[2] += e.n_used as f64 * e.pack.bytes as f64;
        }
    }
    let n_attn = (0..h.n_layer).filter(|&l| attn(l)).count() as f64;
    b[1] += n_attn * pos * h.n_head_kv as f64 * 2.0 * (h.head_dim as f64 + 4.0);
    if let Some(s) = &h.ssm {
        b[0] += (h.n_layer as f64 - n_attn) * 2.0 * (s.n_value_heads * s.ds * s.ds) as f64 * 2.0;
    }
    b
}
