//! torc-repeat <model.torc> <text file> [len] [blocks] [runs] [ctx]: whether a prompt gives the same bits run after
//! run. The first `len` tokens of the text (repeated as needed) are read in blocks, with every row's logits, and the
//! next token is decoded; `runs` times in one process, each from a cleared state, the block sizes (one, or a list such
//! as 512,1024,256) taken in turn. Each run after the first says whether its logits match the first run's bit for bit,
//! and if not, the first prompt position that differs and the largest difference in the decoded token's logits. On the
//! CPU every run must match.
//!
//!   torc-repeat model.torc text.txt 1800 512,1024,256 30 4096

use std::time::Instant;

fn hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: torc-repeat <model.torc> <text file> [len] [blocks] [runs] [ctx]");
        std::process::exit(2);
    }
    let num = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (len, runs, ctx) = (num(3, 1800), num(5, 8), num(6, 4096) as u32);
    // one block size, or several (512,1024) taken in turn
    let blocks: Vec<usize> =
        a.get(4).map_or("1024", |v| v.as_str()).split(',').filter_map(|v| v.parse().ok()).collect();
    model::pool::init(0);
    let fail = |e: &dyn std::fmt::Display| -> ! {
        eprintln!("torc-repeat: {e}");
        std::process::exit(1)
    };
    let tok = model::tok::Tokenizer::open(&a[1]).unwrap_or_else(|e| fail(&e));
    let mut m = model::Model::load(&a[1], ctx).unwrap_or_else(|e| fail(&e));
    let nv = m.n_vocab().unwrap_or_else(|e| fail(&e));
    let text = std::fs::read_to_string(&a[2]).unwrap_or_else(|e| fail(&e));
    let mut ids = Vec::new();
    while ids.len() <= len {
        ids.extend(tok.encode(&text, false).unwrap_or_else(|e| fail(&e)));
    }
    // per run: each prompt row's hash, then the decoded token's
    let mut first: Option<(Vec<u64>, Vec<f32>)> = None;
    let mut bad = 0;
    for r in 0..runs {
        m.clear_states();
        let block = blocks[r % blocks.len()];
        let t = Instant::now();
        let mut rows: Vec<u64> = Vec::with_capacity(len + 1);
        m.prefill_blocks(&ids[..len], block, true, &mut |_, lg| {
            for row in lg.chunks(nv) {
                rows.push(hash(row));
            }
        })
        .unwrap_or_else(|e| fail(&e));
        let mut lg = vec![0.0f32; nv];
        m.decode(ids[len], len as u32, &mut lg).unwrap_or_else(|e| fail(&e));
        rows.push(hash(&lg));
        let secs = t.elapsed().as_secs_f64();
        match &first {
            None => {
                println!("run {r} (block {block}): the reference ({secs:.1} s)");
                first = Some((rows, lg));
            }
            Some((r0, l0)) => match rows.iter().zip(r0).position(|(x, y)| x != y) {
                None => println!("run {r} (block {block}): same bits ({secs:.1} s)"),
                Some(p) => {
                    bad += 1;
                    let d = lg.iter().zip(l0).fold(0f32, |d, (x, y)| d.max((x - y).abs()));
                    let rows_bad = rows.iter().zip(r0).filter(|(x, y)| x != y).count();
                    println!(
                        "run {r} (block {block}): DIFFERS from position {p} (block {}, row {} in it), {rows_bad} of {} rows; the \
                         decoded token's logits up to {d:.4} apart ({secs:.1} s)",
                        p / block,
                        p % block,
                        rows.len()
                    );
                }
            },
        }
    }
    println!("{bad} of {} runs differ", runs.saturating_sub(1));
}
