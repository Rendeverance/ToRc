//! torc-chat <model.torc> "<user message>" [max new tokens] [threads]: one ChatML turn, greedy, printing the reply as
//! it is made - torc-server's tokenizer, template and loop, without the server. TORC_PIPE and the other settings work
//! as in torc-run; TORC_PIPE_PULL=1 pulls the worker's states back after the prompt.

use std::io::Write;

fn main() {
    model::set_npu_env();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: torc-chat <model.torc> \"<message>\" [max new tokens] [threads]");
        std::process::exit(2);
    }
    if let Err(e) = run(&a) {
        eprintln!("torc-chat: {e}");
        std::process::exit(1)
    }
}

fn run(a: &[String]) -> format::Result<()> {
    let max: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(256);
    // the thread pool before the load (0: one thread per fast core)
    model::pool::init(a.get(4).and_then(|v| v.parse().ok()).unwrap_or(0));
    let tok = model::tok::Tokenizer::open(&a[1])?;
    // the message as text, as the server takes a user's: a turn marker in it can't end the turn
    let text = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", tok.defuse(&a[2], &[]));
    let ids = tok.encode(&text, true)?;
    let stops: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.find(s)).collect();
    let mut m = model::Model::load(&a[1], 4096)?;
    let nv = m.n_vocab()?;
    let argmax =
        |l: &[f32]| l.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
    let t0 = std::time::Instant::now();
    let mut lg = vec![0.0f32; ids.len() * nv];
    m.prefill(&ids, 0, &mut lg)?;
    let tp = t0.elapsed().as_secs_f64();
    if model::switches::on("TORC_PIPE_PULL") {
        m.pull_states(ids.len())?;
    }
    let mut t = argmax(&lg[(ids.len() - 1) * nv..]) as u32;
    let mut pos = ids.len() as u32;
    let mut out = std::io::stdout();
    let mut one = vec![0.0f32; nv];
    let (t1, mut n) = (std::time::Instant::now(), 0usize);
    while n < max && !stops.contains(&t) {
        // a closed pipe (the output piped into something that has stopped reading) ends the reply quietly
        if out.write_all(&tok.decode(t)).and_then(|()| out.flush()).is_err() {
            break;
        }
        m.decode(t, pos, &mut one)?;
        t = argmax(&one) as u32;
        pos += 1;
        n += 1;
    }
    let tg = t1.elapsed().as_secs_f64();
    eprintln!(
        "\n[prompt {} tokens in {tp:.2} s = {:.1} tok/s; reply {n} tokens in {tg:.2} s = {:.1} tok/s]",
        ids.len(),
        ids.len() as f64 / tp,
        n as f64 / tg.max(1e-9)
    );
    Ok(())
}
