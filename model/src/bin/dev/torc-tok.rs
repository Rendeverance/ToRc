//! torc-tok <model> <text file> <out.int32> [max]: writes the text's token ids (the model's own tokenizer, no special
//! tokens), the first `max` of them, as little-endian u32 - the prompt files torc-run reads.

use model::tok::Tokenizer;

fn die(e: impl std::fmt::Display) -> ! {
    eprintln!("{e}");
    std::process::exit(1)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: torc-tok <model.torc> <text file> <out.int32> [max]");
        std::process::exit(2);
    }
    let tok = Tokenizer::open(&a[1]).unwrap_or_else(|e| die(e));
    let text = std::fs::read_to_string(&a[2]).expect("text");
    let mut ids = tok.encode(&text, false).expect("encode");
    if let Some(m) = a.get(4).and_then(|v| v.parse::<usize>().ok()) {
        ids.truncate(m);
    }
    let b: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[3], b).expect("write");
    eprintln!("{} tokens -> {}", ids.len(), a[3]);
}
