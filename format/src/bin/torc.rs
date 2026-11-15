//! torc: convert, check and describe model files.

use format::{check, read::Model};
use std::process::ExitCode;

const USAGE: &str = "usage:
  torc convert-gguf <in.gguf> <out.torc> [--hash-source]  (a split GGUF: name its first part)
  torc gguf-info <in.gguf>
  torc check <file.torc>
  torc check <file.torc> --against <in.gguf>          (every tensor and expert against its source bytes)
  torc check <file.torc> --layers <a-b> [ngram-layer]  (a shard: what a machine running layers a..b holds)
  torc info <file.torc>
  torc tensors <file.torc>
  torc meta <file.torc>                                (the metadata, long arrays as their length)
  torc shard-plan <model> <a-b> [ngram-layer]          (what a machine running layers a..b needs)
  torc pack <model> <out|-> <a-b> [ngram-layer]        (those byte ranges as a stream)
  torc unpack <in|-> <out>                             (a stream into a sparse full-length file)";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn span(s: &str) -> format::Result<(u32, u32)> {
    s.split_once('-')
        .and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))
        .ok_or_else(|| format::Error("layers: a-b".into()))
}

fn info(path: &str) -> format::Result<()> {
    let m = Model::open(path)?;
    let h = &m.header;
    println!("{path}: ToRc layout 3, arch {}, {:.2} GiB", h.arch, h.file_bytes as f64 / 1073741824.0);
    println!("model id: {}", hex(&h.model_id));
    println!(
        "resident: {} tensors, {:.2} GiB; experts: {} MoE layers, {:.2} GiB",
        m.tensors.len(),
        m.resident.bytes as f64 / 1073741824.0,
        m.experts.len(),
        m.expert_region.bytes as f64 / 1073741824.0
    );
    let mut kinds = std::collections::BTreeMap::new();
    for b in &m.blocks {
        *kinds.entry(format!("{:?}", b.kind)).or_insert(0) += 1;
    }
    println!("blocks: {kinds:?}");
    if let Some(x) = m.experts.first() {
        println!(
            "layer {}: {} experts, {} used, d_in {}, width {}, pack {} B",
            x.layer, x.n_expert, x.n_used, x.d_in, x.width, x.pack.bytes
        );
        for (what, p) in [("gate", &x.pack.gate), ("up", &x.pack.up), ("down", &x.pack.down)] {
            println!(
                "  {what}: {} type {} {} x {} at {} ({} B), changes {}",
                p.name, p.ty, p.rows, p.cols, p.at, p.bytes, p.origin.changes
            );
        }
    }
    if let Some(s) = &m.source {
        for f in &s.files {
            println!(
                "source: {} ({} B){}",
                f.name,
                f.bytes,
                if f.hash == [0; 32] { String::new() } else { format!(", BLAKE3 {}", hex(&f.hash)) }
            );
        }
        println!("converter: {}", s.converter);
    }
    Ok(())
}

fn run(args: &[String]) -> format::Result<()> {
    match args.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
        ["convert-gguf", src, dst, rest @ ..] if rest.is_empty() || rest == ["--hash-source"] => {
            let r = format::convert::convert_gguf(src, dst, !rest.is_empty())?;
            println!(
                "wrote {dst}: {:.2} GiB, {} tensors, {} MoE layers, {:.1} s",
                r.file_bytes as f64 / 1073741824.0,
                r.tensors,
                r.moe_layers,
                r.seconds
            );
            Ok(())
        }
        ["gguf-info", path] => {
            let g = format::gguf::Gguf::open(path)?;
            println!(
                "{path}: GGUF v{}, arch {}, {} part(s), {} tensors, metadata {} B",
                g.version,
                g.arch,
                g.files.len(),
                g.tensors.len(),
                g.metadata.len()
            );
            for t in &g.tensors {
                println!(
                    "{} type {} dims {:?} {} B (part {})",
                    t.name,
                    t.ty,
                    &t.dims[..t.n_dims as usize],
                    t.bytes,
                    t.file + 1
                );
            }
            Ok(())
        }
        ["check", path, "--against", gguf] => {
            let r = check::check_against(path, gguf)?;
            println!("OK {path} against {gguf}: {} tensors, {} packs, {:.1} s", r.tensors, r.packs, r.seconds);
            Ok(())
        }
        ["check", path, "--layers", s, rest @ ..] if rest.len() <= 1 => {
            let (a, b) = span(s)?;
            let ng = rest.first().and_then(|v| v.parse::<u32>().ok());
            let r = check::check_shard(path, a, b, ng)?;
            // the engine fixes the pipeline split from this file (pipe.rs in the model crate)
            std::fs::write(format!("{path}.layers"), format!("{a}-{b}"))?;
            println!("OK {path} (layers {a}..{b}): {} tensors, {} packs, {:.1} s", r.tensors, r.packs, r.seconds);
            Ok(())
        }
        ["check", path] => {
            let r = check::check(path)?;
            println!("OK {path}: {} tensors, {} packs, {:.1} s", r.tensors, r.packs, r.seconds);
            Ok(())
        }
        ["info", path] => info(path),
        ["tensors", path] => {
            let m = Model::open(path)?;
            for t in &m.tensors {
                println!("{} type {} dims {:?} {} B", t.name, t.ty, &t.dims[..t.n_dims as usize], t.bytes);
            }
            Ok(())
        }
        ["meta", path] => {
            let m = Model::open(path)?;
            for (k, v) in format::meta::Meta::parse(&m.metadata)?.entries {
                match v {
                    format::meta::Value::Array(_, a) if a.len() > 64 => println!("{k} = [{} items]", a.len()),
                    v => println!("{k} = {v:?}"),
                }
            }
            Ok(())
        }
        ["shard-plan", src, s, rest @ ..] | ["pack", src, _, s, rest @ ..] if rest.len() <= 1 => {
            let (a, b) = span(s)?;
            let ng = rest.first().and_then(|v| v.parse::<u32>().ok());
            let m = Model::open(src)?;
            let rs = format::shard::ranges(&m, a, b, ng);
            let need: u64 = rs.iter().map(|r| r.1).sum();
            eprintln!(
                "{src}: layers {a}..{b}{}: {} ranges, {:.2} GB of {:.2} GB",
                ng.map_or(String::new(), |l| format!(" + n-gram layer {l}")),
                rs.len(),
                need as f64 / 1e9,
                m.header.file_bytes as f64 / 1e9
            );
            if args[0] == "pack" {
                let dst = args[2].as_str();
                let n = if dst == "-" {
                    let so = std::io::stdout();
                    let mut w = std::io::BufWriter::with_capacity(8 << 20, so.lock());
                    format::shard::pack(src, &rs, &mut w)?
                } else {
                    // a new file only, as unpack: a model's name given here by mistake would otherwise be emptied
                    let f = std::fs::OpenOptions::new().write(true).create_new(true).open(dst).map_err(|e| {
                        format::Error(format!("cannot create {dst}: {e} (pack writes a new file only)"))
                    })?;
                    let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
                    format::shard::pack(src, &rs, &mut w)?
                };
                eprintln!("packed {:.2} GB", n as f64 / 1e9);
            }
            Ok(())
        }
        ["unpack", src, dst] => {
            let (fb, n) = if *src == "-" {
                let si = std::io::stdin();
                let mut r = std::io::BufReader::with_capacity(8 << 20, si.lock());
                format::shard::unpack(&mut r, dst)?
            } else {
                let mut r = std::io::BufReader::with_capacity(8 << 20, std::fs::File::open(src)?);
                format::shard::unpack(&mut r, dst)?
            };
            eprintln!("wrote {dst}: {:.2} GB long, {:.2} GB of it data (sparse)", fb as f64 / 1e9, n as f64 / 1e9);
            Ok(())
        }
        _ => Err(format::Error(USAGE.into())),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("torc: {e}");
            ExitCode::FAILURE
        }
    }
}
