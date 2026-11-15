//! hwscan - inventory the machine and measure the numbers that decide how fast a model can run on it (RAM bandwidth by
//! thread count, real clock under load, drive random-read speed). What a machine reports about itself is not always
//! what it does, so every figure here is measured.

mod bench;
mod inventory;
mod power;
mod report;
mod sys;

use std::time::Instant;

const USAGE: &str = "\
hwscan - hardware inventory and speed tests for tuning ToRc to a machine

USAGE: hwscan [--json] [--quick] [--no-bench]

  --json      machine-readable output on stdout (progress goes to stderr)
  --quick     shorter speed tests (about 10 s; the full run is about 25 s plus a
              2 GiB temp-file test when the %TEMP% drive holds no model file)
  --no-bench  inventory and power only
  --host      put this machine's name in the report (left out by default, so a
              report can be shared as it is)
  -h, --help  this text
";

fn main() {
    let (mut json, mut quick, mut no_bench, mut host) = (false, false, false, false);
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--json" => json = true,
            "--quick" => quick = true,
            "--no-bench" => no_bench = true,
            "--host" => host = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    let mode = if quick { bench::Mode::Quick } else { bench::Mode::Full };
    let log = |s: &str| eprintln!("[hwscan] {s}");

    log("inventory");
    let inv = inventory::collect();
    let power = power::collect();

    let bench = (!no_bench).then(|| {
        let t = Instant::now();
        // the clock test first: its burst numbers want a rested machine
        let clock = bench::clock::run(&inv.cpu, mode, &log);
        let ram = bench::ram::run(&inv.cpu, mode, &log);
        let drives = bench::disk::run(&inv.drives, mode, &log);
        bench::BenchResults {
            mode: if quick { "quick" } else { "full" },
            ram,
            clock,
            drives,
            gpu: "GPU tests: none (the GPU tier measures itself while it runs)",
            total_seconds: bench::secs(t.elapsed()),
        }
    });

    let (warnings, notes) = report::derive_warnings(&inv, &power, bench.as_ref());
    let r = report::Report {
        tool: "hwscan",
        version: env!("CARGO_PKG_VERSION"),
        generated_utc: report::utc_now(),
        host: if host { std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).ok() } else { None },
        os: std::env::consts::OS,
        inventory: inv,
        power,
        bench,
        warnings,
        notes,
    };
    if json {
        match serde_json::to_string_pretty(&r) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("json: {e}");
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", report::render_human(&r));
    }
}
