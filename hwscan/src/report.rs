//! report.rs - the report: warnings derived from the data, and the human-readable rendering. (`--json` serialises the
//! same `Report` struct.)

use crate::bench::BenchResults;
use crate::inventory::Inventory;
use crate::power::PowerInfo;
use serde::Serialize;
use std::fmt::Write as _;

#[derive(Debug, Serialize)]
pub struct Report {
    pub tool: &'static str,
    pub version: &'static str,
    pub generated_utc: String,
    pub host: Option<String>,
    pub os: &'static str,
    pub inventory: Inventory,
    pub power: PowerInfo,
    pub bench: Option<BenchResults>,
    /// Things that cap the engine. Loud.
    pub warnings: Vec<String>,
    /// Facts worth knowing that are not problems.
    pub notes: Vec<String>,
}

pub fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

/// UTC "YYYY-MM-DDTHH:MM:SSZ" without a date crate (civil-from-days, Howard Hinnant).
pub fn utc_now() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (s.div_euclid(86_400), s.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem / 60 % 60, rem % 60)
}

pub fn derive_warnings(inv: &Inventory, power: &PowerInfo, bench: Option<&BenchResults>) -> (Vec<String>, Vec<String>) {
    let (mut w, mut n) = (Vec::new(), Vec::new());

    if power.energy_saver_on() {
        w.push(
            "WINDOWS ENERGY SAVER IS ON (EnergySaverState=1). It can hold the CPU far below its rated clock \
             and cut decode speed by a third or more. Turn it off: Settings > System > Power & battery > Energy saver."
                .into(),
        );
    } else if power.saver_flag_from_power_status == Some(true) {
        w.push("GetSystemPowerStatus reports battery/energy saver active although EnergySaverState is not 1.".into());
    }
    if power.on_ac == Some(false) {
        w.push(format!(
            "Running on BATTERY ({}%): clocks and power limits are lower than on AC.",
            power.battery_percent.map_or("?".into(), |p| p.to_string())
        ));
    }
    if power.overlay_ac_name.as_deref() == Some("Best power efficiency") {
        w.push("Power mode on AC is 'Best power efficiency': expect lower clocks.".into());
    }
    for d in &inv.drives {
        if let (Some(dir), true) = (&d.models_dir, d.is_usb()) {
            w.push(format!(
                "Model directory {dir} is on a USB-bus drive ({}): expert misses will read at USB speed.",
                d.product.as_deref().unwrap_or("?")
            ));
        }
    }
    for g in &inv.gpus.gpus {
        if g.resizable_bar == Some(false) {
            w.push(format!(
                "{}: resizable BAR looks OFF (host-visible VRAM {} of {}).",
                g.name,
                gib(g.host_visible_device_local_bytes),
                gib(g.device_local_bytes)
            ));
        }
    }
    if let Some(e) = &inv.memory.error {
        n.push(format!("RAM module query failed: {e}"));
    }
    if inv.npus.npus.is_empty() {
        n.push("No NPU found (best effort: a disabled or driverless NPU is invisible).".into());
    }

    if let Some(b) = bench {
        if let Some(c) = &b.clock {
            let burst = c.single_per_core.iter().copied().fold(0.0, f64::max);
            if let Some(nom) = inv.cpu.nominal_mhz {
                let nom = nom as f64 / 1000.0;
                if burst < 0.9 * nom {
                    w.push(format!(
                        "Best single-core clock under load is {burst:.2} GHz, below the {nom:.2} GHz nominal: something is capping the CPU."
                    ));
                }
            }
            // single core: the short-visit burst against the sustained run; all cores: the first sample against the
            // last
            for (what, start, end) in
                [("1 core", burst, c.single_ghz_median), ("all cores", c.all_ghz_first, c.all_ghz_last)]
            {
                if start > 0.0 && end < 0.9 * start {
                    w.push(format!(
                        "Clock drops under sustained load ({what}): {start:.2} -> {end:.2} GHz ({:.0}%). A power/thermal \
                         limit (vendor performance mode, PPT, temperature) caps sustained speed; plan from the sustained figure.",
                        100.0 * (end / start - 1.0)
                    ));
                }
            }
        }
        if let Some(r) = &b.ram {
            if r.rows.iter().any(|row| !row.all_pinned) {
                w.push("Some benchmark threads could not be pinned: RAM numbers may be pessimistic.".into());
            }
            let best_phys = r
                .rows
                .iter()
                .filter(|x| x.placement.starts_with("one per"))
                .max_by(|a, b| a.gbs_median.total_cmp(&b.gbs_median));
            let smt = r.rows.iter().find(|x| x.placement.starts_with("all logical"));
            if let (Some(p), Some(s)) = (best_phys, smt)
                && s.gbs_median < 0.97 * p.gbs_median
            {
                n.push(format!(
                    "SMT lowers read bandwidth ({:.1} GB/s at {} threads vs {:.1} at {} cores): count cores, not threads.",
                    s.gbs_median, s.threads, p.gbs_median, p.threads
                ));
            }
            n.push(format!(
                "RAM read ceiling: {:.1} GB/s (best at {} threads; within 5% from {} thread(s)). This is a pure streaming read: a dequantising matvec needs more cores to reach the same ceiling.",
                r.best_gbs, r.best_threads, r.saturates_at_threads
            ));
            let noisy: Vec<String> = r
                .rows
                .iter()
                .filter(|x| x.passes > 0 && x.gbs_min < 0.85 * x.gbs_median)
                .map(|x| x.threads.to_string())
                .collect();
            if !noisy.is_empty() {
                n.push(format!(
                    "RAM passes were noisy at {} thread(s) (min < 85% of median): other processes preempted pinned \
                     threads. Noise only lowers a pass, so the max column is the better ceiling estimate there.",
                    noisy.join("/")
                ));
            }
            if let Some(t) = inv.memory.theoretical_gbs_if_64bit_channels {
                n.push(format!(
                    "That is {:.0}% of the {:.1} GB/s SMBIOS peak (speed x 8 B x {} channel labels).",
                    100.0 * r.best_gbs / t,
                    t,
                    inv.memory.channels.len()
                ));
            }
        }
        for d in &b.drives {
            if let Some(e) = &d.error {
                n.push(format!("Drive test {}: {e}", d.root));
            }
        }
    }
    (w, n)
}

fn opt<T: std::fmt::Display>(v: &Option<T>) -> String {
    v.as_ref().map_or("?".into(), |x| x.to_string())
}

pub fn render_human(r: &Report) -> String {
    let mut o = String::new();
    let inv = &r.inventory;
    let host = r.host.as_ref().map_or(String::new(), |h| format!("  host {h}"));
    let _ = writeln!(o, "hwscan {}  {}{host}  ({})", r.version, r.generated_utc, r.os);

    if !r.warnings.is_empty() {
        let _ = writeln!(o, "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNINGS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        for w in &r.warnings {
            let _ = writeln!(o, "!! {w}");
        }
        let _ = writeln!(o, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }

    // 1. Inventory
    let c = &inv.cpu;
    let _ = writeln!(o, "\n== 1. Inventory ==");
    let _ = writeln!(o, "CPU     {}  [{}]", opt(&c.brand), c.arch);
    let _ = writeln!(
        o,
        "        {} physical cores, {} logical CPUs{}; nominal {} MHz; SIMD: {}",
        c.physical_cores,
        c.logical_cpus,
        if c.hybrid { " (hybrid: mixed core classes)" } else { "" },
        opt(&c.nominal_mhz),
        if c.simd.is_empty() { "-".into() } else { c.simd.join(" ") }
    );
    let map: Vec<String> = c
        .cores
        .iter()
        .map(|k| {
            let ids: Vec<String> = k.logical_ids.iter().map(|x| x.to_string()).collect();
            if c.hybrid {
                format!("c{}[{}]:e{}", k.id, ids.join(","), k.efficiency_class)
            } else {
                format!("c{}[{}]", k.id, ids.join(","))
            }
        })
        .collect();
    let _ = writeln!(o, "        core->logical: {}", map.join(" "));
    if !c.caches.is_empty() {
        let cs: Vec<String> = c
            .caches
            .iter()
            .filter(|x| x.kind != "instruction")
            .map(|x| format!("L{} {} {} KiB x{}", x.level, x.kind, x.size_bytes >> 10, x.instances))
            .collect();
        let _ = writeln!(o, "        caches: {}", cs.join(", "));
    }

    let m = &inv.memory;
    let _ = writeln!(
        o,
        "RAM     {} visible, {} installed in {} module(s); channels: {}; speed {} MT/s",
        m.visible_bytes.map_or("?".into(), gib),
        m.installed_bytes.map_or("?".into(), gib),
        m.modules.len(),
        if m.channels.is_empty() {
            "unknown".into()
        } else {
            format!("{} ({})", m.channels.len(), m.channels.join(", "))
        },
        opt(&m.speed_mts)
    );
    for md in &m.modules {
        let _ = writeln!(
            o,
            "        - {} {} {} MT/s (rated {}) bank '{}' slot '{}' {}",
            md.capacity_bytes.map_or("?".into(), gib),
            md.memory_type,
            opt(&md.configured_speed_mts),
            opt(&md.speed_mts),
            opt(&md.bank_label),
            opt(&md.device_locator),
            opt(&md.part_number)
        );
    }
    if let Some(t) = m.theoretical_gbs_if_64bit_channels {
        let _ = writeln!(o, "        SMBIOS peak if each channel label is 64-bit: {t:.1} GB/s");
    }

    let _ = writeln!(o, "Drives");
    for d in &inv.drives {
        let _ = writeln!(
            o,
            "        {} {:<5} {:<34} free {} of {}  {}{}",
            d.root,
            d.bus,
            d.product.as_deref().unwrap_or("?"),
            d.free_bytes.map_or("?".into(), gib),
            d.total_bytes.map_or("?".into(), gib),
            opt(&d.filesystem),
            d.models_dir.as_ref().map_or(String::new(), |m| format!("  MODELS: {m}"))
        );
    }

    let g = &inv.gpus;
    let _ = writeln!(o, "GPUs    (Vulkan loader {})", opt(&g.loader_version));
    if let Some(e) = &g.error {
        let _ = writeln!(o, "        error: {e}");
    }
    for gpu in &g.gpus {
        let _ = writeln!(
            o,
            "        {} [{} {}, Vulkan {}, driver {} {}]",
            gpu.name,
            gpu.vendor,
            gpu.kind,
            gpu.vulkan_version,
            opt(&gpu.driver_name),
            opt(&gpu.driver_info)
        );
        for h in &gpu.heaps {
            let _ = writeln!(
                o,
                "          heap {}: {:>9} {}  types: {}",
                h.index,
                gib(h.size_bytes),
                if h.device_local { "DEVICE_LOCAL" } else { "host" },
                if h.type_flags.is_empty() { "-".into() } else { h.type_flags.join("|") }
            );
        }
        let rebar = match gpu.resizable_bar {
            Some(true) => format!(
                "YES ({} of {} host-visible)",
                gib(gpu.host_visible_device_local_bytes),
                gib(gpu.device_local_bytes)
            ),
            Some(false) => format!("NO (only {} host-visible)", gib(gpu.host_visible_device_local_bytes)),
            None => "n/a (integrated: shares system RAM)".into(),
        };
        let sg = match (gpu.subgroup_size, gpu.subgroup_size_min, gpu.subgroup_size_max) {
            (Some(s), Some(a), Some(b)) if a != b => format!("{s} (range {a}-{b})"),
            (Some(s), _, _) => s.to_string(),
            _ => "?".into(),
        };
        let _ = writeln!(
            o,
            "          resizable BAR: {rebar}; external_memory_host: {}{}; subgroup {sg}",
            if gpu.external_memory_host { "yes" } else { "no" },
            gpu.min_imported_host_pointer_alignment.map_or(String::new(), |a| format!(" (align {a})"))
        );
        if !gpu.notable_extensions.is_empty() {
            let _ = writeln!(o, "          ext: {}", gpu.notable_extensions.join(" "));
        }
    }

    let _ = writeln!(o, "NPUs    ({})", inv.npus.method);
    if let Some(e) = &inv.npus.error {
        let _ = writeln!(o, "        error: {e}");
    }
    if inv.npus.npus.is_empty() {
        let _ = writeln!(o, "        none found");
    }
    for n in &inv.npus.npus {
        let _ = writeln!(o, "        {} [{}] status {} ({})", n.name, opt(&n.class), opt(&n.status), n.matched);
    }

    // 2. Power
    let p = &r.power;
    let _ = writeln!(o, "\n== 2. Power ==");
    let _ = writeln!(
        o,
        "Supply        {}{}",
        match p.on_ac {
            Some(true) => "AC",
            Some(false) => "BATTERY",
            None => "unknown",
        },
        p.battery_percent.map_or(String::new(), |b| format!(" (battery {b}%)"))
    );
    let _ =
        writeln!(o, "Energy Saver  {}  [EnergySaverState raw = {}]", p.energy_saver, opt(&p.energy_saver_state_raw));
    let _ = writeln!(o, "Scheme        {} {}", opt(&p.active_scheme_name), opt(&p.active_scheme_guid));
    let _ = writeln!(o, "Overlay AC    {} {}", opt(&p.overlay_ac_name), opt(&p.overlay_ac_guid));
    let _ = writeln!(o, "Overlay DC    {} {}", opt(&p.overlay_dc_name), opt(&p.overlay_dc_guid));
    let _ = writeln!(
        o,
        "Boost (AC)    {}; EPP (AC) {}%  [scheme values; overlay may override]",
        opt(&p.boost_mode_ac),
        opt(&p.epp_ac)
    );

    // 3. Speed tests
    let _ = writeln!(o, "\n== 3. Speed tests ==");
    match &r.bench {
        None => {
            let _ = writeln!(o, "skipped (--no-bench)");
        }
        Some(b) => {
            let _ = writeln!(o, "mode {}, total {:.1} s", b.mode, b.total_seconds);
            if let Some(ram) = &b.ram {
                let _ = writeln!(
                    o,
                    "a. RAM read bandwidth  ({} MiB buffer, {}, median of passes, GB = 1e9 B, {:.1} s)",
                    ram.buffer_bytes >> 20,
                    ram.kernel,
                    ram.seconds
                );
                for row in &ram.rows {
                    let _ = writeln!(
                        o,
                        "   {:>3} threads  {:>6.1} GB/s  (min {:>5.1}, max {:>5.1}, {} passes)  {}{}",
                        row.threads,
                        row.gbs_median,
                        row.gbs_min,
                        row.gbs_max,
                        row.passes,
                        row.placement,
                        if row.all_pinned { "" } else { "  [NOT ALL PINNED]" }
                    );
                }
                let _ = writeln!(
                    o,
                    "   best: {:.1} GB/s at {} threads; within 5% of best from {} thread(s)",
                    ram.best_gbs, ram.best_threads, ram.saturates_at_threads
                );
            }
            if let Some(c) = &b.clock {
                let _ = writeln!(o, "b. Real clock under load  ({}, {:.1} s)", c.method, c.seconds);
                let alone: Vec<String> = c.single_per_core.iter().map(|x| format!("{x:.2}")).collect();
                let _ = writeln!(o, "   each core alone (best short visit): {}", alone.join(" "));
                let _ = writeln!(
                    o,
                    "   1 core sustained (cpu {}, {:.1} s):  start {:.2} -> median {:.2} -> end {:.2} GHz",
                    c.single_core_logical_id,
                    c.single_seconds,
                    c.single_ghz_first,
                    c.single_ghz_median,
                    c.single_ghz_last
                );
                let _ = writeln!(
                    o,
                    "   {} cores sustained ({:.1} s):        start {:.2} -> median {:.2} -> end {:.2} GHz (per-core medians {:.2}-{:.2})",
                    c.all_threads,
                    c.all_seconds,
                    c.all_ghz_first,
                    c.all_ghz_median,
                    c.all_ghz_last,
                    c.all_ghz_min_core,
                    c.all_ghz_max_core
                );
                let per: Vec<String> = c.all_per_core.iter().map(|x| format!("{x:.2}")).collect();
                let _ = writeln!(o, "   each core, all-on (median):         {}", per.join(" "));
            }
            let _ = writeln!(o, "c. Drive random reads  (2 MiB blocks, 4 KiB-aligned random offsets, unbuffered)");
            if b.drives.is_empty() {
                let _ = writeln!(o, "   no drive tested");
            }
            for d in &b.drives {
                let _ = writeln!(o, "   {} {} - {} {} ({})", d.root, d.bus, d.target_kind, d.target, gib(d.file_bytes));
                if let Some(wm) = d.temp_write_mbs {
                    let _ = writeln!(o, "     temp file write (sequential, write-through): {wm:.0} MB/s");
                }
                for q in [&d.qd1, &d.qd8].into_iter().flatten() {
                    let _ = writeln!(
                        o,
                        "     QD{:<2} {:>7.0} MB/s  latency ms: median {:>6.2}  p90 {:>6.2}  p99 {:>6.2}  max {:>6.2}  ({} reads{})",
                        q.queue_depth,
                        q.mbs,
                        q.lat_ms_median,
                        q.lat_ms_p90,
                        q.lat_ms_p99,
                        q.lat_ms_max,
                        q.reads,
                        if q.capped { ", time-capped" } else { "" }
                    );
                }
                if let Some(e) = &d.error {
                    let _ = writeln!(o, "     error: {e}");
                }
            }
            let _ = writeln!(o, "GPU tests: none (the GPU tier measures itself while it runs)");
        }
    }

    if !r.notes.is_empty() {
        let _ = writeln!(o, "\n== Notes ==");
        for n in &r.notes {
            let _ = writeln!(o, "- {n}");
        }
    }
    o
}

#[cfg(test)]
mod tests {
    #[test]
    fn utc_format() {
        let s = super::utc_now();
        assert_eq!(s.len(), 20);
        assert!(s.ends_with('Z'));
    }
}
