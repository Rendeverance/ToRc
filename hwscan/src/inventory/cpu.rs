//! cpu.rs - CPU inventory: brand, physical cores vs logical CPUs, which logical ids share a core, caches.

use crate::sys::{self, LogicalCpu};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Core {
    /// Index in OS enumeration order.
    pub id: usize,
    /// Windows EfficiencyClass: higher = faster core on hybrid parts.
    pub efficiency_class: u8,
    /// Flat logical CPU ids (group*64 + bit) that share this core.
    pub logical_ids: Vec<u32>,
    #[serde(skip)]
    pub logical: Vec<LogicalCpu>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheSummary {
    pub level: u8,
    pub kind: String,
    pub size_bytes: u64,
    pub instances: usize,
    pub line_bytes: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuInfo {
    pub brand: Option<String>,
    pub arch: &'static str,
    pub nominal_mhz: Option<u32>,
    pub physical_cores: usize,
    pub logical_cpus: usize,
    pub hybrid: bool,
    pub cores: Vec<Core>,
    pub caches: Vec<CacheSummary>,
    /// SIMD the benchmarks can use on this CPU.
    pub simd: Vec<&'static str>,
}

impl CpuInfo {
    /// One logical CPU per physical core, fastest cores first.
    pub fn one_per_core(&self) -> Vec<LogicalCpu> {
        self.cores_fastest_first().iter().filter_map(|c| c.logical.first().copied()).collect()
    }

    /// Every logical CPU, spread so SMT siblings come last.
    pub fn all_logical(&self) -> Vec<LogicalCpu> {
        let cores = self.cores_fastest_first();
        let depth = cores.iter().map(|c| c.logical.len()).max().unwrap_or(0);
        (0..depth).flat_map(|k| cores.iter().filter_map(move |c| c.logical.get(k).copied())).collect()
    }

    fn cores_fastest_first(&self) -> Vec<&Core> {
        let mut v: Vec<&Core> = self.cores.iter().collect();
        v.sort_by_key(|c| (std::cmp::Reverse(c.efficiency_class), c.id));
        v
    }
}

fn simd_features() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            v.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            v.push("avx512f");
        }
        if std::arch::is_x86_feature_detected!("avx512vnni") {
            v.push("avx512vnni");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            v.push("fma");
        }
        if std::arch::is_x86_feature_detected!("f16c") {
            v.push("f16c");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            v.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            v.push("dotprod");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            v.push("i8mm");
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            v.push("fp16");
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
            v.push("sve");
        }
    }
    v
}

pub fn collect() -> CpuInfo {
    let raw = sys::cpu_cores();
    let cores: Vec<Core> = raw
        .into_iter()
        .enumerate()
        .map(|(id, c)| Core {
            id,
            efficiency_class: c.efficiency_class,
            logical_ids: c.logical.iter().map(|l| l.flat_id()).collect(),
            logical: c.logical,
        })
        .collect();
    let logical_cpus = cores.iter().map(|c| c.logical.len()).sum();
    let hybrid = cores.windows(2).any(|w| w[0].efficiency_class != w[1].efficiency_class);

    let mut caches: Vec<CacheSummary> = Vec::new();
    for c in sys::cpu_caches() {
        if let Some(s) =
            caches.iter_mut().find(|s| s.level == c.level && s.kind == c.kind && s.size_bytes == c.size_bytes)
        {
            s.instances += 1;
        } else {
            caches.push(CacheSummary {
                level: c.level,
                kind: c.kind.to_string(),
                size_bytes: c.size_bytes,
                instances: 1,
                line_bytes: c.line_bytes,
            });
        }
    }
    caches.sort_by_key(|c| (c.level, c.kind.clone()));

    CpuInfo {
        brand: sys::cpu_brand(),
        arch: std::env::consts::ARCH,
        nominal_mhz: sys::cpu_nominal_mhz(),
        physical_cores: cores.len(),
        logical_cpus,
        hybrid,
        cores,
        caches,
        simd: simd_features(),
    }
}
