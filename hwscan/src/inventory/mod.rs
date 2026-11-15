//! inventory - section 1: what is in the machine (no benchmarks).

pub mod cpu;
pub mod drives;
pub mod gpu;
pub mod memory;
pub mod npu;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Inventory {
    pub cpu: cpu::CpuInfo,
    pub memory: memory::MemoryInfo,
    pub drives: Vec<drives::Drive>,
    pub gpus: gpu::GpuInfo,
    pub npus: npu::NpuInfo,
}

pub fn collect() -> Inventory {
    Inventory {
        cpu: cpu::collect(),
        memory: memory::collect(),
        drives: drives::collect(),
        gpus: gpu::collect(),
        npus: npu::collect(),
    }
}
