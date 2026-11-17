//! platform - the platform kernels, i.e. what computes on a device: one folder per device class (cpu, npu), and one
//! file per instruction set or backend. The quant types they read are defined in crate::quant.

pub mod cpu;
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
pub mod npu;
