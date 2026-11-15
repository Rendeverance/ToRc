//! tiers - the optional devices beside the CPU: the Snapdragon NPU (npu.rs) and a Vulkan GPU for long prompts'
//! attention (gpu.rs, built with the `gpu` feature).

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod npu;
