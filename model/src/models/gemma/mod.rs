//! gemma - Gemma 4: the model (gemma.rs), its assistant model as drafter (gemma_mtp.rs), the whole model on the
//! Snapdragon NPU (npu_gemma.rs), its chat format (chat.rs) and its server half (serve.rs).

pub mod chat;
#[allow(clippy::module_inception)]
pub mod gemma;
pub mod gemma_mtp;
#[cfg(all(target_arch = "aarch64", windows))]
pub mod npu_gemma;
pub mod serve;
