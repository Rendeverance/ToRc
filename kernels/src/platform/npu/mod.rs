//! npu - the Snapdragon NPU (Hexagon HTP, ARM64 Windows), through GenieX's ggml loaded at run time: its graph API for
//! whole models, and the split tier's device layer (a CPU model's chosen products, run as graphs).
//!
//! Roadmap: Intel's and AMD's NPUs. Only the Snapdragon's is here so far; each needs a back end of its own, and I
//! have neither machine to build one on.

/// GenieX's ggml (the graph API and the HTP backend), loaded at run time.
pub mod ggml;
/// The split tier: a CPU model's chosen products on the NPU.
pub mod split;
