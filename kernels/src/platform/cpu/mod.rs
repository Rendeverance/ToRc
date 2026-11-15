//! cpu - the CPU kernels: NEON (SDOT, SMMLA) on ARM64; AVX2, FMA, F16C and AVX-VNNI on x86-64; and float helpers shared
//! by both.

pub mod fvec;
pub mod mix;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub mod neon_mm;
#[cfg(target_arch = "x86_64")]
pub mod x86;
#[cfg(target_arch = "x86_64")]
pub mod x86_gemm;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
pub mod x86dot;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
pub mod x86mm;
