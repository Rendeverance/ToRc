// build.rs: on Linux, the NPU tier's C++ guard (src/platform/npu/guard.cpp), which turns an exception thrown inside
// ggml into an error. Elsewhere there is nothing to build.
fn main() {
    println!("cargo:rerun-if-changed=src/platform/npu/guard.cpp");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os == "linux" && (arch == "aarch64" || arch == "x86_64") {
        cc::Build::new().cpp(true).file("src/platform/npu/guard.cpp").compile("torc_npu_guard");
    }
}
