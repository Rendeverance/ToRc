#!/usr/bin/env bash
# x86-64 Windows build: rustup's gnu toolchain (msvc builds too), x86-64-v3 from
# .cargo/config.toml. Usage: build-x86.sh [target dir, default target] [more cargo arguments,
# e.g. --features torc-model/gpu]
cd "$(dirname "$0")" && cargo +stable-x86_64-pc-windows-gnu build --release --workspace \
    --target-dir "${1:-target}" "${@:2}"
