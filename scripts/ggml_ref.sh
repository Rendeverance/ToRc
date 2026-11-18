#!/bin/sh
# Build the reference harness the parity test compares the portable quant kernels against.
#   scripts/ggml_ref.sh ~/llama.cpp            # writes target/ggml-ref
# Then: TORC_GGML_REF=target/ggml-ref cargo test -p torc-kernels --test parity
# The two files it links are the formats and their scalar definitions; no ggml build or backend is needed.
set -e
GGML=${1:?usage: ggml_ref.sh <llama.cpp checkout> [output]}
OUT=${2:-target/ggml-ref}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$(dirname "$OUT")"
cc -O2 -fno-strict-aliasing "$HERE/kernels/tests/ggml_ref.c" \
    "$GGML/ggml/src/ggml-quants.c" "$GGML/ggml/src/ggml-cpu/quants.c" \
    -I "$GGML/ggml/include" -I "$GGML/ggml/src" -lm -o "$OUT"
echo "built $OUT from $GGML ($(cd "$GGML" && git describe --tags 2>/dev/null || echo 'not a checkout'))"
