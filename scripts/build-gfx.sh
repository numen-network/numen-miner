#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_TOOLCHAIN="${GPU_TOOLCHAIN:-nightly-2026-08-27}"

OUT="miner/gfx"
mkdir -p "$OUT"

TARGETS=(
    gfx1010 gfx1011 gfx1012
    gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
    gfx1100 gfx1101 gfx1102 gfx1103
    gfx1150 gfx1151 gfx1152 gfx1153
    gfx1200 gfx1201
    gfx803 gfx900 gfx906 gfx90c
)
if [ $# -gt 0 ]; then
    TARGETS=("$@")
fi

HOST="$(rustc +"$GPU_TOOLCHAIN" -vV | sed -n 's/^host: //p')"
BIN="$(rustc +"$GPU_TOOLCHAIN" --print sysroot)/lib/rustlib/$HOST/bin"
PY="$(command -v python || command -v python3)"

# rustc has no way to give a kernel its block size, and LLVM otherwise plans
# every kernel for blocks of 1024. That caps the registers low enough to spill
# the scan kernel's sums. So the kernels go out as LLVM IR, take the block
# size as an attribute, and the object is built from there.
for gfx in "${TARGETS[@]}"; do
    DIR="target/gfx/$gfx"
    mkdir -p "$DIR"
    RUSTFLAGS="-Ctarget-cpu=$gfx -Cllvm-args=-unroll-threshold=2000" \
    CARGO_TARGET_DIR="$DIR" \
    cargo +"$GPU_TOOLCHAIN" rustc \
        -p numen-kernel \
        --release \
        --target amdgcn-amd-amdhsa \
        -Zbuild-std=core,compiler_builtins \
        -Zbuild-std-features=compiler-builtins-mem \
        --crate-type cdylib \
        -- --emit="link,llvm-ir=$DIR/kernel.ll"

    "$PY" - "$DIR/kernel.ll" <<'PY'
import re
import sys

path = sys.argv[1]
ir = open(path).read()
groups = set()
for name in ("poscan_draw", "poscan_scan", "poscan_finish"):
    m = re.search(r"^define amdgpu_kernel void @%s\(.*#(\d+)" % name, ir, re.M)
    if not m:
        sys.exit(f"kernel entry {name} not found")
    groups.add(m.group(1))
for g in groups:
    ir, hits = re.subn(r"^(attributes #%s = \{.*)\}$" % g,
                       r'\1"amdgpu-flat-work-group-size"="128,128" }', ir, count=1, flags=re.M)
    if hits != 1:
        sys.exit(f"attribute group {g} not found")
open(path, "w", newline="\n").write(ir)
PY

    "$BIN/llc" -O3 -mtriple=amdgcn-amd-amdhsa -mcpu="$gfx" --amdhsa-code-object-version=5 \
        --disable-promote-alloca-to-lds -filetype=obj -o "$DIR/kernel.o" "$DIR/kernel.ll"
    "$BIN/rust-lld" -flavor gnu -shared --gc-sections -O1 --strip-debug \
        -o "$OUT/$gfx.hsaco" "$DIR/kernel.o"
    grep -q 'poscan_scan.kd' "$OUT/$gfx.hsaco" || {
        echo "  ERROR kernel entry missing in $gfx" >&2
        exit 1
    }
done

echo
echo "staged ${#TARGETS[@]} code objects in $OUT"
