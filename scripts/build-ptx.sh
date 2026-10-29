#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The GPU ABI and LLVM intrinsics are nightly surfaces. Pin the toolchain so a
# future nightly cannot silently change generated kernels or stop old releases
# from rebuilding.
GPU_TOOLCHAIN="${GPU_TOOLCHAIN:-nightly-2026-08-27}"

# Full unrolling keeps the per nonce arrays in registers. At its default
# threshold LLVM rolls them back into loops over local memory.
RUSTFLAGS="-Ctarget-cpu=sm_70 -Cllvm-args=-unroll-threshold=2000" cargo +"$GPU_TOOLCHAIN" rustc \
    -p numen-kernel \
    --release \
    --target nvptx64-nvidia-cuda \
    -Zbuild-std=core,compiler_builtins \
    -Zbuild-std-features=compiler-builtins-mem \
    --crate-type cdylib

SRC="target/nvptx64-nvidia-cuda/release/numen_kernel.ptx"
OUT="miner/ptx"
mkdir -p "$OUT"

PY="$(command -v python || command -v python3)"

"$PY" - "$SRC" "$OUT" <<'PY'
import re
import sys

src_path, out_dir = sys.argv[1], sys.argv[2]
src = open(src_path).read()

# The scan kernel is sized for three resident blocks of 128, the most its
# shared memory allows, and ptxas derives the register budget from that.
def cap(text):
    text = re.sub(r"^\.(maxntid|minnctapersm) [^\n]*\n", "", text, flags=re.M)
    pat = re.compile(r"(\.visible \.entry poscan_scan\([^)]*\)[^\n]*\n)")
    text, hits = pat.subn(lambda m: m.group(1) + ".maxntid 128, 1, 1\n.minnctapersm 3\n", text, count=1)
    if hits != 1:
        sys.exit("kernel entry poscan_scan not found")
    return text

modern = cap(src)

legacy = modern.replace(".version 7.0", ".version 6.4")
legacy = legacy.replace(".target sm_70", ".target sm_30")
legacy = legacy.replace("ld.global.nc.", "ld.global.")

shf = re.compile(
    r"^(\s*)shf\.l\.wrap\.b32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+),\s*(\d+);", re.M
)

# shf.l.wrap d, a, b, n is the high word of b:a shifted left by n.
def expand(m):
    ind, d, a, b, n = m.groups()
    n = int(n)
    if not 1 <= n <= 31:
        sys.exit(f"unexpected funnel shift {m.group(0).strip()}")
    return (
        f"{ind}{{ .reg .b32 t1; .reg .b32 t2; "
        f"shl.b32 t1, {b}, {n}; shr.u32 t2, {a}, {32 - n}; "
        f"or.b32 {d}, t1, t2; }}"
    )

legacy, n = shf.subn(expand, legacy)
open(f"{out_dir}/modern.ptx", "w", newline="\n").write(modern)
open(f"{out_dir}/legacy.ptx", "w", newline="\n").write(legacy)
print(f"expanded {n} funnel shifts for the legacy blob")
PY

if grep -qE 'shf\.|ld\.global\.nc' "$OUT/legacy.ptx"; then
    echo "  ERROR gated instructions left in the legacy blob" >&2
    exit 1
fi

echo
echo "staged $OUT/modern.ptx"
echo "staged $OUT/legacy.ptx, sm_30 at PTX 6.4"
grep -c 'sqrt.rn.f32' "$OUT/modern.ptx" | xargs echo "  hardware sqrt sites"
