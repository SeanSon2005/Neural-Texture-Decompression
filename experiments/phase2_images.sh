#!/usr/bin/env bash
# Decode a full 2**L square image of every material on Vortex (ntc_tcu, full tiles), check it
# against the reference decoder, write a PNG sheet per material and PSNR vs the source.
# Needs the binaries from phase2_check.sh and vectors from `python -m ntc.vortex_io prepare`.
# Usage: experiments/phase2_images.sh [build_dir]     HIDDEN=32 LOG=7 JOBS=8 PY=~/ntd-venv/bin/python
set -euo pipefail
BUILD=${1:-$HOME/ntd-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HIDDEN=${HIDDEN:-32}
LOG=${LOG:-7}
JOBS=${JOBS:-8}
PY=${PY:-$HOME/ntd-venv/bin/python}
EXP="$ROOT/data/export/h$HIDDEN"
OUT="$BUILD/p2img_h$HIDDEN"
PNG="$ROOT/results/phase2_images/h$HIDDEN"
mkdir -p "$OUT" "$PNG"
export LD_LIBRARY_PATH="$BUILD/sw/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VORTEX_DRIVER=simx
NMAT=$(awk '$1 == "materials" {print $2}' "$EXP/vortex.txt")

decode() {
  local m=$1 dir="$OUT/m$1"
  mkdir -p "$dir"; cd "$dir"
  "$BUILD/tests/regression/ntc_tcu/ntc_tcu" -k "$BUILD/tests/regression/ntc_tcu/kernel.vxbin" \
    -e "$EXP" -v "img${m}_$LOG" -d dump.bin > run.log 2>&1 || true
  echo "material $m: $(grep -m1 -E 'PASSED|FAILED|Error' run.log)  $(grep -m1 '^PERF:' run.log)"
}
export -f decode
export BUILD OUT EXP LOG
seq 0 $((NMAT - 1)) | xargs -P "$JOBS" -I{} bash -c 'decode {}'

cd "$ROOT"
"$PY" - "$EXP" "$OUT" "$PNG" "$LOG" "$NMAT" <<'EOF'
import csv, os, sys
from ntc.vortex_io import check_image
exp, out, png, log, nmat = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rows = []
for m in range(nmat):
    r = check_image(exp, f"img{m}_{log}", os.path.join(out, f"m{m}", "dump.bin"), "data/raw",
                    os.path.join(png, "{name}.png"))
    rows.append(r)
    print(r)
path = os.path.join(os.path.dirname(png), f"psnr_{os.path.basename(png)}.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("wrote", path)
sys.exit(1 if any(r["mismatches"] for r in rows) else 0)
EOF
