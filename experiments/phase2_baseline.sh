#!/usr/bin/env bash
# Phase 2 baselines: cycles/request and instruction mix, scalar vs tensor, tensor tile
# occupancy 1/8 .. 8/8, plus simulator speed. Runs the binaries phase2_check.sh built
# (run that first), each point in its own directory so points run in parallel.
# Usage: experiments/phase2_baseline.sh [build_dir]       (default ~/ntd-build)
#   HIDDEN="32"  N=1024  MAT=0  JOBS=$(nproc)  PY=~/ntd-venv/bin/python
set -euo pipefail
BUILD=${1:-$HOME/ntd-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HIDDEN=${HIDDEN:-32}
N=${N:-1024}
MAT=${MAT:-0}
JOBS=${JOBS:-$(nproc)}
PY=${PY:-$HOME/ntd-venv/bin/python}
OUT="$BUILD/p2base"
rm -rf "$OUT"; mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BUILD/sw/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VORTEX_DRIVER=simx VORTEX_PROFILING=1   # 1 = core class: adds the instruction mix

points="$OUT/points.txt"
: > "$points"
for h in $HIDDEN; do
  exp="$ROOT/data/export/h$h"
  img="img${MAT}_7"   # one material, raster order: every tile can be full
  echo "scalar_h${h}_${img} ntc_scalar -e $exp -v $img -n $N" >> "$points"
  echo "scalar_h${h}_mixed ntc_scalar -e $exp -v mixed -n $N" >> "$points"
  for o in 1 2 3 4 5 6 7 8; do
    echo "tcu_h${h}_${img}_o$o ntc_tcu -e $exp -v $img -n $N -o $o" >> "$points"
  done
  echo "tcu_h${h}_mixed ntc_tcu -e $exp -v mixed -n $N" >> "$points"   # FIFO over interleaved materials
done

run_point() {
  local name=$1 app=$2; shift 2
  local dir="$OUT/$name"; mkdir -p "$dir"; cd "$dir"
  local t0; t0=$(date +%s.%N)
  "$BUILD/tests/regression/$app/$app" -k "$BUILD/tests/regression/$app/kernel.vxbin" "$@" > run.log 2>&1 || true
  echo "WALL: $(echo "$(date +%s.%N) - $t0" | bc)" >> run.log
  echo "$name: $(grep -m1 -E 'PASSED|FAILED|Error' run.log || echo '(no result)')"
}
export -f run_point
export BUILD OUT
xargs -P "$JOBS" -L 1 bash -c 'run_point "$@"' _ < "$points"

"$PY" "$ROOT/experiments/phase2_table.py" "$OUT" --csv "$ROOT/results/phase2_baseline.csv"
