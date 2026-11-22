#!/usr/bin/env bash
# Phase 2 checks on SimX: ntc_scalar and ntc_tcu decode the exported vectors bit-exactly.
# Usage: experiments/phase2_check.sh [build_dir]   (default ~/ntd-build)
#   HIDDEN="16 32 64"   exports to check (data/export/h<H>)
#   VECS="mixed"        vector sets (vec_<name>_req.bin), see `python -m ntc.vortex_io prepare`
#   SCALAR_N=4096       request cap for the (slow) scalar decoder; TCU always runs the full set
#   ONLY=regex          subset of test names
set -uo pipefail
BUILD=${1:-$HOME/ntd-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export MAKEFLAGS=${MAKEFLAGS:--j$(nproc)}
HIDDEN=${HIDDEN:-"16 32 64"}
VECS=${VECS:-mixed}
SCALAR_N=${SCALAR_N:-4096}
ONLY=${ONLY:-}
CFG="-DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE"
cd "$BUILD"
for p in ntc_scalar ntc_tcu; do   # out-of-tree build dirs need the Makefile copied in
  mkdir -p "tests/regression/$p"
  cp "$ROOT/vortex/tests/regression/$p/Makefile" "tests/regression/$p/Makefile"
done

fail=0
run() {  # name, app, args
  local name=$1 app=$2 args=$3
  [[ -n "$ONLY" && ! "$name" =~ $ONLY ]] && return 0
  local log="$BUILD/p2_$name.log"
  local t0=$SECONDS
  if CONFIGS="$CFG" ./ci/blackbox.sh --driver=simx --app="$app" --threads=8 --args="$args" > "$log" 2>&1 \
     && grep -q "PASSED" "$log"; then
    echo "PASS $name  $(grep -m1 -E '^PERF:' "$log")  $(grep -m1 -o 'Requests checked: [0-9]*' "$log")  wall=$((SECONDS - t0))s"
  else
    echo "FAIL $name  (see $log)"; grep -E "mismatch|Error|error:" "$log" | head -5; tail -3 "$log"; fail=1
  fi
}

for h in $HIDDEN; do
  for v in $VECS; do
    run "scalar_h${h}_$v" ntc_scalar "-e $ROOT/data/export/h$h -v $v -n $SCALAR_N"
    run "tcu_h${h}_$v"    ntc_tcu    "-e $ROOT/data/export/h$h -v $v"
  done
done
exit $fail
