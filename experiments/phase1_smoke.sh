#!/usr/bin/env bash
# Phase 1 checks on SimX: stock smoke tests plus the TEX -> int8 TCU spike.
# Usage: experiments/phase1_smoke.sh [build_dir]   (default ~/ntd-build)
set -uo pipefail
BUILD=${1:-$HOME/ntd-build}
export MAKEFLAGS=${MAKEFLAGS:--j$(nproc)}
cd "$BUILD"

INT8="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE"
fail=0
ONLY=${ONLY:-}   # regex on test names, e.g. ONLY='sgemm|ntc' to run a subset
run() {  # name, then the command
  local name=$1; shift
  [[ -n "$ONLY" && ! "$name" =~ $ONLY ]] && return 0
  local log="$BUILD/smoke_$name.log"
  if "$@" > "$log" 2>&1 && grep -q "PASSED" "$log"; then
    echo "PASS $name  $(grep -m1 -E 'PERF:' "$log" || true)"
  else
    echo "FAIL $name  (see $log)"; tail -5 "$log"; fail=1
  fi
}

run demo ./ci/blackbox.sh --driver=simx --app=demo
run gfx_tex ./ci/blackbox.sh --driver=simx --app=gfx_tex

if [[ -z "$ONLY" || "sgemm_tcu_int8" =~ $ONLY ]]; then
  make -s -C tests/regression/sgemm_tcu clean > /dev/null 2>&1
  CONFIGS="-DVX_CFG_NUM_THREADS=8 $INT8 -DITYPE=int8 -DOTYPE=int32" make -s -C tests/regression/sgemm_tcu > /dev/null 2>&1
fi
CONFIGS="$INT8 -DITYPE=int8 -DOTYPE=int32" run sgemm_tcu_int8 \
  ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --threads=8

for seed in 1 2 3; do
  CONFIGS="-DVX_CFG_EXT_TEX_ENABLE $INT8" run "ntc_spike_s$seed" \
    ./ci/blackbox.sh --driver=simx --app=ntc_spike --threads=8 --args="-s$seed"
done
CONFIGS="-DVX_CFG_EXT_TEX_ENABLE $INT8" run ntc_spike_point_repeat \
  ./ci/blackbox.sh --driver=simx --app=ntc_spike --threads=8 --args="-g0 -w1"
exit $fail
