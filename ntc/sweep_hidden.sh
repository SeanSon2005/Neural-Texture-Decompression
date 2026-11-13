#!/usr/bin/env bash
# Train + export the hidden-width sweep (H in {16, 32, 64}) used to pick H.
# Usage (from the repo root, inside WSL/Linux): ntc/sweep_hidden.sh [python]
set -euo pipefail
PY=${1:-python3}
mkdir -p data/export
for h in 16 32 64; do
  "$PY" -m ntc.train --out "data/export/h$h" --hidden "$h" \
        --steps-float 12000 --steps-qat 4000 > "data/export/train_h$h.log" 2>&1
  echo "h$h done"
  grep -E "psnr_all|ratio" "data/export/train_h$h.log"
done
