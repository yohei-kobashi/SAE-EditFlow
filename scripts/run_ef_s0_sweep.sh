#!/bin/bash
# S0 (EDIT_FLOWS_ZERO §5, revised process): decode done with the paper's
# own tools on the CHAMPION (pilot) — CFG sweep on λ/Q, temperature, and
# bo{K} (best-of-K stochastic + directional SAE-achievement selection).
# true-only to keep the sweep cheap (~2h); the winning config gets a full
# three-condition confirmation afterwards. Probe resumes per pair.
# Run on miyabi with a GPU session:  bash scripts/run_ef_s0_sweep.sh
set -eo pipefail
cd "$(dirname "$0")/.."

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}
OUT=$V6/editflow_pilot/probe_s0

python scripts/editflow_probe.py \
    --llm2vec-dir "$LLM2VEC" \
    --editflow-ckpt "$V6/editflow_pilot/editflow-final.pt" \
    --output-dir "$OUT" \
    --cond-scope local --blocklist "$BLOCKLIST" \
    --k-amp 64 --k-sup 64 --sample-size 200 \
    --steps 48 --steer-lambda 1 \
    --conditions true \
    --decode thr0.02,det@cfg2,det@cfg3,thr0.02@cfg2,thr0.02@cfg3,bo4@temp0.7,bo4@temp0.7@cfg2 \
    --device cuda

echo "==================== S0 SWEEP DONE ===================="
sed -n '/## Decode quality/,$p' "$OUT/probe_report.md"
echo "Reference: pilot recal2 thr0.02 exact 0.1407 / sim 0.6335 (the"
echo "  hack to beat with principled tools). thr0.02 row here = control"
echo "  (same config, resume-invariant RNG). Winner advances to a"
echo "  three-condition confirmation + becomes the S-series decode stack."