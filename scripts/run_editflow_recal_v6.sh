#!/bin/bash
# SAE-EF re-probe, round 2. Round 1 (probe_recal) established: (i) tagger
# count-oracle 0.7472 ≈ EF λ-IoU 0.7337 — WHERE ranking is at PARITY, the
# old "0.30" bar was decision-calibration loss; (ii) the rate head is ~10x
# under target AND does not track w(t) growth (ratio 0.119@t=0.3 →
# 0.027@t=0.9, saturating ≈0.25); (iii) the thr decode numbers were VOID —
# a stall-exit bug ended every decode at t≈0.12, before the firing window.
# This rerun uses the fixed stall exit (late-half only) and a sweep placed
# around the measured saturation. ~30 min:  bash scripts/run_editflow_recal_v6.sh
set -eo pipefail
cd "$(dirname "$0")/.."

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}
OUT=$V6/editflow_pilot/probe_recal2

python scripts/editflow_probe.py \
    --llm2vec-dir "$LLM2VEC" \
    --editflow-ckpt "$V6/editflow_pilot/editflow-final.pt" \
    --output-dir "$OUT" \
    --cond-scope local --blocklist "$BLOCKLIST" \
    --k-amp 64 --k-sup 64 --sample-size 200 \
    --steps 48 --decode thr0.02,thr0.05,thr0.1 --steer-lambda 1 \
    --device cuda

echo "==================== EF RECAL-2 DONE ===================="
sed -n '/## Decode quality/,$p' "$OUT/probe_report.md"
echo "Reading:"
echo "  - true thr rows now reflect a decode that reaches the firing window."
echo "    If exact/sim recover -> calibration was the whole decode failure;"
echo "    if copy stays ~1.0 -> the saturated rates never clear the floor"
echo "    anywhere -> retrain with larger rate-head LR + stronger t signal."
echo "  - empty/random no_edit must stay 1.00 under the fixed stall exit."