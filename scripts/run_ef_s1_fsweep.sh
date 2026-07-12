#!/bin/bash
# S1 follow-up (EDIT_FLOWS_ZERO §5): decode-only F sweep BELOW 0.25 on the
# S1 hazard checkpoint. The S1 probe's frontier was monotone in F
# (thr0.25 0.1407 > thr0.5 0.0704 > thr0.75 0.0352 exact) with the best F
# at the sweep edge — champion judgment vs S0 (pilot thr0.02+greedyQ
# 0.1859/0.6622) needs the F<0.25 region. Three conditions ON PURPOSE:
# the pilot showed random-leak grows as F drops, so the low-F rows need
# their premise controls next to them. bo4 rides along because stoch
# firing (p = 1-exp(-h·λ)) is no longer vanishing on a hazard model —
# first meaningful test of best-of-K + SAE-gain selection (true-only).
# ~1.5h on a GPU session:  bash scripts/run_ef_s1_fsweep.sh
set -eo pipefail
cd "$(dirname "$0")/.."

V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}
OUT=$V6/editflow_s1/probe_fsweep

# Mid-run kills resume per pair (records.partial.jsonl); this guard makes
# a rerun AFTER completion a no-op instead of a redo.
if [ ! -f "$OUT/probe_report.md" ]; then
python scripts/editflow_probe.py \
    --llm2vec-dir "$LLM2VEC" \
    --editflow-ckpt "$V6/editflow_s1/editflow-final.pt" \
    --output-dir "$OUT" \
    --cond-scope local --blocklist "$BLOCKLIST" \
    --k-amp 64 --k-sup 64 --sample-size 200 \
    --steps 48 --steer-lambda 1 \
    --decode thr0.05,thr0.1,thr0.15,bo4@temp0.7,bo4@temp0.7@cfg2 \
    --device cuda
fi

echo "==================== S1 F-SWEEP DONE ===================="
sed -n '/## Decode quality/,$p' "$OUT/probe_report.md"
echo "Judgment: pick the F whose true exact/sim reaches the S0 champion"
echo "  (pilot thr0.02+greedyQ: exact 0.1859 / sim 0.6622) while keeping"
echo "  empty no_edit = 1.00 and random no_edit >= the pilot's 0.88."
echo "  S1 wins -> S2 builds on hazard; S1 only ties -> hazard still"
echo "  advances on lambda-IoU 0.7692 + structural premise safety, but"
echo "  record the decode tie honestly."
