#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

# S1 (EDIT_FLOWS_ZERO §5, revised process): hazard-factorized rate head —
# λ = w(t)·sigmoid(head), the analytic hazard × a learned P(pending).
# Everything else matches the PILOT recipe (pooled cond, difflib align,
# no FiLM) and warm-starts FROM the pilot checkpoint, so the delta vs
# pilot recal2 attributes to the parameterization alone. 30k is fair
# here because the head has LESS to learn (only P, not w(t)·P).
# Probe: thr{F} now reads as p ≥ F (calibrated probability) — no hack;
# det's expected-count also becomes meaningful. ~2.5h; resubmit until
# "S1 DONE".
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
OUT=$V6/editflow_s1

if [ ! -f "$OUT/editflow-final.pt" ]; then
    python train_editflow.py \
        --corruption-dir "$V4/corruption" \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --init-from-editflow "$V6/editflow_pilot/editflow-final.pt" \
        --rate-param hazard \
        --max-steps 30000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --dev-batches 96 --eval-steps 2000 \
        --batch-size 8 --num-workers 2 \
        --resume --device cuda
fi

if [ ! -f "$OUT/probe/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$OUT/editflow-final.pt" \
        --output-dir "$OUT/probe" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 200 \
        --steps 48 --decode det,thr0.25,thr0.5,thr0.75 --steer-lambda 1 \
        --device cuda
fi

echo "==================== S1 DONE ===================="
sed -n '/## Gate (a)/,$p' "$OUT/probe/probe_report.md" | head -60
echo "Gates: (i) on-dist ratio column now IS mean p — check it is"
echo "  informative (well above empty) and t-stable; (ii) thr0.5 (= p>=0.5)"
echo "  exact/sim >= pilot recal2 thr0.02 (0.1407/0.6335) WITHOUT the"
echo "  magnitude hack; (iii) empty/random no_edit = 1.00."