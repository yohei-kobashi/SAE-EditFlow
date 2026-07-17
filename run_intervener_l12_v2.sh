#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Intervener L12 v2 — residual parameterization around the steer champion.
# v1 (runs/prod_gemma_v6/intervener_l12) collapsed to the copy attractor:
# probe500 exact 0.0200 (bar steer0.5 = 0.2385), true==random, copy 0.65,
# |delta|~12 << budget 45.6. Diagnosis: x1~x0 so the NLL is dominated by
# unchanged tokens; from identity-init the cheapest solution is "copy".
# v2 changes:
#   * --steer-alpha-base 0.5: intervention = 0.5*dvec at ALL positions
#     (the exact C1' champion rendering) + zero-init learned corrections
#     => training STARTS at 0.2385 and can only be shaped from there.
#   * --edit-weight 4.0: CE upweight on the response tokens that differ
#     from src (LCP/LCS trim) — the copy attractor countermeasure.
# Batch:      cd ~/SAE-LEWIS && qsub run_intervener_l12_v2.sh  (login node)
# Multi-session: checkpoints every 2000 steps, --resume; resubmit until
# "INTERVENER L12 V2 DONE". Probe500 runs automatically after training.

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
OUT=$V6/intervener_l12_v2

if [ ! -f "$OUT/intervener-final.pt" ]; then
    python train_intervener.py \
        --corruption-dir "$V4/corruption" \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --inject-layer 12 \
        --steer-alpha-base 0.5 --edit-weight 4.0 \
        --max-steps 40000 \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 \
        --norm-alpha 0.5 --norm-reg-w 0.05 --null-norm-w 0.1 \
        --dev-batches 48 --eval-steps 2000 --save-steps 2000 \
        --resume --device cuda
fi

if [ -f "$OUT/intervener-final.pt" ] && [ ! -f "$OUT/probe500/report.md" ]; then
    python scripts/eval_clamp_baseline.py \
        --llm2vec-dir "$LLM2VEC" --blocklist "$BLOCKLIST" \
        --output-dir "$OUT/probe500" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --intervention learned \
        --intervener-ckpt "$OUT/intervener-final.pt" \
        --device cuda
fi

if [ -f "$OUT/probe500/report.md" ]; then
    echo "==================== INTERVENER L12 V2 DONE ===================="
    cat "$OUT/probe500/report.md"
    echo
    echo "Bar: steer0.5 exact 0.2385@499 (same frame; v2 INIT = exactly"
    echo "this). Controls: empty ~ copy, random: copy is fine (mismatch"
    echo "null teaches cancellation of the base under wrong specs)."
fi
