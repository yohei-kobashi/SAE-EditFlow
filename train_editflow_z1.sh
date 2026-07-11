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

# SAE-EF Z1 (EDIT_FLOWS_ZERO.md): two arms, sequential, SAME warm start
# (v6 editor) and budget (30k), so the probe delta attributes cleanly:
#   Z1a  t-FiLM on the λ head + separate rate-head LR (3e-3) + Z2
#        true-alignment teacher — fixes the pilot's rate saturation at
#        the source (λ≈0.25 vs target w(t)→9, README §13.8).
#   Z1b  Z1a recipe + feature-token conditioning (one prefix token per
#        commanded feature: W_dec base + sign + magnitude) — the binding
#        fix (essential problem #1).
# Each arm: train ~2h + probe ~30min. Probes decode det,thr0.02,thr0.05:
# det tests whether SOURCE calibration now fires without the thr patch.
# Re-submit until "Z1 DONE". ~5h total.
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy

train_arm () {  # $1 = out dir, $2... = extra args
    local OUT=$1; shift
    if [ ! -f "$OUT/editflow-final.pt" ]; then
        python train_editflow.py \
            --corruption-dir "$V4/corruption" \
            --dev-corruption-dir "$V4/corruption_seldev" \
            --llm2vec-dir "$LLM2VEC" \
            --output-dir "$OUT" \
            --init-from-editor "$V6/editor/editor-final.pt" \
            --max-steps 30000 \
            --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
            --dev-batches 96 --eval-steps 2000 \
            --batch-size 8 --num-workers 2 \
            --t-film --true-align --rate-head-lr 3e-3 \
            --resume --device cuda "$@"
    fi
    if [ ! -f "$OUT/probe/probe_report.md" ]; then
        python scripts/editflow_probe.py \
            --llm2vec-dir "$LLM2VEC" \
            --editflow-ckpt "$OUT/editflow-final.pt" \
            --output-dir "$OUT/probe" \
            --cond-scope local --blocklist "$BLOCKLIST" \
            --k-amp 64 --k-sup 64 --sample-size 200 \
            --steps 48 --decode det,thr0.02,thr0.05 --steer-lambda 1 \
            --device cuda
    fi
}

train_arm "$V6/editflow_z1a"
train_arm "$V6/editflow_z1b" --cond-mode feature-tokens

echo "==================== Z1 DONE ===================="
for arm in z1a z1b; do
    echo "---- $arm ----"
    sed -n '/## Gate (a)/,/## Multi-site/p' \
        "$V6/editflow_$arm/probe/probe_report.md"
done
echo "Reference (pilot, recal2): λ-IoU 0.7337; thr0.02 exact 0.1407 /"
echo "  sim 0.6335; thr0.1 sim 0.6424; ratio 0.119→0.027 (saturated)."
echo "Z1a gate: rate-calibration ratios → ~1 and det exact >= recal2 thr."
echo "Z1b gate: λ-IoU >= 0.73 kept AND exact/sim > Z1a (esp. 2-3 / 4-8)."