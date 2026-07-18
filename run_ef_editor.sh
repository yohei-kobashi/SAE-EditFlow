#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# EF through-LM editor — one layer per job (EF_LM_LOSS_PLAN §2/§5).
# Batch:  cd ~/SAE-LEWIS && qsub -v LAYER=12 run_ef_editor.sh
#         (then LAYER=4 / LAYER=20 after the L12 gate)
# Stages inside one job (all resumable; resubmit past a walltime kill):
#   1. train to 10k steps -> probe100 (fail-fast: bare-frame liveness)
#   2. resume train to 40k -> probe500 with arms ef,steer,raw
# Needs run_ef_sidecar.sh outputs (corruption_z_l$LAYER dirs).

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

LAYER=${LAYER:-12}
case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy ;;
    12) SAE="layer_12/width_16k/average_l0_82/params.npz"
        BLK=runs/blocklist/blocklist.npy ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy ;;
    *)  echo "unsupported LAYER=$LAYER"; exit 1 ;;
esac

LLM2VEC=runs/mcgill_gemma_repro_3k/final
CACHE=runs/prod_gemma_v4/corruption_z_l$LAYER
DEV=runs/prod_gemma_v4/corruption_seldev_z_l$LAYER
OUT=runs/prod_gemma_v6/eflm_l$LAYER

[ -f "$CACHE/meta.json" ] || { echo "sidecar missing: $CACHE"; exit 1; }

TRAIN_ARGS=(--corruption-dir "$CACHE" --dev-corruption-dir "$DEV"
    --llm2vec-dir "$LLM2VEC" --output-dir "$OUT"
    --inject-layer "$LAYER" --sae-path "$SAE"
    --batch-size 4 --grad-accum-steps 2 --num-workers 2
    --k-top 32 --k-amp log:1-32 --k-sup log:1-32
    --empty-prob 0.08 --mismatch-null-prob 0.12 --t0-prob 0.5
    --norm-alpha 0.5 --norm-reg-w 0.05 --null-norm-w 0.1
    --dev-batches 48 --eval-steps 2000 --save-steps 2000
    --resume --device cuda)

PROBE_ARGS=(--llm2vec-dir "$LLM2VEC" --sae-path "$SAE"
    --sae-layer "$LAYER" --blocklist "$BLK"
    --k-amp 64 --k-sup 64 --steer-alpha 0.5 --device cuda)

# Stage 1 — fail-fast at 10k steps.
if [ ! -f "$OUT/eflm-final.pt" ] && [ ! -f "$OUT/eflm-step10000.pt" ]; then
    python train_ef_editor.py "${TRAIN_ARGS[@]}" --max-steps 10000
    # the 10k run writes final/last as best-dev copies; remove them so the
    # full run's guard and resume see an unfinished training
    rm -f "$OUT/eflm-final.pt" "$OUT/eflm-last.pt"
fi
if [ -f "$OUT/eflm-step10000.pt" ] && [ ! -f "$OUT/probe100/report.md" ] \
   && [ ! -f "$OUT/eflm-final.pt" ]; then
    python scripts/eval_ef_bare.py "${PROBE_ARGS[@]}" \
        --ef-ckpt "$OUT/eflm-best.pt" --output-dir "$OUT/probe100" \
        --sample-size 100 --arms ef,raw
    echo "-------- probe100 (fail-fast) above — training continues --------"
fi

# Stage 2 — full training.
if [ ! -f "$OUT/eflm-final.pt" ]; then
    python train_ef_editor.py "${TRAIN_ARGS[@]}" --max-steps 40000
fi

# Stage 3 — probe500 with all arms.
if [ -f "$OUT/eflm-final.pt" ] && [ ! -f "$OUT/probe500/report.md" ]; then
    python scripts/eval_ef_bare.py "${PROBE_ARGS[@]}" \
        --ef-ckpt "$OUT/eflm-final.pt" --output-dir "$OUT/probe500" \
        --sample-size 500 --arms ef,steer,raw
fi

if [ -f "$OUT/probe500/report.md" ]; then
    echo "==================== EF-LM L$LAYER DONE ===================="
    cat "$OUT/probe500/report.md"
    echo
    echo "Gate: ef(true) exact > steer(true) exact in the SAME bare frame;"
    echo "empty/random ~ raw (no spurious edits); lam-IoU true-specific."
fi
