#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -W group_list=go25
#PBS -j oe

# KL/NLL counterfactual consistency for the ef arm at one layer.
# qsub -N klc4 -v LAYER=4 run_klc_ef.sh   (same invocation as klc_ef_l12,
# see reports/04 §3; guarded by report.md so resubmission is a no-op).

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

LAYER=${LAYER:-4}
case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy ;;
    12) SAE="layer_12/width_16k/average_l0_82/params.npz"
        BLK=runs/blocklist/blocklist.npy ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy ;;
    *)  echo "unknown LAYER=$LAYER"; exit 1 ;;
esac
CKPT=${CKPT:-runs/prod_gemma_v6/eflm_l${LAYER}_v5f2/eflm-final.pt}
OUT=runs/prod_gemma_v6/klc_ef_l$LAYER

if [ ! -f "$OUT/report.md" ]; then
    python scripts/eval_kl_consistency.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --sae-path "$SAE" --sae-layer "$LAYER" --blocklist "$BLK" \
        --sample-size 500 --arms ef --ef-ckpt "$CKPT" \
        --output-dir "$OUT" --device cuda
fi
echo "KLC-EF L$LAYER DONE"
