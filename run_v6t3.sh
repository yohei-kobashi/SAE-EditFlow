#!/bin/bash -l

# v6 ablation study, arm T3-ONLY (user 2026-07-23): champion nb recipe +
# T3 insertion-loss boost 1.5, WITHOUT the T1 group-mean augmentation.
# Isolates whether T3 alone reproduces the v6 enhancement gain and/or
# is the culprit of the v6 ablation regression.
# Run inside interact-g (2h sessions; run_ef_editor.sh --resume makes the
# chain restartable): bash run_v6t3.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
    FLOW_INIT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
    NORM_REG_W=0.0 NULL_NORM_W=0.0 \
    INS_BOOST=1.5 \
    OUT_SUFFIX=_v6t3 MAX_STEPS=40000 bash run_ef_editor.sh

# feature-spec verdict evals (ctx spec = current best construction)
P=runs/prod_gemma_v6
FS=runs/feature_specs
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then EXTRA=--reverse-pairs; else EXTRA=""; fi
    if [ ! -f $P/fs_v6t3_l12$DIRX/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_specctx.json \
            --fspec-scale 3.5 --conditions true,random --arms ef \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_v6t3/eflm-final.pt \
            --sample-size 500 --device cuda \
            --output-dir $P/fs_v6t3_l12$DIRX $EXTRA
    fi
done

echo "==================== V6T3-DONE ===================="
