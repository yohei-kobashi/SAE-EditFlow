#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -W group_list=go25
#PBS -j oe

# v6 SCRATCH training (user 2026-07-22): champion nb recipe + T1
# pseudo-feature group-mean augmentation (AGG_AUG=0.35, related-sample
# mix) + T3 insertion-loss boost (1.5). L12 from scratch, 40k steps.
# Then feature-spec eval (ctx spec, both directions) for the verdict.
# Batch: qsub -N efv6 run_v6.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

CACHE=runs/prod_gemma_v4/corruption_z_l12
if [ ! -f $CACHE/group_means.json ]; then
    python scripts/build_group_means.py \
        --cache $CACHE --out $CACHE/group_means.json
fi

LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
    FLOW_INIT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
    NORM_REG_W=0.0 NULL_NORM_W=0.0 \
    AGG_AUG=0.35 AGG_CLUSTER=$CACHE/group_means.json INS_BOOST=1.5 \
    OUT_SUFFIX=_v6 MAX_STEPS=40000 bash run_ef_editor.sh

# feature-spec verdict evals (ctx spec = current best construction)
P=runs/prod_gemma_v6
FS=runs/feature_specs
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then EXTRA=--reverse-pairs; else EXTRA=""; fi
    if [ ! -f $P/fs_v6_l12$DIRX/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_specctx.json \
            --fspec-scale 3.5 --conditions true,random --arms ef \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_v6/eflm-final.pt \
            --sample-size 500 --device cuda \
            --output-dir $P/fs_v6_l12$DIRX $EXTRA
    fi
done

echo "==================== V6-DONE ===================="
