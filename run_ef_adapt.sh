#!/bin/bash
# Improvement ③ (user-approved): aggregated-spec adaptation fine-tune.
# Seeds the nb champion (L12 40k) and continues to 48k with the dilution
# augmentation (AGG_AUG=0.35: own*w + others-mean*(1-w), w~U(0.3,0.7)),
# then evaluates with the feature-level spec (sup + amp, dev scale).
# Run inside interact-g (chain marker EF-ADAPT-DONE).

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
SRC=$P/eflm_l12_v5f_nobudget
OUT=$P/eflm_l12_nb_aggaug
S3CKPT=runs/prod_gemma_v6/editflow_s3/editflow-final.pt

mkdir -p "$OUT"
cp -n $SRC/eflm-step40000.pt $SRC/eflm-step40000.state.pt \
      $SRC/best.json "$OUT/" 2>/dev/null || true

if [ ! -f $OUT/probe500/report.md ]; then
    LAYER=12 FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
        FLOW_INIT=$S3CKPT NORM_REG_W=0.0 NULL_NORM_W=0.0 \
        AGG_AUG=0.35 OUT_SUFFIX=_nb_aggaug MAX_STEPS=48000 \
        bash run_ef_editor.sh
fi

# feature-spec eval (the judgment that matters)
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SAE=layer_12/width_16k/average_l0_82/params.npz
BLK=runs/blocklist/blocklist.npy
SC=$(cat $P/fs_scale_l12.txt)
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then EXTRA=--reverse-pairs; else EXTRA=""; fi
    if [ ! -f $P/fs_aggaug_l12$DIRX/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_spec.json \
            --fspec-scale $SC --conditions true,random --arms ef \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 \
            --ef-ckpt "$OUT/eflm-final.pt" --sample-size 500 --device cuda \
            --output-dir $P/fs_aggaug_l12$DIRX $EXTRA
    fi
done

echo "==================== EF-ADAPT-DONE ===================="
