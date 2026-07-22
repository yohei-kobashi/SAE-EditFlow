#!/bin/bash
# ⑦ in-context specs: build -> dev scale sweep (both dirs) -> eval500.
# interact-g chain, marker CTX-EVAL-DONE. All guarded.
cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail
git pull || true
P=runs/prod_gemma_v6
FS=runs/feature_specs
SPLIT=runs/tables/eval_split.json

if [ ! -f $FS/l12_specctx.json ]; then
    python scripts/build_feature_specs_ctx.py \
        --out-dir $FS --split $SPLIT --layers 12 --device cuda
fi

EF () {
    OUT=$1; shift
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_specctx.json \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_v5f_nobudget/eflm-final.pt \
            --arms ef --device cuda --output-dir $P/$OUT "$@"
    fi
}
exact_of () {
    grep -E "^\| true \| ef " "$1" | head -1 | awk -F'|' '{print $4}' | tr -d ' '
}
# dev scale sweeps
for S in 1.5 2.5 3.5 5.0; do
    EF fs_cdev_l12_s${S/./}     --pool-dev $SPLIT --sample-size 100 \
       --conditions true --fspec-scale $S
    EF fs_cdev_l12_s${S/./}_amp --pool-dev $SPLIT --sample-size 100 \
       --conditions true --fspec-scale $S --reverse-pairs
done
for D in "" "_amp"; do
    F=$P/fs_ctx_scale_l12$D.txt
    if [ ! -f $F ]; then
        BEST=$(for S in 1.5 2.5 3.5 5.0; do
            printf "%s %s\n" "$S" \
              "$(exact_of $P/fs_cdev_l12_s${S/./}$D/report.md)"
        done | sort -k2 -gr | head -1 | cut -d' ' -f1)
        echo "$BEST" > $F
    fi
done
SCS=$(cat $P/fs_ctx_scale_l12.txt)
SCA=$(cat $P/fs_ctx_scale_l12_amp.txt)
echo "[ctx] scales: sup=$SCS amp=$SCA"
EF fs_ctx_l12     --sample-size 500 --conditions true,random --fspec-scale $SCS
EF fs_ctx_l12_amp --sample-size 500 --conditions true,random --fspec-scale $SCA --reverse-pairs
echo "==================== CTX-EVAL-DONE ===================="
