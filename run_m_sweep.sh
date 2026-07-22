#!/bin/bash
# finer retrieval-m dev sweep (user 2026-07-22 (b)): m in {1,2,3,5,8,15}
# on the 100-pair pool-dev, BOTH directions (amp at scale 2.5, sup at 3.5).
# Existing cells (amp m1/m5/m15, sup m5) are reused via guards.
cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail
git pull || true
P=runs/prod_gemma_v6
FS=runs/feature_specs
SPLIT=runs/tables/eval_split.json
EF () {
    OUT=$1; shift
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_spec.json \
            --fspec-retrieve $FS/l12_retrieve.json \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt $P/eflm_l12_v5f_nobudget/eflm-final.pt \
            --arms ef --conditions true --pool-dev $SPLIT \
            --sample-size 100 --device cuda --output-dir $P/$OUT "$@"
    fi
}
for M in 1 2 3 5 8 15; do
    EF fs_adev_l12_m$M       --reverse-pairs --fspec-scale 2.5 --retrieve-m $M
    EF fs_adev_l12_m${M}_sup --fspec-scale 3.5 --retrieve-m $M
done
echo "== m-sweep summary (dev exact) =="
for M in 1 2 3 5 8 15; do
    A=$(grep -E '^\| true \| ef ' $P/fs_adev_l12_m$M/report.md | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
    S=$(grep -E '^\| true \| ef ' $P/fs_adev_l12_m${M}_sup/report.md | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
    echo "m=$M  enh(dev)=$A  abl(dev)=$S"
done
echo "==================== M-SWEEP-DONE ===================="
