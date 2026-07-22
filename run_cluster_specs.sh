#!/bin/bash
# ① cluster-expanded specs: table -> dev share sweep (both dirs, on the
# base mean spec at current scales) -> eval500 both dirs.
# interact-g chain, marker CLX-EVAL-DONE. All guarded.
cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail
git pull || true
P=runs/prod_gemma_v6
FS=runs/feature_specs
SPLIT=runs/tables/eval_split.json

if [ ! -f $FS/l12_clusters.json ]; then
    python scripts/build_cluster_table.py \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --out $FS/l12_clusters.json --device cuda
fi

EF () {
    OUT=$1; shift
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_spec.json \
            --cluster-expand $FS/l12_clusters.json \
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
for SH in 0.3 0.7; do
    EF fs_xdev_l12_sh${SH/./}     --pool-dev $SPLIT --sample-size 100 \
       --conditions true --fspec-scale 3.5 --cluster-share $SH
    EF fs_xdev_l12_sh${SH/./}_amp --pool-dev $SPLIT --sample-size 100 \
       --conditions true --fspec-scale 2.5 --cluster-share $SH --reverse-pairs
done
for D in "" "_amp"; do
    F=$P/fs_clx_share_l12$D.txt
    if [ ! -f $F ]; then
        BEST=$(for SH in 0.3 0.7; do
            printf "%s %s\n" "$SH" \
              "$(exact_of $P/fs_xdev_l12_sh${SH/./}$D/report.md)"
        done | sort -k2 -gr | head -1 | cut -d' ' -f1)
        echo "$BEST" > $F
    fi
done
SHS=$(cat $P/fs_clx_share_l12.txt)
SHA=$(cat $P/fs_clx_share_l12_amp.txt)
echo "[clx] shares: sup=$SHS amp=$SHA"
EF fs_clx_l12     --sample-size 500 --conditions true,random \
   --fspec-scale 3.5 --cluster-share $SHS
EF fs_clx_l12_amp --sample-size 500 --conditions true,random \
   --fspec-scale 2.5 --cluster-share $SHA --reverse-pairs
echo "==================== CLX-EVAL-DONE ===================="
