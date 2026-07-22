#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Improvement ② (user-approved): spec narrowing x consistency filter, L12.
#  A. build 8 variants (k8/16/32/64 x nofilter/c70)
#  B. pool-dev probe (100 pairs, sup) each variant -> pick dev-best
#  C. eval-500 sup for the 4 nofilter k's (the ToDo① narrowing curve,
#     ALL reported) + the dev-best variant if filtered
#  D. eval-500 amp for the dev-best variant
# Batch: qsub -N specvar run_spec_variants.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SAE=layer_12/width_16k/average_l0_82/params.npz
BLK=runs/blocklist/blocklist.npy
CKPT=$P/eflm_l12_v5f_nobudget/eflm-final.pt
SPLIT=runs/tables/eval_split.json
SC=$(cat $P/fs_scale_l12.txt)

if [ ! -f $FS/l12_spec_k8_c70.json ]; then
    python scripts/build_spec_variants.py \
        --pairs $FS/l12_pairs.jsonl --split $SPLIT \
        --base $FS/l12_spec.json --out-prefix $FS/l12_spec
fi

EF () {  # EF <outdir> <specjson> <extra...>
    OUT=$1; SPEC=$2; shift 2
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $SPEC --fspec-scale $SC \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 \
            --ef-ckpt "$CKPT" --arms ef --device cuda \
            --output-dir $P/$OUT "$@"
    fi
}
exact_of () {
    grep -E "^\| true \| ef " "$1" | head -1 | awk -F'|' '{print $4}' | tr -d ' '
}

# ---- B. dev probes ------------------------------------------------------
for V in k8 k16 k32 k64 k8_c70 k16_c70 k32_c70 k64_c70; do
    EF fs_kdev_l12_$V $FS/l12_spec_$V.json \
       --pool-dev $SPLIT --sample-size 100 --conditions true
done
if [ ! -f $P/fs_kbest_l12.txt ]; then
    BEST=$(for V in k8 k16 k32 k64 k8_c70 k16_c70 k32_c70 k64_c70; do
        printf "%s %s\n" "$V" "$(exact_of $P/fs_kdev_l12_$V/report.md)"
    done | sort -k2 -gr | head -1 | cut -d' ' -f1)
    echo "$BEST" > $P/fs_kbest_l12.txt
fi
BEST=$(cat $P/fs_kbest_l12.txt)
echo "[specvar] dev-best variant: $BEST"

# ---- C. eval narrowing curve (sup) --------------------------------------
for K in 8 16 32 64; do
    EF fs_k${K}_l12 $FS/l12_spec_k$K.json \
       --sample-size 500 --conditions true,random
done
case "$BEST" in *_c70)
    EF fs_kbestv_l12 $FS/l12_spec_$BEST.json \
       --sample-size 500 --conditions true,random ;;
esac

# ---- D. dev-best amp ----------------------------------------------------
EF fs_kbestv_l12_amp $FS/l12_spec_$BEST.json \
   --sample-size 500 --conditions true,random --reverse-pairs

echo "==================== SPEC-VARIANTS-DONE ===================="
