#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Appendix reference: ef feature-spec at FIXED scale 1.0 (no hyperparameter
# selection), user decision 2026-07-22 — quantifies the calibration's
# contribution transparently. L12 sup already measured (fspec_probe_l12,
# 0.0822); remaining cells: L4 sup/amp, L20 sup/amp, L12 amp.
# Batch: qsub -N fss1 run_fspec_scale1.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final

run_cell () {  # run_cell <layer> <outdir> <extra...>
    LAYER=$1; OUT=$2; shift 2
    case "$LAYER" in
      4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
          BLK=runs/blocklist_l4/blocklist.npy
          CKPT=$P/eflm_l4_v5f2_nobudget_80k/eflm-final.pt ;;
      12) SAE="layer_12/width_16k/average_l0_82/params.npz"
          BLK=runs/blocklist/blocklist.npy
          CKPT=$P/eflm_l12_v5f_nobudget/eflm-final.pt ;;
      20) SAE="layer_20/width_16k/average_l0_71/params.npz"
          BLK=runs/blocklist_l20/blocklist.npy
          CKPT=$P/eflm_l20_v5f2_80k/eflm-final.pt ;;
    esac
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l${LAYER}_spec.json \
            --fspec-scale 1.0 --conditions true,random \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer $LAYER \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 \
            --ef-ckpt "$CKPT" --arms ef --sample-size 500 --device cuda \
            --output-dir $P/$OUT "$@"
    fi
}

run_cell 4  fs_s1_l4
run_cell 4  fs_s1_l4_amp  --reverse-pairs
run_cell 20 fs_s1_l20
run_cell 20 fs_s1_l20_amp --reverse-pairs
run_cell 12 fs_s1_l12_amp --reverse-pairs

echo "==================== FS-SCALE1-DONE ===================="
