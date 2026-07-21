#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Feature-spec rollout, one layer per job (user 2026-07-22: 3層×両方向
# exact + FIC生成に進む). Stages (all guarded/resumable):
#   A. per-layer fspec-scale selection on a 100-pair POOL-dev sample
#      (scales 1.5/2.5/3.5 — hyperparameter never touches the eval 500)
#   B. exact sup:  ef+steer+raw, feature-spec at the selected scale
#   C. exact amp:  same with --reverse-pairs
#   D. LinguaLens complete (FRC-r3 clamp) sup + amp
#   E. AxBench complete (AUROC-r1 steer)  sup + amp
# FIC(repeat) judging happens later on prepost from the B/C/D records.
# Batch: qsub -N fsrolN -v LAYER=N run_fspec_rollout.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

LAYER=${LAYER:-12}
P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SPLIT=runs/tables/eval_split.json

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

EF () {  # EF <outdir> <extra...>
    OUT=$1; shift
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l${LAYER}_spec.json \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer $LAYER \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 --steer-alpha 0.5 \
            --ef-ckpt "$CKPT" --device cuda \
            --output-dir $P/$OUT "$@"
    fi
}
CB () {  # CB <outdir> <intervention> <sets> <values> <extra...>
    OUT=$1; IV=$2; SETS=$3; VALS=$4; shift 4
    if [ ! -f $P/$OUT/report.md ]; then
        python scripts/eval_clamp_baseline.py \
            --intervention $IV --feature-sets $SETS --clamp-values $VALS \
            --conditions true,empty,random \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer $LAYER \
            --blocklist "$BLK" --sample-size 500 --device cuda \
            --output-dir $P/$OUT "$@"
    fi
}
exact_of () {
    grep -E "^\| true \| ef " "$1" | head -1 | awk -F'|' '{print $4}' | tr -d ' '
}

# ---- A. scale selection on pool-dev (never the eval 500) ---------------
SCALE_F=$P/fs_scale_l$LAYER.txt
if [ ! -f $SCALE_F ]; then
    for S in 1.5 2.5 3.5; do
        EF fs_dev_l${LAYER}_s${S/./} --pool-dev $SPLIT --sample-size 100 \
           --arms ef --conditions true --fspec-scale $S
    done
    BEST=$(for S in 1.5 2.5 3.5; do
        printf "%s %s\n" "$S" \
          "$(exact_of $P/fs_dev_l${LAYER}_s${S/./}/report.md)"
    done | sort -k2 -gr | head -1 | cut -d' ' -f1)
    echo "$BEST" > $SCALE_F
fi
SC=$(cat $SCALE_F)
echo "[rollout] L$LAYER selected fspec-scale=$SC (pool-dev)"

# ---- B/C. exact both directions (ef + steer + raw) ---------------------
EF fs_probe_l$LAYER      --arms ef,steer,raw --sample-size 500 --fspec-scale $SC
EF fs_probe_l${LAYER}_amp --arms ef,steer,raw --sample-size 500 --fspec-scale $SC --reverse-pairs

# ---- D. LinguaLens complete: FRC-r3 clamp ------------------------------
CB fs_clamp_l$LAYER       clamp $FS/l${LAYER}_frc_r3.json  5,10,20
CB fs_clamp_l${LAYER}_amp clamp $FS/l${LAYER}_frc_r3.json  5,10,20 --reverse-pairs

# ---- E. AxBench complete: AUROC-r1 steer -------------------------------
CB fs_axb_l$LAYER       steer $FS/l${LAYER}_auroc_r1.json 0.5,1
CB fs_axb_l${LAYER}_amp steer $FS/l${LAYER}_auroc_r1.json 0.5,1 --reverse-pairs

echo "==================== FS-ROLLOUT-L$LAYER-DONE ===================="
