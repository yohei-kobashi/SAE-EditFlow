#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Per-feature identification on the canonical pool (user 2026-07-22:
# eval 500 / identification 4,451, ALL arms aligned). One GPU pass emits,
# for L4/L12/L20 at once: ef feature-spec (pool-mean delta) + LinguaLens
# FRC top-r + AxBench AUROC top-r + split-half stability.
# Batch: cd ~/SAE-LEWIS && qsub -N bspecs run_build_specs.sh
# Resumable (per-layer pair sidecars) — resubmit past a walltime kill.

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

python scripts/build_feature_specs.py \
    --out-dir runs/feature_specs \
    --split runs/tables/eval_split.json \
    --device cuda

echo "==================== BUILD-SPECS-JOB-DONE ===================="
