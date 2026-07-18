#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# EF_LM_LOSS_PLAN §3 — data prep for L4/L12/L20 training:
#   1. per-layer grammaticality blocklists (L4, L20; L12 exists)
#   2. multi-layer z sidecar caches (ONE forward -> all 3 layers) for
#      the training cache subset + the full seldev cache.
# Resumable: blocklists are guarded by their output files; the sidecar
# skips shards already written in every layer dir.
# Batch: cd ~/SAE-LEWIS && qsub run_ef_sidecar.sh   (login node)

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

SAE_L4="layer_4/width_16k/average_l0_60/params.npz"
SAE_L12="layer_12/width_16k/average_l0_82/params.npz"
SAE_L20="layer_20/width_16k/average_l0_71/params.npz"

if [ ! -f runs/blocklist_l4/blocklist.npy ]; then
    python scripts/build_grammaticality_blocklist.py \
        --output-dir runs/blocklist_l4 \
        --sae-path "$SAE_L4" --sae-layer 4 \
        --pairs-per-paradigm 200 --device cuda
fi
if [ ! -f runs/blocklist_l20/blocklist.npy ]; then
    python scripts/build_grammaticality_blocklist.py \
        --output-dir runs/blocklist_l20 \
        --sae-path "$SAE_L20" --sae-layer 20 \
        --pairs-per-paradigm 200 --device cuda
fi

BLOCKS="runs/blocklist_l4/blocklist.npy,runs/blocklist/blocklist.npy,runs/blocklist_l20/blocklist.npy"
PATHS="$SAE_L4,$SAE_L12,$SAE_L20"

python scripts/make_z_sidecar.py \
    --cache-dir runs/prod_gemma_v4/corruption \
    --out-root runs/prod_gemma_v4/corruption_z \
    --layers 4,12,20 --sae-paths "$PATHS" --blocklists "$BLOCKS" \
    --max-records 300000 --device cuda

python scripts/make_z_sidecar.py \
    --cache-dir runs/prod_gemma_v4/corruption_seldev \
    --out-root runs/prod_gemma_v4/corruption_seldev_z \
    --layers 4,12,20 --sae-paths "$PATHS" --blocklists "$BLOCKS" \
    --max-records 0 --device cuda

echo "==================== EF SIDECAR DONE ===================="
echo "next: qsub -v LAYER=12 run_ef_editor.sh  (then 4 / 20 after gate)"
