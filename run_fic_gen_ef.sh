#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# FIC generation, ef arm only, BARE frame (user 2026-07-21: ②missing
# inference on short-g). Appends into the same fic_l12 records.jsonl —
# keys are arm-scoped so this never collides with the steer/clamp/
# prompting rows already generated. ef ckpt = the bare-frame champion v4
# (frame-matched; the repeat-frame FIC side reuses v5f probe records).
# Resume-safe; rerun me after a walltime kill.

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

python scripts/eval_fic_gen.py \
    --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
    --sae-path layer_12/width_16k/average_l0_82/params.npz \
    --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
    --arms ef --ef-ckpt runs/prod_gemma_v6/eflm_l12_v4/eflm-final.pt \
    --output-dir runs/prod_gemma_v6/fic_l12 --device cuda

echo "FIC-GEN-EF DONE"
