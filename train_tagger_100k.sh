#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

# B-1: tagger-only retrain at 100k steps (v5 used 30k). Every other
# hyperparameter matches the v5 production run exactly, so the delta vs
# runs/prod_gemma_v5/tagger isolates training-steps. Motivation (§13.6):
# the v5 tagger merely REDISTRIBUTED its 30k-step budget over the
# broadened v4d distribution (new families up, TENSE/NUMBER/ASPECT
# down, aggregate flat at 0.683), while the 100k-step editor absorbed
# both. The editor is reused as-is.
V4=./runs/prod_gemma_v4
V5=./runs/prod_gemma_v5
LLM2VEC=runs/mcgill_gemma_repro_3k/final

python train_tagger.py \
    --corruption-dir "$V5/corruption" \
    --llm2vec-dir "$LLM2VEC" \
    --output-dir "$V5/tagger_100k" \
    --init-proj-a-from "$V5/editor/editor-final.pt" \
    --max-steps 100000 \
    --warmup-steps 500 \
    --proj-a-freeze-steps 500 \
    --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
    --batch-size 8 \
    --num-workers 4 \
    --save-steps 2000 \
    --logging-steps 50 \
    --estimate-class-weights-batches 200 \
    --dev-corruption-dir "$V4/corruption_seldev" \
    --eval-steps 2000 --dev-batches 384 \
    --device cuda \
    --seed 42

# Evaluate at the v5 operating point (64/64, ins 0.9) with the
# per-family breakdown, next to the 30k baseline in eval_dev_k64_fam.
python eval_tagger_editor.py \
    --corruption-dir "$V4/corruption_dev" \
    --llm2vec-dir "$LLM2VEC" \
    --tagger-ckpt "$V5/tagger_100k/tagger-final.pt" \
    --output-dir "$V5/eval_dev_k64_tagger100k" \
    --k-top 64 --k-amp 64 --k-sup 64 --ins-threshold 0.9 \
    --per-family --max-samples 4000 \
    --device cuda
