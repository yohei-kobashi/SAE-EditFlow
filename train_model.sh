#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -W group_list=gj26
#PBS -j oe

source start_gpu_nodes.sh
cd ~/SAE-LEWIS

WORKERS=6 \
OUT_DIR=./runs/prod_gemma_v4/corruption \
LLM2VEC_DIR=runs/mcgill_gemma_repro_3k/final \
CORRUPTION_SAMPLES=500000 CORRUPTION_SHARD=2000 \
BLOCKLIST=runs/blocklist/blocklist.npy \
bash scripts/corruption_parallel.sh

# 1. 選択用 / 報告用 dev の分離
mkdir -p runs/prod_gemma_v4/corruption_seldev runs/prod_gemma_v4/corruption_dev
mv runs/prod_gemma_v4/corruption/shard-w0-00001.jsonl.gz runs/prod_gemma_v4/corruption_seldev/
for i in 1 2 3 4 5; do mv runs/prod_gemma_v4/corruption/shard-w$i-00001.jsonl.gz runs/prod_gemma_v4/corruption_dev/; done
cp runs/prod_gemma_v4/corruption/meta.json runs/prod_gemma_v4/corruption_seldev/
cp runs/prod_gemma_v4/corruption/meta.json runs/prod_gemma_v4/corruption_dev/

# 2. dev 監視付き学習(editor-final = best dev 状態)
RUN_DIR=./runs/prod_gemma_v4 \
EDITOR_STEPS=100000 TAGGER_STEPS=30000 \
DEV_CORRUPTION_DIR=./runs/prod_gemma_v4/corruption_seldev \
LLM2VEC_DIR=runs/mcgill_gemma_repro_3k/final \
SIMCSE_DIR=runs/mcgill_gemma_repro_3k/final \
bash scripts/run_production.sh

# 3. 評価3本
python eval_tagger_editor.py --corruption-dir runs/prod_gemma_v4/corruption_dev ... --output-dir runs/prod_gemma_v4/eval_dev
python scripts/measure_editor_ceiling.py --corruption-dir runs/prod_gemma_v4/corruption_dev ... --output-dir runs/prod_gemma_v4/ceiling
python eval_lingualens.py ... --output-dir runs/prod_gemma_v4/eval_lingualens --sample-size 100
