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

# v5 run: v4d transform families + composition + k-draw 1-8.
# The +200k top-up is generated in a FRESH dir (corruption_v5topup) — the
# v4 cache's meta.json makes corruption_parallel.sh treat the old dir as
# complete, and merging new worker shards into it would collide with the
# existing shard-w{i}-NNNNN names. SKIP_SENTENCES continues the Dolma
# stream past the v4 high-water mark, so no sentence is corrupted twice.
# Models/evals go under prod_gemma_v5 (corruption is symlinked in).
V4=./runs/prod_gemma_v4
V5=./runs/prod_gemma_v5
TOPUP=$V4/corruption_v5topup
LLM2VEC=runs/mcgill_gemma_repro_3k/final

# 0. Generate +200k with the v4d code (new families, compose, t_family),
#    move the FIRST new shard per worker into the dev splits, merge the
#    rest into the training cache under the shard-v5-* prefix.
#    v5_split.done is the completion marker for the whole phase.
if [ ! -f "$V4/v5_split.done" ]; then
    rm -f "$V4/v5_pre_topup.list"      # marker from an older script revision
    SEEN=$(python -c "import json;print(json.load(open('$V4/corruption/meta.json'))['sentences_seen'])")
    mkdir -p "$TOPUP"
    cp -n "$V4/corruption/unigram.json" "$TOPUP/" 2>/dev/null || true

    WORKERS=6 \
    OUT_DIR=$TOPUP \
    LLM2VEC_DIR=$LLM2VEC \
    CORRUPTION_SAMPLES=200000 CORRUPTION_SHARD=2000 \
    BLOCKLIST=runs/blocklist/blocklist.npy \
    TRANSFORM_COMPOSE_PROB=0.15 \
    SKIP_SENTENCES=$SEEN \
    bash scripts/corruption_parallel.sh

    # Dev-split extension: first new shard per worker. The v5 prefix
    # avoids name collisions with the existing dev shards AND sorts
    # before shard-w* ('v' < 'w'), so the dev monitor's first
    # --dev-batches cover the new families.
    for i in 0 1 2 3 4 5; do
        new=$(ls "$TOPUP"/shard-w$i-*.jsonl.gz 2>/dev/null | sort | head -1)
        if [ -z "$new" ]; then
            echo "[v5] no shard for worker $i in $TOPUP — top-up incomplete?" >&2
            exit 1
        fi
        base=$(basename "$new")                 # shard-wI-NNNNN.jsonl.gz
        if [ "$i" = 0 ]; then
            mv "$new" "$V4/corruption_seldev/shard-v5-${base#shard-}"
        else
            mv "$new" "$V4/corruption_dev/shard-v5-${base#shard-}"
        fi
    done
    # Remaining top-up shards join the training cache.
    for f in "$TOPUP"/shard-w*.jsonl.gz; do
        [ -e "$f" ] || continue
        mv "$f" "$V4/corruption/shard-v5-$(basename "${f#*shard-}")"
    done
    cp "$TOPUP/meta.json" "$V4/corruption/meta_v5topup.json"
    touch "$V4/v5_split.done"
fi

# 1. v5 training (fresh tagger/editor; corruption + SAE cache reused).
#    DEV_BATCHES=384 spans the new-family seldev shard (2000 recs) plus
#    ~1k of the v4 shard, so best-checkpoint selection sees both.
mkdir -p "$V5"
[ -e "$V5/corruption" ] || ln -s ../prod_gemma_v4/corruption "$V5/corruption"

RUN_DIR=$V5 \
EDITOR_STEPS=100000 TAGGER_STEPS=30000 \
DEV_CORRUPTION_DIR=$V4/corruption_seldev \
DEV_BATCHES=384 \
K_DRAW=1-8 \
LLM2VEC_DIR=$LLM2VEC \
SIMCSE_DIR=$LLM2VEC \
bash scripts/run_production.sh

# 2. Evaluation at the swept operating point (k_top=8, k=8, ins 0.9;
#    confirmed on the v4 reporting dev — sweep_report.md).
python eval_tagger_editor.py \
    --corruption-dir "$V4/corruption_dev" \
    --llm2vec-dir "$LLM2VEC" \
    --tagger-ckpt "$V5/tagger/tagger-final.pt" \
    --editor-ckpt "$V5/editor/editor-final.pt" \
    --output-dir "$V5/eval_dev_k8" \
    --k-top 8 --k-amp 8 --k-sup 8 --ins-threshold 0.9 \
    --max-samples 4000 \
    --device cuda

python scripts/measure_editor_ceiling.py \
    --corruption-dir "$V4/corruption_dev" \
    --llm2vec-dir "$LLM2VEC" \
    --editor-ckpt "$V5/editor/editor-final.pt" \
    --output-dir "$V5/ceiling" \
    --max-samples 4000 \
    --device cuda

python eval_lingualens.py \
    --llm2vec-dir "$LLM2VEC" \
    --tagger-ckpt "$V5/tagger/tagger-final.pt" \
    --editor-ckpt "$V5/editor/editor-final.pt" \
    --output-dir "$V5/eval_lingualens" \
    --sae-path layer_12/width_16k/average_l0_82/params.npz \
    --k-amp 8 --k-sup 8 --ins-threshold 0.9 \
    --sample-size 100 --refine-passes 3 --refine-recompute \
    --fluency-gate 0.5 \
    --dump-details \
    --device cuda
