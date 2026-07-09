#!/bin/bash
# End-to-end LinguaLens with the §13.6 fixes: edit-local conditioning
# extraction + blocklist + lens fill bias, at λ=1 and λ=2.
# Reference models: tagger_100k + v5 editor. ~15 min per run.
set -eo pipefail
cd "$(dirname "$0")/.."

V5=./runs/prod_gemma_v5
LLM2VEC=runs/mcgill_gemma_repro_3k/final

for LAM in 1 2; do
    python eval_lingualens.py \
        --llm2vec-dir "$LLM2VEC" \
        --tagger-ckpt "$V5/tagger_100k/tagger-final.pt" \
        --editor-ckpt "$V5/editor/editor-final.pt" \
        --output-dir "$V5/eval_lingualens_local_lens$LAM" \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --cond-scope local \
        --blocklist runs/blocklist/blocklist.npy \
        --steer-lambda "$LAM" \
        --k-amp 64 --k-sup 64 --ins-threshold 0.9 \
        --sample-size 100 --refine-passes 3 --refine-recompute \
        --fluency-gate 0.5 --dump-details --device cuda
done

echo "==================== E2E VERDICT ===================="
for LAM in 1 2; do
    echo "---- lambda=$LAM ----"
    sed -n '/| condition |/,/^$/p' "$V5/eval_lingualens_local_lens$LAM/report.md"
done
