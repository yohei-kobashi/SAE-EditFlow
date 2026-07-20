#!/bin/bash
# FIC judge (gpt-4o) over both frames — CPU, runs inside a prepost
# interactive session (user 2026-07-21: ①reuse exact-frame generations
# and judge on prepost; the driver relaunches me until FIC-JUDGE-DONE).
# Resume-safe: the judge cache makes reruns cheap; bare ef rows are
# picked up automatically once the short-g generation lands.

cd ~/SAE-LEWIS
source env-c/bin/activate

set -eo pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

python scripts/eval_fic_judge.py \
    --bare-dir runs/prod_gemma_v6/fic_l12 \
    --repeat-probe500 runs/prod_gemma_v6/eflm_l12_v5f/probe500/records.jsonl \
    --repeat-clamp runs/prod_gemma_v6/clamp_baseline500/records.jsonl \
    --repeat-a3 runs/prod_gemma_v6/a3prime_edit/records.jsonl \
    --dir-map runs/tables/lingualens_dirmap_en.json \
    --output-dir runs/prod_gemma_v6/fic_judge_l12
