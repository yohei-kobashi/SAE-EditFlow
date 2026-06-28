#!/usr/bin/env bash
# scripts/eval_llm2vec.sh
#
# Run eval_llm2vec.py against a trained checkpoint and dump the report
# to stdout. Thin convenience wrapper; eval_llm2vec.py is the source of
# truth for all measurements.
#
# Usage:
#   LLM2VEC_DIR=./runs/smoke-XXX/llm2vec bash scripts/eval_llm2vec.sh
#
#   # Skip the base-LLM comparison (saves one model load)
#   LLM2VEC_DIR=./runs/smoke-XXX/llm2vec BASELINE_LLM=none \
#     bash scripts/eval_llm2vec.sh
#
# Environment variables (override any of these):
#   LLM2VEC_DIR        (required) checkpoint path
#   OUTPUT_DIR         default: $LLM2VEC_DIR/eval
#   BASELINE_LLM       HF id, or "none" to skip. Defaults to llm2vec_meta.json's
#                       base_llm field if present.
#   DOLMA_CACHE        default: $LLM2VEC_DIR/eval/dolma_cache
#   DOLMA_MAX_FILES    default: 1
#   N_SENTENCES        held-out sentences for evals (1)+(2) (default 500)
#   N_BIDIR_CAUSAL     subset for eval (3) (default 100)
#   MAX_SEQ_LENGTH     default: 256
#   MLM_PROBABILITY    default: 0.15
#   DEVICE             default: cuda
#   SEED               default: 42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LLM2VEC_DIR=${LLM2VEC_DIR:?LLM2VEC_DIR is required (point at a trained checkpoint)}
OUTPUT_DIR=${OUTPUT_DIR:-"$LLM2VEC_DIR/eval"}
DOLMA_CACHE=${DOLMA_CACHE:-"$OUTPUT_DIR/dolma_cache"}
DOLMA_MAX_FILES=${DOLMA_MAX_FILES:-1}
N_SENTENCES=${N_SENTENCES:-500}
N_BIDIR_CAUSAL=${N_BIDIR_CAUSAL:-100}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
MLM_PROBABILITY=${MLM_PROBABILITY:-0.15}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

EXTRA_ARGS=()
if [[ -n "${BASELINE_LLM:-}" ]]; then
    EXTRA_ARGS+=(--baseline-llm "$BASELINE_LLM")
fi

mkdir -p "$OUTPUT_DIR"

cat <<EOF
===============================================================
 [eval-llm2vec] config
===============================================================
LLM2VEC_DIR     = $LLM2VEC_DIR
OUTPUT_DIR      = $OUTPUT_DIR
BASELINE_LLM    = ${BASELINE_LLM:-<from llm2vec_meta.json>}
DOLMA_CACHE     = $DOLMA_CACHE
DOLMA_MAX_FILES = $DOLMA_MAX_FILES
N_SENTENCES     = $N_SENTENCES
N_BIDIR_CAUSAL  = $N_BIDIR_CAUSAL
MAX_SEQ_LENGTH  = $MAX_SEQ_LENGTH
MLM_PROBABILITY = $MLM_PROBABILITY
DEVICE          = $DEVICE        SEED = $SEED
EOF

python eval_llm2vec.py \
    --llm2vec-dir "$LLM2VEC_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --data-cache-dir "$DOLMA_CACHE" \
    --max-files "$DOLMA_MAX_FILES" \
    --n-sentences "$N_SENTENCES" \
    --n-bidir-causal "$N_BIDIR_CAUSAL" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --mlm-probability "$MLM_PROBABILITY" \
    --device "$DEVICE" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"

cat <<EOF

===============================================================
 [eval-llm2vec] report
===============================================================
EOF
cat "$OUTPUT_DIR/eval_report.md"

cat <<EOF

Artifacts:
  Metrics JSON : $OUTPUT_DIR/eval_metrics.json
  Report MD    : $OUTPUT_DIR/eval_report.md
EOF
