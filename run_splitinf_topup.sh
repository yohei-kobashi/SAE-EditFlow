#!/bin/bash
# P4 booster: SPLITINF-only harvest — run INSIDE interact-g, only if S6's
# MOVE gate fails on geometry (17 records are likely too thin).
#
# Why this is ~10-100x faster than another general v7 session:
#   * --sentence-regex rejects sentences lacking BOTH a to-infinitive shape
#     and an -ly token BEFORE tokenization/parsing (the general pass paid a
#     ~10ms spaCy parse for every sentence; the pattern occurs in ~0.14%).
#   * --transform-families SPLITINF alone: no family competition, and
#     family-priority-pick is irrelevant — every proposing sentence yields
#     the geometry we need.
#   * --transform-prob 1.0: never spend the sentence on word corruption.
#
# TARGET is CUMULATIVE over corruption_v7topup (12000 already there), so
# TARGET=13000 asks for +1000 records; raise per session as needed.
set -eo pipefail
cd "$(dirname "$0")"

OUT=runs/prod_gemma_v4/corruption_v7topup
MAIN=runs/prod_gemma_v4/corruption
TARGET=${TARGET:-13000}
BUDGET=${BUDGET:-5700}

set +e
timeout "$BUDGET" python corruption.py \
    --out-dir "$OUT" \
    --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
    --blocklist runs/blocklist/blocklist.npy \
    --target-samples "$TARGET" --samples-per-shard 500 \
    --sentence-regex '(?i)\bto\s+(?:\w+ly\s+\w|\w+(?:\s+\S+){0,6}?\s+\w{3,}ly\b)' \
    --transform-prob 1.0 \
    --transform-families "SPLITINF" \
    --repl-mlm-topk 24 --del-mlm-topk 24 --del-top1-prob 0.25 \
    --seed 777 \
    --device cuda
RC=$?
set -e
[ $RC -eq 124 ] && echo "[splitinf] BUDGET hit — rerun with the same TARGET to continue"

echo "==================== SPLITINF MERGE ===================="
n=0
for s in "$OUT"/shard-*.jsonl.gz; do
    [ -e "$s" ] || continue
    base=$(basename "$s" | sed 's/^shard-/shard-v7-/')
    ln -sf "$(readlink -f "$s")" "$MAIN/$base"
    n=$((n+1))
done
echo "[splitinf] linked $n shard(s); SPLITINF cumulative:"
for s in "$OUT"/shard-*.jsonl.gz; do [ -e "$s" ] && zcat "$s"; done \
    | grep -c '"t_family": "SPLITINF"' || true
echo "==================== SPLITINF SESSION DONE ===================="
