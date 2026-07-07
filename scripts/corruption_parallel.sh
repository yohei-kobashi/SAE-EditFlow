#!/usr/bin/env bash
# scripts/corruption_parallel.sh
#
# Parallel corruption data generation: WORKERS corruption.py processes on
# ONE GPU. The single-process pipeline is latency-bound (batch-1 forwards,
# GPU utilization typically <30%), so co-scheduling workers gives a
# near-linear speedup. Each worker takes the sentences whose eligible-index
# satisfies `idx % WORKERS == i` — a disjoint, complete partition of the
# stream — and writes to $OUT_DIR/worker-$i/.
#
# Pre-existing single-process shards in $OUT_DIR are KEPT: workers are told
# to --skip-sentences past the highest source_sent_id already present, so
# no source sentence is corrupted twice.
#
# When every worker reaches its per-worker target, worker shards are moved
# into $OUT_DIR as shard-w{i}-*.jsonl.gz and a merged meta.json is written.
# Downstream (CorruptionDataset, run_production.sh's stage-02 skip check,
# the §13.5 dev-cache split via `sentences_seen`) sees a normal cache.
#
# Re-running after a walltime kill resumes: per-worker resume is
# corruption.py's own (sentence-indexed); the merge only runs once all
# workers have finished. If $OUT_DIR/meta.json already exists the cache is
# complete and this script exits immediately.
#
# NOTE: do NOT run run_production.sh at the same time — its stage 02 would
# spawn a competing single-process run in the same OUT_DIR. Run this script
# to completion first; run_production.sh then skips stage 02 (meta.json
# present) and proceeds to the editor stage.
#
# Usage (env mirrors run_production.sh stage 02):
#   WORKERS=4 \
#   OUT_DIR=./runs/prod_gemma/corruption \
#   LLM2VEC_DIR=runs/mcgill_gemma_repro_3k/final \
#   bash scripts/corruption_parallel.sh
#
# Memory: each worker holds a truncated SAE-Gemma (~3 GB), a causal Gemma
# (~5 GB) and ModernBERT (~0.3 GB) in bf16 → ~9 GB/worker incl. activations.
# 4 workers fit comfortably on an 80-96 GB GPU; 6-8 are possible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

WORKERS=${WORKERS:-4}
OUT_DIR=${OUT_DIR:-"./runs/prod/corruption"}
: "${LLM2VEC_DIR:?set LLM2VEC_DIR to the merged LLM2Vec checkpoint dir}"

DOLMA_CACHE=${DOLMA_CACHE:-"./shared_cache/dolma"}
DOLMA_MAX_FILES=${DOLMA_MAX_FILES:-32}
CORRUPTION_SAMPLES=${CORRUPTION_SAMPLES:-100000}
CORRUPTION_SHARD=${CORRUPTION_SHARD:-10000}
LLM=${LLM:-"google/gemma-2-2b"}
SAE_REPO=${SAE_REPO:-"google/gemma-scope-2b-pt-res"}
SAE_PATH=${SAE_PATH:-"layer_12/width_16k/average_l0_82/params.npz"}
SAE_LAYER=${SAE_LAYER:-12}
MLM_MODEL=${MLM_MODEL:-"modernbert-base"}
SPACY_MODEL=${SPACY_MODEL:-"en_core_web_sm"}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}
# Floor for --skip-sentences (top-up into a FRESH dir: pass the previous
# cache's meta.json "sentences_seen" so the new run continues the Dolma
# stream instead of re-corrupting it from the start).
SKIP_SENTENCES=${SKIP_SENTENCES:-0}
# Per-op-type PROPOSAL weights (defaults match corruption.py argparse).
# Acceptance is not type-neutral (e.g. the SLOR gate hits INS's natural-word
# deletions harder than MLM-proposed REPL/DEL edits), so bump a type's weight
# to compensate when its ACCEPTED share lands below the calibrated target.
OP_WEIGHT_REPL=${OP_WEIGHT_REPL:-0.60}
OP_WEIGHT_INS=${OP_WEIGHT_INS:-0.34}
OP_WEIGHT_DEL=${OP_WEIGHT_DEL:-0.06}
# v4 transformation ops + confound-controlled conditioning
TRANSFORM_PROB=${TRANSFORM_PROB:-0.35}
TRANSFORM_FAMILIES=${TRANSFORM_FAMILIES:-all}
TRANSFORM_COMPOSE_PROB=${TRANSFORM_COMPOSE_PROB:-0.15}
BLOCKLIST=${BLOCKLIST:-""}
BLOCKLIST_ARGS=()
if [[ -n "$BLOCKLIST" ]]; then
    BLOCKLIST_ARGS=(--blocklist "$BLOCKLIST")
fi

mkdir -p "$OUT_DIR"

if [[ -f "$OUT_DIR/meta.json" ]]; then
    echo "[parallel] $OUT_DIR/meta.json already exists — cache complete, nothing to do."
    exit 0
fi

# ---- Scan pre-existing (single-process) shards ---------------------------- #
read -r EXISTING LAST_SID < <(python - "$OUT_DIR" <<'PY'
import gzip, json, sys
from pathlib import Path
out = Path(sys.argv[1])
n, last = 0, 0
for p in sorted(out.glob("shard-*.jsonl.gz")):
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                sid = obj.get("source_sent_id", "")
                if sid.startswith("dolma:s"):
                    try:
                        last = max(last, int(sid.split("s", 1)[1]))
                    except ValueError:
                        pass
    except (OSError, EOFError):
        continue  # truncated shard from a killed run — readable prefix counted
print(n, last)
PY
)
if (( SKIP_SENTENCES > LAST_SID )); then
    LAST_SID=$SKIP_SENTENCES
    echo "[parallel] skip-sentences floor applied: $LAST_SID"
fi
REMAINING=$(( CORRUPTION_SAMPLES - EXISTING ))
echo "[parallel] existing samples=$EXISTING (last source sentence $LAST_SID); remaining=$REMAINING"
if (( REMAINING <= 0 )); then
    echo "[parallel] target already reached by existing shards, but meta.json is"
    echo "           missing. Run one worker to finalize is not supported here —"
    echo "           lower CORRUPTION_SAMPLES or inspect $OUT_DIR."
    exit 1
fi
PER_WORKER=$(( (REMAINING + WORKERS - 1) / WORKERS ))
echo "[parallel] launching $WORKERS workers, --target-samples $PER_WORKER each"

# Share the SLOR unigram baseline when the previous run already built it
# (avoids each worker re-scanning 5k sentences). Without it each worker
# builds its own — identical content (same seed/stream), just slower start.
UNIGRAM_ARGS=()
if [[ -f "$OUT_DIR/unigram.json" ]]; then
    UNIGRAM_ARGS=(--unigram-cache "$OUT_DIR/unigram.json")
fi

# ---- Launch --------------------------------------------------------------- #
PIDS=()
for i in $(seq 0 $((WORKERS - 1))); do
    WDIR="$OUT_DIR/worker-$i"
    mkdir -p "$WDIR"
    python corruption.py \
        --data-cache-dir "$DOLMA_CACHE" \
        --max-files "$DOLMA_MAX_FILES" \
        --out-dir "$WDIR" \
        --llm2vec-dir "$LLM2VEC_DIR" \
        --llm "$LLM" \
        --sae-repo "$SAE_REPO" \
        --sae-path "$SAE_PATH" \
        --sae-layer "$SAE_LAYER" \
        --mlm-model "$MLM_MODEL" \
        --spacy-model "$SPACY_MODEL" \
        --op-weight-repl "$OP_WEIGHT_REPL" \
        --op-weight-ins "$OP_WEIGHT_INS" \
        --op-weight-del "$OP_WEIGHT_DEL" \
        --transform-prob "$TRANSFORM_PROB" \
        --transform-families "$TRANSFORM_FAMILIES" \
        --transform-compose-prob "$TRANSFORM_COMPOSE_PROB" \
        "${BLOCKLIST_ARGS[@]}" \
        --target-samples "$PER_WORKER" \
        --samples-per-shard "$CORRUPTION_SHARD" \
        --sentence-stride "$WORKERS" \
        --sentence-offset "$i" \
        --skip-sentences "$LAST_SID" \
        --device "$DEVICE" \
        --seed "$SEED" \
        "${UNIGRAM_ARGS[@]}" \
        > "$OUT_DIR/worker-$i.log" 2>&1 &
    PIDS+=($!)
    echo "[parallel] worker $i pid=${PIDS[i]} log=$OUT_DIR/worker-$i.log"
    sleep 3   # stagger model loading
done

FAIL=0
for i in $(seq 0 $((WORKERS - 1))); do
    if ! wait "${PIDS[i]}"; then
        echo "[parallel] worker $i FAILED — see $OUT_DIR/worker-$i.log"
        FAIL=1
    fi
done
if (( FAIL )); then
    echo "[parallel] one or more workers failed; re-run this script to resume."
    exit 1
fi

# ---- Merge ---------------------------------------------------------------- #
python - "$OUT_DIR" "$WORKERS" "$EXISTING" "$LAST_SID" <<'PY'
import json, shutil, sys
from pathlib import Path

out = Path(sys.argv[1])
W = int(sys.argv[2])
existing = int(sys.argv[3])
skip = int(sys.argv[4])

base = None
total = 0
max_seen = 0
for i in range(W):
    wd = out / f"worker-{i}"
    meta_p = wd / "meta.json"
    if not meta_p.exists():
        sys.exit(f"[merge] {meta_p} missing — worker {i} did not finish; "
                 f"re-run corruption_parallel.sh to resume it.")
    m = json.loads(meta_p.read_text())
    if base is None:
        base = m
    total += int(m.get("samples_written", 0))
    max_seen = max(max_seen, int(m.get("sentences_seen", 0)))
    moved = 0
    for sp in sorted(wd.glob("shard-*.jsonl.gz")):
        dst = out / f"shard-w{i}-{sp.name.split('-', 1)[1]}"
        shutil.move(str(sp), dst)
        moved += 1
    print(f"[merge] worker {i}: {m.get('samples_written', 0)} samples, "
          f"{moved} shard(s) moved")

base["samples_written"] = existing + total
base["sentences_seen"] = max_seen          # global high-water mark → dev split
base["sentence_stride"] = 1                # merged cache covers all residues
base["sentence_offset"] = 0
base["parallel"] = {"workers": W, "pre_existing_samples": existing,
                    "workers_skip_sentences": skip}
(out / "meta.json").write_text(json.dumps(base, indent=2))
print(f"[merge] wrote {out}/meta.json  total_samples={existing + total}  "
      f"sentences_seen={max_seen}")
PY

echo "[parallel] done. Downstream consumes $OUT_DIR unchanged;"
echo "           run_production.sh will now skip stage 02."
