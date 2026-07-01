#!/usr/bin/env bash
# scripts/vendor_mcgill_llm2vec.sh
#
# Vendor the McGill-NLP/llm2vec training scripts into this repo so we can
# rerun the paper recipe locally with our own configs and modifications.
#
# Why vendor (not `pip install` and go)?
#   - `pip install llm2vec` gives us the library (encoder classes, LLM2Vec
#     wrapper) but the TRAINING scripts (`experiments/run_mntp.py`,
#     `run_simcse.py`, LoRA JSON configs, custom collators) live in the
#     repo, NOT the package. We need those files to reproduce training.
#   - Pinning a commit means we know exactly what code we're running.
#     Upstream can move freely without breaking us.
#
# What lands where:
#   vendored/mcgill_llm2vec/                       ← full clone (shallow)
#   vendored/mcgill_llm2vec/experiments/           ← training entrypoints
#   vendored/mcgill_llm2vec/train_configs/         ← JSON configs
#   vendored/mcgill_llm2vec/llm2vec/models/        ← BiGemma2Model etc.
#
# Usage:
#   bash scripts/vendor_mcgill_llm2vec.sh
#   # or pin to a specific commit
#   PIN_COMMIT=abc1234 bash scripts/vendor_mcgill_llm2vec.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENDOR_DIR=${VENDOR_DIR:-"$REPO_ROOT/vendored/mcgill_llm2vec"}
UPSTREAM_URL=${UPSTREAM_URL:-"https://github.com/McGill-NLP/llm2vec.git"}
# Empty PIN_COMMIT = latest main. Pin for reproducibility once you've
# confirmed a specific commit works.
PIN_COMMIT=${PIN_COMMIT:-""}

if [[ -d "$VENDOR_DIR/.git" ]]; then
    echo "[vendor] $VENDOR_DIR already cloned. To re-fetch, delete and re-run:"
    echo "  rm -rf $VENDOR_DIR"
    echo "  bash scripts/vendor_mcgill_llm2vec.sh"
    exit 0
fi

mkdir -p "$(dirname "$VENDOR_DIR")"

echo "[vendor] cloning $UPSTREAM_URL"
if [[ -n "$PIN_COMMIT" ]]; then
    # Deep clone if pinning — shallow clone can't check out arbitrary commits.
    git clone --quiet "$UPSTREAM_URL" "$VENDOR_DIR"
    cd "$VENDOR_DIR"
    git checkout --quiet "$PIN_COMMIT"
    cd "$REPO_ROOT"
    echo "[vendor] pinned to $PIN_COMMIT"
else
    git clone --quiet --depth=1 "$UPSTREAM_URL" "$VENDOR_DIR"
    echo "[vendor] cloned latest main (shallow)"
fi

# Record the vendored commit hash for auditability. Not committed via
# .gitignore below, but kept for local reference.
cd "$VENDOR_DIR"
VENDORED_COMMIT=$(git rev-parse HEAD)
echo "$VENDORED_COMMIT" > "$VENDOR_DIR/.vendored_commit"
cd "$REPO_ROOT"

# Add vendored/ to .gitignore so we don't accidentally commit the clone.
if ! grep -q "^vendored/" .gitignore 2>/dev/null; then
    echo "vendored/" >> .gitignore
    echo "[vendor] added vendored/ to .gitignore"
fi

# ----- Inspection: show the key files that will drive training -----
echo
echo "===================================================================="
echo " Vendored files inventory"
echo "===================================================================="
echo "  commit : $VENDORED_COMMIT"
echo

echo "  Training entrypoints (experiments/):"
ls -la "$VENDOR_DIR/experiments/" 2>/dev/null | grep -E '\.py$' | awk '{print "    " $NF}'

echo
echo "  Training configs (train_configs/):"
if [[ -d "$VENDOR_DIR/train_configs" ]]; then
    find "$VENDOR_DIR/train_configs" -type f -name "*.json" | sort | \
        sed "s|$VENDOR_DIR/||" | awk '{print "    " $0}'
fi

echo
echo "  Bidirectional model implementations (llm2vec/models/):"
if [[ -d "$VENDOR_DIR/llm2vec/models" ]]; then
    ls "$VENDOR_DIR/llm2vec/models/" | grep -E '^bidirectional_.*\.py$' | \
        awk '{print "    llm2vec/models/" $0}'
fi

cat <<EOF

Next: inspect the Gemma-2 pieces we'll build on top of.

    ls -la $VENDOR_DIR/train_configs/mntp/     # to find the closest Gemma config
    less $VENDOR_DIR/experiments/run_mntp.py    # to understand CLI
    less $VENDOR_DIR/llm2vec/models/bidirectional_gemma.py   # bidir subclass

Once we know the config shape, the next script (train_mcgill_llm2vec.sh)
runs the vendored training pipeline against a Gemma-2-2b config in the
isolated venv.
EOF
