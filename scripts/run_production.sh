#!/usr/bin/env bash
# scripts/run_production.sh
#
# Production-scale end-to-end SAE-LEWIS training pipeline.
#
# Re-running with the same RUN_DIR resumes from where the previous
# invocation left off:
#   - Completed stages (output marker present) are skipped outright by
#     this script, same as smoke_pipeline.sh.
#   - Mid-stage interruption is picked up by each script's per-stage
#     `--resume` default (corruption, train_llm2vec, train_tagger,
#     train_editor_phaseA, train_length_head, precompute_sae).
#
# Usage:
#   bash scripts/run_production.sh                       # ./runs/prod, all defaults
#   RUN_DIR=./runs/exp1 bash scripts/run_production.sh   # custom output root
#   DOLMA_MAX_FILES=64 bash scripts/run_production.sh    # bump Dolma slice
#
# To genuinely start fresh, either (a) point RUN_DIR somewhere new or
# (b) pass FORCE_FRESH=1 to wipe and re-run every stage:
#   FORCE_FRESH=1 RUN_DIR=./runs/prod bash scripts/run_production.sh
#
# Environment overrides (defaults match the argparse production defaults
# of each script; calibrated for an 80GB GPU):
#
#   RUN_DIR                 output root (DEFAULT: ./runs/prod -- NOT
#                            timestamped, so re-running the same command
#                            resumes the same run)
#   SHARED_CACHE_ROOT       root for Dolma + SAE caches; SHARED across runs
#                            (default: ./shared_cache). Override to put on
#                            a fast scratch disk.
#   DOLMA_CACHE             Dolma shard dir (default: $SHARED_CACHE_ROOT/dolma)
#   SAE_CACHE               SAE cache dir; keyed by (LLM, SAE path, max-sents)
#                            so multiple configs coexist as siblings
#                            (default: $SHARED_CACHE_ROOT/sae/<key>)
#   DEVICE                  cuda | cpu (default: cuda)
#   SEED                    (default: 42)
#
#   DOLMA_MAX_FILES         # of Dolma shards to download (default: 32)
#   SAE_MAX_SENTS           cap on stage-0 SAE cache size (default: 100000)
#   LLM2VEC_STEPS           MNTP steps (default: 10000)
#   LLM2VEC_BATCH           per-device batch for MNTP (default: 8)
#   LLM2VEC_ACCUM           grad-accum-steps for MNTP (default: 4)
#   SIMCSE_STEPS            SimCSE steps (default: 2000)
#   SIMCSE_BATCH            per-device batch for SimCSE (default: 32)
#   SIMCSE_LR               SimCSE learning rate (default: 1e-6)
#   SIMCSE_TEMP             NT-Xent temperature (default: 0.05)
#   SIMCSE_DROPOUT          injected attention dropout (default: 0.1)
#   SIMCSE_MAX_SEQ_LEN      sequence length for SimCSE (default: 128)
#   SIMCSE_GRAD_CKPT        1 → --gradient-checkpointing (default: 0)
#   SKIP_SIMCSE             1 → skip SimCSE; downstream consumes MNTP dir directly
#                            (LLM2Vec paper: Bi+MNTP only ≈ 0.55-0.65 on STS-B;
#                             SimCSE adds another 0.1+; only skip for ablation)
#   CORRUPTION_SAMPLES      stage-2 target sample count (default: 100000)
#   CORRUPTION_SHARD        samples per corruption shard (default: 10000)
#   TAGGER_STEPS            tagger training steps (default: 10000)
#   TAGGER_BATCH            (default: 8)
#   EDITOR_STEPS            editor Phase A steps (default: 20000)
#   EDITOR_BATCH            (default: 8)
#   LENGTH_STEPS            length head steps (default: 5000)
#   LENGTH_BATCH            (default: 8)
#   NUM_WORKERS             data-loader workers (default: 4)
#
#   LLM                     causal LLM HF id   (default: google/gemma-2-2b)
#   SAE_REPO                Gemma Scope HF repo (default: google/gemma-scope-2b-pt-res)
#   SAE_PATH                file inside SAE_REPO (default: layer_12/width_16k/average_l0_82/params.npz)
#   SAE_LAYER               (default: 12)
#   MLM_MODEL               corruption MLM preset (default: modernbert-base)
#   SPACY_MODEL             spaCy POS model (default: en_core_web_sm)
#
#   FORCE_FRESH             1 = wipe RUN_DIR before starting; pass --no-resume
#                            to every stage. Use this when you want to redo
#                            an experiment under the same RUN_DIR.
#
# Per-stage logs land in $RUN_DIR/logs/<stage>.log. Per-stage wall time
# in $RUN_DIR/timing.tsv. Per-stage GPU peak / utilization in
# $RUN_DIR/gpu.tsv (sampled at 1Hz via nvidia-smi).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RUN_DIR=${RUN_DIR:-"./runs/prod"}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

DOLMA_MAX_FILES=${DOLMA_MAX_FILES:-32}
SAE_MAX_SENTS=${SAE_MAX_SENTS:-100000}
# Canonical LLM2Vec uses LoRA. Defaults below match the McGill-NLP recipe
# (r=16, alpha=32, MNTP 1k @ 3e-4, SimCSE 1k @ 3e-5, batch 128).
# For full-FT ablation: USE_LORA=0 LLM2VEC_STEPS=10000 LLM2VEC_LR=1e-5
# SIMCSE_STEPS=2000 SIMCSE_LR=1e-6 SIMCSE_BATCH=32.
USE_LORA=${USE_LORA:-1}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LLM2VEC_STEPS=${LLM2VEC_STEPS:-1000}
LLM2VEC_BATCH=${LLM2VEC_BATCH:-8}
LLM2VEC_ACCUM=${LLM2VEC_ACCUM:-4}
LLM2VEC_LR=${LLM2VEC_LR:-3e-4}
SIMCSE_STEPS=${SIMCSE_STEPS:-1000}
SIMCSE_BATCH=${SIMCSE_BATCH:-128}
SIMCSE_LR=${SIMCSE_LR:-3e-5}
SIMCSE_TEMP=${SIMCSE_TEMP:-0.05}
SIMCSE_DROPOUT=${SIMCSE_DROPOUT:-0.1}
SIMCSE_MAX_SEQ_LEN=${SIMCSE_MAX_SEQ_LEN:-128}
# SimCSE retains TWO forwards in autograd, doubling activation memory vs
# MNTP. At batch=128 + Gemma-2B this is ~85 GB and OOMs on 95 GB H200.
# Canonical LLM2Vec uses gradient_checkpointing here; auto-enable on the
# LoRA path. The full-FT ablation batch (default 32) doesn't need it.
if [[ "$USE_LORA" == "1" ]]; then
    SIMCSE_GRAD_CKPT=${SIMCSE_GRAD_CKPT:-1}
else
    SIMCSE_GRAD_CKPT=${SIMCSE_GRAD_CKPT:-0}
fi
SKIP_SIMCSE=${SKIP_SIMCSE:-0}
CORRUPTION_SAMPLES=${CORRUPTION_SAMPLES:-100000}
CORRUPTION_SHARD=${CORRUPTION_SHARD:-10000}
# Dev-monitored best-checkpoint selection (stages 03/04). Point
# DEV_CORRUPTION_DIR at a SELECTION split (distinct from the reporting dev
# cache) to enable; editor-final.pt / tagger-final.pt then hold the best
# dev state instead of the last step.
DEV_CORRUPTION_DIR=${DEV_CORRUPTION_DIR:-""}
EVAL_STEPS=${EVAL_STEPS:-2000}
DEV_BATCHES=${DEV_BATCHES:-64}
DEV_ARGS=()
if [[ -n "$DEV_CORRUPTION_DIR" ]]; then
    DEV_ARGS=(--dev-corruption-dir "$DEV_CORRUPTION_DIR" --eval-steps "$EVAL_STEPS"
              --dev-batches "$DEV_BATCHES")
fi

K_DRAW=${K_DRAW:-"1-8"}
K_TOP=${K_TOP:-8}
EXCLUDE_FAMILIES=${EXCLUDE_FAMILIES:-""}
LOFO_ARGS=()
if [[ -n "$EXCLUDE_FAMILIES" ]]; then
    LOFO_ARGS=(--exclude-families "$EXCLUDE_FAMILIES")
fi
TRANSFORM_COMPOSE_PROB=${TRANSFORM_COMPOSE_PROB:-0.15}
TAGGER_STEPS=${TAGGER_STEPS:-10000}
TAGGER_BATCH=${TAGGER_BATCH:-8}
EDITOR_STEPS=${EDITOR_STEPS:-20000}
EDITOR_BATCH=${EDITOR_BATCH:-8}
LENGTH_STEPS=${LENGTH_STEPS:-5000}
LENGTH_BATCH=${LENGTH_BATCH:-8}
NUM_WORKERS=${NUM_WORKERS:-4}

LLM=${LLM:-"google/gemma-2-2b"}
SAE_REPO=${SAE_REPO:-"google/gemma-scope-2b-pt-res"}
SAE_PATH=${SAE_PATH:-"layer_12/width_16k/average_l0_82/params.npz"}
SAE_LAYER=${SAE_LAYER:-12}
MLM_MODEL=${MLM_MODEL:-"modernbert-base"}
SPACY_MODEL=${SPACY_MODEL:-"en_core_web_sm"}

FORCE_FRESH=${FORCE_FRESH:-0}

# When FORCE_FRESH=1 we pass --no-resume to every stage AND remove any
# existing outputs so the outer "skip if exists" checks don't kick in.
if [[ "$FORCE_FRESH" == "1" ]]; then
    if [[ -d "$RUN_DIR" ]]; then
        echo "[prod] FORCE_FRESH=1: removing $RUN_DIR"
        rm -rf "$RUN_DIR"
    fi
    RESUME_FLAG="--no-resume"
else
    RESUME_FLAG=""
fi

# --------------------------------------------------------------------------- #
# Output layout
# --------------------------------------------------------------------------- #
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
SUMMARY="$RUN_DIR/timing.tsv"
if [[ ! -f "$SUMMARY" ]]; then
    printf "stage\tstart\tend\telapsed_sec\tstatus\n" > "$SUMMARY"
fi
GPU_TSV="$RUN_DIR/gpu.tsv"
if [[ ! -f "$GPU_TSV" ]]; then
    printf "stage\tpeak_used_mib\ttotal_mib\tavg_util_pct\tn_samples\n" > "$GPU_TSV"
fi

GPU_AVAILABLE=0
if [[ "${DEVICE,,}" == cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_AVAILABLE=1
fi
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
GPU_MON_PID=
GPU_MON_FILE=
trap 'if [[ -n "${GPU_MON_PID:-}" ]]; then kill "$GPU_MON_PID" 2>/dev/null || true; fi' EXIT

# --------------------------------------------------------------------------- #
# Shared caches (deliberately OUTSIDE $RUN_DIR so they survive FORCE_FRESH
# and are reused across runs).
#   - Dolma shards   : identical bytes regardless of training config
#   - SAE features   : reusable when (LLM, SAE config, max-sentences) match;
#                       precompute_sae.py has all-or-nothing resume keyed on
#                       meta.json, so we key the path on the same fields and
#                       let mismatching configs coexist in sibling dirs.
# Override SHARED_CACHE_ROOT to redirect both to a scratch disk:
#   SHARED_CACHE_ROOT=/scratch/$USER bash scripts/run_production.sh
# --------------------------------------------------------------------------- #
SHARED_CACHE_ROOT=${SHARED_CACHE_ROOT:-"./shared_cache"}
DOLMA_CACHE=${DOLMA_CACHE:-"$SHARED_CACHE_ROOT/dolma"}

_LLM_SLUG="$(echo "$LLM" | tr '/' '-')"
_SAE_PATH_SLUG="$(dirname "$SAE_PATH" | tr '/' '-')"
SAE_CACHE_KEY="${_LLM_SLUG}_${_SAE_PATH_SLUG}_n${SAE_MAX_SENTS}"
SAE_CACHE=${SAE_CACHE:-"$SHARED_CACHE_ROOT/sae/$SAE_CACHE_KEY"}

# Overridable so an externally-trained LLM2Vec checkpoint (e.g. the McGill
# vendored route's merged output, train_mcgill_llm2vec.sh → $RUN_ROOT/final)
# can be dropped in: its llm2vec_meta.json makes stages 01/01b skip.
LLM2VEC_DIR=${LLM2VEC_DIR:-"$RUN_DIR/llm2vec"}
SIMCSE_DIR=${SIMCSE_DIR:-"$RUN_DIR/llm2vec_simcse"}
CORRUPTION_DIR="$RUN_DIR/corruption"
TAGGER_DIR="$RUN_DIR/tagger"
EDITOR_DIR="$RUN_DIR/editor"
LENGTH_DIR="$RUN_DIR/length_head"
EVAL_DIR="$RUN_DIR/eval"

# Downstream stages (corruption / tagger / editor / length / evaluate) all
# consume an LLM2Vec-style HF checkpoint via --llm2vec-dir. We normally
# point this at the SimCSE output (canonical LLM2Vec end-product); set
# SKIP_SIMCSE=1 to consume the MNTP-only dir for ablation.
if [[ "$SKIP_SIMCSE" == "1" ]]; then
    DOWNSTREAM_LLM2VEC_DIR="$LLM2VEC_DIR"
else
    DOWNSTREAM_LLM2VEC_DIR="$SIMCSE_DIR"
fi

TAGGER_CKPT="$TAGGER_DIR/tagger-final.pt"
EDITOR_CKPT="$EDITOR_DIR/editor-final.pt"
LENGTH_CKPT="$LENGTH_DIR/length-final.pt"

# --------------------------------------------------------------------------- #
# Helpers (mirror smoke_pipeline.sh; refactor if you change one and not both)
# --------------------------------------------------------------------------- #
banner() {
    printf '\n===============================================================\n'
    printf ' %s\n' "$1"
    printf '===============================================================\n'
}

record() {
    local stage=$1 start=${2:-} end=${3:-} elapsed=${4:-0} status=$5
    printf "%s\t%s\t%s\t%s\t%s\n" "$stage" "$start" "$end" "$elapsed" "$status" \
        >> "$SUMMARY"
}

start_gpu_monitor() {
    GPU_MON_PID=
    GPU_MON_FILE=
    [[ "$GPU_AVAILABLE" -eq 1 ]] || return 0
    local name=$1
    GPU_MON_FILE="$LOG_DIR/gpu_${name}.tsv"
    : > "$GPU_MON_FILE"
    (
        while true; do
            nvidia-smi -i "$GPU_ID" \
                --query-gpu=memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null \
                | awk -F', *' 'NF>=3 {printf "%s\t%s\t%s\n", $1, $2, $3}' \
                >> "$GPU_MON_FILE"
            sleep 5
        done
    ) &
    GPU_MON_PID=$!
}

stop_gpu_monitor() {
    local name=$1
    if [[ -n "${GPU_MON_PID:-}" ]]; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        wait "$GPU_MON_PID" 2>/dev/null || true
        GPU_MON_PID=
    fi
    [[ -s "${GPU_MON_FILE:-/dev/null}" ]] || return 0
    awk -F'\t' -v stage="$name" '
        { if ($1 + 0 > peak) peak = $1 + 0
          if ($2 + 0 > total) total = $2 + 0
          sum_util += $3 + 0; n++ }
        END {
            avg = (n > 0 ? sum_util / n : 0)
            printf "%s\t%d\t%d\t%.1f\t%d\n", stage, peak, total, avg, n
        }
    ' "$GPU_MON_FILE" >> "$GPU_TSV"
}

run_stage() {
    local name=$1; shift
    local log="$LOG_DIR/${name}.log"
    banner "[prod] stage: $name"
    echo "[prod] cmd: $*" | tee "$log"
    local start
    start=$(date +%s)
    start_gpu_monitor "$name"
    # Run, capturing stdout/stderr to both terminal and log. We disable
    # `set -e` around the pipe so PIPESTATUS is readable; we do NOT
    # use `|| true` because that would mask failures as 0.
    set +e
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    stop_gpu_monitor "$name"
    local end
    end=$(date +%s)
    local elapsed=$((end - start))
    if [[ "$rc" -ne 0 ]]; then
        record "$name" "$start" "$end" "$elapsed" "fail(rc=$rc)"
        printf '[prod] %-22s FAILED rc=%d  see %s\n' "$name" "$rc" "$log" >&2
        echo "[prod] re-run the same command to resume from the last "\
             "checkpoint" >&2
        exit "$rc"
    fi
    record "$name" "$start" "$end" "$elapsed" "ok"
    printf '[prod] %-22s elapsed=%dh%02dm%02ds  log=%s\n' \
        "$name" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) \
        $((elapsed % 60)) "$log"
}

skip_stage() {
    local name=$1 reason=$2
    record "$name" "" "" 0 "skip"
    printf '[prod] %-22s SKIPPED  %s\n' "$name" "$reason"
}

# --------------------------------------------------------------------------- #
# Print config
# --------------------------------------------------------------------------- #
banner "[prod] config"
cat <<EOF
RUN_DIR              = $RUN_DIR     (resume by default; FORCE_FRESH=$FORCE_FRESH)
DEVICE               = $DEVICE        SEED = $SEED
DOLMA_MAX_FILES      = $DOLMA_MAX_FILES
SAE_MAX_SENTS        = $SAE_MAX_SENTS
LLM2VEC_STEPS / BS   = $LLM2VEC_STEPS / $LLM2VEC_BATCH (accum $LLM2VEC_ACCUM, effective $((LLM2VEC_BATCH * LLM2VEC_ACCUM)))
SIMCSE_STEPS / BS    = $SIMCSE_STEPS / $SIMCSE_BATCH  (LR $SIMCSE_LR, τ $SIMCSE_TEMP, dropout $SIMCSE_DROPOUT)
SKIP_SIMCSE          = $SKIP_SIMCSE  → downstream LLM2Vec dir = $DOWNSTREAM_LLM2VEC_DIR
CORRUPTION samples   = $CORRUPTION_SAMPLES   (shard size $CORRUPTION_SHARD)
TAGGER_STEPS / BS    = $TAGGER_STEPS / $TAGGER_BATCH
EDITOR_STEPS / BS    = $EDITOR_STEPS / $EDITOR_BATCH
LENGTH_STEPS / BS    = $LENGTH_STEPS / $LENGTH_BATCH
NUM_WORKERS          = $NUM_WORKERS
LLM                  = $LLM
SAE_REPO             = $SAE_REPO
SAE_PATH             = $SAE_PATH
SAE_LAYER            = $SAE_LAYER
MLM_MODEL            = $MLM_MODEL
SPACY_MODEL          = $SPACY_MODEL
EOF

# Time the overall run separately from the timing.tsv per-stage rows.
PROD_T0=$(date +%s)

# --------------------------------------------------------------------------- #
# Stage 0: precompute SAE cache (Dolma → sentences → SAE top-L per token)
# --------------------------------------------------------------------------- #
if [[ -f "$SAE_CACHE/meta.json" ]]; then
    skip_stage "00_precompute_sae" "exists at $SAE_CACHE"
else
    run_stage "00_precompute_sae" \
        python precompute_sae.py \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --out-dir "$SAE_CACHE" \
            --max-sentences "$SAE_MAX_SENTS" \
            --batch-size 32 \
            --llm "$LLM" \
            --sae-repo "$SAE_REPO" \
            --sae-path "$SAE_PATH" \
            --sae-layer "$SAE_LAYER" \
            --device "$DEVICE" \
            --seed "$SEED" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 1: LLM2Vec MNTP — turn causal Gemma into a bidirectional encoder
# --------------------------------------------------------------------------- #
if [[ -f "$LLM2VEC_DIR/llm2vec_meta.json" ]]; then
    skip_stage "01_train_llm2vec" "exists at $LLM2VEC_DIR"
else
    LLM2VEC_EXTRA=()
    if [[ "$USE_LORA" == "1" ]]; then
        LLM2VEC_EXTRA+=(--use-lora
                        --lora-r "$LORA_R"
                        --lora-alpha "$LORA_ALPHA"
                        --lora-dropout "$LORA_DROPOUT")
        # LoRA recipe: shorter warmup + save, since max_steps default is 1k.
        LLM2VEC_WARMUP_ARG=100
        LLM2VEC_SAVE_ARG=500
    else
        LLM2VEC_EXTRA+=(--no-use-lora)
        LLM2VEC_WARMUP_ARG=1000
        LLM2VEC_SAVE_ARG=1000
    fi
    run_stage "01_train_llm2vec" \
        python train_llm2vec.py \
            --llm "$LLM" \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --output-dir "$LLM2VEC_DIR" \
            --max-steps "$LLM2VEC_STEPS" \
            --warmup-steps "$LLM2VEC_WARMUP_ARG" \
            --learning-rate "$LLM2VEC_LR" \
            --per-device-batch-size "$LLM2VEC_BATCH" \
            --grad-accum-steps "$LLM2VEC_ACCUM" \
            --save-steps "$LLM2VEC_SAVE_ARG" \
            --logging-steps 50 \
            --num-workers "$NUM_WORKERS" \
            --seed "$SEED" \
            "${LLM2VEC_EXTRA[@]}" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 1b: SimCSE — canonical LLM2Vec finishes Bi+MNTP with unsupervised
# contrastive (NT-Xent / in-batch negatives). Outputs a fresh HF-format
# checkpoint at $SIMCSE_DIR that downstream stages consume.
# --------------------------------------------------------------------------- #
SIMCSE_DONE=0
if [[ -f "$SIMCSE_DIR/llm2vec_meta.json" ]] && \
   [[ "$(python -c "import json,sys; print('simcse' in json.load(open('$SIMCSE_DIR/llm2vec_meta.json')))" 2>/dev/null)" == "True" ]]; then
    SIMCSE_DONE=1
fi
if [[ "$SKIP_SIMCSE" == "1" ]]; then
    skip_stage "01b_train_simcse" "SKIP_SIMCSE=1; downstream uses $LLM2VEC_DIR"
elif [[ "$SIMCSE_DONE" == "1" ]]; then
    skip_stage "01b_train_simcse" "exists at $SIMCSE_DIR"
else
    SIMCSE_EXTRA=()
    if [[ "$SIMCSE_GRAD_CKPT" == "1" ]]; then
        SIMCSE_EXTRA+=(--gradient-checkpointing)
    fi
    if [[ "$USE_LORA" == "1" ]]; then
        SIMCSE_EXTRA+=(--use-lora
                        --lora-r "$LORA_R"
                        --lora-alpha "$LORA_ALPHA"
                        --lora-dropout "$LORA_DROPOUT")
    else
        SIMCSE_EXTRA+=(--no-use-lora)
    fi
    run_stage "01b_train_simcse" \
        python train_simcse.py \
            --llm2vec-dir "$LLM2VEC_DIR" \
            --output-dir "$SIMCSE_DIR" \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --max-steps "$SIMCSE_STEPS" \
            --warmup-steps 100 \
            --learning-rate "$SIMCSE_LR" \
            --per-device-batch-size "$SIMCSE_BATCH" \
            --temperature "$SIMCSE_TEMP" \
            --dropout "$SIMCSE_DROPOUT" \
            --max-seq-length "$SIMCSE_MAX_SEQ_LEN" \
            --save-steps 500 \
            --logging-steps 50 \
            --num-workers "$NUM_WORKERS" \
            --device "$DEVICE" \
            --seed "$SEED" \
            "${SIMCSE_EXTRA[@]}" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 2: corruption — generate (X, X') training pairs with the
# Bi+MNTP+SimCSE encoder + Gemma Scope SAE conditioning. Longest stage.
# --------------------------------------------------------------------------- #
if [[ -f "$CORRUPTION_DIR/meta.json" ]]; then
    skip_stage "02_corruption" "exists at $CORRUPTION_DIR"
else
    run_stage "02_corruption" \
        python corruption.py \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --out-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$DOWNSTREAM_LLM2VEC_DIR" \
            --llm "$LLM" \
            --sae-repo "$SAE_REPO" \
            --sae-path "$SAE_PATH" \
            --sae-layer "$SAE_LAYER" \
            --mlm-model "$MLM_MODEL" \
            --spacy-model "$SPACY_MODEL" \
            --target-samples "$CORRUPTION_SAMPLES" \
            --samples-per-shard "$CORRUPTION_SHARD" \
            --transform-compose-prob "$TRANSFORM_COMPOSE_PROB" \
            --device "$DEVICE" \
            --seed "$SEED" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 3: editor (Phase A) — unified corruption pretraining
#   Runs BEFORE the tagger: the editor's all-position CE gives Proj_A the
#   richest gradient, and the tagger warm-starts its conditioning interface
#   (Proj_A / type_emb / cond_scale) from editor-final.pt (README C2).
# --------------------------------------------------------------------------- #
if [[ -f "$EDITOR_CKPT" ]]; then
    skip_stage "03_train_editor_phaseA" "exists at $EDITOR_CKPT"
else
    run_stage "03_train_editor_phaseA" \
        python train_editor_phaseA.py \
            --corruption-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$DOWNSTREAM_LLM2VEC_DIR" \
            --output-dir "$EDITOR_DIR" \
            --max-steps "$EDITOR_STEPS" \
            --warmup-steps 500 \
            --proj-a-freeze-steps 1000 \
            --k-top "$K_TOP" --k-amp "$K_DRAW" --k-sup "$K_DRAW" \
            "${LOFO_ARGS[@]}" \
            --batch-size "$EDITOR_BATCH" \
            --num-workers "$NUM_WORKERS" \
            --save-steps 2000 \
            --logging-steps 50 \
            "${DEV_ARGS[@]}" \
            --device "$DEVICE" \
            --seed "$SEED" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 4: tagger — per-token KEEP/REPL/INS/DEL classifier
# --------------------------------------------------------------------------- #
if [[ -f "$TAGGER_CKPT" ]]; then
    skip_stage "04_train_tagger" "exists at $TAGGER_CKPT"
else
    run_stage "04_train_tagger" \
        python train_tagger.py \
            --corruption-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$DOWNSTREAM_LLM2VEC_DIR" \
            --output-dir "$TAGGER_DIR" \
            --init-proj-a-from "$EDITOR_CKPT" \
            --max-steps "$TAGGER_STEPS" \
            --warmup-steps 500 \
            --proj-a-freeze-steps 500 \
            --k-top "$K_TOP" --k-amp "$K_DRAW" --k-sup "$K_DRAW" \
            "${LOFO_ARGS[@]}" \
            --batch-size "$TAGGER_BATCH" \
            --num-workers "$NUM_WORKERS" \
            --save-steps 2000 \
            --logging-steps 50 \
            --estimate-class-weights-batches 200 \
            "${DEV_ARGS[@]}" \
            --device "$DEVICE" \
            --seed "$SEED" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 5: length head — predict INS span length at gap positions
# --------------------------------------------------------------------------- #
if [[ -f "$LENGTH_CKPT" ]]; then
    skip_stage "05_train_length_head" "exists at $LENGTH_CKPT"
else
    run_stage "05_train_length_head" \
        python train_length_head.py \
            --corruption-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$DOWNSTREAM_LLM2VEC_DIR" \
            --editor-ckpt "$EDITOR_CKPT" \
            --output-dir "$LENGTH_DIR" \
            --max-steps "$LENGTH_STEPS" \
            --warmup-steps 200 \
            --batch-size "$LENGTH_BATCH" \
            --num-workers "$NUM_WORKERS" \
            --save-steps 1000 \
            --logging-steps 50 \
            --device "$DEVICE" \
            --seed "$SEED" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 6: sanity-check intervention on a single example
# (evaluate_intervention.py has no checkpoint; cheap, always re-runs)
# --------------------------------------------------------------------------- #
mkdir -p "$EVAL_DIR"
run_stage "06_evaluate" \
    python evaluate_intervention.py \
        --llm2vec-dir "$DOWNSTREAM_LLM2VEC_DIR" \
        --tagger-ckpt "$TAGGER_CKPT" \
        --editor-ckpt "$EDITOR_CKPT" \
        --mu "$SAE_CACHE/mu.npy" \
        --llm "$LLM" \
        --sae-repo "$SAE_REPO" \
        --sae-path "$SAE_PATH" \
        --sae-layer "$SAE_LAYER" \
        --text "The quick brown fox jumps over the lazy dog." \
        --spec "+1234" \
        --l-max 5 \
        --device "$DEVICE"

# --------------------------------------------------------------------------- #
# Wrap up
# --------------------------------------------------------------------------- #
PROD_T1=$(date +%s)
TOTAL=$((PROD_T1 - PROD_T0))

banner "[prod] DONE"
printf 'Total wall time this invocation: %dh%02dm%02ds\n' \
    $((TOTAL / 3600)) $(((TOTAL % 3600) / 60)) $((TOTAL % 60))
echo
echo "Shared caches (outside $RUN_DIR — reused across runs):"
echo "  Dolma shards      : $DOLMA_CACHE"
echo "  SAE features      : $SAE_CACHE"
echo
echo "Artifacts under $RUN_DIR:"
echo "  LLM2Vec MNTP      : $LLM2VEC_DIR"
echo "  LLM2Vec MNTP+SimCSE: $SIMCSE_DIR  (skipped if SKIP_SIMCSE=1)"
echo "  Downstream uses   : $DOWNSTREAM_LLM2VEC_DIR"
echo "  Corruption data   : $CORRUPTION_DIR"
echo "  Tagger            : $TAGGER_CKPT"
echo "  Editor            : $EDITOR_CKPT"
echo "  Length head       : $LENGTH_CKPT"
echo "  Per-stage logs    : $LOG_DIR/"
echo "  Timing summary    : $SUMMARY"
echo "  GPU summary       : $GPU_TSV"
echo
echo "If a stage failed, fix the underlying issue and re-run the SAME"
echo "command. Completed stages will be skipped; the failed stage will"
echo "resume from its last per-stage checkpoint."
