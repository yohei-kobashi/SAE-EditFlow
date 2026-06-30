#!/usr/bin/env bash
# scripts/train_eval_llm2vec.sh
#
# LLM2Vec を学習し、続けて学習効果を 5 項目で自動評価するパイプライン。
#
#   stage 1 : train_llm2vec.py  (MNTP, bidir patch, [INS]/[DEL] 追加)
#   stage 2 : train_simcse.py   (unsupervised contrastive, NT-Xent)
#   stage 3 : eval_llm2vec.py   (MNTP loss / causal PPL drift /
#                                bidir-vs-causal cosine / special-token /
#                                MTEB STS-B)
#
# 同じ RUN_DIR で再実行すると stage 1/2 は最新 checkpoint から再開
# (両 trainer ともに --resume をデフォルト ON にしている)。
# stage 3 (評価) は毎回再実行する — 計算は軽く、報告は上書きされる。
#
# 評価対象は stage 2 完了後の SimCSE 出力 (= $SIMCSE_DIR)。
# SKIP_SIMCSE=1 のときは MNTP 出力 (= $LLM2VEC_DIR) を評価する。
#
# Usage:
#   bash scripts/train_eval_llm2vec.sh
#   RUN_DIR=./runs/llm2vec-exp1 bash scripts/train_eval_llm2vec.sh
#   LLM2VEC_STEPS=2000 SIMCSE_STEPS=500 bash scripts/train_eval_llm2vec.sh
#   FORCE_FRESH=1 bash scripts/train_eval_llm2vec.sh             # 全部やり直し
#   SKIP_TRAIN=1 SIMCSE_DIR=./runs/prod/llm2vec_simcse \
#     bash scripts/train_eval_llm2vec.sh                         # 評価のみ
#   SKIP_SIMCSE=1 bash scripts/train_eval_llm2vec.sh             # MNTP only
#
# Environment overrides:
#   RUN_DIR              出力ルート (default: ./runs/llm2vec_eval)
#   LLM2VEC_DIR          MNTP 出力先 (default: $RUN_DIR/llm2vec)
#   SIMCSE_DIR           SimCSE 出力先 (default: $RUN_DIR/llm2vec_simcse)
#   EVAL_DIR             評価出力先  (default: $RUN_DIR/eval)
#   SHARED_CACHE_ROOT    Dolma cache の共通ルート (default: ./shared_cache)。
#                         RUN_DIR の外に置くので FORCE_FRESH=1 でも残る。
#   DOLMA_CACHE          Dolma shards (default: $SHARED_CACHE_ROOT/dolma)
#   DEVICE               cuda | cpu  (default: cuda)
#   SEED                 (default: 42)
#
#   --- MNTP (stage 1) ---
#   LLM                  HF base model id (default: google/gemma-2-2b)
#   DOLMA_MAX_FILES      Dolma shards to download (default: 32)
#   LLM2VEC_STEPS        MNTP steps (default: 10000)
#   LLM2VEC_BATCH        per-device batch (default: 8)
#   LLM2VEC_ACCUM        grad-accum-steps (default: 4)
#   LLM2VEC_LR           learning rate (default: 1e-5)
#   LLM2VEC_WARMUP       warmup steps (default: 1000)
#   LLM2VEC_SAVE_STEPS   checkpoint cadence (default: 1000)
#   NUM_WORKERS          data-loader workers (default: 4)
#   MAX_SEQ_LENGTH       (default: 256)
#   MLM_PROBABILITY      (default: 0.15)
#
#   --- SimCSE (stage 2) ---
#   SIMCSE_STEPS         (default: 2000)  optimizer steps; canonical 1-2k
#   SIMCSE_BATCH         per-device batch (default: 32) — larger = more negatives
#   SIMCSE_LR            (default: 1e-6)  very low for full FT
#   SIMCSE_WARMUP        (default: 100)
#   SIMCSE_TEMP          NT-Xent τ (default: 0.05)
#   SIMCSE_DROPOUT       injected attention dropout (default: 0.1) — Gemma-2
#                         default is 0.0, must be >0 or loss collapses
#   SIMCSE_SAVE_STEPS    (default: 500)
#   SIMCSE_MAX_SEQ_LEN   (default: 128)  SimCSE typically uses shorter seqs
#   SIMCSE_GRAD_CKPT     1 → --gradient-checkpointing (default: 0)
#
#   --- evaluation (stage 3) ---
#   N_SENTENCES          held-out sentences for eval 1+2 (default: 500)
#   N_BIDIR_CAUSAL       sentences for eval 3 (default: 100)
#   POOLING              mean | last | weighted_mean (default: mean) —
#                         should match SIMCSE_POOLING
#   SIMCSE_POOLING       pooling used by SimCSE training (default: mean)
#   MTEB_TASKS           space-separated task list (default: "STSBenchmark")
#                         empty string disables eval 5
#   MTEB_BATCH_SIZE      (default: 8)
#   SKIP_MTEB            1 → skip eval 5 (default: 0)
#   BASELINE_LLM         override; "none" to skip base-LLM causal-PPL load
#
#   --- pipeline control ---
#   FORCE_FRESH          1 = wipe RUN_DIR + pass --no-resume to all trainers
#   SKIP_TRAIN           1 = stage 1+2 を両方 skip。SIMCSE_DIR を指定すれば
#                         既存 checkpoint への評価が走る
#   SKIP_SIMCSE          1 = stage 2 を skip。stage 3 は $LLM2VEC_DIR を評価
#
# Per-stage log:  $RUN_DIR/logs/<stage>.log
# Timing summary: $RUN_DIR/timing.tsv
# Final report:   $EVAL_DIR/eval_report.md   (末尾で stdout にも cat される)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RUN_DIR=${RUN_DIR:-"./runs/llm2vec_eval"}
LLM2VEC_DIR=${LLM2VEC_DIR:-"$RUN_DIR/llm2vec"}
SIMCSE_DIR=${SIMCSE_DIR:-"$RUN_DIR/llm2vec_simcse"}
EVAL_DIR=${EVAL_DIR:-"$RUN_DIR/eval"}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

LLM=${LLM:-"google/gemma-2-2b"}
DOLMA_MAX_FILES=${DOLMA_MAX_FILES:-32}
# Canonical LLM2Vec uses LoRA. The defaults below match the McGill-NLP
# train_configs (r=16, alpha=32, MNTP 1k steps @ 3e-4, SimCSE 1k steps @ 3e-5).
# To reproduce the previous full-FT runs: USE_LORA=0 LLM2VEC_LR=1e-5
# LLM2VEC_STEPS=10000 SIMCSE_LR=1e-6 SIMCSE_STEPS=2000.
USE_LORA=${USE_LORA:-1}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LLM2VEC_STEPS=${LLM2VEC_STEPS:-1000}
LLM2VEC_BATCH=${LLM2VEC_BATCH:-8}
LLM2VEC_ACCUM=${LLM2VEC_ACCUM:-4}
LLM2VEC_LR=${LLM2VEC_LR:-3e-4}
LLM2VEC_WARMUP=${LLM2VEC_WARMUP:-100}
LLM2VEC_SAVE_STEPS=${LLM2VEC_SAVE_STEPS:-500}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-256}
MLM_PROBABILITY=${MLM_PROBABILITY:-0.15}

SIMCSE_STEPS=${SIMCSE_STEPS:-1000}
SIMCSE_BATCH=${SIMCSE_BATCH:-128}
SIMCSE_LR=${SIMCSE_LR:-3e-5}
SIMCSE_WARMUP=${SIMCSE_WARMUP:-100}
SIMCSE_TEMP=${SIMCSE_TEMP:-0.05}
SIMCSE_DROPOUT=${SIMCSE_DROPOUT:-0.1}
SIMCSE_SAVE_STEPS=${SIMCSE_SAVE_STEPS:-500}
SIMCSE_MAX_SEQ_LEN=${SIMCSE_MAX_SEQ_LEN:-128}
SIMCSE_POOLING=${SIMCSE_POOLING:-mean}
SIMCSE_GRAD_CKPT=${SIMCSE_GRAD_CKPT:-0}

N_SENTENCES=${N_SENTENCES:-500}
N_BIDIR_CAUSAL=${N_BIDIR_CAUSAL:-100}
POOLING=${POOLING:-mean}
MTEB_TASKS=${MTEB_TASKS:-"STSBenchmark"}
MTEB_BATCH_SIZE=${MTEB_BATCH_SIZE:-8}
SKIP_MTEB=${SKIP_MTEB:-0}

FORCE_FRESH=${FORCE_FRESH:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_SIMCSE=${SKIP_SIMCSE:-0}

# FORCE_FRESH=1: rm -rf RUN_DIR + pass --no-resume to every trainer
RESUME_FLAG=""
if [[ "$FORCE_FRESH" == "1" ]]; then
    if [[ "$SKIP_TRAIN" == "1" ]]; then
        echo "[train-eval] FORCE_FRESH=1 と SKIP_TRAIN=1 を同時指定しても" \
             "学習は skip されます。出力先の中身は wipe しません。" >&2
    else
        if [[ -d "$RUN_DIR" ]]; then
            echo "[train-eval] FORCE_FRESH=1: removing $RUN_DIR"
            rm -rf "$RUN_DIR"
        fi
        RESUME_FLAG="--no-resume"
    fi
fi

# 評価対象: 通常は SimCSE 後。SKIP_SIMCSE=1 なら MNTP 後。
if [[ "$SKIP_SIMCSE" == "1" ]]; then
    EVAL_TARGET_DIR="$LLM2VEC_DIR"
else
    EVAL_TARGET_DIR="$SIMCSE_DIR"
fi

# --------------------------------------------------------------------------- #
# Output layout
# --------------------------------------------------------------------------- #
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# Shared Dolma cache OUTSIDE $RUN_DIR — bytes are identical regardless of
# training config, so re-downloading per-run is pure waste. FORCE_FRESH=1
# does NOT wipe this. Override DOLMA_CACHE explicitly to put it per-run.
SHARED_CACHE_ROOT=${SHARED_CACHE_ROOT:-"./shared_cache"}
DOLMA_CACHE=${DOLMA_CACHE:-"$SHARED_CACHE_ROOT/dolma"}

SUMMARY="$RUN_DIR/timing.tsv"
if [[ ! -f "$SUMMARY" ]]; then
    printf "stage\tstart\tend\telapsed_sec\tstatus\n" > "$SUMMARY"
fi

# --------------------------------------------------------------------------- #
# Helpers (same shape as run_production.sh)
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

run_stage() {
    local name=$1; shift
    local log="$LOG_DIR/${name}.log"
    banner "[train-eval] stage: $name"
    echo "[train-eval] cmd: $*" | tee "$log"
    local start
    start=$(date +%s)
    set +e
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    local end
    end=$(date +%s)
    local elapsed=$((end - start))
    if [[ "$rc" -ne 0 ]]; then
        record "$name" "$start" "$end" "$elapsed" "fail(rc=$rc)"
        printf '[train-eval] %-22s FAILED rc=%d  see %s\n' "$name" "$rc" "$log" >&2
        echo "[train-eval] 同じコマンドを再実行すれば最新 checkpoint から続行します" >&2
        exit "$rc"
    fi
    record "$name" "$start" "$end" "$elapsed" "ok"
    printf '[train-eval] %-22s elapsed=%dh%02dm%02ds  log=%s\n' \
        "$name" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) \
        $((elapsed % 60)) "$log"
}

skip_stage() {
    local name=$1 reason=$2
    record "$name" "" "" 0 "skip"
    printf '[train-eval] %-22s SKIPPED  %s\n' "$name" "$reason"
}

# --------------------------------------------------------------------------- #
# Print config
# --------------------------------------------------------------------------- #
banner "[train-eval] config"
cat <<EOF
RUN_DIR              = $RUN_DIR     (resume by default; FORCE_FRESH=$FORCE_FRESH)
LLM2VEC_DIR          = $LLM2VEC_DIR
SIMCSE_DIR           = $SIMCSE_DIR
EVAL_TARGET_DIR      = $EVAL_TARGET_DIR
EVAL_DIR             = $EVAL_DIR
DEVICE               = $DEVICE        SEED = $SEED
LLM                  = $LLM
DOLMA_MAX_FILES      = $DOLMA_MAX_FILES
LLM2VEC_STEPS / BS   = $LLM2VEC_STEPS / $LLM2VEC_BATCH (accum $LLM2VEC_ACCUM, effective $((LLM2VEC_BATCH * LLM2VEC_ACCUM)))
LLM2VEC_LR / WARMUP  = $LLM2VEC_LR / $LLM2VEC_WARMUP
LLM2VEC_SAVE_STEPS   = $LLM2VEC_SAVE_STEPS
SIMCSE_STEPS / BS    = $SIMCSE_STEPS / $SIMCSE_BATCH
SIMCSE_LR / WARMUP   = $SIMCSE_LR / $SIMCSE_WARMUP
SIMCSE_TEMP / DROP   = $SIMCSE_TEMP / $SIMCSE_DROPOUT
SIMCSE_MAX_SEQ_LEN   = $SIMCSE_MAX_SEQ_LEN
SIMCSE_POOLING       = $SIMCSE_POOLING
SIMCSE_GRAD_CKPT     = $SIMCSE_GRAD_CKPT
NUM_WORKERS          = $NUM_WORKERS
MAX_SEQ_LENGTH       = $MAX_SEQ_LENGTH
MLM_PROBABILITY      = $MLM_PROBABILITY
N_SENTENCES          = $N_SENTENCES
N_BIDIR_CAUSAL       = $N_BIDIR_CAUSAL
POOLING (eval)       = $POOLING
MTEB_TASKS           = $MTEB_TASKS
MTEB_BATCH_SIZE      = $MTEB_BATCH_SIZE
SKIP_MTEB            = $SKIP_MTEB
BASELINE_LLM         = ${BASELINE_LLM:-<llm2vec_meta.json から自動>}
SKIP_TRAIN           = $SKIP_TRAIN
SKIP_SIMCSE          = $SKIP_SIMCSE
EOF

T0=$(date +%s)

# --------------------------------------------------------------------------- #
# Stage 1: train LLM2Vec (MNTP)
# --------------------------------------------------------------------------- #
if [[ "$SKIP_TRAIN" == "1" ]]; then
    skip_stage "01_train_llm2vec" "SKIP_TRAIN=1"
elif [[ -f "$LLM2VEC_DIR/llm2vec_meta.json" ]]; then
    skip_stage "01_train_llm2vec" "exists at $LLM2VEC_DIR (FORCE_FRESH=1 で再学習)"
else
    LLM2VEC_EXTRA=()
    if [[ "$USE_LORA" == "1" ]]; then
        LLM2VEC_EXTRA+=(--use-lora
                        --lora-r "$LORA_R"
                        --lora-alpha "$LORA_ALPHA"
                        --lora-dropout "$LORA_DROPOUT")
    else
        LLM2VEC_EXTRA+=(--no-use-lora)
    fi
    run_stage "01_train_llm2vec" \
        python train_llm2vec.py \
            --llm "$LLM" \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --output-dir "$LLM2VEC_DIR" \
            --max-steps "$LLM2VEC_STEPS" \
            --warmup-steps "$LLM2VEC_WARMUP" \
            --learning-rate "$LLM2VEC_LR" \
            --per-device-batch-size "$LLM2VEC_BATCH" \
            --grad-accum-steps "$LLM2VEC_ACCUM" \
            --save-steps "$LLM2VEC_SAVE_STEPS" \
            --logging-steps 50 \
            --max-seq-length "$MAX_SEQ_LENGTH" \
            --mlm-probability "$MLM_PROBABILITY" \
            --num-workers "$NUM_WORKERS" \
            --seed "$SEED" \
            "${LLM2VEC_EXTRA[@]}" \
            $RESUME_FLAG
fi

# --------------------------------------------------------------------------- #
# Stage 2: train SimCSE (unsupervised contrastive on top of MNTP)
# --------------------------------------------------------------------------- #
if [[ "$SKIP_SIMCSE" == "1" ]]; then
    skip_stage "02_train_simcse" "SKIP_SIMCSE=1 — eval will target MNTP dir"
elif [[ "$SKIP_TRAIN" == "1" ]]; then
    skip_stage "02_train_simcse" "SKIP_TRAIN=1"
elif [[ -f "$SIMCSE_DIR/llm2vec_meta.json" ]] && \
     [[ "$(python -c "import json,sys; print('simcse' in json.load(open('$SIMCSE_DIR/llm2vec_meta.json')))" 2>/dev/null)" == "True" ]]; then
    skip_stage "02_train_simcse" "exists at $SIMCSE_DIR (FORCE_FRESH=1 で再学習)"
else
    if [[ ! -f "$LLM2VEC_DIR/llm2vec_meta.json" ]]; then
        echo "[train-eval] stage 02 requires a completed MNTP checkpoint at $LLM2VEC_DIR" >&2
        exit 1
    fi
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
    run_stage "02_train_simcse" \
        python train_simcse.py \
            --llm2vec-dir "$LLM2VEC_DIR" \
            --output-dir "$SIMCSE_DIR" \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --max-steps "$SIMCSE_STEPS" \
            --warmup-steps "$SIMCSE_WARMUP" \
            --learning-rate "$SIMCSE_LR" \
            --per-device-batch-size "$SIMCSE_BATCH" \
            --temperature "$SIMCSE_TEMP" \
            --dropout "$SIMCSE_DROPOUT" \
            --pooling "$SIMCSE_POOLING" \
            --max-seq-length "$SIMCSE_MAX_SEQ_LEN" \
            --save-steps "$SIMCSE_SAVE_STEPS" \
            --logging-steps 50 \
            --num-workers "$NUM_WORKERS" \
            --device "$DEVICE" \
            --seed "$SEED" \
            "${SIMCSE_EXTRA[@]}" \
            $RESUME_FLAG
fi

# Sanity-check eval target
if [[ ! -f "$EVAL_TARGET_DIR/llm2vec_meta.json" ]]; then
    echo "[train-eval] eval target $EVAL_TARGET_DIR/llm2vec_meta.json が見つかりません" >&2
    if [[ "$SKIP_TRAIN" == "1" ]]; then
        echo "[train-eval] SKIP_TRAIN=1 のときは LLM2VEC_DIR / SIMCSE_DIR を学習済み dir に向けてください" >&2
    fi
    exit 1
fi

# --------------------------------------------------------------------------- #
# Stage 3: evaluate (always re-runs; eval is cheap and the report is overwritten)
# --------------------------------------------------------------------------- #
EVAL_EXTRA=()
if [[ -n "${BASELINE_LLM:-}" ]]; then
    EVAL_EXTRA+=(--baseline-llm "$BASELINE_LLM")
fi
if [[ "$SKIP_MTEB" == "1" ]]; then
    EVAL_EXTRA+=(--skip-mteb)
fi

mkdir -p "$EVAL_DIR"

# MTEB_TASKS is intentionally word-split for multi-task input.
# shellcheck disable=SC2086
run_stage "03_eval_llm2vec" \
    python eval_llm2vec.py \
        --llm2vec-dir "$EVAL_TARGET_DIR" \
        --output-dir "$EVAL_DIR" \
        --data-cache-dir "$DOLMA_CACHE" \
        --max-files 1 \
        --n-sentences "$N_SENTENCES" \
        --n-bidir-causal "$N_BIDIR_CAUSAL" \
        --max-seq-length "$MAX_SEQ_LENGTH" \
        --mlm-probability "$MLM_PROBABILITY" \
        --pooling "$POOLING" \
        --mteb-tasks $MTEB_TASKS \
        --mteb-batch-size "$MTEB_BATCH_SIZE" \
        --device "$DEVICE" \
        --seed "$SEED" \
        "${EVAL_EXTRA[@]}"

# --------------------------------------------------------------------------- #
# Wrap up
# --------------------------------------------------------------------------- #
T1=$(date +%s)
TOTAL=$((T1 - T0))

banner "[train-eval] eval_report.md"
if [[ -f "$EVAL_DIR/eval_report.md" ]]; then
    cat "$EVAL_DIR/eval_report.md"
else
    echo "(eval_report.md not found at $EVAL_DIR — eval failed?)"
fi

banner "[train-eval] DONE"
printf 'Total wall time this invocation: %dh%02dm%02ds\n' \
    $((TOTAL / 3600)) $(((TOTAL % 3600) / 60)) $((TOTAL % 60))
echo
echo "Artifacts:"
echo "  MNTP ckpt        : $LLM2VEC_DIR"
echo "  SimCSE ckpt      : $SIMCSE_DIR"
echo "  Eval target      : $EVAL_TARGET_DIR"
echo "  Metrics JSON     : $EVAL_DIR/eval_metrics.json"
echo "  Report MD        : $EVAL_DIR/eval_report.md"
echo "  Per-stage logs   : $LOG_DIR/"
echo "  Timing summary   : $SUMMARY"
echo
echo "学習効果のクイック判定 (詳細は eval_report.md):"
echo "  - eval 1 MNTP loss が <8 なら MNTP/SimCSE は走った (random init ≈ 12)"
echo "  - eval 2 causal-PPL drift が +5〜+30% 程度なら causal LM の挙動は健全"
echo "  - eval 3 mean cosine が <0.999 なら bidir patch が効いている"
echo "  - eval 5 STSBenchmark Spearman が >0.75 なら canonical 帯 (Bi+MNTP+SimCSE)"
echo "    (Bi+MNTP のみだと <0.65 が普通; SimCSE が乗ると 0.1+ 上がる)"
