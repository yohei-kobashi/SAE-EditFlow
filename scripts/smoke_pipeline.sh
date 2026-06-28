#!/usr/bin/env bash
# scripts/smoke_pipeline.sh
#
# Runs the full SAE-LEWIS pipeline end-to-end at SMALL SCALE so each stage's
# per-step wall time can be measured. Per-stage outputs are kept; if you run
# this script again with the same RUN_DIR, completed stages are skipped.
#
# At the end, the script extrapolates per-stage time to the production-scale
# step / sample counts encoded in the argparse defaults of each training
# script, so you can read off a rough estimate of a real run's duration.
#
# Usage:
#   bash scripts/smoke_pipeline.sh
#   RUN_DIR=./runs/myrun bash scripts/smoke_pipeline.sh
#   LLM2VEC_STEPS=500 BATCH_SIZE=8 bash scripts/smoke_pipeline.sh
#
# Environment variables (override any of these):
#   RUN_DIR                 output root (default: ./runs/smoke-<ts>)
#   DEVICE                  cuda | cpu (default: cuda)
#   SEED                    (default: 42)
#   DOLMA_MAX_FILES         # of Dolma shards to download (default: 1)
#   SAE_MAX_SENTS           cap on stage-0 SAE cache size (default: 2000)
#   LLM2VEC_STEPS           MNTP steps (default: 200)
#   LLM2VEC_BATCH           per-device batch for MNTP (default: 2)
#   CORRUPTION_SAMPLES      stage-2 target sample count (default: 1000)
#   CORRUPTION_SHARD        samples per corruption shard (default: 500)
#   TAGGER_STEPS            (default: 200)
#   EDITOR_STEPS            (default: 200)
#   LENGTH_STEPS            (default: 200)
#   BATCH_SIZE              tagger / editor / length head batch (default: 4)
#   NUM_WORKERS             data loader workers (default: 4)
#   EVAL_TEXT               text for stage-6 inference (default: a single sentence)
#   EVAL_SPEC               intervention spec for stage-6 (default: "+1234")
#   LLM                     causal LLM HF id        (default: google/gemma-2-2b)
#   SAE_REPO                Gemma Scope HF repo     (default: google/gemma-scope-2b-pt-res)
#   SAE_PATH                file inside SAE_REPO    (default: layer_12/width_16k/average_l0_82/params.npz)
#                            For 2b-pt-res layer_12/width_16k the available L0 values
#                            are {22, 41, 82, 176, 445}; pick a different file by
#                            overriding this var.
#   SAE_LAYER               (default: 12)
#   MLM_MODEL               corruption MLM preset   (default: modernbert-base)
#   SPACY_MODEL             spaCy model for POS tagging used by INS position
#                            selection (default: en_core_web_sm). UPOS tags are
#                            language-agnostic, so ja_core_news_sm / etc. work.
#   CALIBRATION             1 → run corruption.py in --calibration-mode
#                            (records (N, ppl_ratio, sae_shift) without applying
#                            the gate; useful for fitting ppl_per_op_factor /
#                            sae_per_op_min / sae_per_op_max).
#
# Each stage's stdout/stderr is captured under $RUN_DIR/logs/<stage>.log.
# Per-stage wall time and status land in $RUN_DIR/timing.tsv.
# Per-stage GPU memory peaks (sampled via `nvidia-smi` every 1s while the
# stage is running) land in $RUN_DIR/gpu.tsv and are used to derive a
# recommended batch size at the end of the run.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Locate repo root regardless of where the script is invoked from
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RUN_DIR=${RUN_DIR:-"./runs/smoke-$(date +%Y%m%d-%H%M%S)"}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

DOLMA_MAX_FILES=${DOLMA_MAX_FILES:-1}
SAE_MAX_SENTS=${SAE_MAX_SENTS:-2000}
LLM2VEC_STEPS=${LLM2VEC_STEPS:-200}
LLM2VEC_BATCH=${LLM2VEC_BATCH:-2}
CORRUPTION_SAMPLES=${CORRUPTION_SAMPLES:-1000}
CORRUPTION_SHARD=${CORRUPTION_SHARD:-500}
TAGGER_STEPS=${TAGGER_STEPS:-200}
EDITOR_STEPS=${EDITOR_STEPS:-200}
LENGTH_STEPS=${LENGTH_STEPS:-200}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}

EVAL_TEXT=${EVAL_TEXT:-"The quick brown fox jumps over the lazy dog."}
EVAL_SPEC=${EVAL_SPEC:-"+1234"}

LLM=${LLM:-"google/gemma-2-2b"}
SAE_REPO=${SAE_REPO:-"google/gemma-scope-2b-pt-res"}
SAE_PATH=${SAE_PATH:-"layer_12/width_16k/average_l0_82/params.npz"}
SAE_LAYER=${SAE_LAYER:-12}
MLM_MODEL=${MLM_MODEL:-"modernbert-base"}
SPACY_MODEL=${SPACY_MODEL:-"en_core_web_sm"}
CALIBRATION=${CALIBRATION:-0}

# Production-scale defaults baked into the training scripts' argparses.
# These are used only for the extrapolation table at the end.
PROD_LLM2VEC_STEPS=10000
PROD_LLM2VEC_BATCH=8
PROD_LLM2VEC_ACCUM=4
PROD_CORRUPTION_SAMPLES=100000
PROD_TAGGER_STEPS=10000
PROD_TAGGER_BATCH=8
PROD_EDITOR_STEPS=20000
PROD_EDITOR_BATCH=8
PROD_LENGTH_STEPS=5000
PROD_LENGTH_BATCH=8

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

# Decide whether GPU monitoring is possible. Honour CUDA_VISIBLE_DEVICES so we
# sample the GPU the training process is actually using (first one if a list).
GPU_AVAILABLE=0
if [[ "${DEVICE,,}" == cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_AVAILABLE=1
fi
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
GPU_MON_PID=
GPU_MON_FILE=
trap 'if [[ -n "${GPU_MON_PID:-}" ]]; then kill "$GPU_MON_PID" 2>/dev/null || true; fi' EXIT

DOLMA_CACHE="$RUN_DIR/dolma_cache"
SAE_CACHE="$RUN_DIR/sae_cache"
LLM2VEC_DIR="$RUN_DIR/llm2vec"
CORRUPTION_DIR="$RUN_DIR/corruption"
TAGGER_DIR="$RUN_DIR/tagger"
EDITOR_DIR="$RUN_DIR/editor"
LENGTH_DIR="$RUN_DIR/length_head"

TAGGER_CKPT="$TAGGER_DIR/tagger-final.pt"
EDITOR_CKPT="$EDITOR_DIR/editor-final.pt"
LENGTH_CKPT="$LENGTH_DIR/length-final.pt"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
banner() {
    printf '\n===============================================================\n'
    printf ' %s\n' "$1"
    printf '===============================================================\n'
}

# Append elapsed seconds for `stage` to the timing TSV. Status is "ok" / "skip"
# / "fail(rc=N)". `start`/`end` are unix seconds (or empty when skipped).
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
            sleep 1
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
    banner "[smoke] stage: $name"
    echo "[smoke] cmd: $*" | tee "$log"
    local start
    start=$(date +%s)
    start_gpu_monitor "$name"
    # Run the python command, capturing stdout/stderr to both terminal and log.
    # Temporarily disable `set -e` so the script doesn't exit before we read
    # PIPESTATUS. We must NOT chain "|| true" here — that would replace
    # PIPESTATUS with `true`'s status (0) and we'd record every failure as ok.
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
        printf '[smoke] %-22s FAILED rc=%d  see %s\n' "$name" "$rc" "$log" >&2
        exit "$rc"
    fi
    record "$name" "$start" "$end" "$elapsed" "ok"
    printf '[smoke] %-22s elapsed=%dm%02ds  log=%s\n' \
        "$name" $((elapsed / 60)) $((elapsed % 60)) "$log"
}

skip_stage() {
    local name=$1 reason=$2
    record "$name" "" "" 0 "skip"
    printf '[smoke] %-22s SKIPPED  %s\n' "$name" "$reason"
}

# --------------------------------------------------------------------------- #
# Print config
# --------------------------------------------------------------------------- #
banner "[smoke] config"
cat <<EOF
RUN_DIR              = $RUN_DIR
DEVICE               = $DEVICE        SEED = $SEED
DOLMA_MAX_FILES      = $DOLMA_MAX_FILES
SAE_MAX_SENTS        = $SAE_MAX_SENTS
LLM2VEC_STEPS / BS   = $LLM2VEC_STEPS / $LLM2VEC_BATCH
CORRUPTION samples   = $CORRUPTION_SAMPLES   (shard size $CORRUPTION_SHARD)
TAGGER_STEPS         = $TAGGER_STEPS
EDITOR_STEPS         = $EDITOR_STEPS
LENGTH_STEPS         = $LENGTH_STEPS
BATCH_SIZE           = $BATCH_SIZE
NUM_WORKERS          = $NUM_WORKERS
EVAL_TEXT            = $EVAL_TEXT
EVAL_SPEC            = $EVAL_SPEC
LLM                  = $LLM
SAE_REPO             = $SAE_REPO
SAE_PATH             = $SAE_PATH
SAE_LAYER            = $SAE_LAYER
MLM_MODEL            = $MLM_MODEL
SPACY_MODEL          = $SPACY_MODEL
CALIBRATION          = $CALIBRATION
EOF

# --------------------------------------------------------------------------- #
# Stage 0: SAE cache
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
            --batch-size 16 \
            --llm "$LLM" \
            --sae-repo "$SAE_REPO" \
            --sae-path "$SAE_PATH" \
            --sae-layer "$SAE_LAYER" \
            --device "$DEVICE" \
            --seed "$SEED"
fi

# --------------------------------------------------------------------------- #
# Stage 1: LLM2Vec MNTP
# --------------------------------------------------------------------------- #
if [[ -f "$LLM2VEC_DIR/llm2vec_meta.json" ]]; then
    skip_stage "01_train_llm2vec" "exists at $LLM2VEC_DIR"
else
    run_stage "01_train_llm2vec" \
        python train_llm2vec.py \
            --llm "$LLM" \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --output-dir "$LLM2VEC_DIR" \
            --max-steps "$LLM2VEC_STEPS" \
            --warmup-steps 20 \
            --per-device-batch-size "$LLM2VEC_BATCH" \
            --grad-accum-steps 1 \
            --save-steps "$LLM2VEC_STEPS" \
            --logging-steps 20 \
            --num-workers "$NUM_WORKERS" \
            --seed "$SEED"
fi

# --------------------------------------------------------------------------- #
# Stage 2: corruption
# --------------------------------------------------------------------------- #
CORRUPTION_EXTRA_ARGS=()
if [[ "$CALIBRATION" == "1" ]]; then
    CORRUPTION_EXTRA_ARGS+=(--calibration-mode
                            --calibration-out "$CORRUPTION_DIR/calibration.jsonl")
fi

if [[ -f "$CORRUPTION_DIR/meta.json" ]]; then
    skip_stage "02_corruption" "exists at $CORRUPTION_DIR"
else
    run_stage "02_corruption" \
        python corruption.py \
            --data-cache-dir "$DOLMA_CACHE" \
            --max-files "$DOLMA_MAX_FILES" \
            --out-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$LLM2VEC_DIR" \
            --llm "$LLM" \
            --sae-repo "$SAE_REPO" \
            --sae-path "$SAE_PATH" \
            --sae-layer "$SAE_LAYER" \
            --mlm-model "$MLM_MODEL" \
            --spacy-model "$SPACY_MODEL" \
            --target-samples "$CORRUPTION_SAMPLES" \
            --samples-per-shard "$CORRUPTION_SHARD" \
            --device "$DEVICE" \
            --seed "$SEED" \
            "${CORRUPTION_EXTRA_ARGS[@]}"
fi

# --------------------------------------------------------------------------- #
# Stage 3: tagger
# --------------------------------------------------------------------------- #
if [[ -f "$TAGGER_CKPT" ]]; then
    skip_stage "03_train_tagger" "exists at $TAGGER_CKPT"
else
    run_stage "03_train_tagger" \
        python train_tagger.py \
            --corruption-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$LLM2VEC_DIR" \
            --output-dir "$TAGGER_DIR" \
            --max-steps "$TAGGER_STEPS" \
            --warmup-steps 20 \
            --proj-a-freeze-steps 50 \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --save-steps "$TAGGER_STEPS" \
            --logging-steps 20 \
            --estimate-class-weights-batches 50 \
            --device "$DEVICE" \
            --seed "$SEED"
fi

# --------------------------------------------------------------------------- #
# Stage 4: editor phase A
# --------------------------------------------------------------------------- #
if [[ -f "$EDITOR_CKPT" ]]; then
    skip_stage "04_train_editor_phaseA" "exists at $EDITOR_CKPT"
else
    run_stage "04_train_editor_phaseA" \
        python train_editor_phaseA.py \
            --corruption-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$LLM2VEC_DIR" \
            --output-dir "$EDITOR_DIR" \
            --max-steps "$EDITOR_STEPS" \
            --warmup-steps 20 \
            --proj-a-freeze-steps 50 \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --save-steps "$EDITOR_STEPS" \
            --logging-steps 20 \
            --device "$DEVICE" \
            --seed "$SEED"
fi

# --------------------------------------------------------------------------- #
# Stage 5: length head (ablation)
# --------------------------------------------------------------------------- #
if [[ -f "$LENGTH_CKPT" ]]; then
    skip_stage "05_train_length_head" "exists at $LENGTH_CKPT"
else
    run_stage "05_train_length_head" \
        python train_length_head.py \
            --corruption-dir "$CORRUPTION_DIR" \
            --llm2vec-dir "$LLM2VEC_DIR" \
            --editor-ckpt "$EDITOR_CKPT" \
            --output-dir "$LENGTH_DIR" \
            --max-steps "$LENGTH_STEPS" \
            --warmup-steps 20 \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --save-steps "$LENGTH_STEPS" \
            --logging-steps 20 \
            --device "$DEVICE" \
            --seed "$SEED"
fi

# --------------------------------------------------------------------------- #
# Stage 6: evaluate (single example, just to time the inference path)
# --------------------------------------------------------------------------- #
run_stage "06_evaluate" \
    python evaluate_intervention.py \
        --llm2vec-dir "$LLM2VEC_DIR" \
        --tagger-ckpt "$TAGGER_CKPT" \
        --editor-ckpt "$EDITOR_CKPT" \
        --mu "$SAE_CACHE/mu.npy" \
        --llm "$LLM" \
        --sae-repo "$SAE_REPO" \
        --sae-path "$SAE_PATH" \
        --sae-layer "$SAE_LAYER" \
        --text "$EVAL_TEXT" \
        --spec $EVAL_SPEC \
        --l-max 5 \
        --device "$DEVICE"

# --------------------------------------------------------------------------- #
# Timing summary + production-scale extrapolation
# --------------------------------------------------------------------------- #
banner "[smoke] timing summary (this run)"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY"
else
    cat "$SUMMARY"
fi

# Sum elapsed across all rows
total_sec=$(awk -F'\t' 'NR>1 {sum+=$4} END {print sum+0}' "$SUMMARY")
printf '\n[smoke] total wall time this run: %dm%02ds\n' \
    $((total_sec / 60)) $((total_sec % 60))

# Helper: read a stage's elapsed seconds from $SUMMARY (0 if missing or skipped)
elapsed_of() {
    awk -F'\t' -v s="$1" '$1==s && $5=="ok" {print $4; found=1} END {if (!found) print 0}' \
        "$SUMMARY"
}

# Helper: print "stage  small_sec → prod  h:m"
extrap_row() {
    local name=$1 small_sec=$2 ratio=$3
    awk -v n="$name" -v s="$small_sec" -v r="$ratio" 'BEGIN {
        prod = s * r
        printf "  %-22s  smoke=%6ds  ×%-7.2f  → prod ≈ %02dh%02dm\n",
            n, s, r, int(prod/3600), int((prod%3600)/60)
    }'
}

banner "[smoke] linear extrapolation to production scale"
cat <<EOF
Production-scale defaults assumed:
  LLM2Vec MNTP   :  $PROD_LLM2VEC_STEPS steps × batch $PROD_LLM2VEC_BATCH × accum $PROD_LLM2VEC_ACCUM
  Corruption     :  $PROD_CORRUPTION_SAMPLES samples
  Tagger         :  $PROD_TAGGER_STEPS steps × batch $PROD_TAGGER_BATCH
  Editor (Phase A):  $PROD_EDITOR_STEPS steps × batch $PROD_EDITOR_BATCH
  Length head    :  $PROD_LENGTH_STEPS steps × batch $PROD_LENGTH_BATCH

Extrapolation assumes per-step / per-sample time is independent of step
or batch size — this is approximate; larger batches usually run a bit
sub-linearly, and rejection-sampled corruption depends on yield.
EOF
echo

llm2vec_sec=$(elapsed_of "01_train_llm2vec")
corr_sec=$(elapsed_of    "02_corruption")
tag_sec=$(elapsed_of     "03_train_tagger")
ed_sec=$(elapsed_of      "04_train_editor_phaseA")
len_sec=$(elapsed_of     "05_train_length_head")

llm2vec_ratio=$(awk -v ps="$PROD_LLM2VEC_STEPS" -v pb="$PROD_LLM2VEC_BATCH" \
                    -v pa="$PROD_LLM2VEC_ACCUM" \
                    -v ss="$LLM2VEC_STEPS"     -v sb="$LLM2VEC_BATCH" \
    'BEGIN { print (ps*pb*pa)/(ss*sb*1) }')
corr_ratio=$(awk    -v p="$PROD_CORRUPTION_SAMPLES" -v s="$CORRUPTION_SAMPLES" \
    'BEGIN { print p/s }')
tag_ratio=$(awk     -v ps="$PROD_TAGGER_STEPS"  -v pb="$PROD_TAGGER_BATCH" \
                    -v ss="$TAGGER_STEPS"       -v sb="$BATCH_SIZE" \
    'BEGIN { print (ps*pb)/(ss*sb) }')
ed_ratio=$(awk      -v ps="$PROD_EDITOR_STEPS"  -v pb="$PROD_EDITOR_BATCH" \
                    -v ss="$EDITOR_STEPS"       -v sb="$BATCH_SIZE" \
    'BEGIN { print (ps*pb)/(ss*sb) }')
len_ratio=$(awk     -v ps="$PROD_LENGTH_STEPS"  -v pb="$PROD_LENGTH_BATCH" \
                    -v ss="$LENGTH_STEPS"       -v sb="$BATCH_SIZE" \
    'BEGIN { print (ps*pb)/(ss*sb) }')

extrap_row "01_train_llm2vec"      "$llm2vec_sec" "$llm2vec_ratio"
extrap_row "02_corruption"         "$corr_sec"    "$corr_ratio"
extrap_row "03_train_tagger"       "$tag_sec"     "$tag_ratio"
extrap_row "04_train_editor_phaseA" "$ed_sec"     "$ed_ratio"
extrap_row "05_train_length_head"  "$len_sec"     "$len_ratio"

prod_total=$(awk -v a="$llm2vec_sec" -v ar="$llm2vec_ratio" \
                 -v b="$corr_sec"    -v br="$corr_ratio" \
                 -v c="$tag_sec"     -v cr="$tag_ratio" \
                 -v d="$ed_sec"      -v dr="$ed_ratio" \
                 -v e="$len_sec"     -v er="$len_ratio" \
    'BEGIN { print int(a*ar + b*br + c*cr + d*dr + e*er) }')
printf '\n[smoke] training stages (1..5) at prod scale: ≈ %dh%02dm  (%d sec)\n' \
    $((prod_total / 3600)) $(((prod_total % 3600) / 60)) "$prod_total"

cat <<EOF

Notes:
  - Stage 0 (precompute_sae) is NOT included in the prod estimate because
    its scaling depends on how many Dolma sentences you cache; multiply its
    smoke wall time by (target_sentences / $SAE_MAX_SENTS).
  - Stage 2 (corruption) is rejection-sampled. The estimate assumes the
    accept rate stays similar; check meta.json["bucket_yields"] in the
    smoke run before trusting the extrapolation.
  - Per-step time can grow super-linearly with batch size (cache pressure
    on small GPUs) or sub-linearly (idle CUDA cores on big GPUs). Treat
    the numbers above as an order-of-magnitude estimate, not an SLA.
EOF

# --------------------------------------------------------------------------- #
# GPU memory profile + batch-size recommendation
# --------------------------------------------------------------------------- #
# Only show this block if at least one stage produced GPU samples.
if [[ "$GPU_AVAILABLE" -eq 1 ]] && [[ "$(wc -l < "$GPU_TSV")" -gt 1 ]]; then
    banner "[smoke] GPU memory peaks (per stage)"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$GPU_TSV"
    else
        cat "$GPU_TSV"
    fi

    # Look up a stage's recorded peak / total (0 if missing).
    gpu_field_of() {
        local stage=$1 field=$2
        awk -F'\t' -v s="$stage" -v f="$field" \
            '$1==s {print $f; found=1} END {if (!found) print 0}' "$GPU_TSV"
    }

    # B_rec = floor(B_smoke * (Total / Peak) * SAFETY), floor at 1.
    # SAFETY leaves headroom for fragmentation, larger sequences, eval-time
    # spikes, and the activation-vs-fixed-cost mismatch of this heuristic.
    SAFETY=0.85

    recommend_batch() {
        local stage=$1 b_smoke=$2 label=$3
        local peak total
        peak=$(gpu_field_of "$stage" 2)
        total=$(gpu_field_of "$stage" 3)
        if [[ "$peak" -le 0 || "$total" -le 0 ]]; then
            printf '  %-26s  (no GPU samples)\n' "$label"
            return
        fi
        awk -v lab="$label" -v b_smoke="$b_smoke" \
            -v peak="$peak" -v total="$total" -v safety="$SAFETY" 'BEGIN {
            headroom = total / peak
            b_rec = int(b_smoke * headroom * safety)
            if (b_rec < 1) b_rec = 1
            pct = 100.0 * peak / total
            printf "  %-26s  smoke B=%-3d  peak %5dMiB / %5dMiB (%4.1f%%)  →  recommended B≈%d\n",
                lab, b_smoke, peak, total, pct, b_rec
        }'
    }

    banner "[smoke] recommended batch sizes for production"
    recommend_batch "01_train_llm2vec"       "$LLM2VEC_BATCH" "LLM2Vec MNTP (per-device)"
    recommend_batch "03_train_tagger"        "$BATCH_SIZE"    "Tagger"
    recommend_batch "04_train_editor_phaseA" "$BATCH_SIZE"    "Editor (Phase A)"
    recommend_batch "05_train_length_head"   "$BATCH_SIZE"    "Length head"

    cat <<'NOTE'

How the recommendation is derived:
  B_rec = floor(B_smoke × (Total / Peak) × 0.85)

  This treats the smoke-batch peak as if it were all activation memory. In
  reality, model weights + optimizer state + KV cache are a fixed cost that
  doesn't scale with batch, so the *true* maximum batch is usually higher
  than B_rec. Treat B_rec as a safe starting point — try doubling it and
  watch nvidia-smi before committing to a longer run.

Other guidance:
  - LLM2Vec: effective batch = per-device-batch × grad-accum-steps. If a
    larger per-device batch fits, you can keep effective batch constant by
    lowering --grad-accum-steps proportionally (fewer accum steps → faster).
  - Multi-GPU DDP: divide B_rec by world size for the per-device setting.
  - If average GPU utilisation (avg_util_pct column) is well below ~70%,
    the bottleneck is data loading or CPU pre-processing, not GPU compute.
    In that case, raise --num-workers before raising batch size — extra
    memory headroom won't help if the GPU sits idle waiting for batches.
NOTE
fi

cat <<EOF

Artifacts:
  Run dir         : $RUN_DIR
  Per-stage logs  : $LOG_DIR/
  Timing TSV      : $SUMMARY
  GPU samples TSV : $GPU_TSV   (peak/total/util per stage)
  Per-stage GPU traces: $LOG_DIR/gpu_<stage>.tsv  (1-Hz nvidia-smi samples)
EOF
