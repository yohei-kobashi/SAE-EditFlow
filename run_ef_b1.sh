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

# B1 baseline (claim C1): LinguaLens direct SAE intervention, reproduced
# faithfully from THU-KEG/LinguaLens + OpenSAE code on OUR backbone
# (gemma-2-2b-it + Gemma Scope layer-12/16k): "set" intervention with
# force-insert, reconstruction REPLACEMENT of the residual at every
# position of every step (prompt_only=False), control = reconstruction
# passthrough (their multiply-x1) = our empty `recon` mode; `raw` mode
# isolates reconstruction damage. Clamp sweep {5,10,20} (their value:
# 10, Llama scale) + task-native clampZ (commanded magnitudes).
# Enhancement (z_amp) and ablation-to-0 (z_sup) apply simultaneously.
# ~2-3h (7 generations x 500 pairs); resumes per pair.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
PIPE=$V6/eval_lingualens_final/records.jsonl
OUT=$V6/clamp_baseline500

if [ ! -f "$OUT/report.md" ]; then
    python scripts/eval_clamp_baseline.py \
        --llm2vec-dir "$LLM2VEC" \
        --blocklist "$BLOCKLIST" \
        --output-dir "$OUT" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --clamp-values 5,10,20 \
        --device cuda
fi

echo "==================== B1 DONE ===================="
cat "$OUT/report.md"
echo
echo "-------- matched-pair vs pipeline (+ 300-pair holdout) --------"
python scripts/compare_ef_pipeline.py \
    --ef "$OUT/records.jsonl" --pipeline "$PIPE"
python scripts/compare_ef_pipeline.py \
    --ef "$OUT/records.jsonl" --pipeline "$PIPE" \
    --exclude "$V6/editflow_s3/probe/records.jsonl"
echo
echo "Reading (claim C1 — same 500 pairs as S4/M1/B2):"
echo "  EF S3 thr0.1 = 0.1904/0.6192; B2 prompt = 0.1242/0.6118;"
echo "  pipeline = 0.1102/0.6681; input-copy sim = 0.6116."
echo "  - Expected: clamp exact ~ 0 with degraded sim — the continuous"
echo "    bias cannot express a targeted minimal edit (C1's premise)."
echo "  - `recon` (empty) copy rate reads reconstruction damage: if the"
echo "    rewriter can no longer copy through the SAE reconstruction,"
echo "    report it — it bounds what ANY clamp variant can preserve."
echo "  - The LinguaLens-basis judgment is the fair half: run"
echo "    run_ef_frr.sh afterwards (B1 rows included) — their method"
echo "    may realize features directionally (their 48-72% regime)"
echo "    while failing exact; C1 is judged on exact AND FRR jointly."
echo "  - clampZ vs clamp{5,10,20}: whether a task-informed magnitude"
echo "    rescues the method (best-shot fairness)."
