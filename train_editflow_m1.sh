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

# M1 (EDIT_FLOWS_ZERO §5): MOVE op on the S3 champion. Content-identical
# DEL/INS run pairs are reinterpreted as single MOV ops (V7 A2 gold rule):
# a 4th rate channel fires at the source token and an insert-after pointer
# picks the destination — one firing replaces the DEL-fire x INS-fire x
# Q-regenerates-content product that made reorderings a structural zero
# (headroom pre-measurement: champion 0/12 on strict MOVE pairs at every
# decode mode; +0.024 total exact upper bound, +0.048 in the 2-3 bucket).
# Warm-start from S3 is effectively exact: shared lam_head rows copy over,
# the MOV row inits at bias -6 (p~0.0025), pointer heads start fresh.
# Probe runs at 500 pairs (the S4/pipeline sample) so the MOVE gate is
# judged on the same 12 pairs the pre-measurement flagged.
# ~7h total; both stages resume — resubmit until "M1 DONE".
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
PIPE=$V6/eval_lingualens_final/records.jsonl
OUT=$V6/editflow_m1

if [ ! -f "$OUT/editflow-final.pt" ]; then
    python train_editflow.py \
        --corruption-dir "$V4/corruption" \
        --dev-corruption-dir "$V4/corruption_seldev" \
        --llm2vec-dir "$LLM2VEC" \
        --output-dir "$OUT" \
        --init-from-editflow "$V6/editflow_s3/editflow-final.pt" \
        --rate-param hazard \
        --cond-mode feature-tokens \
        --true-align \
        --lora-r 32 \
        --lam-prop 4.0 \
        --move-ops \
        --max-steps 50000 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --dev-batches 96 --eval-steps 4000 \
        --batch-size 8 --num-workers 2 \
        --resume --device cuda
fi

if [ ! -f "$OUT/probe500/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$OUT/editflow-final.pt" \
        --output-dir "$OUT/probe500" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --steps 48 --steer-lambda 1 \
        --decode det,thr0.05,thr0.1,thr0.25,thr0.5 \
        --device cuda
fi

echo "==================== M1 DONE ===================="
sed -n '/## Gate (a)/,$p' "$OUT/probe500/probe_report.md" | head -70
echo
echo "-------- MOVE gate (the point of M1) --------"
python scripts/measure_move_headroom.py \
    --llm2vec-dir "$LLM2VEC" \
    --records "$OUT/probe500/records.jsonl"
echo
echo "-------- matched-pair vs pipeline (+ 300-pair holdout) --------"
python scripts/compare_ef_pipeline.py \
    --ef "$OUT/probe500/records.jsonl" --pipeline "$PIPE"
python scripts/compare_ef_pipeline.py \
    --ef "$OUT/probe500/records.jsonl" --pipeline "$PIPE" \
    --exclude "$V6/editflow_s3/probe/records.jsonl"
echo
echo "Gates vs S3 champion on the same 500 pairs (thr0.1 0.1904/0.6192,"
echo "  thr0.5 0.1683/0.6533, empty 1.00, random 0.8677@thr0.1,"
echo "  2-3 exact 0.2095, 4-8 0.0678, lambda-IoU 0.7449):"
echo "  (i) THE POINT — strict MOVE pairs (12 on this sample): exact > 0,"
echo "      realistic target >= 4/12 (headroom table above);"
echo "  (ii) headline thr0.1 >= 0.1904 (no regression from the MOV"
echo "      channel or its rate mass);"
echo "  (iii) empty no_edit 1.00 all F; random >= 0.87 at the operating"
echo "      point; (iv) 2-3 exact >= 0.23 (capture >= half the in-bucket"
echo "      headroom); lambda-IoU >= 0.74."
echo "  Pass -> MOVE enters the production op set + paper claims 'inter-"
echo "  token relation manipulation' with direct evidence. Fail on (i)"
echo "  with (ii)-(iii) held -> pointer did not learn from cache"
echo "  reordering families; record and keep S3 as champion."
