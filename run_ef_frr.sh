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

# LinguaLens-basis evaluation (FRR — judged feature realization) over the
# EXISTING outputs of every system on the same 500 pairs. LinguaLens
# evaluates interventions by LLM-judged feature-prominence direction
# (GPT-4o in the paper), not exact match; this is the fairness metric
# their method (B1) and the prompt baseline can score on. Judge here =
# local gemma-2-9b-it for the pilot read; for the paper table set
# JUDGE=openai:gpt-4o and export OPENAI_API_KEY, then rerun — judgments
# and the gold cache are PER-JUDGE (separate directories).
# All judgments resume per pair; ~2500 short judge calls total (~1h).
V6=./runs/prod_gemma_v6
JUDGE=${JUDGE:-hf:google/gemma-2-9b-it}
TAG=$(echo "$JUDGE" | tr ':/' '__')
FRR=runs/frr/$TAG
GOLD=$FRR/gold.jsonl

NREF=$V6/editflow_s3/probe500/records.jsonl   # n_ops source for buckets

run_frr () {  # label records mode condition
    echo "-------- FRR: $1 (condition $4) --------"
    python scripts/judge_feature_realization.py \
        --records "$2" --mode "$3" --condition "$4" \
        --gold-cache "$GOLD" --judge "$JUDGE" \
        --n-ops-ref "$NREF" \
        --out "$FRR/$1.jsonl" --device cuda
}

# operating points, true condition
run_frr ef_s3_thr01   "$V6/editflow_s3/probe500/records.jsonl"      "thr0.1"  true
run_frr ef_s3_thr05   "$V6/editflow_s3/probe500/records.jsonl"      "thr0.5"  true
run_frr pipeline      "$V6/eval_lingualens_final/records.jsonl"     ""        true
run_frr b2_prompt8    "$V6/prompt_baseline500/records.jsonl"        "prompt8" true

# random-conditioning control (the paper's control-group analog):
# FRR here reads as the judge/prior false-positive floor.
# NOTE: the v6 pipeline eval ran --conditions true,empty (default), so a
# pipeline-random row would need an eval_lingualens rerun — recorded as
# an accepted asymmetry unless the paper table demands it.
run_frr ef_s3_thr01_rnd "$V6/editflow_s3/probe500/records.jsonl"    "thr0.1"  random
run_frr ef_s3_thr05_rnd "$V6/editflow_s3/probe500/records.jsonl"    "thr0.5"  random
run_frr b2_prompt8_rnd  "$V6/prompt_baseline500/records.jsonl"      "prompt8" random
# B2 empty: it edits ~52% of sentences with NO request — does that
# unconditional editing accidentally realize features? (EF/pipeline
# empty rows are copies -> FRR trivially ~0, stated analytically.)
run_frr b2_prompt8_empty "$V6/prompt_baseline500/records.jsonl"     "prompt8" empty

# B1 (LinguaLens clamp) — THE system this metric exists for; rows are
# skipped gracefully if run_ef_b1.sh has not produced records yet.
if [ -f "$V6/clamp_baseline500/records.jsonl" ]; then
    run_frr b1_clamp10     "$V6/clamp_baseline500/records.jsonl" "clamp10" true
    run_frr b1_clampZ      "$V6/clamp_baseline500/records.jsonl" "clampZ"  true
    run_frr b1_clamp10_rnd "$V6/clamp_baseline500/records.jsonl" "clamp10" random
    run_frr b1_recon_empty "$V6/clamp_baseline500/records.jsonl" "recon"   empty
else
    echo "[frr] B1 records not found — run run_ef_b1.sh first, then rerun"
fi

echo "==================== FRR DONE ===================="
echo "Reading: FRR = P(judged prominence moved in the gold direction),"
echo "  pairs with judge-equal gold excluded; copies score unrealized"
echo "  (unless gold direction is src-more, which requires an edit)."
echo "  - exact is a LOWER bound on FRR by construction: every exact"
echo "    match realizes the feature. The interesting quantity is the"
echo "    FRR - exact gap = directionally-correct-but-inexact edits."
echo "  - B2 was built for this metric (free rewrite toward a described"
echo "    property); if B2 FRR ~ EF FRR at much lower exact, the paper"
echo "    story becomes: equal steering power, only EF does it as a"
echo "    minimal grammatical edit (premise-safe, exact)."
echo "  - random rows: false-positive floor; true-condition FRR must"
echo "    clear them decisively for the judged claim to stand."
