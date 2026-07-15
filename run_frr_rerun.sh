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

# Re-judge the SYSTEM comparisons after the shared-rng fix.
#
# The bug: judge_feature_realization.py drew the A/B presentation order
# for the gold and system comparisons from ONE rng stream, so the system
# order depended on gold-CACHE STATE. run_frr_final.sh judges routed
# first, so routed spent draw #1 on gold and drew #2 for itself, while
# ef32/steer/steer_rnd found gold cached, skipped that call, and drew #1
# — the SAME order gold used. A position-biased judge therefore agreed
# with itself more often on ef32/steer than on routed, INFLATING their
# FRR. Since ef32 tops FRR and routed tops exact, the artifact pushed in
# exactly the direction of the paper's multi-axis claim; it has to go.
#
# GOLD IS STILL VALID (rng_gold reproduces the old first draw), so this
# re-runs system judgments only and keeps gold.jsonl. The old files are
# preserved as *.orderbug.jsonl and diffed against the new ones: same
# judge, same system, order correlated vs independent, so
#   FRR(orderbug) - FRR(fixed) = the position-bias inflation, measured.
#
# Usage:  JUDGE=openai:gpt-4o bash run_frr_rerun.sh      (CPU, ~3000 calls)
#         qsub -v JUDGE=hf:google/gemma-2-9b-it run_frr_rerun.sh   (GPU)
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
JUDGE=${JUDGE:-hf:google/gemma-2-9b-it}
TAG=$(echo "$JUDGE" | tr ':/' '__')
FRR=runs/frr_final/$TAG
GOLD=$FRR/gold.jsonl
ROUTED=$V6/routed_system/records.jsonl
[ -d "$FRR" ] || { echo "no $FRR — run run_frr_final.sh first"; exit 1; }
[ -f "$GOLD" ] || { echo "no gold cache at $GOLD"; exit 1; }
echo "[rerun] judge=$JUDGE  gold kept: $(wc -l < "$GOLD") judgments"

# park the order-correlated files (once — a second run must not clobber
# the originals with already-fixed output)
for s in routed ef32 steer steer_rnd; do
    if [ -f "$FRR/$s.jsonl" ] && [ ! -f "$FRR/$s.orderbug.jsonl" ]; then
        mv "$FRR/$s.jsonl" "$FRR/$s.orderbug.jsonl"
        echo "[rerun] parked $s -> $s.orderbug.jsonl"
    fi
    rm -f "$FRR/.done.$s"
done

frr () {  # label records mode condition
    if [ ! -f "$FRR/.done.$1" ]; then
        python scripts/judge_feature_realization.py \
            --records "$2" --mode "$3" --condition "$4" \
            --gold-cache "$GOLD" --judge "$JUDGE" \
            --n-ops-ref "$V6/ksweep500/records.jsonl" \
            --out "$FRR/$1.jsonl" --device cuda
        touch "$FRR/.done.$1"
    fi
}
frr routed    "$ROUTED"                                "routed"   true
frr ef32      "$V6/ksweep500/records.jsonl"            "k32"      true
frr steer     "$V6/steer_baseline500/records.jsonl"    "steer0.5" true
frr steer_rnd "$V6/steer_baseline500/records.jsonl"    "steer0.5" random

echo "==================== FRR RERUN DONE ===================="
python scripts/frr_per_feature.py \
    --frr routed="$FRR/routed.jsonl" \
    --frr ef32="$FRR/ef32.jsonl" \
    --frr steer="$FRR/steer.jsonl" \
    --frr steer_rnd="$FRR/steer_rnd.jsonl" \
    --out runs/tables/frr_per_feature_$TAG

echo
echo "============ POSITION-BIAS INFLATION (old vs fixed) ============"
echo "FRR a = order-correlated-with-gold (buggy), b = independent (fixed)."
echo "a - b is the inflation. Expect ~0 for routed (it always drew an"
echo "independent order) and >0 for ef32/steer if the judge is position-"
echo "biased — that difference is the artifact, now measured not assumed."
for s in routed ef32 steer steer_rnd; do
    if [ -f "$FRR/$s.orderbug.jsonl" ]; then
        echo "--- $s"
        python scripts/judge_agreement.py \
            --a "$FRR/$s.orderbug.jsonl" --b "$FRR/$s.jsonl"
    fi
done

echo
echo "==================== SELF-CONSISTENCY (fixed) ===================="
python scripts/judge_selfconsistency.py --label "$TAG" \
    --sys ef32="$FRR/ef32.jsonl,$V6/ksweep500/records.jsonl,k32" \
    --sys routed="$FRR/routed.jsonl,$V6/routed_system/records.jsonl,routed" \
    --sys steer="$FRR/steer.jsonl,$V6/steer_baseline500/records.jsonl,steer0.5" \
    --out "runs/tables/judge_selfconsistency_$TAG"
echo "All three columns are now honest order-randomized readings; before"
echo "the fix only the routed column was."

echo
echo "==================== PAIRED FRR TESTS ===================="
python scripts/frr_paired_test.py --label "$TAG" \
    --frr ef32="$FRR/ef32.jsonl" \
    --frr routed="$FRR/routed.jsonl" \
    --frr steer="$FRR/steer.jsonl" \
    --out "runs/tables/frr_paired_$TAG"
