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

# P-L: localized steering with REGENERATION — the untested cell that targets
# B3's 0.2337 directly. No training, no corruption categories.
#
# The localization insight was verified on the readout (the mask selects
# exactly where the suppressed features fire) but never tested with the
# effector that actually works (free regeneration). The reading/writing
# split: steering every PROMPT position corrupts the model's READING of the
# parts it must preserve — mask the prefill to phenomenon-active positions;
# generated tokens are always steered (writing is what steering is for).
#
# Also sweeps alpha finely around the known cliff (0.5 -> 1 halves exact):
# the operating curve may peak off-grid.
#
# Spec = our instance-level edit-local delta (the info-matched lane, same as
# B3). Bars: B3 all-positions 0.2337; empty must stay copy~1; true>>random.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final

for SC in local all; do
    OUT=$V6/steer_${SC}500_fine
    if [ ! -f "$OUT/report.md" ]; then
        echo "-------- steer scope=$SC (fine alpha)"
        python scripts/eval_clamp_baseline.py \
            --llm2vec-dir "$LLM2VEC" \
            --output-dir "$OUT" \
            --intervention steer --scope "$SC" \
            --clamp-values 0.25,0.375,0.5,0.75 \
            --conditions true,empty,random \
            --sample-size 500 --device cuda
    fi
done

echo
echo "==================== B3-LOCAL DONE ===================="
for SC in local all; do
    R="$V6/steer_${SC}500_fine/report.md"
    [ -f "$R" ] && { echo; echo "======== steer_$SC"; cat "$R"; }
done
echo
echo "Reading: steer_local vs steer_all at matched alpha isolates the"
echo "reading/writing split; steer_all at 0.25/0.375/0.75 checks whether"
echo "0.2337 (alpha=0.5) was even B3's peak. Best cell vs 0.2337 is the"
echo "intervention-lane improvement; it feeds C1' and, if it wins, becomes"
echo "the router's fallback too (routed would inherit the gain)."
