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

# CONFIRMATION measurement for the router system (phase 1 passed on the
# 300-pair holdout: count-rule T=1 = 0.2700 vs steer0.5 0.2400, but k=32,
# alpha=0.5 and T were all chosen on the original 500 pairs). This job
# extends the two systems the FINAL RULE uses (EF k=32 decode + steer0.5)
# to the first 1000 pairs of the same seed-42 shuffle — the NEW ~500
# pairs (idx 501-1000) have never been touched by any selection — then
# re-runs the router analysis with prefix = the ORIGINAL 499 (tuning) so
# the holdout block IS the fresh sample. The rule is FROZEN before this
# job: count-rule T=1 (head=ef32 if its own output has <=1 hunk, else
# steer0.5); band-rule hunks==1 is the pre-registered secondary.
# Both underlying scripts resume per pair — resubmit until CONFIRM DONE.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy

# EF k=32 decode on pairs 1..1000 (resumes past the original 500;
# k-grid 32 only — new pairs get exactly the mode the rule needs)
python scripts/eval_k_sweep.py \
    --llm2vec-dir "$LLM2VEC" \
    --editflow-ckpt "$V6/editflow_s3/editflow-final.pt" \
    --blocklist "$BLOCKLIST" \
    --output-dir "$V6/ksweep500" \
    --k-grid 32 --decode thr0.1 \
    --sample-size 1000 --device cuda

# steer0.5 on pairs 1..1000 (alpha 0.5 only; true/empty/random controls)
python scripts/eval_clamp_baseline.py \
    --llm2vec-dir "$LLM2VEC" --blocklist "$BLOCKLIST" \
    --output-dir "$V6/steer_baseline500" \
    --k-amp 64 --k-sup 64 --sample-size 1000 \
    --intervention steer --clamp-values 0.5 --device cuda

echo "==================== CONFIRM DONE ===================="
# prefix = the ORIGINAL 499 (all tuning); HOLDOUT block = fresh ~500
python scripts/analyze_router.py \
    --llm2vec-dir "$LLM2VEC" --blocklist "$BLOCKLIST" \
    --cand ef32="$V6/ksweep500/records.jsonl":k32 \
    --cand steer="$V6/steer_baseline500/records.jsonl":steer0.5 \
    --prefix-records "$V6/editflow_s3/probe500/records.jsonl" \
    --count-cand ef32 --route-to steer \
    --out runs/router/confirm1000.json --device cuda
echo
echo "READ: the HOLDOUT block is the untouched fresh ~500 pairs."
echo "  Pre-registered primary rule: count-rule T=1. Win condition:"
echo "  its holdout exact > steer's holdout exact (and > every single"
echo "  system's). Also read the discordant +w/-l counts: the paired"
echo "  sign test over w+l discordant pairs is the significance readout"
echo "  (combined with the original 300-pair holdout: +9 net there)."