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

# P-B CONTROL: P-B as run cannot support "detecting != commanding".
#
# The problem, found 2026-07-15 by separating train time from test time:
#   training  conditioning = instance delta of the corruption pair, k drawn at
#             random from the top-64 candidates (intervene.py diff_to_sparse)
#   P-B eval  conditioning = instance delta MASKED to a phenomenon-level FRC set
# The mask is applied at eval only — no training script takes --feature-sets.
# So P-B hands the model a conditioning distribution it never saw, and a 10x
# collapse is what distribution shift produces regardless of whether concept
# features carry the information. As it stands P-B supports only "a model
# trained on instance specs cannot consume concept specs", which is nearly
# tautological.
#
# Worse, P-B ran with --conditions true ONLY, so it has no size-matched
# control. randomize_intervention gives exactly that for free: "same nonzero
# count and (permuted) magnitudes at random indices". So:
#
#   FRC-true ~= FRC-random  =>  the collapse is entirely the sparsity/shift.
#                               FRC's CONTENT contributes nothing and P-B
#                               proves nothing about FRC.
#   FRC-true >> FRC-random  =>  FRC content carries signal even under
#                               mismatch — the claim survives, weakened.
#   FRC-true << FRC-random  =>  FRC selection is WORSE than a random set of
#                               the same size. That is a real and striking
#                               finding, and the strongest version of P-B.
#
# Second control, free from P-D: P-B intersect's spec is delta AND FRC, i.e. a
# SUBSET of the instance delta. Training saw random subsets of the top-64, so
# the mismatch is mostly about SIZE. Compare P-B against the k-curve at the
# matched |delta AND FRC| rather than against k=32.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=${BLOCKLIST:-runs/blocklist/blocklist.npy}

for FM in intersect pure; do
    OUT="$V6/editflow_s3/probe500_frc_${FM}_ctrl"
    if [ ! -f "$OUT/probe_report.md" ]; then
        echo "-------- FRC $FM + random control"
        python scripts/editflow_probe.py \
            --llm2vec-dir "$LLM2VEC" \
            --editflow-ckpt "$V6/editflow_s3/editflow-final.pt" \
            --output-dir "$OUT" \
            --cond-scope local --blocklist "$BLOCKLIST" \
            --k-amp 64 --k-sup 64 --sample-size 500 \
            --steps 48 --steer-lambda 1 --decode thr0.1 \
            --feature-sets runs/frc/identified_l12_16k.json \
            --feature-mode "$FM" \
            --conditions true,random \
            --device cuda
    fi
done

echo
echo "==================== P-B CONTROL DONE ===================="
for FM in intersect pure; do
    R="$V6/editflow_s3/probe500_frc_${FM}_ctrl/probe_report.md"
    if [ -f "$R" ]; then
        echo; echo "======== $R"; cat "$R"
    fi
done
echo "Read exact for condition=true vs condition=random in each"
echo "probe500_frc_*_ctrl/probe_report.md. random has the SAME nonzero count"
echo "and magnitude distribution as true, only the feature IDENTITIES differ,"
echo "so the gap between them is exactly what FRC's content buys at that"
echo "sparsity — with the distribution shift held constant."
echo
echo "Then compare true against runs/tables/kcurve_exact.md at the k matching"
echo "the mean |delta AND FRC|. If P-B ~= the k-curve at that size, P-B is a"
echo "size effect and the paper must not read it as 'phenomenon features"
echo "cannot command'."
