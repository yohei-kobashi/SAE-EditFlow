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

# P-K: both papers' COMPLETE protocols, end-to-end, on exact.
#
# The gap this fills (2026-07-16): every intervention number we have uses OUR
# instance-level edit-local delta spec — B1 0.1743 is LinguaLens's MECHANISM
# with our spec, B3 0.2337 is standard steering with our spec. Nobody has
# measured the baselines' own end-to-end protocols on this task:
#
#   frc_r3   + clamp10 + regenerate  =  LinguaLens complete (FRC top-3, set)
#   auroc_r1 + steer   + regenerate  =  AxBench complete (best latent, add)
#   (+ the mechanism cross, so spec and mechanism separate)
#
# P-J predicts these sit near zero (frc_r3 fires 0.17, auroc_r1 fires 0.06
# through the causal readout). If so, the honest intervention-axis claim is:
#
#   "Holding the intervention mechanism fixed, replacing the phenomenon-level
#    specification (FRC top-3 / single AUROC latent) with our instance-level
#    edit-local delta lifts exact from ~0 to 0.17-0.23. What determines an
#    intervention's editing power is the SPECIFICATION, not the mechanism."
#
# — an intervention-only claim, no editor anywhere, which beats LinguaLens's
# and AxBench's protocols by construction and loses to nothing but prompting
# (acceptable per the research goal). The routed 0.2839 stays a CONDITIONING
# headline and must not be sold as intervention.
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final

e2e () {  # tag feature-sets intervention values
    OUT=$V6/protocol_e2e/$1
    if [ ! -f "$OUT/report.md" ]; then
        echo "-------- $1"
        python scripts/eval_clamp_baseline.py \
            --llm2vec-dir "$LLM2VEC" \
            --output-dir "$OUT" \
            --feature-sets "$2" \
            --intervention "$3" \
            --clamp-values "$4" \
            --conditions true,empty,random \
            --sample-size 500 --device cuda
    fi
}
# the two papers' own protocols
e2e lingualens_frc_r3_clamp "runs/frc/identified_l12_16k_r3.json"   clamp "5,10,20"
e2e axbench_auroc_r1_steer  "runs/auroc/identified_l12_16k_r1.json" steer "0.5,1"
# the mechanism cross — separates spec from mechanism
e2e cross_frc_r3_steer      "runs/frc/identified_l12_16k_r3.json"   steer "0.5,1"
e2e cross_auroc_r1_clamp    "runs/auroc/identified_l12_16k_r1.json" clamp "5,10,20"

echo
echo "==================== PROTOCOL E2E DONE ===================="
for d in lingualens_frc_r3_clamp axbench_auroc_r1_steer cross_frc_r3_steer cross_auroc_r1_clamp; do
    R="$V6/protocol_e2e/$d/report.md"
    [ -f "$R" ] && { echo; echo "======== $d"; cat "$R"; }
done
echo
echo "The 2x2 to assemble (exact, best value per cell):"
echo "                      phenomenon spec        our instance spec"
echo "  clamp+regenerate    lingualens_frc_r3_...  B1 = 0.1743"
echo "  steer+regenerate    cross_frc_r3_steer /   B3 = 0.2337"
echo "                      axbench_auroc_r1_..."
echo "Read row-wise for the mechanism effect, column-wise for the SPEC effect."
echo "P-J predicts the left column sits near zero; if so, the intervention"
echo "claim is a specification claim, measured with the editor nowhere in it."
