#!/bin/bash
# P-N: judge the saved B1/B3/protocol outputs on the PAPERS' OWN metrics.
# CPU + OpenAI API only (no regeneration) — run inside prepost with env-c:
#   qsub -I -l select=1 -W group_list=go25 -q prepost
#   cd SAE-LEWIS && source env-c/bin/activate && bash run_paper_metrics.sh
#
# LinguaLens metric (their judge gpt-4o): feature-prominence presence;
#   ablation success P(Y=0), normalized effect E_abl vs the random-features
#   arm. Positive E_abl on ll_protocol/b1_ours = the clamp implementation
#   reproduces their qualitative claim on their own metric.
# AxBench metric (their judge gpt-4o-mini): harmonic mean of 0-2 subscores
#   (concept / instruct / fluency). b3_ours should beat the uninformed
#   rewrite on concept with fluency held up.
set -eo pipefail
cd "$(dirname "$0")"
V6=runs/prod_gemma_v6

python scripts/judge_paper_metrics.py \
    --run b1_ours="$V6/clamp_baseline500/records.jsonl:clamp10" \
    --run ll_protocol="$V6/protocol_e2e/lingualens_frc_r3_clamp/records.jsonl:clamp10" \
    --run b3_ours="$V6/steer_baseline500/records.jsonl:steer0.5" \
    --run ax_protocol="$V6/protocol_e2e/axbench_auroc_r1_steer/records.jsonl:steer1" \
    --out-dir runs/paper_metrics

echo "==================== PAPER METRICS DONE ===================="
