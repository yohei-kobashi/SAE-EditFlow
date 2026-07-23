#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-c
#PBS -l select=1
#PBS -l walltime=6:00:00
#PBS -W group_list=gj26
#PBS -j oe

# AxBench steering reproduction — JUDGE (CPU batch on short-c; aarch64,
# so env-c works. prepost rejects batch qsub — interactive only).
# Batch:      cd ~/SAE-LEWIS && qsub run_axbench_repro_judge.sh  (login node)
# Interactive: qsub -I -l select=1 -W group_list=go25 -q prepost
#   cd SAE-LEWIS && git pull && bash run_axbench_repro_judge.sh
# Judge = gpt-4o-mini with the official rubric templates (verbatim).
# Lazy judging: selection half over all 14 factors, holdout at best factor.
# Resumable (judge cache) — resubmit past a walltime kill.
cd ~/SAE-LEWIS
source env-c/bin/activate
set -eo pipefail
git pull || true
CONFIG=${CONFIG:-prod_2b_l20_v1}

# 2026-07-18: parse_rating required the literal "[[x]]" wrapper but
# gpt-4o-mini answers "Rating: 0" bare -> ALL 16k cached scores were
# parse-failure zeros. Those cache entries are POISON for resume;
# quarantine once (records are healthy and stay).
JC=runs/axbench_repro/judge_cache_${CONFIG}_gpt-4o-mini.jsonl
if [ -f "$JC" ] && [ ! -f "$JC.parsebug-v2" ]; then
    mv "$JC" "$JC.parsebug-v2"
    rm -f "runs/axbench_repro/report_${CONFIG}.md"
    echo "[axjudge] quarantined parse-bug cache -> $JC.parsebug-v2"
fi

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }
[ -f "runs/axbench_repro/records_${CONFIG}.jsonl" ] || { echo "run run_axbench_repro.sh first"; exit 1; }

python scripts/judge_axbench_repro.py \
    --run-dir runs/axbench_repro --config "$CONFIG"

echo "==================== AXBENCH-REPRO JUDGE DONE ===================="
