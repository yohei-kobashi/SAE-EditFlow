#!/bin/bash
# Judge-reliability readouts over the FINISHED FRR files (CPU, seconds —
# run interactively, no qsub needed). Two independent checks:
#
#   1. self-consistency: FRR restricted to exact-match pairs. On those the
#      system judgment judge(src,out) IS the gold judgment judge(src,tgt),
#      so a consistent judge scores 1.0 by construction. 1 - it = the FRR
#      noise floor, and it ranks judge QUALITY without human labels.
#   2. agreement: same system, two judges, idx-joined — realized-verdict
#      and raw-direction agreement (the paper's judge-invariance line).
set -eo pipefail

V6=./runs/prod_gemma_v6
FF=runs/frr_final
G=$FF/hf_google_gemma-2-9b-it
O=$FF/openai_gpt-4o
N=$FF/openai_gpt-5.4-nano

sc () {  # dir label
    [ -f "$1/ef32.jsonl" ] || { echo "[skip] $2 (no $1/ef32.jsonl)"; return; }
    python scripts/judge_selfconsistency.py --label "$2" \
        --sys ef32="$1/ef32.jsonl,$V6/ksweep500/records.jsonl,k32" \
        --sys routed="$1/routed.jsonl,$V6/routed_system/records.jsonl,routed" \
        --sys steer="$1/steer.jsonl,$V6/steer_baseline500/records.jsonl,steer0.5" \
        --out "runs/tables/judge_selfconsistency_$2"
    echo
    python scripts/frr_paired_test.py --label "$2" \
        --frr ef32="$1/ef32.jsonl" \
        --frr routed="$1/routed.jsonl" \
        --frr steer="$1/steer.jsonl" \
        --out "runs/tables/frr_paired_$2"
    echo
}
sc "$G" hf_google_gemma-2-9b-it
sc "$O" openai_gpt-4o
sc "$N" openai_gpt-5.4-nano

echo "==================== AGREEMENT ===================="
ag () {  # a b name
    [ -f "$1" ] && [ -f "$2" ] || { echo "[skip] $3"; return; }
    echo "--- $3"
    python scripts/judge_agreement.py --a "$1" --b "$2"
}
for s in routed ef32 steer; do
    ag "$G/$s.jsonl" "$O/$s.jsonl" "$s: gemma-2-9b-it vs gpt-4o"
    ag "$O/$s.jsonl" "$N/$s.jsonl" "$s: gpt-4o vs gpt-5.4-nano"
    ag "$G/$s.jsonl" "$N/$s.jsonl" "$s: gemma-2-9b-it vs gpt-5.4-nano"
done
echo "==================== JUDGE CHECKS DONE ===================="
