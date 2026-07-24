#!/bin/bash
# FIC judge for the v3a blend (fs_v3a_final_l12{,_amp}) -> fic_v3a_l12{,_amp}
cd ~/SAE-LEWIS
source env-c/bin/activate
set -o pipefail
git pull || true
[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
P=runs/prod_gemma_v6
for SUF in "" "_amp"; do
    SRC=$P/fs_v3a_final_l12$SUF/records.jsonl
    OUT=$P/fic_v3a_l12$SUF
    [ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
    mkdir -p "$OUT"
    [ -f "$OUT/records_merged.jsonl" ] || python scripts/merge_ef_records.py \
        --base $P/fs_probe_l12$SUF/records.jsonl --ef "$SRC" \
        --out "$OUT/records_merged.jsonl"
    [ -f "$OUT/report.md" ] || python scripts/eval_fic_judge.py \
        --repeat-probe500 "$OUT/records_merged.jsonl" --output-dir "$OUT"
done
echo "==================== FIC-V3A-DONE ===================="
grep -E "^\| ef \|" $P/fic_v3a_l12/report.md $P/fic_v3a_l12_amp/report.md
