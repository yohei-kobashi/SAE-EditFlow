#!/bin/bash
# FIC wave (prepost, user 2026-07-23 「可能なものからprepostで」):
# judges, as their records land:
#   * repeat-frame baselines (clampset/axbsteer) via run_bl_judge.sh
#   * adopted ef (T2)      fs_v6t2_l{L}{,_amp} -> fic_ad_l{L}{,_amp}
#   * pool-adapted ef (T4) fs_t4_l{L}{,_amp}   -> fic_t4_l{L}{,_amp}
# for L in 4 12 20. Polls every 15 min inside one prepost session; the
# chain driver relaunches sessions until ALL-FIC-DONE.
# Run inside prepost: bash run_fic_wave.sh

cd ~/SAE-LEWIS
source env-c/bin/activate

set -o pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

P=runs/prod_gemma_v6

judge_ef_cell () {   # $1=ef-src dir tag  $2=out tag  $3=layer  $4=suf
    local SRC=$P/$1$4/records.jsonl
    local PROBE=$P/fs_probe_l$3$4/records.jsonl
    local OUT=$P/$2$4
    [ -f "$SRC" ] || return 1
    [ -f "$OUT/report.md" ] && return 0
    mkdir -p "$OUT"
    [ -f "$OUT/records_merged.jsonl" ] || python scripts/merge_ef_records.py \
        --base "$PROBE" --ef "$SRC" --out "$OUT/records_merged.jsonl"
    python scripts/eval_fic_judge.py \
        --repeat-probe500 "$OUT/records_merged.jsonl" \
        --output-dir "$OUT"
    [ -f "$OUT/report.md" ]
}

for CYCLE in $(seq 1 18); do
    echo "[wave] cycle $CYCLE $(date +%H:%M)"
    bash run_bl_judge.sh || true
    ALL=1
    for L in 12 4 20; do
        for SUF in "" "_amp"; do
            judge_ef_cell fs_v6t2_l$L fic_ad_l$L "$L" "$SUF" || ALL=0
            [ -f $P/fic_bl_l$L$SUF/report.md ] || ALL=0
        done
    done
    # L12-only ef variants (T4 v1 = negative-result row; T4v2 = defended)
    for SUF in "" "_amp"; do
        judge_ef_cell fs_t4_l12   fic_t4_l12   12 "$SUF" || ALL=0
        judge_ef_cell fs_t4v2_l12 fic_t4v2_l12 12 "$SUF" || ALL=0
    done
    if [ "$ALL" = "1" ]; then
        echo "==================== ALL-FIC-DONE ===================="
        exit 0
    fi
    sleep 900
done
echo "[wave] session budget spent — chain will relaunch"
