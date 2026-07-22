#!/bin/bash
# FIC judging for the AxBench-faithful arm (prepost polling runner;
# chain marker FIC-AXB-DONE). Cells reuse the fs_probe records as the
# judged base (ef/steer rows come from cache — no new API cost) and
# attach the AxBench records with the layer's dev-selected factor key:
#   sup: fs_axb_l{L}      key steer1   (0.5/1 sweep, main column)
#   amp: fs_axbE_l{L}_amp key steer1 (L4/L20) / steer0.6 (L12)
# Convention note (recorded in 04): the frame control is the repeat-frame
# raw from the probe records — same convention as the clamp rows judged
# in pass 1 (movement measured vs the source; PB nets out frame effects).

cd ~/SAE-LEWIS
source env-c/bin/activate

set -o pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

P=runs/prod_gemma_v6
CELLS=(
  "fs_probe_l4       fs_axb_l4        steer1    fic_fs_axb_l4"
  "fs_probe_l4_amp   fs_axbE_l4_amp   steer0.6  fic_fs_axb_l4_amp"
  "fs_probe_l12      fs_axb_l12       steer1    fic_fs_axb_l12"
  "fs_probe_l12_amp  fs_axbE_l12_amp  steer0.6  fic_fs_axb_l12_amp"
  "fs_probe_l20      fs_axb_l20       steer1    fic_fs_axb_l20"
  "fs_probe_l20_amp  fs_axbE_l20_amp  steer0.6  fic_fs_axb_l20_amp"
)

while true; do
    git pull -q || true
    done_n=0
    for CFG in "${CELLS[@]}"; do
        set -- $CFG
        PROBE=$P/$1/records.jsonl
        AXB=$P/$2/records.jsonl
        KEY=$3
        OUT=$P/$4
        { [ -f "$PROBE" ] && [ -f "$AXB" ]; } || continue
        echo "==== judging $2 ($KEY) -> $4 ===="
        if python scripts/eval_fic_judge.py \
              --repeat-probe500 "$PROBE" \
              --repeat-axb "$AXB" --axb-key "$KEY" \
              --dir-map runs/tables/lingualens_dirmap_en.json \
              --output-dir "$OUT" | tee /tmp/ficaxb_last.log \
           && grep -q "FIC-JUDGE-DONE" /tmp/ficaxb_last.log \
           && { case "$4" in *_amp) grep -qE "^\| axbench \| -?[0-9]" "$OUT/report.md" ;;
                  *) grep -qE "^\| axbench \| — \| -?[0-9]" "$OUT/report.md" ;; esac; }; then
            done_n=$((done_n+1))
        fi
    done
    echo "[fic-axb] complete cells: $done_n / ${#CELLS[@]}"
    if [ "$done_n" -eq "${#CELLS[@]}" ]; then
        echo "==================== FIC-AXB-DONE ===================="
        break
    fi
    sleep 900
done
