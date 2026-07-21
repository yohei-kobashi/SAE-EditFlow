#!/bin/bash
# FIC(repeat) judging for the feature-spec protocol (prepost polling
# runner; the chain driver relaunches me until FIC-FS-ALL-DONE).
# Judges each layer x direction as its records land:
#   ef+steer  from fs_probe_l{L}[_amp]   (feature-spec, selected scale)
#   clamp     from fs_clamp_l{L}[_amp]   (LinguaLens FRC-r3, pool)
#   prompting from a3prime_edit / amp_a3prime_l12 (layer-independent,
#             protocol-unchanged — attached to the L12 judge only)

cd ~/SAE-LEWIS
source env-c/bin/activate

set -o pipefail
git pull || true

[ -n "$OPENAI_API_KEY" ] || { [ -f .openai_key ] && export OPENAI_API_KEY=$(cat .openai_key); }
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

P=runs/prod_gemma_v6
CELLS=(
  "fs_probe_l4       fs_clamp_l4       -                fic_fs_l4"
  "fs_probe_l4_amp   fs_clamp_l4_amp   -                fic_fs_l4_amp"
  "fs_probe_l12      fs_clamp_l12      a3prime_edit     fic_fs_l12"
  "fs_probe_l12_amp  fs_clamp_l12_amp  amp_a3prime_l12  fic_fs_l12_amp"
  "fs_probe_l20      fs_clamp_l20      -                fic_fs_l20"
  "fs_probe_l20_amp  fs_clamp_l20_amp  -                fic_fs_l20_amp"
)

while true; do
    git pull -q || true
    done_n=0
    for CFG in "${CELLS[@]}"; do
        set -- $CFG
        PROBE=$P/$1/records.jsonl; CLAMP=$P/$2/records.jsonl
        A3=$([ "$3" = "-" ] || echo $P/$3/records.jsonl)
        OUT=$P/$4
        [ -f "$PROBE" ] || continue
        echo "==== judging $1 -> $4 ===="
        if python scripts/eval_fic_judge.py \
              --repeat-probe500 "$PROBE" \
              $([ -f "$CLAMP" ] && echo --repeat-clamp "$CLAMP") \
              $([ -n "$A3" ] && [ -f "$A3" ] && echo --repeat-a3 "$A3") \
              --dir-map runs/tables/lingualens_dirmap_en.json \
              --output-dir "$OUT" | tee /tmp/ficfs_last.log \
           && grep -q "FIC-JUDGE-DONE" /tmp/ficfs_last.log \
           && [ -f "$CLAMP" ]; then
            done_n=$((done_n+1))
        fi
    done
    echo "[fic-fs] complete cells: $done_n / ${#CELLS[@]}"
    if [ "$done_n" -eq "${#CELLS[@]}" ]; then
        echo "==================== FIC-FS-ALL-DONE ===================="
        break
    fi
    sleep 900
done
