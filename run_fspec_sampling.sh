#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=go25
#PBS -j oe

# Sampling-robustness probe for exact (user 2026-07-22): greedy is the
# production convention, but does the arm ranking survive temp-1.0
# sampling, and how much headroom does pass@5 reveal? L12, ef arm,
# sup+amp, 5 seeds. Aggregation -> runs/tables/fs_sampling_l12.md
# (mean exact per seed + pass@5 joined by pair idx).
# Batch: qsub -N fssamp run_fspec_sampling.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail
git pull || true

# DIRS=sup / amp / sup,amp (default) — lets two workers split directions
DIRS=${DIRS:-sup,amp}

P=runs/prod_gemma_v6
FS=runs/feature_specs
L2V=runs/mcgill_gemma_repro_3k/final
SAE=layer_12/width_16k/average_l0_82/params.npz
BLK=runs/blocklist/blocklist.npy
CKPT=$P/eflm_l12_v5f_nobudget/eflm-final.pt
SC=$(cat $P/fs_scale_l12.txt)

for R in 1 2 3 4 5; do
  for DIRX in "" "_amp"; do
    if [ -z "$DIRX" ] && ! echo "$DIRS" | grep -q sup; then continue; fi
    if [ -n "$DIRX" ] && ! echo "$DIRS" | grep -q amp; then continue; fi
    OUT=$P/fs_tmp1_l12_r${R}${DIRX}
    if [ -n "$DIRX" ]; then EXTRA=--reverse-pairs; else EXTRA=""; fi
    if [ ! -f $OUT/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_spec.json \
            --fspec-scale $SC --temperature 1.0 --gen-seed $R \
            --conditions true,random --arms ef \
            --llm2vec-dir "$L2V" --sae-path "$SAE" --sae-layer 12 \
            --blocklist "$BLK" --k-amp 64 --k-sup 64 \
            --ef-ckpt "$CKPT" --sample-size 500 --device cuda \
            --output-dir $OUT $EXTRA
    fi
  done
done

# aggregate only when ALL 10 cells exist (last finisher does it)
NDONE=$(ls $P/fs_tmp1_l12_r*/report.md 2>/dev/null | wc -l)
if [ "$NDONE" -lt 10 ]; then
    echo "[fssamp] $NDONE/10 cells done — aggregation deferred"
    echo "==================== FS-SAMPLING-PART-DONE ===================="
    exit 0
fi
python3 - <<'PY'
import json
from pathlib import Path
P = Path("runs/prod_gemma_v6")
lines = ["# exact sampling robustness (L12, ef, feature-spec, temp 1.0, "
         "5 seeds)", ""]
for sfx, name in (("", "sup"), ("_amp", "amp")):
    per_seed, hit = [], {}
    for r in range(1, 6):
        f = P / f"fs_tmp1_l12_r{r}{sfx}" / "records.jsonl"
        if not f.exists():
            continue
        n = ok = 0
        for line in open(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            e = rec["outputs"].get("true", {}).get("ef", {}).get("exact")
            if e is None:
                continue
            n += 1
            ok += int(e == 1.0)
            hit[rec["idx"]] = hit.get(rec["idx"], 0) or int(e == 1.0)
        per_seed.append(ok / max(n, 1))
    mean = sum(per_seed) / max(len(per_seed), 1)
    p5 = sum(hit.values()) / max(len(hit), 1)
    lines.append(f"## {name}: per-seed {['%.4f' % v for v in per_seed]}")
    lines.append(f"mean exact (temp1.0) = {mean:.4f} | pass@5 = {p5:.4f}")
    lines.append("")
Path("runs/tables/fs_sampling_l12.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "==================== FS-SAMPLING-DONE ===================="
