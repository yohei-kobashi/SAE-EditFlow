#!/bin/bash -l

# v3a stage 3: fine alpha search toward the specificity constraint.
# alpha in {0.2, 0.3} for the v1final blend; DEV-selected (relative rule:
# rmax <= T2's own dev rmax) to avoid test-set tuning; single eval500
# verification of the selected point -> fs_v3a_final_l12{,_amp}.
# Run inside interact-g: bash run_v3a3.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS
set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
SPLIT=runs/tables/eval_split.json
T2=$P/eflm_l12_v6t2/eflm-final.pt
BSRC=$P/eflm_l12_t4_v6t2/eflm-final.pt
BL=$P/v3a_blends2

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

for A in 0.2 0.3; do
    CK=$BL/v1final_a$A.pt
    [ -f "$CK" ] || python scripts/blend_ckpt.py \
        --a "$T2" --b "$BSRC" --alpha $A --out "$CK"
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/v3adev_v1final_a$A$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$CK" \
            --pool-dev $SPLIT --sample-size 200 --output-dir "$O" $X
    done
done

SEL=$(python - "$P" <<'PY'
import re, sys
P = sys.argv[1]
def cell(tag):
    nets, rands = [], []
    for suf in ("", "_amp"):
        t = open(f"{P}/v3adev_{tag}{suf}/report.md").read()
        tr = float(re.search(r"\| true \| ef \| ([0-9.]+)", t).group(1))
        rd = float(re.search(r"\| random \| ef \| ([0-9.]+)", t).group(1))
        nets.append(tr - rd); rands.append(rd)
    return sum(nets) / 2, max(rands)
ref, refr = cell("t2ref")
best = None
for a in ("0.2", "0.3"):
    net, rmax = cell(f"v1final_a{a}")
    feas = rmax <= refr + 1e-9
    print(f"#  a={a}: net={net:.4f} rmax={rmax:.4f} feas(rel<= {refr:.3f})={feas}",
          file=sys.stderr)
    if feas and net > ref and (best is None or net > best[1]):
        best = (a, net)
print(best[0] if best else "NONE")
PY
)
echo "[v3a3] selected alpha: $SEL"
if [ "$SEL" != "NONE" ]; then
    CK=$BL/v1final_a$SEL.pt
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_v3a_final_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$CK" \
            --sample-size 500 --output-dir "$O" $X
    done
    for d in $P/fs_v3a_final_l12 $P/fs_v3a_final_l12_amp; do
        echo "--- $d"; grep -E "^\| (true|random) \| ef" $d/report.md || true
    done
fi
echo "==================== V3A3-DONE ===================="
