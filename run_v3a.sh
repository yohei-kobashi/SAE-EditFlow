#!/bin/bash -l

# v3a (user 2026-07-24): WiSE-FT interpolation between the zero-shot T2
# editor and the T4 checkpoints, searching the constraint frontier that
# the coarse T4v2 grid missed.
#   sources: t4v2 step1500 (best dev net, rmax 0.025) / t4v1 final
#   alpha in {0.4, 0.6, 0.8}; T2 dev reference measured identically
#   selection: rmax <= 0.02 AND mean dev net > T2 dev reference
# Run inside interact-g: bash run_v3a.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
SPLIT=runs/tables/eval_split.json
T2=$P/eflm_l12_v6t2/eflm-final.pt
BL=$P/v3a_blends
mkdir -p $BL

EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)

dev_eval () {  # $1 ckpt, $2 outtag
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/v3adev_$2$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$1" \
            --pool-dev $SPLIT --sample-size 200 --output-dir "$O" $X
    done
}

# T2 reference on the same dev protocol
dev_eval "$T2" t2ref

for SRC in v2s1500 v1final; do
    case $SRC in
        v2s1500) BSRC=$P/eflm_l12_t4v2/eflm-step1500.pt ;;
        v1final) BSRC=$P/eflm_l12_t4_v6t2/eflm-final.pt ;;
    esac
    for A in 0.4 0.6 0.8; do
        CK=$BL/${SRC}_a$A.pt
        [ -f "$CK" ] || python scripts/blend_ckpt.py \
            --a "$T2" --b "$BSRC" --alpha $A --out "$CK"
        dev_eval "$CK" ${SRC}_a$A
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
print(f"# T2 dev ref: net={ref:.4f} rmax={refr:.4f}", file=sys.stderr)
best = None
for src in ("v2s1500", "v1final"):
    for a in ("0.4", "0.6", "0.8"):
        try:
            net, rmax = cell(f"{src}_a{a}")
        except Exception:
            continue
        feas = rmax <= 0.02
        print(f"#  {src} a={a}: net={net:.4f} rmax={rmax:.4f} "
              f"feas={feas}", file=sys.stderr)
        if feas and (best is None or net > best[2]):
            best = (src, a, net, rmax)
if best is None:
    print("NONE none 0 0")
else:
    win = "WIN" if best[2] > ref else "NOGAIN"
    print(f"{best[0]} {best[1]} {best[2]:.4f} {win}")
PY
)
echo "[v3a] selection: $SEL"
SRC=$(echo "$SEL" | awk '{print $1}')
A=$(echo "$SEL" | awk '{print $2}')
VERDICT=$(echo "$SEL" | awk '{print $4}')

if [ "$SRC" != "NONE" ]; then
    CK=$BL/${SRC}_a$A.pt
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/fs_v3a_l12$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$CK" \
            --sample-size 500 --output-dir "$O" $X
    done
fi
echo "[v3a] final: $SRC a=$A ($VERDICT)"
echo "==================== V3A-DONE ===================="
