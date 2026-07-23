#!/bin/bash -l

# T4v2 (user-approved 2026-07-24): pool adaptation with the specificity
# constraint "random floor must stay at the zero-shot level".
#   A  replay mix   (~20% T4 rows / ~80% ctx corruption rows)
#   B+ negatives    mismatch-null 0.30 + SCRAMBLED-spec copy rows in the
#                   cache (the eval 'random' construction, taught as null)
#   C  low LR       3e-5 / backbone 1e-5, 2k steps
#   E' selection    per-ckpt dev-200 evals; best mean net SUBJECT TO
#                   max(random_abl, random_enh) <= 0.02; none feasible ->
#                   T4v2 auto-rejected (reported)
# Run inside interact-g: bash run_t4v2.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
CTX=$V4/corruption_zctx_l12
CTXDEV=$V4/corruption_seldev_zctx_l12
T4C=$V4/t4_ctx_l12_v2
MIX=$V4/t4v2_mix_l12
OUT=$P/eflm_l12_t4v2
SPLIT=runs/tables/eval_split.json

# ---- 1. cache with scrambled-null rows ----------------------------------
if ! python -c "import json,sys;m=json.load(open('$T4C/meta.json'));sys.exit(0 if 'd_sae' in m else 1)" 2>/dev/null; then
    python scripts/make_t4_cache.py \
        --spec $FS/l12_specctx.json --split $SPLIT \
        --out $T4C --scale 3.5 --scramble-prob 0.5 \
        --meta-from $CTX/meta.json
fi

# ---- 2. replay mix dir ---------------------------------------------------
if [ ! -f $MIX/build.done ]; then
    mkdir -p $MIX
    cp $T4C/meta.json $MIX/meta.json
    for f in $CTX/shard-*.jsonl.gz; do
        ln -sf "$(readlink -f $f)" "$MIX/$(basename $f)"
    done
    for f in $T4C/shard-*.jsonl.gz; do
        for i in 1 2 3 4 5 6 7 8; do
            ln -sf "$(readlink -f $f)" \
                "$MIX/$(basename ${f%.jsonl.gz})-dup$i.jsonl.gz"
        done
    done
    touch $MIX/build.done
fi

# ---- 3. constrained fine-tune -------------------------------------------
if [ ! -f "$OUT/eflm-step2000.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $MIX --dev-corruption-dir $CTXDEV \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$OUT" \
        --inject-layer 12 \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.30 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat \
        --init-ckpt $P/eflm_l12_v6t2/eflm-final.pt \
        --learning-rate 3e-5 --backbone-lr 1e-5 \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 2000 --resume --device cuda
fi

# ---- 4. E' constrained ckpt selection (dev-200, both dirs) ---------------
EVC=(--frame repeat --feature-spec $FS/l12_specctx.json --fspec-scale 3.5
     --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
     --sae-path layer_12/width_16k/average_l0_82/params.npz
     --sae-layer 12 --blocklist runs/blocklist/blocklist.npy
     --k-amp 64 --k-sup 64 --conditions true,random --device cuda)
for ST in 500 1000 1500 2000; do
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/t4v2dev_s$ST$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVC[@]}" --ef-ckpt "$OUT/eflm-step$ST.pt" \
            --pool-dev $SPLIT --sample-size 200 --output-dir "$O" $X
    done
done
SEL=$(python - "$P" <<'PY'
import re, sys
P = sys.argv[1]
best = (None, -1.0, None)     # (step, mean_net, feasible)
rows = []
for st in (500, 1000, 1500, 2000):
    nets, rands = [], []
    for suf in ("", "_amp"):
        try:
            t = open(f"{P}/t4v2dev_s{st}{suf}/report.md").read()
            tr = float(re.search(r"\| true \| ef \| ([0-9.]+)", t).group(1))
            rd = float(re.search(r"\| random \| ef \| ([0-9.]+)", t).group(1))
            nets.append(tr - rd); rands.append(rd)
        except Exception:
            pass
    if len(nets) != 2:
        continue
    feas = max(rands) <= 0.02
    rows.append((st, sum(nets) / 2, max(rands), feas))
feasible = [r for r in rows if r[3]]
pool = feasible if feasible else rows
pool.sort(key=lambda r: -r[1])
st, net, rmax, feas = pool[0]
print(f"{st} {'FEASIBLE' if feas else 'INFEASIBLE'} net={net:.4f} rmax={rmax:.4f}")
for r in rows:
    print(f"#  step{r[0]}: net={r[1]:.4f} rmax={r[2]:.4f} feas={r[3]}",
          file=sys.stderr)
PY
)
echo "[t4v2] selection: $SEL"
ST=$(echo "$SEL" | awk '{print $1}')
FEAS=$(echo "$SEL" | awk '{print $2}')

# ---- 5. eval500 with the selected ckpt ----------------------------------
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    O=$P/fs_t4v2_l12$DIRX
    [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
        "${EVC[@]}" --ef-ckpt "$OUT/eflm-step$ST.pt" \
        --sample-size 500 --output-dir "$O" $X
done

echo "[t4v2] final: step$ST ($FEAS)"
echo "==================== T4V2-DONE ===================="
