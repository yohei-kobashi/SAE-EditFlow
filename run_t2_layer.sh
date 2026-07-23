#!/bin/bash -l

# L4/L20 rollout of the ADOPTED config (user 2026-07-23): T2 (ctx-cache
# scratch 40k) + (7) ctx spec with per-layer dev scale + T4 adaptation.
# Stages (all resumable, one interact-g chain per layer):
#   0. (7) ctx specs for the layer (build_feature_specs_ctx --layers L)
#   1. ctx z sidecar for the training + dev caches (make_z_sidecar_ctx)
#   2. T2 scratch 40k (champion nb recipe on ctx cache)
#   3. (7) scale dev-selection (dev-200, true only, both dirs, mean)
#   4. eval500 fs_v6t2_l{L}{,_amp} (true,random)
#   5. T4: train-section cache -> 4k fine-tune -> fs_t4_l{L}{,_amp}
# Run: LAYER=4 bash run_t2_layer.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

LAYER=${LAYER:?set LAYER=4|20}
case "$LAYER" in
    4)  SAE="layer_4/width_16k/average_l0_60/params.npz"
        BLK=runs/blocklist_l4/blocklist.npy
        FLOW=runs/prod_gemma_v6/editflow_s3_l4/editflow-final.pt
        GRID="1.5 2.5 3.5" ;;
    20) SAE="layer_20/width_16k/average_l0_71/params.npz"
        BLK=runs/blocklist_l20/blocklist.npy
        FLOW=runs/prod_gemma_v6/editflow_s3_l20/editflow-final.pt
        GRID="1.0 1.5 2.5" ;;
    *)  echo "unsupported LAYER=$LAYER"; exit 1 ;;
esac
P=runs/prod_gemma_v6
FS=runs/feature_specs
V4=runs/prod_gemma_v4
SPLIT=runs/tables/eval_split.json

# ---- 0. (7) ctx specs for this layer -------------------------------------
if [ ! -f $FS/l${LAYER}_specctx.json ]; then
    python scripts/build_feature_specs_ctx.py \
        --out-dir $FS --split $SPLIT --layers $LAYER
fi

# ---- 1. ctx sidecars -----------------------------------------------------
CTX=$V4/corruption_zctx_l$LAYER
CTXDEV=$V4/corruption_seldev_zctx_l$LAYER
if [ ! -f $CTX/build.done ]; then
    python scripts/make_z_sidecar_ctx.py \
        --cache-dir $V4/corruption_z_l$LAYER --out-dir $CTX \
        --layer $LAYER && touch $CTX/build.done
fi
if [ ! -f $CTXDEV/build.done ]; then
    python scripts/make_z_sidecar_ctx.py \
        --cache-dir $V4/corruption_seldev_z_l$LAYER --out-dir $CTXDEV \
        --layer $LAYER && touch $CTXDEV/build.done
fi

# ---- 2. T2 scratch 40k ---------------------------------------------------
CACHE_DIR=$CTX DEV_DIR=$CTXDEV \
    LAYER=$LAYER FRAME=repeat EDIT_ONLY=1 LAM_SUP=0.2 \
    FLOW_INIT=$FLOW NORM_REG_W=0.0 NULL_NORM_W=0.0 \
    OUT_SUFFIX=_v6t2 MAX_STEPS=40000 bash run_ef_editor.sh

CK=$P/eflm_l${LAYER}_v6t2/eflm-final.pt

EVAL_COMMON=(--frame repeat --feature-spec $FS/l${LAYER}_specctx.json
    --arms ef --llm2vec-dir runs/mcgill_gemma_repro_3k/final
    --sae-path "$SAE" --sae-layer "$LAYER" --blocklist "$BLK"
    --k-amp 64 --k-sup 64 --ef-ckpt "$CK" --device cuda)

# ---- 3. scale dev-selection ---------------------------------------------
for SC in $GRID; do
    for DIRX in "" "_amp"; do
        if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
        O=$P/t2dev_l${LAYER}_s$SC$DIRX
        [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
            "${EVAL_COMMON[@]}" --fspec-scale $SC --conditions true \
            --pool-dev $SPLIT --sample-size 200 --output-dir "$O" $X
    done
done
SC=$(python - "$P" "$LAYER" $GRID <<'PY'
import re, sys
P, L, grid = sys.argv[1], sys.argv[2], sys.argv[3:]
best, bv = grid[0], -1.0
for sc in grid:
    vals = []
    for suf in ("", "_amp"):
        try:
            t = open(f"{P}/t2dev_l{L}_s{sc}{suf}/report.md").read()
            vals.append(float(re.search(
                r"\| true \| ef \| ([0-9.]+)", t).group(1)))
        except Exception:
            pass
    if len(vals) == 2 and sum(vals) / 2 > bv:
        bv, best = sum(vals) / 2, sc
print(best)
PY
)
echo "[t2layer] L$LAYER selected ctx scale = $SC"

# ---- 4. eval500 ----------------------------------------------------------
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    O=$P/fs_v6t2_l$LAYER$DIRX
    [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
        "${EVAL_COMMON[@]}" --fspec-scale $SC --conditions true,random \
        --sample-size 500 --output-dir "$O" $X
done

# ---- 5. T4 (user 2026-07-24: SKIPPED for L4/L20 — DO_T4=1 to enable) ----
if [ "${DO_T4:-0}" != "1" ]; then
    echo "[t2layer] T4 stage skipped (user decision)"
    echo "==================== T2T4-L$LAYER-DONE ===================="
    exit 0
fi
T4C=$V4/t4_ctx_l$LAYER
if ! python -c "import json,sys;m=json.load(open('$T4C/meta.json'));sys.exit(0 if 'd_sae' in m else 1)" 2>/dev/null; then
    python scripts/make_t4_cache.py \
        --spec $FS/l${LAYER}_specctx.json --split $SPLIT \
        --out $T4C --scale $SC --meta-from $CTX/meta.json
fi
OUT=$P/eflm_l${LAYER}_t4_v6t2
if [ ! -f "$OUT/eflm-final.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $T4C --dev-corruption-dir $CTXDEV \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$OUT" \
        --inject-layer $LAYER --sae-path "$SAE" \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat --init-ckpt "$CK" \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 4000 --resume --device cuda
fi
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    O=$P/fs_t4_l$LAYER$DIRX
    [ -f $O/report.md ] || python scripts/eval_ef_bare.py \
        "${EVAL_COMMON[@]}" --fspec-scale $SC --conditions true,random \
        --ef-ckpt "$OUT/eflm-final.pt" \
        --sample-size 500 --output-dir "$O" $X
done

echo "==================== T2T4-L$LAYER-DONE ===================="
