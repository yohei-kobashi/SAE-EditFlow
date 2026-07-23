#!/bin/bash -l

# T4 = LinguaLens-train adaptation (user 2026-07-23, auto-start rule:
# adopt the better of {T2+T3, T2} by min(abl, enh), then fine-tune THAT
# model on the train-section cache — pool-adapted row of the 2-row story).
#   env ADOPT = v6t23 | v6t2   (set by the decision driver)
# Stages (all resumable): T4 cache build -> 4k-step fine-tune from the
# adopted checkpoint -> (7)-spec evals both directions.
# Run inside interact-g: ADOPT=v6t23 bash run_t4.sh

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

ADOPT=${ADOPT:?set ADOPT=v6t23|v6t2}
P=runs/prod_gemma_v6
FS=runs/feature_specs
T4C=runs/prod_gemma_v4/t4_ctx_l12
CTXDEV=runs/prod_gemma_v4/corruption_seldev_zctx_l12
OUT=$P/eflm_l12_t4_$ADOPT

# Stage 1 — cache
if ! python -c "import json,sys;m=json.load(open('$T4C/meta.json'));sys.exit(0 if 'd_sae' in m else 1)" 2>/dev/null; then
    python scripts/make_t4_cache.py \
        --spec $FS/l12_specctx.json --split runs/tables/eval_split.json \
        --out $T4C --scale 3.5 \
        --meta-from runs/prod_gemma_v4/corruption_zctx_l12/meta.json
fi

# Stage 2 — fine-tune (bypasses run_ef_editor: its stage-1 hardcodes 10k)
EXTRA=()
[ "$ADOPT" = "v6t23" ] && EXTRA+=(--ins-loss-boost 1.5)
if [ ! -f "$OUT/eflm-final.pt" ]; then
    python train_ef_editor.py \
        --corruption-dir $T4C --dev-corruption-dir $CTXDEV \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --output-dir "$OUT" \
        --inject-layer 12 \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --batch-size 4 --grad-accum-steps 2 --num-workers 2 \
        --k-top 32 --k-amp log:1-32 --k-sup log:1-32 \
        --empty-prob 0.08 --mismatch-null-prob 0.12 --t0-prob 0.5 \
        --norm-alpha 0.5 --norm-reg-w 0.0 --null-norm-w 0.0 \
        --edit-only-loss --bg-weight 0.1 --lam-sup-w 0.2 \
        --frame repeat \
        --init-ckpt $P/eflm_l12_$ADOPT/eflm-final.pt \
        --dev-batches 48 --eval-steps 500 --save-steps 500 \
        --max-steps 4000 --resume --device cuda "${EXTRA[@]}"
fi

# Stage 3 — (7)-spec evals (pool-adapted row)
for DIRX in "" "_amp"; do
    if [ -n "$DIRX" ]; then X=--reverse-pairs; else X=""; fi
    if [ ! -f $P/fs_t4_l12$DIRX/report.md ]; then
        python scripts/eval_ef_bare.py \
            --frame repeat --feature-spec $FS/l12_specctx.json \
            --fspec-scale 3.5 --conditions true,random --arms ef \
            --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
            --sae-path layer_12/width_16k/average_l0_82/params.npz \
            --sae-layer 12 --blocklist runs/blocklist/blocklist.npy \
            --k-amp 64 --k-sup 64 \
            --ef-ckpt "$OUT/eflm-final.pt" \
            --sample-size 500 --device cuda \
            --output-dir $P/fs_t4_l12$DIRX $X
    fi
done

echo "==================== T4-DONE (base=$ADOPT) ===================="
