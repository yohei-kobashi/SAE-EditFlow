#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

cd ~/
source start_gpu_nodes.sh
cd ~/SAE-LEWIS

set -eo pipefail

# IMPROVING THE INTERVENTION (2026-07-16). Target, per the research goal:
# OUR intervention must beat LinguaLens's clamp and the AxBench steering
# family on exact (and ideally FRR). Losing to prompting is acceptable.
# So the bar is not B1 — the bar is B3.
#
#   B1 clamp10  exact 0.1743   LinguaLens-faithful
#   B3 steer0.5 exact 0.2337   <- THE BAR
#
# B1's three handicaps, each removable, in order of known size:
#
#  1. RECONSTRUCTION. B1 does encode->set->decode, replacing the residual
#     with the SAE reconstruction everywhere. B3 adds the delta and touches
#     nothing else. That difference alone is +0.059 exact, already measured.
#     -> --intervention delta
#  2. SCOPE. LinguaLens intervenes at EVERY position because it does not know
#     where the phenomenon lives. A minimal-pair edit is local, so
#     intervening everywhere corrupts what should be preserved. The SAE says
#     where: the suppressed features FIRE at particular source tokens.
#     No training, no target — just the source's activation pattern.
#     -> --scope local        (UNTESTED — this is the new idea)
#  3. FREE REGENERATION. B1/B3 make the LM rewrite the whole sentence, so
#     every unchanged token is a chance to lose exact. Read the edit out of
#     the intervened LM's own head instead, teacher-forced.
#     -> the readout itself   (UNTESTED)
#
# Plus the three things EF used at its best that the readout was missing:
#  4. ITERATION. EF edits, RE-READS x_t, edits again (48 steps). A single
#     shot cannot fix a token whose context only goes wrong after an earlier
#     edit, and cannot decompose a multi-token insertion into single ones.
#     -> --steps 8
#  5. TOP-RATE FIRING, not everything past a bar (EF's det/thr modes).
#     -> --max-ops-per-step 4
#  6. THE OP VOCABULARY. The readout was SUB-only, so half the phenomena were
#     structurally unreachable. INS/DEL now come from the intervened head
#     itself: DEL when it prefers the token AFTER this one; INS when it wants
#     v but still wants the original too. apply_step_ops applies them — a
#     PURE FUNCTION with no parameters, so the causal claim is untouched.
#
# Everything here is an intervention on gemma-2-2b's layer-12 residual
# stream, and nothing here is trained. The causal claim survives all three.
V6=./runs/prod_gemma_v6
# v2: WHAT = pmi (the token the intervention promoted most, top-50-restricted)
# + the clamp-path position mask actually consumed. v1 (clamp_readout500/) is
# kept untouched — it IS the --what int ablation arm, and its verdict stands:
# empty perfect, WHERE causal (true fires 2-3x random), WHAT dominated by the
# LM prior (sim fell below the 0.6033 copy baseline at every setting).
R=$V6/clamp_readout500_v2

# ---- the 2x2: {clamp, delta} x {all, local}, all with the readout --------
# clamp+all is B1's intervention with only handicap 3 removed, so it isolates
# the readout's contribution against B1's 0.1743 directly.
for IV in clamp delta; do
    for SC in all local; do
        D="$R/${IV}_${SC}"
        if [ ! -f "$D/report.md" ]; then
            echo "-------- intervention=$IV scope=$SC"
            V=$([ "$IV" = clamp ] && echo "5,10,20" || echo "0.5,1,2")
            python scripts/eval_clamp_readout.py \
                --output-dir "$D" \
                --intervention "$IV" --scope "$SC" \
                --clamp-values "$V" --delta-thr -1.0 \
                --steps 8 --max-ops-per-step 4 \
                --conditions true,empty,random \
                --sample-size 500 --device cuda
        fi
    done
done

# ---- threshold sweep on the winning cell (delta+local a priori) ----------
for THR in -0.5 -2.0 -4.0; do
    D="$R/delta_local_thr${THR}"
    if [ ! -f "$D/report.md" ]; then
        python scripts/eval_clamp_readout.py \
            --output-dir "$D" --intervention delta --scope local \
            --clamp-values 0.5,1,2 --delta-thr "$THR" \
            --steps 8 --max-ops-per-step 4 \
            --conditions true,empty,random \
            --sample-size 500 --device cuda
    fi
done

echo
echo "==================== B1 IMPROVEMENT DONE ===================="
echo "Read the four report.md in $R/{clamp,delta}_{all,local}/ as a ladder:"
echo
echo "  clamp + all   -> B1's intervention, readout instead of regeneration."
echo "                   vs B1's 0.1743 = what the readout alone buys."
echo "  delta + all   -> also drops reconstruction. vs B3's 0.2337 = what the"
echo "                   readout buys on top of the intervention B3 already uses."
echo "  delta + local -> also drops the everywhere-scope. THE candidate."
echo "  clamp + local -> localization without the delta fix; separates the two."
echo
echo "The bar to clear is B3 = 0.2337. Beating B1 (0.1743) is not enough —"
echo "B3 already does, with no editor and no readout."
echo
echo "Then, in every report:"
echo "  * empty MUST be exact~0 / copy 1.00 / fires 0.00. With delta the spec"
echo "    is zero so the hook adds a zero vector: Delta is identically 0,"
echo "    structurally. If empty is not clean, nothing else can be trusted."
echo "  * true vs random is the causal test — same count, same magnitudes,"
echo "    only the feature IDENTITIES differ."
echo "  * n_masked reports how many positions scope=local touched. If it is"
echo "    ~= sequence length, localization did nothing and the comparison"
echo "    against scope=all is vacuous."
echo
echo "Ablate iteration with --steps 1 (the old single-shot readout) if the"
echo "iterative numbers look good — EF's gain came from re-reading, and we"
echo "should know how much of ours does too."
