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

# Paper-ToDo master (items 1-6): every stage is guarded by its output
# file and every underlying script resumes per pair, so this job can be
# resubmitted until "ALL PAPER-TODO STAGES DONE" prints. Total ~12-16h
# => expect 2-3 submissions.
#   A  B1  LinguaLens clamp (faithful OpenSAE mechanics)      ~2.5h
#   B  B3  steering vector (same frame, delta-add)            ~2h
#   C  FRR LinguaLens-basis judgments, all systems + controls ~1.5h
#   D  P-A/P-C conditioning k-sweep on the champion           ~3h
#   E  P-B  FRC identification + identified-set conditioning  ~3h
#   F  SLOR grammaticality pass over all baseline outputs     ~1h
#   G  comparison printouts (compare ext., no GPU)            ~min
V4=./runs/prod_gemma_v4
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
PIPE=$V6/eval_lingualens_final/records.jsonl
EXPL=runs/np_explanations/gemma-2-2b_12-res-16k.json
S3REC=$V6/editflow_s3/probe500/records.jsonl
UNIG=$V4/corruption/unigram.json

# ---- Stage A: B1 (LinguaLens clamp) ---------------------------------
if [ ! -f "$V6/clamp_baseline500/report.md" ]; then
    python scripts/eval_clamp_baseline.py \
        --llm2vec-dir "$LLM2VEC" --blocklist "$BLOCKLIST" \
        --output-dir "$V6/clamp_baseline500" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --intervention clamp --clamp-values 5,10,20 --device cuda
fi

# ---- Stage B: B3 (steering vector) ----------------------------------
if [ ! -f "$V6/steer_baseline500/report.md" ]; then
    python scripts/eval_clamp_baseline.py \
        --llm2vec-dir "$LLM2VEC" --blocklist "$BLOCKLIST" \
        --output-dir "$V6/steer_baseline500" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --intervention steer --clamp-values 0.5,1,2,4 --device cuda
fi

# ---- Stage C: FRR (LinguaLens-basis judgments) -----------------------
JUDGE=${JUDGE:-hf:google/gemma-2-9b-it}
TAG=$(echo "$JUDGE" | tr ':/' '__')
FRR=runs/frr/$TAG
GOLD=$FRR/gold.jsonl
frr () {  # label records mode condition
    if [ ! -f "$FRR/.done.$1" ]; then
        python scripts/judge_feature_realization.py \
            --records "$2" --mode "$3" --condition "$4" \
            --gold-cache "$GOLD" --judge "$JUDGE" \
            --n-ops-ref "$S3REC" --out "$FRR/$1.jsonl" --device cuda
        touch "$FRR/.done.$1"
    fi
}
mkdir -p "$FRR"
frr ef_s3_thr01     "$S3REC"                            "thr0.1"  true
frr ef_s3_thr05     "$S3REC"                            "thr0.5"  true
frr pipeline        "$PIPE"                             ""        true
frr b2_prompt8      "$V6/prompt_baseline500/records.jsonl" "prompt8" true
frr b1_clamp10      "$V6/clamp_baseline500/records.jsonl" "clamp10" true
frr b1_clampZ       "$V6/clamp_baseline500/records.jsonl" "clampZ"  true
frr b3_steer1       "$V6/steer_baseline500/records.jsonl" "steer1"  true
frr ef_s3_thr01_rnd "$S3REC"                            "thr0.1"  random
frr ef_s3_thr05_rnd "$S3REC"                            "thr0.5"  random
frr b2_prompt8_rnd  "$V6/prompt_baseline500/records.jsonl" "prompt8" random
frr b1_clamp10_rnd  "$V6/clamp_baseline500/records.jsonl" "clamp10" random
frr b3_steer1_rnd   "$V6/steer_baseline500/records.jsonl" "steer1"  random
frr b2_prompt8_empty "$V6/prompt_baseline500/records.jsonl" "prompt8" empty
frr b1_recon_empty  "$V6/clamp_baseline500/records.jsonl" "recon"   empty

# ---- Stage D: P-A/P-C conditioning k-sweep ---------------------------
if [ ! -f "$V6/ksweep500/report.md" ]; then
    python scripts/eval_k_sweep.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$V6/editflow_s3/editflow-final.pt" \
        --blocklist "$BLOCKLIST" \
        --output-dir "$V6/ksweep500" \
        --k-grid 1,2,4,8,16,32,64 --decode thr0.1 \
        --sample-size 500 --device cuda
fi

# ---- Stage E: P-B FRC identification + identified conditioning -------
if [ ! -f "runs/frc/identified_l12_16k.json" ]; then
    python scripts/identify_features_frc.py \
        --out runs/frc/identified_l12_16k.json \
        --explanations "$EXPL" --device cuda
fi
for FM in intersect pure; do
    if [ ! -f "$V6/editflow_s3/probe500_frc_$FM/probe_report.md" ]; then
        python scripts/editflow_probe.py \
            --llm2vec-dir "$LLM2VEC" \
            --editflow-ckpt "$V6/editflow_s3/editflow-final.pt" \
            --output-dir "$V6/editflow_s3/probe500_frc_$FM" \
            --cond-scope local --blocklist "$BLOCKLIST" \
            --k-amp 64 --k-sup 64 --sample-size 500 \
            --steps 48 --steer-lambda 1 \
            --decode thr0.1,thr0.5 \
            --feature-sets runs/frc/identified_l12_16k.json \
            --feature-mode "$FM" \
            --conditions true \
            --device cuda
    fi
done

# ---- Stage F: SLOR grammaticality pass -------------------------------
slor () {  # tag records modes condition
    if [ ! -f "runs/slor/$1.json" ]; then
        python scripts/score_slor.py --records "$2" --modes "$3" \
            --condition "$4" --unigram "$UNIG" \
            --out "runs/slor/$1.json" --device cuda
    fi
}
slor ef_s3    "$S3REC"                              "thr0.1,thr0.5,det" true
slor pipeline "$PIPE"                               ""                  true
slor b2       "$V6/prompt_baseline500/records.jsonl" "prompt8,prompt16" true
slor b1       "$V6/clamp_baseline500/records.jsonl" "clamp5,clamp10,clamp20,clampZ" true
slor b1_ctrl  "$V6/clamp_baseline500/records.jsonl" "recon,raw"         empty
slor b3       "$V6/steer_baseline500/records.jsonl" "steer0.5,steer1,steer2,steer4" true

# ---- Stage G: comparisons (CPU, always re-printed) --------------------
echo "==================== ALL PAPER-TODO STAGES DONE ===================="
echo
echo "######## B1 report ########";  cat "$V6/clamp_baseline500/report.md"
echo
echo "######## B3 report ########";  cat "$V6/steer_baseline500/report.md"
echo
echo "######## B1/B3 vs pipeline (matched + holdout) ########"
for D in clamp_baseline500 steer_baseline500; do
    python scripts/compare_ef_pipeline.py \
        --ef "$V6/$D/records.jsonl" --pipeline "$PIPE"
    python scripts/compare_ef_pipeline.py \
        --ef "$V6/$D/records.jsonl" --pipeline "$PIPE" \
        --exclude "$V6/editflow_s3/probe/records.jsonl"
done
echo
echo "######## EF (champion) vs baselines DIRECT paired stats ########"
python scripts/compare_ef_pipeline.py --ef "$S3REC" \
    --pipeline "$V6/prompt_baseline500/records.jsonl" --pipeline-mode prompt8
python scripts/compare_ef_pipeline.py --ef "$S3REC" \
    --pipeline "$V6/clamp_baseline500/records.jsonl" --pipeline-mode clamp10
python scripts/compare_ef_pipeline.py --ef "$S3REC" \
    --pipeline "$V6/steer_baseline500/records.jsonl" --pipeline-mode steer1
echo
echo "######## FRR summaries ########"
for f in "$FRR"/*.jsonl; do
    [ "$f" = "$GOLD" ] && continue
    n=$(wc -l < "$f")
    ok=$(grep -c '"realized": true' "$f" || true)
    ex=$(grep -c '"realized": null' "$f" || true)
    echo "$(basename "$f" .jsonl): realized $ok / scored $((n-ex)) (gold-equal excluded $ex)"
done
echo
echo "######## P-A/P-C k-sweep ########"; cat "$V6/ksweep500/report.md"
echo
echo "######## P-B identified-set conditioning (vs full-diff thr0.1 0.1904) ########"
for FM in intersect pure; do
    echo "--- feature-mode $FM ---"
    sed -n '/## Decode quality/,/## Multi/p' \
        "$V6/editflow_s3/probe500_frc_$FM/probe_report.md" | head -12
done
echo
echo "######## SLOR ########"
for f in runs/slor/*.json; do echo "--- $f"; cat "$f"; echo; done
echo
echo "P-B/P-D comparison note: overlap between FRC-identified sets and"
echo "the small-k diff features that P-A shows are sufficient is computed"
echo "offline from runs/frc/identified_l12_16k.json + ksweep records"
echo "(n_amp/n_sup + gains are in the records) — paste results for the"
echo "verdict; the causal minimal-set (P-D) rides the same records."
