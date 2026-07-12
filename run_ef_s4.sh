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

# S4 (EDIT_FLOWS_ZERO §5): the 500-pair judgment vs the v6 pipeline.
# S3 split the frontier — S3 thr0.1 is the exact champion (0.2010/0.6177),
# S2 thr0.5 keeps the sim crown (0.1859/0.6766) — so BOTH checkpoints run
# on the SAME 500 pairs. The probe and eval_lingualens.py shuffle the
# LinguaLens indices with the same seed, so --sample-size 500 reproduces
# the pipeline e2e sample exactly and compare_ef_pipeline.py joins the
# records on idx (matched pairs, no cross-sample noise). The 200-pair
# probe sample is a prefix of the 500, so --exclude on the probe records
# yields a 300-pair holdout that the operating-F selection never saw.
# ~6h total; each probe resumes per pair — resubmit until "S4 DONE".
V6=./runs/prod_gemma_v6
LLM2VEC=runs/mcgill_gemma_repro_3k/final
BLOCKLIST=runs/blocklist/blocklist.npy
PIPE=$V6/eval_lingualens_final/records.jsonl

# sim-side champion: S2 (operating point thr0.5; det = zero-knob ref)
if [ ! -f "$V6/editflow_s2/probe500/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$V6/editflow_s2/editflow-final.pt" \
        --output-dir "$V6/editflow_s2/probe500" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --steps 48 --steer-lambda 1 \
        --decode det,thr0.25,thr0.5 \
        --device cuda
fi

# exact-side champion: S3 (operating point thr0.1; thr0.05 = exact-max ref)
if [ ! -f "$V6/editflow_s3/probe500/probe_report.md" ]; then
    python scripts/editflow_probe.py \
        --llm2vec-dir "$LLM2VEC" \
        --editflow-ckpt "$V6/editflow_s3/editflow-final.pt" \
        --output-dir "$V6/editflow_s3/probe500" \
        --cond-scope local --blocklist "$BLOCKLIST" \
        --k-amp 64 --k-sup 64 --sample-size 500 \
        --steps 48 --steer-lambda 1 \
        --decode det,thr0.05,thr0.1,thr0.5 \
        --device cuda
fi

echo "==================== S4 DONE ===================="
for M in s2 s3; do
    echo
    echo "======== editflow_$M vs pipeline: ALL matched pairs ========"
    python scripts/compare_ef_pipeline.py \
        --ef "$V6/editflow_$M/probe500/records.jsonl" \
        --pipeline "$PIPE"
    echo
    echo "======== editflow_$M vs pipeline: 300-pair HOLDOUT ========"
    python scripts/compare_ef_pipeline.py \
        --ef "$V6/editflow_$M/probe500/records.jsonl" \
        --pipeline "$PIPE" \
        --exclude "$V6/editflow_$M/probe/records.jsonl"
done
echo
echo "Pre-registered outcomes (operating points: S2 thr0.5, S3 thr0.1):"
echo "  (a) an EF mode >= pipeline on BOTH exact and sim (matched pairs,"
echo "      holdout consistent) -> EF replaces the pipeline. Candidate:"
echo "      S2 thr0.5 (cross-sample 0.186/0.6766 vs 0.112/0.6687 — the"
echo "      sim edge is the open question);"
echo "  (b) EF wins exact, pipeline keeps sim -> record the frontier"
echo "      (S3 thr0.1 exact +76% is the far point);"
echo "  (c) parity on both -> wrap the EF chapter, reopen V7_PLAN."
echo "  Watch: random ranking control ~0.37 (feature-token cost) and"
echo "  random no_edit at the operating points (bars: >= 0.88)."
