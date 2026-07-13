"""
P-A + P-C (PAPER_OUTLINE / activation-identification track): how many
conditioning features does the EF champion actually need?

For every pair, decode at k_amp = k_sup = k for k in --k-grid (top-|dz|
order, the probe's own conditioning path). Training drew k ~
log-uniform{1..32}, so small k is IN-distribution. Per mode `k{m}` we
record text/exact/sim/copy AND the unsupervised directional SAE
achievement (sae_gain — the bo-K selector), which enables:

  P-A (analysis) : exact-vs-k frontier; r95 = smallest k reaching 95%
                   of the full-k exact; ORACLE per-pair minimal-k
                   distribution (labelled analysis only — selecting k by
                   gold match is test-label peeking, never a method);
  P-C (method)   : deployable minimal-k selection WITHOUT labels —
                   smallest k whose sae_gain >= tau (else largest k),
                   evaluated post-hoc from the same records for a tau
                   grid. No extra decodes.

Usage (miyabi):
    python scripts/eval_k_sweep.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --editflow-ckpt runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
        --blocklist runs/blocklist/blocklist.npy \
        --output-dir runs/prod_gemma_v6/ksweep500 --device cuda
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer                         # noqa: E402

from editflow import load_editflow_from_checkpoint             # noqa: E402
from editflow_ops import align_pair, slot_ops                  # noqa: E402
from editflow_probe import decode_flow                         # noqa: E402
from eval_lingualens import (                                  # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk, pair_metrics,
    sae_z_with_offsets,
)
from model import SAEFeatureExtractor, load_sae_w_dec          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--editflow-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--blocklist", default="")
    p.add_argument("--k-grid", default="1,2,4,8,16,32,64")
    p.add_argument("--decode", default="thr0.1",
                   help="single thr{F} decode (the champion's operating "
                        "point)")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--steer-lambda", type=float, default=1.0)
    p.add_argument("--w-max", type=float, default=20.0)
    p.add_argument("--max-ops-per-step", type=int, default=8)
    p.add_argument("--max-grow", type=int, default=24)
    p.add_argument("--taus", default="0.3,0.5,0.7",
                   help="P-C selection thresholds on sae_gain")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


def bname(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4-8" if n <= 8 else "9+"


def main():
    args = parse_args()
    k_grid = [int(x) for x in args.k_grid.split(",") if x]
    taus = [float(x) for x in args.taus.split(",") if x]
    thr_frac = float(args.decode.replace("thr", ""))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen = order[:min(args.sample_size, len(order))]
    print(f"[ksweep] {len(ds)} pairs, sampling {len(chosen)}; "
          f"k grid {k_grid}, decode {args.decode}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    suppress = [tokenizer.mask_token_id,
                tokenizer.convert_tokens_to_ids("[INS]"),
                tokenizer.convert_tokens_to_ids("[SEP]"),
                tokenizer.convert_tokens_to_ids("[DEL]"),
                tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id]
    suppress = sorted({int(s) for s in suppress if s is not None})

    model = load_editflow_from_checkpoint(
        args.llm2vec_dir, args.editflow_ckpt, dtype=dtype,
    ).to(args.device).eval()
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    blk = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        blk = torch.as_tensor(np.asarray(_bl, dtype=np.int64))
    w_dec = load_sae_w_dec(args.sae_repo, args.sae_path).to(args.device)
    head_w = model.lm_head.weight.detach().float().to(args.device)

    def lens_bias(za_v, zs_v):
        d = (za_v.to(args.device) - zs_v.to(args.device)) @ w_dec
        lb = head_w @ d
        s = lb.std()
        if float(s) < 1e-6:
            return None
        return args.steer_lambda * lb / (s + 1e-8)

    partial_path = out_dir / "records.partial.jsonl"
    records, done_idx = [], set()
    if partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(r)
                done_idx.add(int(r["idx"]))
        print(f"[ksweep] RESUME: {len(records)} pairs")
    pf = open(partial_path, "a")

    for step_i, k in enumerate(chosen):
        if int(k) in done_idx:
            continue
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids
        slots = align_pair(src_ids, tgt_ids)
        n_ops = len(slot_ops(slots))
        if n_ops == 0:
            continue

        with torch.no_grad():
            s_off, z_s = sae_z_with_offsets(extractor, src, args.device)
            t_off, z_t = sae_z_with_offsets(extractor, tgt, args.device)
            om_s = [tuple(o) for o in tokenizer(
                src, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            om_t = [tuple(o) for o in tokenizer(
                tgt, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            opcodes = difflib.SequenceMatcher(
                None, src_ids, tgt_ids, autojunk=False).get_opcodes()
            sr, tr = edit_char_ranges(opcodes, om_s, om_t)
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
            z_in_global = extractor.pool_max_topk(
                extractor.encode_text(src), args.pool_topk).float().cpu()

        def sae_gain(out_ids, za_v, zs_v):
            am, sm = za_v > 0, zs_v > 0
            total = float(za_v[am].sum() + zs_v[sm].sum())
            if total <= 0:
                return 0.0
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            with torch.no_grad():
                z_out = extractor.pool_max_topk(
                    extractor.encode_text(text),
                    args.pool_topk).float().cpu()
            delta = z_out - z_in_global
            g = torch.clamp(delta[am], -za_v[am], za_v[am]).sum()
            g = g + torch.clamp(-delta[sm], -zs_v[sm], zs_v[sm]).sum()
            return float(g) / (total + 1e-8)

        rec = {"idx": int(k), "src": src, "tgt": tgt, "n_ops": n_ops,
               "outputs": {"true": {}}}
        for m in k_grid:
            za, zs = diff_intervention(z_src, z_tgt, m, m)
            lb = lens_bias(za, zs)
            out_ids = decode_flow(
                model, src_ids, za.unsqueeze(0).to(args.device),
                zs.unsqueeze(0).to(args.device), steps=args.steps,
                device=args.device, mode="thr", thr_frac=thr_frac,
                w_max=args.w_max, lens_bias=lb,
                max_ops_per_step=args.max_ops_per_step,
                max_grow=args.max_grow, suppress_ids=suppress)
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            pm = pair_metrics(out_text, src, tgt)
            rec["outputs"]["true"][f"k{m}"] = {
                "text": out_text, "exact": pm["exact_match"],
                "sim_target": pm["sim_target"], "copy": pm["copy_rate"],
                "no_edit": float(out_ids == src_ids),
                "gain": sae_gain(out_ids, za, zs),
                "n_amp": int((za > 0).sum()), "n_sup": int((zs > 0).sum())}
        records.append(rec)
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        if (step_i + 1) % 10 == 0:
            print(f"[ksweep] {step_i + 1}/{len(chosen)} "
                  f"({len(records)} scored)")
    pf.close()

    # ---- report ----------------------------------------------------------
    lines = ["# Conditioning k-sweep (P-A frontier + P-C selection)", ""]
    lines.append(f"pairs: {len(records)}; decode {args.decode}; "
                 f"ckpt {args.editflow_ckpt}")
    lines += ["", "## P-A — exact/sim vs k (top-|dz| order)", "",
              "| k | exact | sim | copy | mean gain |", "|---|---|---|---|---|"]
    for m in k_grid:
        rows = [r["outputs"]["true"][f"k{m}"] for r in records]
        lines.append(f"| {m} | {np.mean([r['exact'] for r in rows]):.4f} | "
                     f"{np.mean([r['sim_target'] for r in rows]):.4f} | "
                     f"{np.mean([r['copy'] for r in rows]):.4f} | "
                     f"{np.mean([r['gain'] for r in rows]):.4f} |")
    full = np.mean([r["outputs"]["true"][f"k{k_grid[-1]}"]["exact"]
                    for r in records])
    r95 = next((m for m in k_grid if np.mean(
        [r["outputs"]["true"][f"k{m}"]["exact"] for r in records])
        >= 0.95 * full), k_grid[-1])
    lines += ["", f"r95 (smallest k reaching 95% of k={k_grid[-1]} exact "
                  f"{full:.4f}): **{r95}**"]

    # oracle per-pair minimal k (ANALYSIS ONLY)
    dist = defaultdict(int)
    for r in records:
        mk = next((m for m in k_grid
                   if r["outputs"]["true"][f"k{m}"]["exact"] > 0), None)
        dist["never" if mk is None else str(mk)] += 1
    lines += ["", "## Oracle per-pair minimal k (analysis only — "
                  "label-peeking, not a method)", "",
              "| min k | pairs |", "|---|---|"]
    for key in [str(m) for m in k_grid] + ["never"]:
        if dist.get(key):
            lines.append(f"| {key} | {dist[key]} |")

    # P-C: unsupervised smallest-k-with-gain>=tau selection
    lines += ["", "## P-C — deployable selection: smallest k with "
                  "sae_gain >= tau (else largest)", "",
              "| tau | exact | sim | mean chosen k |", "|---|---|---|---|"]
    for tau in taus:
        ex_l, sim_l, kk = [], [], []
        for r in records:
            sel = next((m for m in k_grid
                        if r["outputs"]["true"][f"k{m}"]["gain"] >= tau),
                       k_grid[-1])
            o = r["outputs"]["true"][f"k{sel}"]
            ex_l.append(o["exact"])
            sim_l.append(o["sim_target"])
            kk.append(sel)
        lines.append(f"| {tau:g} | {np.mean(ex_l):.4f} | "
                     f"{np.mean(sim_l):.4f} | {np.mean(kk):.1f} |")

    report = "\n".join(lines)
    print(report)
    (out_dir / "report.md").write_text(report + "\n")
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)
    print(f"[ksweep] wrote {out_dir}/report.md, records.jsonl")


if __name__ == "__main__":
    main()
