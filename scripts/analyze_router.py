"""
Phase 0 of the win-overall-exact strategy: measure the routing headroom
between the EF champion and the steered-regeneration baselines from
EXISTING records (no new generation).

Per pair we take candidate outputs (EF thr0.1, steer0.5, clamp10, ...),
and compute:
  * overlap structure and the ORACLE union exact (routing ceiling);
  * the UNSUPERVISED router: recompute each candidate's directional SAE
    achievement (sae_gain — no gold involved) and pick argmax, ties ->
    the FIRST candidate (EF), which structurally preserves EF's
    empty->no-edit (empty conditioning => all gains 0 => EF's copy);
  * everything reported separately on the 200-pair tuning prefix and
    the 300-pair holdout (router rules must be chosen on the prefix).

Usage (miyabi):
    python scripts/analyze_router.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --blocklist runs/blocklist/blocklist.npy \
        --cand ef=runs/prod_gemma_v6/editflow_s3/probe500/records.jsonl:thr0.1 \
        --cand steer=runs/prod_gemma_v6/steer_baseline500/records.jsonl:steer0.5 \
        --cand clamp=runs/prod_gemma_v6/clamp_baseline500/records.jsonl:clamp10 \
        --prefix-records runs/prod_gemma_v6/editflow_s3/probe/records.jsonl \
        --out runs/router/phase0.json --device cuda
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer                         # noqa: E402

from eval_lingualens import (                                  # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk,
    sae_z_with_offsets,
)
from model import SAEFeatureExtractor                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--cand", action="append", required=True,
                   help="name=records.jsonl:mode — FIRST candidate is the "
                        "tie-break default (use EF for premise safety)")
    p.add_argument("--prefix-records", required=True,
                   help="records whose idx set is the TUNING prefix "
                        "(e.g. the 200-pair probe); the rest is holdout")
    p.add_argument("--condition", default="true")
    p.add_argument("--out", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--blocklist", default="")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def bname(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4-8" if n <= 8 else "9+"


def main():
    args = parse_args()
    cands = []
    for spec in args.cand:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        with open(path) as f:
            recs = {r["idx"]: r for r in
                    (json.loads(l) for l in f if l.strip())}
        cands.append((name, recs, mode))
        print(f"[router] {name}: {len(recs)} records, mode {mode}")
    with open(args.prefix_records) as f:
        prefix_idx = {json.loads(l)["idx"] for l in f if l.strip()}
    common = sorted(set.intersection(*[set(r) for _, r, _ in cands]))
    print(f"[router] common pairs: {len(common)} "
          f"(prefix {sum(1 for k in common if k in prefix_idx)})")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    blk = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        blk = torch.as_tensor(np.asarray(_bl, dtype=np.int64))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out_path.with_suffix(".rows.jsonl")
    done = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[int(r["idx"])] = r
        print(f"[router] RESUME: {len(done)} rows")
    cf = open(cache_path, "a")

    @torch.no_grad()
    def pooled(text):
        return extractor.pool_max_topk(
            extractor.encode_text(text), args.pool_topk).float().cpu()

    rows = []
    for i, k in enumerate(common):
        if k in done:
            rows.append(done[k])
            continue
        base = cands[0][1][k]
        src = base.get("src") or base.get("source")
        tgt = base.get("tgt") or base.get("target")
        n_ops = base.get("n_ops", 1)
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids
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
        za, zs = diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)
        am, sm = za > 0, zs > 0
        total = float(za[am].sum() + zs[sm].sum())
        z_in = pooled(src)

        row = {"idx": int(k), "n_ops": n_ops, "cands": {}}
        for name, recs, mode in cands:
            node = recs[k]["outputs"][args.condition].get(mode)
            if not isinstance(node, dict):
                continue
            text = node["text"]
            if total <= 0:
                gain = 0.0
            else:
                z_out = pooled(text)
                delta = z_out - z_in
                g = torch.clamp(delta[am], -za[am], za[am]).sum()
                g = g + torch.clamp(-delta[sm], -zs[sm], zs[sm]).sum()
                gain = float(g) / (total + 1e-8)
            row["cands"][name] = {"exact": node["exact"],
                                  "sim": node["sim_target"],
                                  "gain": gain}
        rows.append(row)
        done[k] = row
        cf.write(json.dumps(row) + "\n")
        cf.flush()
        if (i + 1) % 25 == 0:
            print(f"[router] {i + 1}/{len(common)}")
    cf.close()

    names = [n for n, _, _ in cands]

    def report(subset, label):
        lines = [f"## {label} (n={len(subset)})", ""]
        for n in names:
            ex = np.mean([r["cands"][n]["exact"] for r in subset
                          if n in r["cands"]])
            lines.append(f"- {n}: exact {ex:.4f}")
        oracle = np.mean([max(c["exact"] for c in r["cands"].values())
                          for r in subset])
        lines.append(f"- **oracle union: {oracle:.4f}**")
        # unsupervised router: argmax gain, tie -> first candidate
        r_ex, picks = [], defaultdict(int)
        for r in subset:
            best, best_g = names[0], -1e9
            for n in names:
                if n in r["cands"] and r["cands"][n]["gain"] > best_g + 1e-9:
                    best, best_g = n, r["cands"][n]["gain"]
            picks[best] += 1
            r_ex.append(r["cands"][best]["exact"])
        lines.append(f"- **gain-router: {np.mean(r_ex):.4f}** "
                     f"(picks: {dict(picks)})")
        # per-bucket
        byb = defaultdict(list)
        for r in subset:
            byb[bname(int(r["n_ops"]))].append(r)
        for b in ("1", "2-3", "4-8", "9+"):
            rs = byb.get(b, [])
            if not rs:
                continue
            cells = " ".join(
                f"{n}={np.mean([r['cands'][n]['exact'] for r in rs]):.3f}"
                for n in names)
            orc = np.mean([max(c["exact"] for c in r["cands"].values())
                           for r in rs])
            lines.append(f"  - {b} (n={len(rs)}): {cells} "
                         f"oracle={orc:.3f}")
        lines.append("")
        return lines

    tune = [r for r in rows if r["idx"] in prefix_idx]
    hold = [r for r in rows if r["idx"] not in prefix_idx]
    lines = ["# Router phase-0: headroom and unsupervised selection", ""]
    lines += report(rows, "ALL")
    lines += report(tune, "TUNING PREFIX")
    lines += report(hold, "HOLDOUT")
    report_text = "\n".join(lines)
    print(report_text)
    out_path.write_text(json.dumps({"rows": rows}, indent=0))
    out_path.with_suffix(".md").write_text(report_text + "\n")
    print(f"[router] wrote {out_path} (+ .md)")


if __name__ == "__main__":
    main()
