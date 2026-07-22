"""Spec variants for improvement ② (user-approved 2026-07-22):
top-k narrowing x sign-consistency filtering of the pool-mean feature spec.

From the build_feature_specs sidecar, per feature and per component:
  support     = fraction of pool pairs where the component's delta != 0
  consistency = max(#positive, #negative) / (#positive + #negative)
Variants: k in {8,16,32,64} x {nofilter, c70 = keep components with
consistency >= 0.7 and support >= 0.1}. mean values unchanged (still the
plain pool mean); only the KEPT set changes. mean_norm is recomputed on
the kept top-k vector so the eval-time norm-median rescale stays
comparable across variants.

Usage:
    python scripts/build_spec_variants.py \
        --pairs runs/feature_specs/l12_pairs.jsonl \
        --split runs/tables/eval_split.json \
        --base runs/feature_specs/l12_spec.json \
        --out-prefix runs/feature_specs/l12_spec
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--base", required=True,
                   help="l{L}_spec.json (for norm_median passthrough)")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--ks", default="8,16,32,64")
    p.add_argument("--cons", type=float, default=0.7)
    p.add_argument("--min-support", type=float, default=0.1)
    args = p.parse_args()

    eval_idx = set(json.loads(Path(args.split).read_text())["eval_idx"])
    base = json.loads(Path(args.base).read_text())

    acc = defaultdict(lambda: defaultdict(float))
    pos = defaultdict(lambda: defaultdict(int))
    neg = defaultdict(lambda: defaultdict(int))
    npairs = defaultdict(int)
    with open(args.pairs) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r["idx"]) in eval_idx:
                continue
            ph = r["feature"]
            npairs[ph] += 1
            for i, v in r["delta"].items():
                i = int(i)
                acc[ph][i] += float(v)
                if v > 0:
                    pos[ph][i] += 1
                elif v < 0:
                    neg[ph][i] += 1

    ks = [int(x) for x in args.ks.split(",")]
    for k in ks:
        for tag, use_filter in (("", False), ("_c70", True)):
            out = {}
            for ph, comp in acc.items():
                n = npairs[ph]
                cand = []
                for i, s in comp.items():
                    np_, nn = pos[ph][i], neg[ph][i]
                    supp = (np_ + nn) / n
                    cons = max(np_, nn) / max(np_ + nn, 1)
                    if use_filter and (cons < args.cons
                                       or supp < args.min_support):
                        continue
                    cand.append((i, s / n))
                cand.sort(key=lambda x: -abs(x[1]))
                keep = cand[:k]
                vec = np.zeros(1)  # norm only
                mn = float(np.sqrt(sum(v * v for _, v in keep)))
                out[ph] = {
                    "n": n,
                    "spec": {int(i): round(float(v), 5) for i, v in keep},
                    "mean_norm": round(mn, 5),
                    "norm_median": base[ph]["norm_median"],
                    "splithalf_cos": base[ph].get("splithalf_cos"),
                }
            path = f"{args.out_prefix}_k{k}{tag}.json"
            Path(path).write_text(json.dumps(out))
            kept = sum(len(v["spec"]) for v in out.values()) / len(out)
            print(f"[variants] {path}: {len(out)} features, "
                  f"avg kept {kept:.1f}")
    print("VARIANTS-BUILT")


if __name__ == "__main__":
    main()
