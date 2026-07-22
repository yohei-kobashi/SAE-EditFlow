"""v6/T1: pseudo-feature group means over the training corruption cache.

Groups every training sample by the DOMINANT latent of its SAE delta
(argmax |z_X' − z_X|) and stores each group's mean delta (sparse top-K).
Training-time augmentation mixes a pair's own delta with its group mean —
the feature-mean statistics of evaluation, reproduced with RELATED
samples (unlike the failed unrelated-batch mix, the signal survives).

Usage:
    python scripts/build_group_means.py \
        --cache runs/prod_gemma_v4/corruption_z_l12 \
        --out runs/prod_gemma_v4/corruption_z_l12/group_means.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sparse_delta(row, d_sae):
    d = np.zeros(d_sae, dtype=np.float64)
    for e in row["z_X_prime_topk"]:
        d[int(e["f"])] += float(e["v"])
    for e in row["z_X_topk"]:
        d[int(e["f"])] -= float(e["v"])
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--d-sae", type=int, default=16384)
    p.add_argument("--top-store", type=int, default=128)
    p.add_argument("--min-group", type=int, default=5,
                   help="groups smaller than this are dropped (their "
                        "mean is dominated by the sample itself)")
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.cache}/shard-*.jsonl.gz"))
    acc = defaultdict(lambda: np.zeros(args.d_sae, dtype=np.float64))
    cnt = defaultdict(int)
    n = 0
    for f in files:
        for line in gzip.open(f, "rt"):
            if not line.strip():
                continue
            r = json.loads(line)
            d = sparse_delta(r, args.d_sae)
            if not np.abs(d).any():
                continue
            dom = int(np.argmax(np.abs(d)))
            acc[dom] += d
            cnt[dom] += 1
            n += 1
        print(f"[groups] {f.split('/')[-1]}: {n} samples, "
              f"{len(acc)} groups", flush=True)
    out = {}
    for dom, s in acc.items():
        if cnt[dom] < args.min_group:
            continue
        mean = s / cnt[dom]
        order = np.argsort(-np.abs(mean))[:args.top_store]
        out[int(dom)] = {int(i): round(float(mean[i]), 5)
                         for i in order if abs(mean[i]) > 0}
    Path(args.out).write_text(json.dumps(out))
    sizes = sorted(cnt.values(), reverse=True)
    print(f"[groups] {n} samples -> {len(out)} groups kept "
          f"(>= {args.min_group}); largest {sizes[:5]}, "
          f"median {sizes[len(sizes) // 2]}")
    print("GROUP-MEANS-BUILT")


if __name__ == "__main__":
    main()
