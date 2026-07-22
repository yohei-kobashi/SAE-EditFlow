"""Improvement ① (user-approved 2026-07-22): W_dec neighbor table for
cluster-expanded specs. For every latent, store the decoder-cosine
top-K neighbors above a threshold — the split-siblings identified by the
split-half analysis (audit §5) that a pool-mean spec misses when the
eval sentence uses a different member of the near-tie cluster.

Usage (GPU):
    python scripts/build_cluster_table.py \
        --sae-path layer_12/width_16k/average_l0_82/params.npz \
        --out runs/feature_specs/l12_clusters.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import load_sae                                     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sae = load_sae("jumprelu", args.sae_repo, args.sae_path)
    W = sae.W_dec.float().to(args.device)              # (d_sae, d_llm)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    d = W.shape[0]
    tab = {}
    n_edges = 0
    chunk = 1024
    for s in range(0, d, chunk):
        cos = W[s:s + chunk] @ W.T                     # (chunk, d)
        for r in range(cos.shape[0]):
            i = s + r
            row = cos[r].clone()
            row[i] = -1.0                              # exclude self
            v, j = row.topk(args.top_k)
            keep = v >= args.tau
            if keep.any():
                tab[int(i)] = [[int(jj), round(float(vv), 4)]
                               for jj, vv in zip(j[keep].tolist(),
                                                 v[keep].tolist())]
                n_edges += int(keep.sum())
    Path(args.out).write_text(json.dumps(tab))
    print(f"[clusters] {len(tab)}/{d} latents have neighbors "
          f"(tau={args.tau}, K={args.top_k}); {n_edges} edges "
          f"-> {args.out}")
    print("CLUSTER-TABLE-BUILT")


if __name__ == "__main__":
    main()
