"""Multi-layer z sidecar for the corruption cache (EF_LM_LOSS_PLAN.md §3,
kufuu-1): ONE forward of base gemma-2-2b per sentence yields the hidden
states of ALL target layers at once; each layer's SAE encode is a single
matmul. Rewrites cache records with layer-L conditioning z into
per-layer cache dirs (same shard names, same record fields, only
z_X_topk / z_X_prime_topk replaced) so CorruptionDataset reads them
unchanged.

Pooling reproduces the cache semantics (corruption.py _pooled_dense):
edit-local max-pool over the token positions touched by the x<->x' edit
(difflib on the stored token id lists; the original used char ranges of
the same edits), global max-pool when no edit positions exist
(identity records); per-layer blocklist zeroed before top-64.

Resume: a shard is skipped when its output exists in EVERY layer dir
(tmp-write + rename = atomic).

Usage:
    python scripts/make_z_sidecar.py \
        --cache-dir runs/prod_gemma_v4/corruption \
        --out-root  runs/prod_gemma_v4/corruption_z \
        --layers 4,12,20 \
        --sae-paths layer_4/width_16k/average_l0_60/params.npz,\
layer_12/width_16k/average_l0_82/params.npz,\
layer_20/width_16k/average_l0_71/params.npz \
        --blocklists runs/blocklist_l4/blocklist.npy,\
runs/blocklist/blocklist.npy,runs/blocklist_l20/blocklist.npy \
        --max-records 300000
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModel  # noqa: E402

from model import load_sae          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out-root", required=True,
                   help="per-layer dirs are written as {out-root}_l{L}")
    p.add_argument("--layers", required=True, help="comma ints, e.g. 4,12,20")
    p.add_argument("--sae-paths", required=True,
                   help="comma list aligned with --layers")
    p.add_argument("--blocklists", default="",
                   help="comma list aligned with --layers ('' entry = none)")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--max-records", type=int, default=300000,
                   help="0 = all records")
    p.add_argument("--batch-records", type=int, default=16,
                   help="records per forward batch (2 sequences each)")
    p.add_argument("--cond-topk", type=int, default=64)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def edit_positions(a: List[int], b: List[int]):
    """Token positions touched by the a<->b edit, per side."""
    pa, pb = set(), set()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        pa.update(range(i1, max(i1 + 1, i2)))
        pb.update(range(j1, max(j1 + 1, j2)))
    return sorted(p for p in pa if p < len(a)), \
        sorted(p for p in pb if p < len(b))


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    sae_paths = args.sae_paths.split(",")
    assert len(sae_paths) == len(layers)
    bl_paths = (args.blocklists.split(",") if args.blocklists
                else [""] * len(layers))
    assert len(bl_paths) == len(layers)

    device = args.device
    llm = AutoModel.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16).to(device).eval()
    llm.requires_grad_(False)
    saes, blks = {}, {}
    for L, sp, bp in zip(layers, sae_paths, bl_paths):
        saes[L] = load_sae("jumprelu", args.sae_repo, sp).to(device).eval()
        blks[L] = (torch.as_tensor(np.load(bp).astype(np.int64),
                                   device=device) if bp else None)
        print(f"[sidecar] L{L}: sae={sp} "
              f"blocklist={'%d feats' % len(blks[L]) if blks[L] is not None else 'none'}")

    cache_dir = Path(args.cache_dir)
    out_dirs = {L: Path(f"{args.out_root}_l{L}") for L in layers}
    meta = json.loads((cache_dir / "meta.json").read_text())
    for L, sp, bp in zip(layers, sae_paths, bl_paths):
        out_dirs[L].mkdir(parents=True, exist_ok=True)
        m = dict(meta)
        m["z_sidecar"] = {"sae_repo": args.sae_repo, "sae_path": sp,
                          "sae_layer": L, "blocklist": bp,
                          "source_cache": str(cache_dir),
                          "cond_topk": args.cond_topk}
        (out_dirs[L] / "meta.json").write_text(json.dumps(m, indent=1))

    @torch.no_grad()
    def batch_z(seqs: List[List[int]]) -> Dict[int, List[torch.Tensor]]:
        """One forward -> per-layer dense z per sequence."""
        T = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), T, dtype=torch.long, device=device)
        mask = torch.zeros(len(seqs), T, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=device)
            mask[i, :len(s)] = 1
        out = llm(input_ids=ids, attention_mask=mask,
                  output_hidden_states=True, use_cache=False)
        res: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
        for L in layers:
            h = out.hidden_states[L + 1]          # residual AFTER block L
            z = saes[L].encode(h.to(saes[L].W_enc.dtype))
            for i, s in enumerate(seqs):
                res[L].append(z[i, :len(s)].float())
        return res

    def pooled_topk(z, pos, blk):
        dense = (z[pos].max(dim=0).values if pos else
                 z.max(dim=0).values).clone()
        if blk is not None:
            dense[blk] = 0.0
        k = min(args.cond_topk, dense.numel())
        vals, idx = dense.topk(k)
        keep = vals > 0
        return [{"f": int(f), "v": float(v)}
                for f, v in zip(idx[keep].tolist(), vals[keep].tolist())]

    shards = sorted(cache_dir.glob("shard-*.jsonl.gz"))
    print(f"[sidecar] {len(shards)} shards, layers {layers}, "
          f"cap {args.max_records or 'all'} records")
    n_done = 0
    for shard in shards:
        if args.max_records and n_done >= args.max_records:
            break
        outs = {L: out_dirs[L] / shard.name for L in layers}
        if all(o.exists() for o in outs.values()):
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                n_done += sum(1 for _ in f)
            print(f"[sidecar] skip {shard.name} (done, ~{n_done})")
            continue
        recs = []
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except (OSError, EOFError, gzip.BadGzipFile):
            print(f"[sidecar] {shard.name} truncated — readable prefix")
        if args.max_records:
            recs = recs[:max(0, args.max_records - n_done)]
        writers = {L: gzip.open(str(outs[L]) + ".tmp", "wt",
                                encoding="utf-8") for L in layers}
        for i0 in range(0, len(recs), args.batch_records):
            chunk = recs[i0:i0 + args.batch_records]
            seqs, keep = [], []
            for r in chunk:
                a = list(map(int, r["x_token_ids"]))[:args.max_len]
                b = list(map(int, r["x_prime_token_ids"]))[:args.max_len]
                seqs += [a, b]
                keep.append((r, a, b))
            zs = batch_z(seqs)
            for j, (r, a, b) in enumerate(keep):
                pa, pb = edit_positions(a, b)
                for L in layers:
                    za, zb = zs[L][2 * j], zs[L][2 * j + 1]
                    r2 = dict(r)
                    r2["z_X_topk"] = pooled_topk(za, pa, blks[L])
                    r2["z_X_prime_topk"] = pooled_topk(zb, pb, blks[L])
                    writers[L].write(
                        json.dumps(r2, ensure_ascii=False) + "\n")
        for L in layers:
            writers[L].close()
            Path(str(outs[L]) + ".tmp").rename(outs[L])
        n_done += len(recs)
        print(f"[sidecar] {shard.name}: {len(recs)} recs "
              f"(total {n_done})")
    print(f"[sidecar] DONE — {n_done} records x {len(layers)} layers")


if __name__ == "__main__":
    main()
