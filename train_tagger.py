"""
Train the SAE-LEWIS 4-class tagger on the corruption stream.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from data import CorruptionCollator, CorruptionDataset
from intervene import diff_to_sparse
from lewis_ops import NUM_OPS, OP_NAMES
from tagger import SAETagger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--proj-a-freeze-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=2000)

    p.add_argument("--k-top", type=int, default=8)
    p.add_argument("--empty-cond-prob", type=float, default=0.15)

    p.add_argument("--estimate-class-weights-batches", type=int, default=200,
                   help="Number of warmup batches used to estimate inverse-freq class weights.")
    p.add_argument("--class-weight-smoothing", type=float, default=0.5)

    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _estimate_class_weights(
    loader, n_batches: int, smoothing: float,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    counts = Counter()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        gold = batch["tagger_gold"].view(-1).tolist()
        counts.update([g for g in gold if g != -100])
    total = sum(counts.values()) or 1
    freq = np.array([counts.get(i, 0) / total for i in range(NUM_OPS)], dtype=np.float32)
    # Smoothed inverse-frequency; smoothing bounds the rare-class weight
    # (e.g. smoothing=0.5 → max raw weight = 2.0 before normalization).
    weights = 1.0 / (freq + smoothing)
    weights /= weights.mean()
    counts_by_name = {OP_NAMES[i]: int(counts.get(i, 0)) for i in range(NUM_OPS)}
    return torch.from_numpy(weights.astype(np.float32)), counts_by_name


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    tagger = SAETagger(args.llm2vec_dir, d_sae=d_sae, dtype=dtype).to(args.device)

    for p in tagger.proj_a.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(tagger.parameters(), lr=args.learning_rate)
    sched = get_linear_schedule_with_warmup(
        optim, args.warmup_steps, args.max_steps,
    )

    ds = CorruptionDataset(args.corruption_dir, shuffle=True, seed=args.seed)
    coll = CorruptionCollator(d_sae=d_sae, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=coll,
    )

    # Estimate class weights from a small warmup pass
    print("[tagger] estimating class weights ...")
    cw_iter = iter(DataLoader(
        CorruptionDataset(args.corruption_dir, shuffle=True, seed=args.seed + 1),
        batch_size=args.batch_size, num_workers=0, collate_fn=coll,
    ))
    cw_batches = [next(cw_iter) for _ in range(args.estimate_class_weights_batches)]
    weights, counts_by_name = _estimate_class_weights(
        iter(cw_batches), args.estimate_class_weights_batches,
        args.class_weight_smoothing,
    )
    total_counted = sum(counts_by_name.values()) or 1
    print("[tagger] class counts (estimate):")
    for name in OP_NAMES:
        c = counts_by_name.get(name, 0)
        print(f"          {name:6s}  count={c:>8d}  freq={c/total_counted:.4f}")
    print(f"[tagger] class weights = "
          f"{ {n: round(float(weights[i]), 3) for i, n in enumerate(OP_NAMES)} }")
    weights = weights.to(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    proj_a_unfrozen = False
    loss_window = []

    for batch in loader:
        if step >= args.max_steps:
            break

        if not proj_a_unfrozen and step >= args.proj_a_freeze_steps:
            for p in tagger.proj_a.parameters():
                p.requires_grad_(True)
            proj_a_unfrozen = True
            print(f"[tagger] step={step} unfroze Proj_A")

        z_X = batch["z_X"]
        z_X_prime = batch["z_X_prime"]
        B = z_X.shape[0]
        z_amp = torch.zeros_like(z_X)
        z_sup = torch.zeros_like(z_X)
        for b in range(B):
            k_amp = int(rng.integers(0, 4))
            k_sup = int(rng.integers(0, 4))
            a, s = diff_to_sparse(
                z_X[b], z_X_prime[b],
                k_top=args.k_top, k_amp=k_amp, k_sup=k_sup,
                rng=rng, empty_conditioning_prob=args.empty_cond_prob,
            )
            z_amp[b] = a
            z_sup[b] = s

        out = tagger(
            input_ids=batch["tagger_input_ids"].to(args.device),
            attention_mask=batch["tagger_attention_mask"].to(args.device),
            z_amp=z_amp.to(args.device),
            z_sup=z_sup.to(args.device),
            labels=batch["tagger_gold"].to(args.device),
            class_weights=weights,
        )
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()

        loss_window.append(float(loss.item()))
        if step % args.logging_steps == 0:
            avg = sum(loss_window[-args.logging_steps:]) / max(1, min(len(loss_window), args.logging_steps))
            print(f"[tagger] step={step} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")

        if step > 0 and step % args.save_steps == 0:
            ckpt = out_dir / f"tagger-step{step}.pt"
            tagger.save(str(ckpt))
            print(f"[tagger] saved {ckpt}")

        step += 1

    tagger.save(str(out_dir / "tagger-final.pt"))
    print(f"[tagger] done at step {step}")


if __name__ == "__main__":
    main()
