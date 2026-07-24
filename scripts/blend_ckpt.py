"""v3a (WiSE-FT): linear interpolation of two editor checkpoints.

theta_alpha = (1 - alpha) * theta_A + alpha * theta_B over the trainable
state dict (LoRA + conditioning + heads; the frozen backbone is shared).

Usage:
    python scripts/blend_ckpt.py --a T2.pt --b T4.pt --alpha 0.6 --out X.pt
"""

from __future__ import annotations

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="checkpoint at alpha=0")
    p.add_argument("--b", required=True, help="checkpoint at alpha=1")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    A = torch.load(args.a, map_location="cpu")
    B = torch.load(args.b, map_location="cpu")
    sa, sb = A["trainable"], B["trainable"]
    assert set(sa) == set(sb), "trainable key mismatch"
    out = {}
    for k in sa:
        x, y = sa[k], sb[k]
        out[k] = ((1.0 - args.alpha) * x.float()
                  + args.alpha * y.float()).to(x.dtype)
    torch.save({"trainable": out,
                "blend": {"a": args.a, "b": args.b,
                          "alpha": args.alpha}}, args.out)
    print(f"[blend] alpha={args.alpha}: {args.a} + {args.b} -> {args.out}")


if __name__ == "__main__":
    main()
