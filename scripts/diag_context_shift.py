"""DIAGNOSTIC 7 (reviewer-defense, 2026-07-19): how much do the SAE
features at the src positions shift between the IDENTIFICATION condition
(bare sentence, base gemma-2-2b) and the INJECTION condition (src inside
the repeat prompt, gemma-2-2b-it)?

Decomposition via three encodings of the same sentence:
  A: bare src, base model      (identification condition)
  B: bare src, -it model       (model-shift control)
  C: src in repeat prompt, -it (injection condition)
A vs C = total shift; A vs B = model shift; B vs C = context shift.

Metrics (mean over pairs):
  * pooled top-64 Jaccard between the max-pooled src-position features
  * spec persistence: fraction of A's pooled top-64 still in C's top-64
  * mean per-position top-16 overlap at aligned src positions
Report -> {out}/report.md (transcribe into reports/context_shift_diag.md).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer                          # noqa: E402

from intervener import (REPEAT_PROMPT, REPEAT_PROMPT_SRC_FIRST,  # noqa
                        chat_prompt_ids, find_subseq)
from model import SAEFeatureExtractor                           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--llm-base", default="google/gemma-2-2b")
    p.add_argument("--llm-it", default="google/gemma-2-2b-it")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--pos-topk", type=int, default=16)
    p.add_argument("--src-first", action="store_true",
                   help="use REPEAT_PROMPT_SRC_FIRST (instruction AFTER "
                        "src) as the injection-condition prompt")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def top_set(dense: torch.Tensor, k: int):
    k = min(k, int((dense > 0).sum()))
    if k == 0:
        return set()
    return set(dense.topk(k).indices.tolist())


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    srcs = [ds[int(k)]["sentence1"] for k in order[:args.sample_size]]

    ex_base = SAEFeatureExtractor(
        llm_name=args.llm_base, sae_repo=args.sae_repo,
        sae_path=args.sae_path, sae_layer=args.sae_layer,
    ).to(args.device).eval()
    ex_it = SAEFeatureExtractor(
        llm_name=args.llm_it, sae_repo=args.sae_repo,
        sae_path=args.sae_path, sae_layer=args.sae_layer,
    ).to(args.device).eval()
    it_tok = AutoTokenizer.from_pretrained(args.llm_it)

    rows = []
    for i, src in enumerate(srcs):
        with torch.no_grad():
            zA = ex_base.encode_text(src)              # (Ta, d)
            zB = ex_it.encode_text(src)                # (Tb, d)
            tpl = (REPEAT_PROMPT_SRC_FIRST if args.src_first
                   else REPEAT_PROMPT)
            pids = chat_prompt_ids(it_tok, tpl.format(src=src))
            needle = it_tok(src, add_special_tokens=False).input_ids
            off = 0
            lo = find_subseq(pids, needle)
            if lo is None and len(needle) > 1:
                lo = find_subseq(pids, needle[1:])
                if lo is not None:
                    off = 1
                    needle = needle[1:]
            if lo is None:
                continue
            zC_full = ex_it.encode_token_ids(
                torch.tensor(pids, dtype=torch.long))
            zC = zC_full[lo:lo + len(needle)]          # src span in prompt

        # exclude BOS from the bare pools: its attention-sink mega-
        # activations dominate a global max-pool and would make A/B vs C
        # (a BOS-free span) incomparable. The real spec pipeline pools
        # edit-locally (BOS-free) as well.
        pA = zA[1:].max(dim=0).values.float().cpu()
        pB = zB[1:].max(dim=0).values.float().cpu()
        pC = zC.max(dim=0).values.float().cpu()
        sA, sB, sC = (top_set(p, args.topk) for p in (pA, pB, pC))
        # per-position overlap: bare -it positions vs in-context positions
        # (same tokenizer; bare has BOS at 0, needle may lack first piece)
        n = min(len(needle), zB.shape[0] - 1 - off, zC.shape[0])
        pos_ov = []
        for j in range(n):
            tb = top_set(zB[1 + off + j].float().cpu(), args.pos_topk)
            tc = top_set(zC[j].float().cpu(), args.pos_topk)
            if tb or tc:
                pos_ov.append(len(tb & tc)
                              / max(1, min(len(tb), len(tc))))
        rows.append({
            "jac_AC": jac(sA, sC), "jac_AB": jac(sA, sB),
            "jac_BC": jac(sB, sC),
            "persist_AC": (len(sA & sC) / max(1, len(sA))),
            "pos_ov_BC": float(np.mean(pos_ov)) if pos_ov else float("nan"),
        })
        if (i + 1) % 20 == 0:
            print(f"[diag7] {i + 1}/{len(srcs)}")

    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return float(np.mean(v))

    lines = [
        f"# DIAG 7 — identification vs injection context shift "
        f"(layer {args.sae_layer}, n={len(rows)}, "
        f"prompt={'src-first' if args.src_first else 'instr-first'})", "",
        "A = bare src / base model (identification condition)",
        "B = bare src / -it model (model-shift control)",
        "C = src inside the repeat prompt / -it (injection condition)", "",
        "| metric | mean |", "|---|---|",
        f"| pooled top-{args.topk} Jaccard A vs C (total shift) | "
        f"{m('jac_AC'):.3f} |",
        f"| pooled Jaccard A vs B (model shift only) | {m('jac_AB'):.3f} |",
        f"| pooled Jaccard B vs C (context shift only) | "
        f"{m('jac_BC'):.3f} |",
        f"| A's top-{args.topk} persisting in C (spec persistence) | "
        f"{m('persist_AC'):.3f} |",
        f"| per-position top-{args.pos_topk} overlap B vs C | "
        f"{m('pos_ov_BC'):.3f} |",
    ]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
