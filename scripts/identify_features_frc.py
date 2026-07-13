"""
P-B step 1: LinguaLens-faithful phenomenon-level feature identification
(PS / PN / FRC — the exact computation of their lingualens/metrics.py)
on OUR SAE (Gemma Scope layer-12/16k).

For every English phenomenon (`feature` column) the dataset provides
positive (`sentence1`) / counterfactual (`sentence2`) pairs. Per SAE
feature f:
    PS  = P(f active anywhere in a positive sentence)
    PN  = P(f NOT active anywhere in a counterfactual sentence)
    FRC = harmonic mean of PS and PN
Top-r features by FRC are the phenomenon's identified activations
(their GPT-4o selection step is replaced by attaching Neuronpedia
labels for manual sanity checking).

CONTAMINATION GUARD: the 500 evaluation pairs (same seed-42 shuffle
prefix as every eval) are EXCLUDED from the identification statistics.

Output: --out JSON {phenomenon: [[feature_id, frc], ...]} (top-r) plus
a labelled markdown report next to it. Per-sentence active-feature sets
are cached (resume-safe) in <out>.acts.jsonl.

Usage (miyabi):
    python scripts/identify_features_frc.py \
        --out runs/frc/identified_l12_16k.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import SAEFeatureExtractor                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-sample-size", type=int, default=500,
                   help="the eval prefix to EXCLUDE from identification")
    p.add_argument("--top-r", type=int, default=16)
    p.add_argument("--explanations", default="",
                   help="optional Neuronpedia {index: description} JSON "
                        "for the labelled report")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    eval_idx = set(order[:min(args.eval_sample_size, len(order))])
    print(f"[frc] {len(ds)} pairs; excluding {len(eval_idx)} eval pairs "
          f"from identification")

    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    acts_path = out_path.with_suffix(out_path.suffix + ".acts.jsonl")
    done = {}
    if acts_path.exists():
        with open(acts_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[int(r["idx"])] = r
        print(f"[frc] RESUME: {len(done)} pairs cached")
    af = open(acts_path, "a")

    @torch.no_grad()
    def active_set(text: str):
        z = extractor.encode_text(text)          # (T, d_sae)
        return torch.nonzero((z > 0).any(dim=0)).flatten().tolist()

    n_done = 0
    for k in range(len(ds)):
        if k in eval_idx:
            continue
        if k in done:
            continue
        ex = ds[k]
        row = {"idx": k, "feature": ex.get("feature") or "?",
               "pos": active_set(ex["sentence1"]),
               "neg": active_set(ex["sentence2"])}
        done[k] = row
        af.write(json.dumps(row) + "\n")
        af.flush()
        n_done += 1
        if n_done % 100 == 0:
            print(f"[frc] encoded {n_done} new pairs "
                  f"({len(done)} total)")
    af.close()

    # ---- PS / PN / FRC per phenomenon (LinguaLens metrics.py) -----------
    by_ph = defaultdict(list)
    for r in done.values():
        by_ph[r["feature"]].append(r)
    print(f"[frc] {len(by_ph)} phenomena")

    expl = {}
    if args.explanations and Path(args.explanations).exists():
        expl = json.loads(Path(args.explanations).read_text())

    result = {}
    lines = ["# FRC-identified features per phenomenon "
             "(LinguaLens PS/PN/FRC on Gemma Scope l12/16k)", "",
             f"identification pairs exclude the {len(eval_idx)}-pair "
             f"eval sample; top-{args.top_r} by FRC", ""]
    for ph in sorted(by_ph):
        rows = by_ph[ph]
        n = len(rows)
        pos_count = defaultdict(int)
        neg_count = defaultdict(int)
        for r in rows:
            for f in set(r["pos"]):
                pos_count[f] += 1
            for f in set(r["neg"]):
                neg_count[f] += 1
        scored = []
        for f, pc in pos_count.items():
            ps = pc / n
            pn = 1.0 - neg_count.get(f, 0) / n
            if ps + pn > 0:
                frc = 2 * ps * pn / (ps + pn)
                scored.append((f, frc, ps, pn))
        scored.sort(key=lambda x: -x[1])
        top = scored[:args.top_r]
        result[ph] = [[int(f), round(frc, 4)] for f, frc, _, _ in top]
        lines.append(f"## {ph} (n={n})")
        for f, frc, ps, pn in top[:8]:
            lab = expl.get(str(int(f)), "")[:70]
            lines.append(f"- {int(f)} FRC={frc:.3f} (PS={ps:.2f} "
                         f"PN={pn:.2f}) {lab}")
        lines.append("")

    out_path.write_text(json.dumps(result))
    report = "\n".join(lines)
    out_path.with_suffix(".md").write_text(report + "\n")
    print(f"[frc] wrote {out_path} (+ .md report); "
          f"{len(result)} phenomena")


if __name__ == "__main__":
    main()
