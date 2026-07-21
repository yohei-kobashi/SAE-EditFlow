"""Split-half stability of LinguaLens-style FRC feature identification.

Question (user 2026-07-21): are the FRC-identified activations properties of
the *feature* or of the *specific identification pairs*? LinguaLens itself
computes PS/PN/FRC on ALL ~50 pairs per feature and selects top-r in-sample
(no held-out check, paper §Feature Identification / our audit §1), so the
question is unanswerable from their protocol. This script answers it for our
Gemma Scope reproduction directly from the cached per-sentence active sets:

  For R seeded repeats, each phenomenon's identification pairs are split into
  halves A/B. Top-r features are identified on A (exact identify_features_frc
  scoring), then
    * out-of-sample FRC: the FRC of A's top-r measured on B (shrinkage vs
      their in-sample FRC on A quantifies winner's-curse over the 16k
      candidates);
    * stability: overlap |topA ∩ topB| / r for top-1/3/10 (LinguaLens uses
      top-3 for intervention, top-10 for the LLM-agent verification).

Pure stdlib; streams the 350MB acts cache — safe on a login node.

Usage:
    python3 scripts/analyze_frc_splithalf.py \
        --acts runs/frc/identified_l12_16k_r3.json.acts.jsonl \
        --out runs/tables/frc_splithalf_l12.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-rs", default="1,3,10")
    return p.parse_args()


def frc_scores(rows):
    """{feature: (frc, ps, pn)} over the given rows (identify_features_frc
    scoring: PS = P(active | positive), PN = P(inactive | counterfactual),
    harmonic mean; only features active in >=1 positive get a score)."""
    n = len(rows)
    pos_count: dict[int, int] = defaultdict(int)
    neg_count: dict[int, int] = defaultdict(int)
    for r in rows:
        for f in set(r["pos"]):
            pos_count[f] += 1
        for f in set(r["neg"]):
            neg_count[f] += 1
    out = {}
    for f, pc in pos_count.items():
        ps = pc / n
        pn = 1.0 - neg_count.get(f, 0) / n
        if ps + pn > 0:
            out[f] = (2 * ps * pn / (ps + pn), ps, pn)
    return out


def frc_of(rows, feats):
    """FRC of specific features on rows (0.0 if never active in positives —
    the honest out-of-sample value for a feature that does not transfer)."""
    sc = frc_scores(rows)
    return [sc.get(f, (0.0,))[0] for f in feats]


def main():
    args = parse_args()
    top_rs = [int(x) for x in args.top_rs.split(",")]
    by_ph: dict[str, list] = defaultdict(list)
    with open(args.acts) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                by_ph[r["feature"]].append(
                    {"pos": r["pos"], "neg": r["neg"]})
    print(f"[splithalf] {len(by_ph)} phenomena, "
          f"{sum(len(v) for v in by_ph.values())} pairs")

    rng = random.Random(args.seed)
    # per phenomenon: aggregates over repeats
    agg = {}
    for ph in sorted(by_ph):
        rows = by_ph[ph]
        n = len(rows)
        if n < 8:
            print(f"[splithalf] skip {ph} (n={n})")
            continue
        ins = {r: [] for r in top_rs}     # in-sample FRC of topA (on A)
        oos = {r: [] for r in top_rs}     # out-of-sample FRC of topA (on B)
        ovl = {r: [] for r in top_rs}     # |topA ∩ topB| / r
        for _ in range(args.repeats):
            order = list(range(n))
            rng.shuffle(order)
            A = [rows[i] for i in order[: n // 2]]
            B = [rows[i] for i in order[n // 2:]]
            sa, sb = frc_scores(A), frc_scores(B)
            ra = sorted(sa, key=lambda f: -sa[f][0])
            rb = sorted(sb, key=lambda f: -sb[f][0])
            for r in top_rs:
                ta, tb = ra[:r], rb[:r]
                ins[r].append(sum(sa[f][0] for f in ta) / max(len(ta), 1))
                oos[r].append(sum(frc_of(B, ta)) / max(len(ta), 1))
                ovl[r].append(len(set(ta) & set(tb)) / r)
        agg[ph] = {"n": n,
                   "ins": {r: sum(v) / len(v) for r, v in ins.items()},
                   "oos": {r: sum(v) / len(v) for r, v in oos.items()},
                   "ovl": {r: sum(v) / len(v) for r, v in ovl.items()}}
        print(f"[splithalf] {ph} (n={n}): "
              + " ".join(f"top{r}: in {agg[ph]['ins'][r]:.3f} "
                         f"out {agg[ph]['oos'][r]:.3f} "
                         f"ovl {agg[ph]['ovl'][r]:.2f}" for r in top_rs))

    lines = [f"# FRC split-half stability ({Path(args.acts).name}, "
             f"{args.repeats} repeats, seed {args.seed})", "",
             "in = in-sample FRC of half-A top-r; out = same features' FRC "
             "on held-out half B; ovl = |topA ∩ topB|/r", ""]
    hdr = "| phenomenon | n |"
    sep = "|---|---|"
    for r in top_rs:
        hdr += f" in@{r} | out@{r} | ovl@{r} |"
        sep += "---|---|---|"
    lines += [hdr, sep]
    for ph, a in sorted(agg.items()):
        row = f"| {ph} | {a['n']} |"
        for r in top_rs:
            row += (f" {a['ins'][r]:.3f} | {a['oos'][r]:.3f} |"
                    f" {a['ovl'][r]:.2f} |")
        lines.append(row)
    npn = len(agg)
    row = f"| **mean ({npn} phenomena)** | |"
    for r in top_rs:
        row += (f" {sum(a['ins'][r] for a in agg.values()) / npn:.3f} |"
                f" {sum(a['oos'][r] for a in agg.values()) / npn:.3f} |"
                f" {sum(a['ovl'][r] for a in agg.values()) / npn:.2f} |")
    lines.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[splithalf] wrote {out}")
    print("SPLITHALF-DONE")


if __name__ == "__main__":
    main()
