"""
Paired significance test on FRR between systems judged by the SAME judge
on the SAME pairs (exact McNemar).

Why this and not the self-consistency "noise floor": judge noise is
per-judgment and roughly independent of the system, so over ~980 pairs
it AVERAGES OUT of the aggregate FRR. What it does instead is ATTENUATE
the between-system gap toward chance — a noisy judge understates a real
difference, it does not manufacture one. So the self-consistency rate is
NOT a threshold below which an aggregate gap is meaningless (an earlier
draft of judge_selfconsistency.py said so; it was wrong). It bounds how
much a gap is shrunk, which makes an observed gap a conservative reading
of the true one.

The right question — "is system A's FRR really above system B's?" — is a
paired one: both systems are judged on the SAME pairs by the SAME judge,
so discordant pairs (one realized, the other not) carry all the signal.
Exact McNemar over those pairs answers it without any noise-floor
hand-waving; the judge's noise is already priced into the discordance.

Usage:
    python scripts/frr_paired_test.py --label openai_gpt-4o \
        --frr ef32=runs/frr_final/openai_gpt-4o/ef32.jsonl \
        --frr routed=runs/frr_final/openai_gpt-4o/routed.jsonl \
        --frr steer=runs/frr_final/openai_gpt-4o/steer.jsonl \
        --out runs/tables/frr_paired_openai_gpt-4o
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path


def load(path: str) -> dict:
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["idx"])] = r
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: under H0 the b+c discordant pairs split
    Binom(n, 0.5). Returns 1.0 when there are no discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frr", action="append", required=True,
                   help="label=judgments.jsonl (2 or more; every pair "
                        "is tested)")
    p.add_argument("--label", default="judge")
    p.add_argument("--out", default="")
    args = p.parse_args()

    data = {}
    for spec in args.frr:
        label, path = spec.split("=", 1)
        data[label] = load(path)
        print(f"[paired] {label}: {len(data[label])} judgments")

    L = [f"# Paired FRR tests — {args.label}", "",
         "Exact McNemar over pairs where both systems have a scorable "
         "gold direction. Discordant pairs (one system realized the "
         "feature, the other did not) carry the signal; judge noise is "
         "priced into them, and it attenuates gaps toward chance rather "
         "than fabricating them, so each Δ reads as a conservative "
         "estimate of the true gap.", "",
         "| A vs B | n | FRR A | FRR B | Δ | A only | B only | p (exact) |",
         "|---|---|---|---|---|---|---|---|"]
    for x, y in combinations(data, 2):
        A, B = data[x], data[y]
        common = [k for k in sorted(set(A) & set(B))
                  if A[k].get("realized") is not None
                  and B[k].get("realized") is not None]
        if not common:
            continue
        ra = [bool(A[k]["realized"]) for k in common]
        rb = [bool(B[k]["realized"]) for k in common]
        b = sum(1 for i, j in zip(ra, rb) if i and not j)
        c = sum(1 for i, j in zip(ra, rb) if j and not i)
        fa, fb = sum(ra) / len(ra), sum(rb) / len(rb)
        pv = mcnemar_exact(b, c)
        star = ("***" if pv < 1e-3 else "**" if pv < 1e-2
                else "*" if pv < 0.05 else "n.s.")
        L.append(f"| {x} vs {y} | {len(common)} | {fa:.4f} | {fb:.4f} | "
                 f"{fa - fb:+.4f} | {b} | {c} | "
                 f"{pv:.2e} {star} |")

    report = "\n".join(L)
    print()
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + ".md").write_text(report + "\n")
        print(f"\n[paired] wrote {out}.md")


if __name__ == "__main__":
    main()
