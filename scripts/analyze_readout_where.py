"""
P-I WHERE statistics: is true's objection count REALLY above random's,
pair by pair?

The v1/v2 readout runs established the aggregate pattern (true fires 2-3x
random) but only as means. The causal claim deserves the same rigor the FRR
comparison got: a PAIRED test on the same pairs — each idx has both a true
and a random spec of the same size and magnitudes, so the discordant pairs
carry the signal (sign test; no scipy needed, exact binomial).

Also reports the per-pair fire RATIO distribution and, with --dataset, a
per-phenomenon breakdown of the WHERE gap — which phenomena's identified
features move the LM's own predictions, and which don't. That table is the
causal complement of the per-feature exact/FRR tables.

Usage (CPU, instant):
    python scripts/analyze_readout_where.py \
        --records runs/prod_gemma_v6/clamp_readout500_v2/delta_local/records.jsonl \
        --mode delta0.5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb


def sign_test(gt: int, lt: int) -> float:
    """Two-sided exact binomial on discordant pairs."""
    n = gt + lt
    if n == 0:
        return 1.0
    k = min(gt, lt)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--mode", required=True, help="e.g. delta0.5 / clamp10")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--min-n", type=int, default=8)
    p.add_argument("--no-features", action="store_true",
                   help="skip the per-phenomenon table (no dataset load)")
    args = p.parse_args()

    rows = []
    with open(args.records) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            o = r.get("outputs", {})
            t = o.get("true", {}).get(args.mode)
            c = o.get("random", {}).get(args.mode)
            if t is None or c is None:
                continue
            rows.append((int(r["idx"]), int(t["n_fire"]), int(c["n_fire"])))
    if not rows:
        raise SystemExit(f"no pairs with both true and random for "
                         f"mode {args.mode!r}")

    gt = sum(1 for _, a, b in rows if a > b)
    lt = sum(1 for _, a, b in rows if a < b)
    eq = len(rows) - gt - lt
    mt = sum(a for _, a, _ in rows) / len(rows)
    mr = sum(b for _, _, b in rows) / len(rows)
    pv = sign_test(gt, lt)
    print(f"[where] {len(rows)} pairs, mode {args.mode}")
    print(f"[where] mean fires: true {mt:.2f} vs random {mr:.2f} "
          f"(ratio {mt / max(mr, 1e-9):.2f}x)")
    print(f"[where] paired sign test: true>random {gt} / true<random {lt} "
          f"/ tied {eq}  ->  p = {pv:.2e}")
    print("[where] reading: the specs share count and magnitudes; only the "
          "feature IDENTITIES differ. p is the probability the identified "
          "features move the LM's own predictions no more than random ones.")

    if args.no_features:
        return
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    by_ph = defaultdict(list)
    for k, a, b in rows:
        ph = ds[k].get("feature") or "?"
        by_ph[ph].append((a, b))
    print(f"\n| phenomenon | n | true fires | random | Δ | >/< |")
    print("|---|---|---|---|---|---|")
    order = sorted(by_ph.items(),
                   key=lambda kv: -(sum(a - b for a, b in kv[1])
                                    / len(kv[1])))
    for ph, v in order:
        if len(v) < args.min_n:
            continue
        ta = sum(a for a, _ in v) / len(v)
        ra = sum(b for _, b in v) / len(v)
        g = sum(1 for a, b in v if a > b)
        l = sum(1 for a, b in v if a < b)
        print(f"| {ph} | {len(v)} | {ta:.2f} | {ra:.2f} | {ta - ra:+.2f} "
              f"| {g}/{l} |")


if __name__ == "__main__":
    main()
