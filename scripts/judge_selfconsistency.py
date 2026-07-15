"""
Judge self-consistency on exact-match pairs — an empirical noise floor
for FRR that needs NO human labels.

When a system's output is EXACTLY the target, the system judgment
judge(src, out) is literally the SAME comparison as the gold judgment
judge(src, tgt): same feature, same two strings.  A self-consistent
judge therefore scores realized=True on every exact-match pair by
construction, so

    FRR restricted to exact-match pairs  ==  judge self-consistency

and (1 - that) is the judge's disagreement-with-itself rate under
presentation-order randomization plus sampling noise.

What this rate IS: an instrument-quality ranking, obtained without human
labels.  A judge that contradicts itself on identical comparisons cannot
be trusted on the harder non-exact ones, so this is the empirical basis
for choosing the primary judge.  It also bounds ATTENUATION: per-judgment
noise is roughly independent of the system, so it shrinks between-system
FRR gaps toward chance, making every observed gap a conservative reading
of the true one.

What it is NOT: a threshold below which an aggregate FRR gap is
meaningless.  An earlier version of this file claimed that, and it was
wrong — noise that is independent across ~980 pairs AVERAGES OUT of the
aggregate mean rather than swamping it, and it cannot manufacture a gap
that is not there.  For "is A's FRR really above B's?" use the paired
test in scripts/frr_paired_test.py (exact McNemar over the same pairs
and the same judge), not a comparison against this rate.

Also reports FRR on the non-exact subset (where the judge is doing real
work) and the gold-indecisive rate (judge answered "equal" on src/tgt).

Usage:
    python scripts/judge_selfconsistency.py \
        --sys ef32=runs/frr_final/openai_gpt-4o/ef32.jsonl,\
runs/prod_gemma_v6/ksweep500/records.jsonl,k32 \
        --sys routed=runs/frr_final/openai_gpt-4o/routed.jsonl,\
runs/prod_gemma_v6/routed_system/records.jsonl,routed \
        --sys steer=runs/frr_final/openai_gpt-4o/steer.jsonl,\
runs/prod_gemma_v6/steer_baseline500/records.jsonl,steer0.5 \
        --label openai_gpt-4o --out runs/tables/judge_selfconsistency_gpt4o
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def norm(s: str) -> str:
    return " ".join(s.split())


def load_frr(path: str) -> dict:
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["idx"])] = r
    return rows


def load_records(path: str, mode: str, condition: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("src") or r.get("source")
            tgt = r.get("tgt") or r.get("target")
            if src is None or tgt is None:
                continue
            o = (r.get("outputs") or {}).get(condition)
            if o is None:
                continue
            node = o if not mode else o.get(mode)
            if not isinstance(node, dict) or "text" not in node:
                continue
            out[int(r["idx"])] = (src, tgt, node["text"])
    return out


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval — the exact subset is small and p is near 1,
    so the normal approximation would run past 1.0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sys", action="append", required=True,
                   help="label=frr.jsonl,records.jsonl,mode[,condition] "
                        "(condition defaults to 'true'; mode may be empty "
                        "for pipeline-format records)")
    p.add_argument("--label", default="judge",
                   help="judge name, for the report header")
    p.add_argument("--out", default="")
    args = p.parse_args()

    rows = []
    pool_k = pool_n = 0
    ne_k = ne_n = 0
    for spec in args.sys:
        label, rest = spec.split("=", 1)
        parts = rest.split(",")
        if len(parts) == 3:
            frr_path, rec_path, mode = parts
            condition = "true"
        elif len(parts) == 4:
            frr_path, rec_path, mode, condition = parts
        else:
            raise SystemExit(f"bad --sys spec {spec!r}")

        frr = load_frr(frr_path)
        recs = load_records(rec_path, mode, condition)
        common = sorted(set(frr) & set(recs))

        exact, nonexact, indecisive = [], [], 0
        for k in common:
            src, tgt, out_text = recs[k]
            r = frr[k]
            if r.get("realized") is None:
                indecisive += 1
                continue
            (exact if norm(out_text) == norm(tgt)
             else nonexact).append(bool(r["realized"]))

        n_e, k_e = len(exact), sum(exact)
        n_x, k_x = len(nonexact), sum(nonexact)
        lo, hi = wilson(k_e, n_e)
        pool_k += k_e
        pool_n += n_e
        ne_k += k_x
        ne_n += n_x
        rows.append({
            "label": label, "n_common": len(common),
            "indecisive": indecisive,
            "n_exact": n_e, "consist": (k_e / n_e) if n_e else float("nan"),
            "lo": lo, "hi": hi, "flips": n_e - k_e,
            "n_nonexact": n_x,
            "frr_nonexact": (k_x / n_x) if n_x else float("nan"),
        })
        print(f"[sc] {label}: common={len(common)} exact={n_e} "
              f"nonexact={n_x} gold-indecisive={indecisive}")

    plo, phi = wilson(pool_k, pool_n)
    pooled = (pool_k / pool_n) if pool_n else float("nan")
    floor = 1 - pooled

    L = [f"# Judge self-consistency — {args.label}", "",
         "On an exact-match pair the system judgment judge(src, out) IS "
         "the gold judgment judge(src, tgt) (same feature, same strings), "
         "so a self-consistent judge scores realized=True by "
         "construction. FRR on that subset therefore measures the judge "
         "against itself, ranking judges as instruments with no human "
         "labels. It is NOT a threshold for aggregate FRR gaps: noise "
         "independent across ~980 pairs averages out of the mean and "
         "attenuates real gaps toward chance rather than inventing them "
         "(use scripts/frr_paired_test.py to test a gap).", "",
         "| system | exact pairs | self-consistency [95% CI] | flips | "
         "non-exact pairs | FRR (non-exact) | gold-indecisive |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| {r['label']} | {r['n_exact']} | {r['consist']:.4f} "
            f"[{r['lo']:.3f}, {r['hi']:.3f}] | {r['flips']} | "
            f"{r['n_nonexact']} | {r['frr_nonexact']:.4f} | "
            f"{r['indecisive']} |")
    L += ["",
          f"**Pooled self-consistency: {pooled:.4f} "
          f"[{plo:.3f}, {phi:.3f}] (n={pool_n}, flips={pool_n - pool_k})** "
          f"— per-judgment disagreement-with-itself {floor:.3f}.",
          "",
          f"Pooled FRR on the non-exact subset: "
          f"{(ne_k / ne_n) if ne_n else float('nan'):.4f} (n={ne_n}) — "
          "where the judge does real work.",
          "",
          "Reading: self-consistency well below 1.0 means the judge "
          "contradicts itself on identical comparisons under presentation-"
          "order randomization, and cannot be trusted to resolve the "
          "harder non-exact pairs. Compare ACROSS judges to pick the "
          "primary — the most self-consistent judge is the least "
          "attenuating instrument, which is why it should also show the "
          "LARGEST between-system separation. Do not compare this rate "
          "against a between-system gap to decide whether the gap is "
          "real; that is what scripts/frr_paired_test.py is for."]
    report = "\n".join(L)
    print()
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + ".md").write_text(report + "\n")
        print(f"\n[sc] wrote {out}.md")


if __name__ == "__main__":
    main()
