"""
Same-pairs EF vs pipeline comparison — no GPU, seconds.

Both eval_lingualens.py (sample-size 500) and editflow_probe.py
(sample-size 200) shuffle the LinguaLens indices with the same seed, so
the probe's 200 pairs are a PREFIX of the e2e run's 500: their
records.jsonl files can be joined on `idx` for a rigorous matched-pair
comparison (the aggregate tables come from different samples).

Usage (on miyabi):
    python scripts/compare_ef_pipeline.py \
        --ef runs/prod_gemma_v6/editflow_pilot/probe_recal2/records.jsonl \
        --pipeline runs/prod_gemma_v6/eval_lingualens_final/records.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ef", required=True, help="editflow_probe records.jsonl")
    p.add_argument("--pipeline", required=True,
                   help="eval_lingualens records.jsonl")
    p.add_argument("--condition", default="true")
    p.add_argument("--exclude", default="",
                   help="records.jsonl whose idx set is DROPPED from the "
                        "join — e.g. the 200-pair probe that selected the "
                        "operating F, leaving a clean holdout")
    p.add_argument("--pipeline-mode", default="",
                   help="set when --pipeline is a PROBE-format records "
                        "file (EF/B1/B2/B3): the decode-mode key whose "
                        "text/metrics act as the baseline side — enables "
                        "direct EF-vs-baseline paired stats.")
    args = p.parse_args()

    ef = {r["idx"]: r for r in load_jsonl(args.ef)}
    pl = {r["idx"]: r for r in load_jsonl(args.pipeline)}
    common = sorted(set(ef) & set(pl))
    print(f"matched pairs: {len(common)} "
          f"(ef {len(ef)}, pipeline {len(pl)})")
    if args.exclude:
        drop = {r["idx"] for r in load_jsonl(args.exclude)}
        common = [k for k in common if k not in drop]
        print(f"after --exclude ({len(drop)} idx dropped): "
              f"{len(common)} pairs")
    if not common:
        return

    def pl_metrics(pr):
        """Uniform pipeline-side metric access; supports probe-format
        baselines via --pipeline-mode (direct EF-vs-baseline stats)."""
        o = pr["outputs"].get(args.condition)
        if o is None:
            return None
        if args.pipeline_mode:
            m = o.get(args.pipeline_mode)
            if not isinstance(m, dict):
                return None
            return {"exact_match": m["exact"],
                    "sim_target": m["sim_target"],
                    "copy_rate": m.get("copy", float("nan"))}
        return o

    common = [k for k in common if pl_metrics(pl[k]) is not None]

    ef_modes = sorted({m for r in ef.values()
                       for m, v in r["outputs"][args.condition].items()
                       if isinstance(v, dict)})
    rows = defaultdict(lambda: defaultdict(list))
    buckets = ((1, 1, "1"), (2, 3, "2-3"), (4, 8, "4-8"), (9, 10**9, "9+"))

    def bname(n):
        for lo, hi, name in buckets:
            if lo <= n <= hi:
                return name
        return "9+"

    for k in common:
        er, pr = ef[k], pl[k]
        b = bname(er.get("n_ops", 1))
        po = pl_metrics(pr)
        rows["pipeline"]["exact"].append(po["exact_match"])
        rows["pipeline"]["sim"].append(po["sim_target"])
        rows["pipeline"]["copy"].append(po["copy_rate"])
        rows[f"pipeline|{b}"]["exact"].append(po["exact_match"])
        rows[f"pipeline|{b}"]["sim"].append(po["sim_target"])
        for m in ef_modes:
            eo = er["outputs"][args.condition].get(m)
            if not isinstance(eo, dict):
                continue
            rows[f"ef:{m}"]["exact"].append(eo["exact"])
            rows[f"ef:{m}"]["sim"].append(eo["sim_target"])
            rows[f"ef:{m}|{b}"]["exact"].append(eo["exact"])
            rows[f"ef:{m}|{b}"]["sim"].append(eo["sim_target"])

    def mean(v):
        return sum(v) / len(v) if v else float("nan")

    print(f"\n{'system':16s} {'exact':>8s} {'sim':>8s} {'copy':>8s}   n")
    for name in ["pipeline"] + [f"ef:{m}" for m in ef_modes]:
        r = rows[name]
        cp = mean(r["copy"]) if r["copy"] else float("nan")
        print(f"{name:16s} {mean(r['exact']):8.4f} {mean(r['sim']):8.4f} "
              f"{cp:8.4f}   {len(r['exact'])}")

    print("\nby n_ops bucket (exact | sim):")
    hdr = f"{'bucket':8s}" + "".join(
        f"{n:>22s}" for n in ["pipeline"] + [f"ef:{m}" for m in ef_modes])
    print(hdr)
    for _lo, _hi, b in buckets:
        cells = []
        for name in ["pipeline"] + [f"ef:{m}" for m in ef_modes]:
            r = rows[f"{name}|{b}"]
            if r["exact"]:
                cells.append(f"{mean(r['exact']):7.3f}|{mean(r['sim']):.3f}")
            else:
                cells.append("—")
        n_b = len(rows[f"pipeline|{b}"]["exact"])
        if n_b:
            print(f"{b:8s}" + "".join(f"{c:>22s}" for c in cells)
                  + f"   (n={n_b})")

    # Paired per-pair differences vs the pipeline: exact as discordant
    # counts (ef-only wins / pipeline-only wins — the sign-test cells),
    # sim as mean Δ with a 95% normal CI. Same pairs, so this is the
    # rigorous version of comparing the aggregate rows above.
    print("\npaired vs pipeline (per-pair Δ = ef − pipeline):")
    print(f"{'system':16s} {'Δexact':>8s} {'ef+/pl+':>9s} "
          f"{'Δsim':>8s} {'95% CI':>20s}")
    for m in ef_modes:
        de, ds = [], []
        for k in common:
            eo = ef[k]["outputs"][args.condition].get(m)
            if not isinstance(eo, dict):
                continue
            po = pl_metrics(pl[k])
            de.append(eo["exact"] - po["exact_match"])
            ds.append(eo["sim_target"] - po["sim_target"])
        n = len(ds)
        if n < 2:
            continue
        wins = sum(1 for d in de if d > 0)
        losses = sum(1 for d in de if d < 0)
        md = sum(ds) / n
        sd = math.sqrt(sum((x - md) ** 2 for x in ds) / (n - 1))
        half = 1.96 * sd / math.sqrt(n)
        print(f"ef:{m:13s} {sum(de)/n:+8.4f} {wins:>4d}/{losses:<4d} "
              f"{md:+8.4f} [{md-half:+.4f}, {md+half:+.4f}]")


if __name__ == "__main__":
    main()
