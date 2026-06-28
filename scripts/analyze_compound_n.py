"""
scripts/analyze_compound_n.py — summarise calibration.jsonl per N.

Reads `<run_dir>/n{N}/calibration.jsonl` for each N value and produces:

  * Per-N distribution stats for ppl_ratio and sae_shift (count, mean,
    median, percentiles).
  * Per-N op-type histogram (REPL / INS / DEL).
  * Yield simulation under the default sqrt(N) gate AND a sweep of
    alternate ppl_per_op_factor / sae_per_op_min / sae_per_op_max
    settings, so the user can see how the threshold-fit shifts yield
    across N.

Output:
  * Tab-separated `report.tsv` with one row per N.
  * Markdown `report.md` with the same table plus the gate-sweep
    section. Both files are written under <run_dir>.

Usage:
  python scripts/analyze_compound_n.py <run_dir>
  python scripts/analyze_compound_n.py <run_dir> --n-values 0 1 2 3 4 5
  python scripts/analyze_compound_n.py <run_dir> --report-tsv X.tsv --report-md X.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def percentiles(values: List[float], qs: List[float]) -> List[float]:
    """Linear-interpolation percentiles (no numpy dependency). Returns `nan`
    for empty input or all-`nan` input.
    """
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return [float("nan")] * len(qs)
    out: List[float] = []
    for q in qs:
        if q <= 0:
            out.append(clean[0])
            continue
        if q >= 1:
            out.append(clean[-1])
            continue
        pos = q * (len(clean) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(clean[lo])
        else:
            frac = pos - lo
            out.append(clean[lo] * (1 - frac) + clean[hi] * frac)
    return out


def mean(values: List[float]) -> float:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(clean) if clean else float("nan")


# --------------------------------------------------------------------------- #
# Calibration record loader
# --------------------------------------------------------------------------- #
def load_calibration(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------- #
# Per-N summary
# --------------------------------------------------------------------------- #
def summarise_one_N(records: List[Dict]) -> Dict:
    slor_drops = [r.get("slor_drop") for r in records]
    ppl_ratios = [r.get("ppl_ratio") for r in records]
    sae_shifts = [r.get("sae_shift") for r in records]
    sae_topk_changes = [r.get("sae_topk_change") for r in records]
    op_counter: Counter = Counter()
    for r in records:
        for t in r.get("op_types", []):
            op_counter[t] += 1
    n_total = len(records)
    slor_pcts = percentiles(slor_drops, [0.05, 0.25, 0.50, 0.75, 0.95])
    ppl_pcts = percentiles(ppl_ratios, [0.25, 0.50, 0.75, 0.95])
    sae_pcts = percentiles(sae_shifts, [0.05, 0.25, 0.50, 0.75, 0.95])
    topk_pcts = percentiles(sae_topk_changes, [0.05, 0.25, 0.50, 0.75, 0.95])
    # Bucket the discrete topk_change values: 0 / 1 / 2-3 / 4+
    topk_buckets = {"=0": 0, "=1": 0, "2-3": 0, "4+": 0, "unknown": 0}
    for v in sae_topk_changes:
        if v is None:
            topk_buckets["unknown"] += 1
        elif v == 0:
            topk_buckets["=0"] += 1
        elif v == 1:
            topk_buckets["=1"] += 1
        elif v <= 3:
            topk_buckets["2-3"] += 1
        else:
            topk_buckets["4+"] += 1
    return {
        "n_records": n_total,
        "slor_mean": mean(slor_drops),
        "slor_p05": slor_pcts[0],
        "slor_p25": slor_pcts[1],
        "slor_p50": slor_pcts[2],
        "slor_p75": slor_pcts[3],
        "slor_p95": slor_pcts[4],
        # Legacy PPL ratio kept for cross-checking against older runs.
        "ppl_mean": mean(ppl_ratios),
        "ppl_p25": ppl_pcts[0],
        "ppl_p50": ppl_pcts[1],
        "ppl_p75": ppl_pcts[2],
        "ppl_p95": ppl_pcts[3],
        "sae_mean": mean(sae_shifts),
        "sae_p05": sae_pcts[0],
        "sae_p25": sae_pcts[1],
        "sae_p50": sae_pcts[2],
        "sae_p75": sae_pcts[3],
        "sae_p95": sae_pcts[4],
        # Top-K identity-change (new SAE lower-bound metric)
        "topk_mean":  mean(sae_topk_changes),
        "topk_p25":   topk_pcts[1],
        "topk_p50":   topk_pcts[2],
        "topk_p75":   topk_pcts[3],
        "topk_p95":   topk_pcts[4],
        "topk_bucket": topk_buckets,
        "op_counts": dict(op_counter),
    }


# --------------------------------------------------------------------------- #
# Gate simulation
# --------------------------------------------------------------------------- #
def simulate_gate_yield(
    records: List[Dict], N: int,
    slor_drop_per_op: float,
    sae_min_topk_change: int,
    sae_per_op_max: float,
) -> Dict:
    """Compute the fraction of `records` that would pass the gate at the
    given scale constants.

    Fluency:           SLOR drop linear in N.
    SAE lower bound:   top-K identity change >= sae_min_topk_change.
    SAE upper bound:   L2 shift  <=  sae_per_op_max * sqrt(N).
    """
    if N <= 0 or not records:
        return {"n": len(records), "pass_frac": 1.0,
                "rej_slor_inf": 0, "rej_slor_drop": 0,
                "rej_sae_topk": 0, "rej_sae_high": 0}
    slor_drop_max = slor_drop_per_op * N
    sae_max = sae_per_op_max * math.sqrt(N)
    passed = 0
    r_slor_inf = r_slor_drop = r_sae_topk = r_sae_high = 0
    for r in records:
        slor_drop = r.get("slor_drop")
        shift = r.get("sae_shift")
        topk_change = r.get("sae_topk_change")
        if slor_drop is None or not math.isfinite(slor_drop):
            r_slor_inf += 1
            continue
        if slor_drop > slor_drop_max:
            r_slor_drop += 1
            continue
        if topk_change is None or topk_change < sae_min_topk_change:
            r_sae_topk += 1
            continue
        if shift is None or not math.isfinite(shift) or shift > sae_max:
            r_sae_high += 1
            continue
        passed += 1
    total = len(records)
    return {
        "n": total,
        "pass_frac": passed / total,
        "rej_slor_inf": r_slor_inf,
        "rej_slor_drop": r_slor_drop,
        "rej_sae_topk": r_sae_topk,
        "rej_sae_high": r_sae_high,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
TSV_COLUMNS = [
    "N", "records",
    "slor_mean", "slor_p05", "slor_p25", "slor_p50", "slor_p75", "slor_p95",
    "ppl_mean", "ppl_p25", "ppl_p50", "ppl_p75", "ppl_p95",
    "sae_mean", "sae_p05", "sae_p25", "sae_p50", "sae_p75", "sae_p95",
    "topk_mean", "topk_p25", "topk_p50", "topk_p75", "topk_p95",
    "topk_eq0", "topk_eq1", "topk_2_3", "topk_4plus",
    "op_REPL", "op_INS", "op_DEL",
    "yield_default", "slor_drop_max_default",
    "sae_min_topk_change_default", "sae_max_default",
]


def format_tsv_row(N: int, summary: Dict, gate: Dict,
                   slor_drop_per_op: float,
                   sae_min_topk_change: int,
                   sae_per_op_max: float) -> List[str]:
    s = math.sqrt(N) if N > 0 else 0.0
    buckets = summary["topk_bucket"]
    return [
        str(N),
        str(summary["n_records"]),
        f"{summary['slor_mean']:.3f}",
        f"{summary['slor_p05']:.3f}",
        f"{summary['slor_p25']:.3f}",
        f"{summary['slor_p50']:.3f}",
        f"{summary['slor_p75']:.3f}",
        f"{summary['slor_p95']:.3f}",
        f"{summary['ppl_mean']:.3f}",
        f"{summary['ppl_p25']:.3f}",
        f"{summary['ppl_p50']:.3f}",
        f"{summary['ppl_p75']:.3f}",
        f"{summary['ppl_p95']:.3f}",
        f"{summary['sae_mean']:.3f}",
        f"{summary['sae_p05']:.3f}",
        f"{summary['sae_p25']:.3f}",
        f"{summary['sae_p50']:.3f}",
        f"{summary['sae_p75']:.3f}",
        f"{summary['sae_p95']:.3f}",
        f"{summary['topk_mean']:.3f}",
        f"{summary['topk_p25']:.3f}",
        f"{summary['topk_p50']:.3f}",
        f"{summary['topk_p75']:.3f}",
        f"{summary['topk_p95']:.3f}",
        str(buckets["=0"]),
        str(buckets["=1"]),
        str(buckets["2-3"]),
        str(buckets["4+"]),
        str(summary["op_counts"].get("REPL", 0)),
        str(summary["op_counts"].get("INS", 0)),
        str(summary["op_counts"].get("DEL", 0)),
        f"{gate['pass_frac']:.3f}",
        f"{slor_drop_per_op * N:.3f}" if N > 0 else "inf",
        str(sae_min_topk_change),
        f"{sae_per_op_max * s:.3f}",
    ]


def render_md(
    rows: List[Dict],
    sweeps: List[Dict],
    slor_drop_per_op: float,
    sae_min_topk_change: int,
    sae_per_op_max: float,
) -> str:
    out: List[str] = []
    out.append("# Compound-N corruption measurement\n")
    out.append("Default gates (calibrated knobs at the corruption.py defaults):")
    out.append(f"  * `slor_drop_per_op`     = {slor_drop_per_op}  →  slor_drop_max(N) = c · N  (linear)")
    out.append(f"  * `sae_min_topk_change`  = {sae_min_topk_change}  →  top-K(X) must differ from top-K(X') by ≥ this many features")
    out.append(f"  * `sae_per_op_max`       = {sae_per_op_max}  →  sae_max(N) = max · √N (L2 upper bound)\n")

    # Main table — distributions per N. SLOR is the primary fluency signal;
    # PPL ratio is kept for sanity-check against legacy runs.
    out.append("## Per-N distributions (SLOR drop primary, PPL ratio for cross-check)\n")
    out.append("| N | records | SLOR p25 | SLOR p50 | SLOR p75 | SLOR p95 | PPL p50 | PPL p95 | shift p50 | shift p95 |")
    out.append("|---|---------|----------|----------|----------|----------|---------|---------|-----------|-----------|")
    for row in rows:
        s = row["summary"]
        out.append(
            f"| {row['N']} | {s['n_records']} | "
            f"{s['slor_p25']:.3f} | {s['slor_p50']:.3f} | {s['slor_p75']:.3f} | {s['slor_p95']:.3f} | "
            f"{s['ppl_p50']:.2f} | {s['ppl_p95']:.2f} | "
            f"{s['sae_p50']:.2f} | {s['sae_p95']:.2f} |"
        )

    # Top-K identity-change distribution (new SAE lower-bound metric)
    out.append("\n## Top-K SAE feature identity change per N\n")
    out.append("How many of the top-K features differ between top_K(X) and top_K(X'). "
               "0 = activation pattern unchanged (rejected by sae_min); K = completely replaced.\n")
    out.append("| N | records | mean | p25 | p50 | p75 | p95 | =0 | =1 | 2-3 | 4+ |")
    out.append("|---|---------|------|-----|-----|-----|-----|-----|-----|-----|-----|")
    for row in rows:
        s = row["summary"]
        b = s["topk_bucket"]
        out.append(
            f"| {row['N']} | {s['n_records']} | "
            f"{s['topk_mean']:.2f} | {s['topk_p25']:.1f} | {s['topk_p50']:.1f} | "
            f"{s['topk_p75']:.1f} | {s['topk_p95']:.1f} | "
            f"{b['=0']} | {b['=1']} | {b['2-3']} | {b['4+']} |"
        )

    # Default gate yields
    out.append("\n## Yield under the default gate\n")
    out.append("| N | slor_drop_max(N) | min_topk_change | sae_max(N) | yield | rej SLOR inf | rej SLOR drop | rej topK unchanged | rej SAE high |")
    out.append("|---|------------------|-----------------|------------|-------|--------------|---------------|--------------------|--------------|")
    for row in rows:
        N = row["N"]
        s = math.sqrt(N) if N > 0 else 0.0
        slor_drop_max = slor_drop_per_op * N if N > 0 else float("inf")
        sae_max = sae_per_op_max * s
        g = row["gate_default"]
        slor_max_str = "∞" if not math.isfinite(slor_drop_max) else f"{slor_drop_max:.3f}"
        out.append(
            f"| {N} | {slor_max_str} | {sae_min_topk_change} | {sae_max:.2f} | "
            f"{g['pass_frac']*100:.1f}% | {g['rej_slor_inf']} | {g['rej_slor_drop']} | "
            f"{g['rej_sae_topk']} | {g['rej_sae_high']} |"
        )

    # Op-type histogram
    out.append("\n## Per-N op-type histogram\n")
    out.append("Total op counts (sum of all ops across all records at each N).\n")
    out.append("| N | REPL | INS | DEL |")
    out.append("|---|------|-----|-----|")
    for row in rows:
        ops = row["summary"]["op_counts"]
        out.append(
            f"| {row['N']} | {ops.get('REPL', 0)} | "
            f"{ops.get('INS', 0)} | {ops.get('DEL', 0)} |"
        )

    # Sweep
    if sweeps:
        out.append("\n## Gate sweep (yield under alternate scale constants)\n")
        out.append("Each row varies one scale constant while keeping the others at the default.")
        out.append("Shown is the per-N yield (% passing the gate) for the sampled records.\n")
        # Header from the union of N values in any sweep
        n_values_seen = sorted({row["N"] for row in rows})
        header = "| sweep | " + " | ".join(f"N={n}" for n in n_values_seen) + " |"
        sep = "|" + "|".join("---" for _ in range(len(n_values_seen) + 1)) + "|"
        out.append(header)
        out.append(sep)
        for sweep in sweeps:
            yields = sweep["per_n_pass"]
            row = "| " + sweep["label"] + " | " + \
                  " | ".join(
                      f"{yields[n]*100:.1f}%" if n in yields else "-"
                      for n in n_values_seen
                  ) + " |"
            out.append(row)

    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Directory containing n{N}/calibration.jsonl "
                                    "files (produced by smoke_pipeline.sh's "
                                    "MEASURE_COMPOUND_N=1 sweep under "
                                    "<run>/measure_n/).")
    ap.add_argument("--n-values", nargs="*", type=int, default=None,
                    help="N values to look for. Default = autodetect from "
                         "directories named nK under run_dir.")
    ap.add_argument("--slor-drop-per-op", type=float, default=0.10,
                    help="Per-op SLOR drop budget (linear N scaling).")
    ap.add_argument("--sae-min-topk-change", type=int, default=1,
                    help="Min # of features that must differ between "
                         "top-K(X) and top-K(X'). Default 1 (binary).")
    ap.add_argument("--sae-per-op-max", type=float, default=2.50,
                    help="sae_max(N) = c * sqrt(N) (L2 upper bound).")
    ap.add_argument("--report-tsv", help="Path to write TSV summary "
                                          "(default: <run_dir>/report.tsv)")
    ap.add_argument("--report-md", help="Path to write Markdown report "
                                         "(default: <run_dir>/report.md)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    if args.n_values is None:
        n_values: List[int] = []
        for sub in sorted(run_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("n") and (sub / "calibration.jsonl").exists():
                try:
                    n_values.append(int(sub.name[1:]))
                except ValueError:
                    pass
        n_values.sort()
    else:
        n_values = sorted(set(args.n_values))
    if not n_values:
        raise SystemExit(f"no calibration.jsonl files found under {run_dir}")
    print(f"[analyze-N] N values: {n_values}")

    # Load + summarise per-N
    rows: List[Dict] = []
    for N in n_values:
        cal_path = run_dir / f"n{N}" / "calibration.jsonl"
        if not cal_path.exists():
            print(f"[analyze-N] skip N={N}: {cal_path} missing")
            continue
        records = load_calibration(cal_path)
        summary = summarise_one_N(records)
        gate = simulate_gate_yield(
            records, N,
            args.slor_drop_per_op, args.sae_min_topk_change, args.sae_per_op_max,
        )
        rows.append({"N": N, "records": records, "summary": summary,
                     "gate_default": gate})

    # Gate sweep — vary each scale knob independently.
    sweeps: List[Dict] = []
    slor_grid     = [0.05, 0.10, 0.15, 0.20, 0.30]
    topk_grid     = [1, 2, 3, 4]
    sae_max_grid  = [1.50, 2.50, 4.00, 6.00]
    for sl in slor_grid:
        per_n_pass = {}
        for row in rows:
            g = simulate_gate_yield(
                row["records"], row["N"],
                sl, args.sae_min_topk_change, args.sae_per_op_max,
            )
            per_n_pass[row["N"]] = g["pass_frac"]
        sweeps.append({"label": f"slor_drop_per_op={sl}", "per_n_pass": per_n_pass})
    for tk in topk_grid:
        per_n_pass = {}
        for row in rows:
            g = simulate_gate_yield(
                row["records"], row["N"],
                args.slor_drop_per_op, tk, args.sae_per_op_max,
            )
            per_n_pass[row["N"]] = g["pass_frac"]
        sweeps.append({"label": f"sae_min_topk_change={tk}", "per_n_pass": per_n_pass})
    for smax in sae_max_grid:
        per_n_pass = {}
        for row in rows:
            g = simulate_gate_yield(
                row["records"], row["N"],
                args.slor_drop_per_op, args.sae_min_topk_change, smax,
            )
            per_n_pass[row["N"]] = g["pass_frac"]
        sweeps.append({"label": f"sae_max={smax}", "per_n_pass": per_n_pass})

    # TSV
    tsv_path = Path(args.report_tsv) if args.report_tsv else (run_dir / "report.tsv")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "wt", encoding="utf-8") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")
        for row in rows:
            cells = format_tsv_row(
                row["N"], row["summary"], row["gate_default"],
                args.slor_drop_per_op, args.sae_min_topk_change,
                args.sae_per_op_max,
            )
            f.write("\t".join(cells) + "\n")
    print(f"[analyze-N] wrote {tsv_path}")

    # Markdown
    md_path = Path(args.report_md) if args.report_md else (run_dir / "report.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_md(
        rows, sweeps,
        args.slor_drop_per_op, args.sae_min_topk_change, args.sae_per_op_max,
    ))
    print(f"[analyze-N] wrote {md_path}")

    # Console summary
    print()
    print("Per-N summary:")
    print(f"  {'N':>3} | {'records':>7} | {'SLOR p50':>9} | {'topK p50':>8} | "
          f"{'PPL p50':>8} | {'shift p50':>9} | {'yield':>7}")
    for row in rows:
        s = row["summary"]
        g = row["gate_default"]
        print(f"  {row['N']:>3} | {s['n_records']:>7} | "
              f"{s['slor_p50']:>9.3f} | {s['topk_p50']:>8.1f} | "
              f"{s['ppl_p50']:>8.3f} | {s['sae_p50']:>9.3f} | "
              f"{g['pass_frac']*100:>6.1f}%")


if __name__ == "__main__":
    main()
