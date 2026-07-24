"""Integrated FIC (LinguaLens App. E.2) for the feature-spec protocol.

The new-protocol judge cells are single-direction (sup cell -> E_abl,
amp cell -> E_enh), so eval_fic_judge's own FIC column stays empty. This
script merges the DIRECTION-PAIRED judge caches per arm and computes the
integrated per-feature FIC = harmonic mean of the w-penalized E_enh and
E_abl — byte-identical math to eval_fic_judge (eff/penal/fic).

Arm-specific cache sources avoid the steer key collision (the ef-scale
steer judgments in fic_fs_l* share keys with the calibrated fic_fs_steer_*
ones — only the calibrated caches are read for the steer arm).

Usage (miyabi):
    python scripts/combine_fic.py --out runs/tables/fic_fs_integrated.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

P = Path("runs/prod_gemma_v6")
CACHE = "judge_cache_gpt-4o.jsonl"

# arm -> {layer: [cache dirs (sup cell, amp cell)]}
# 2026-07-24 final protocol: ef = ADOPTED T2+(7) (fic_ad_*), clamp/axbench
# = repeat-frame dev-selected baselines (fic_bl_*); steer stays the
# calibrated caches; prompting unchanged. Extra ef variants (T4/T4v2)
# reported separately in 04 §9q.
SOURCES = {
    "ef":        {L: [f"fic_ad_l{L}", f"fic_ad_l{L}_amp"] for L in (4, 12, 20)},
    "clamp":     {L: [f"fic_bl_l{L}", f"fic_bl_l{L}_amp"] for L in (4, 12, 20)},
    "prompting": {12: ["fic_fs_l12", "fic_fs_l12_amp"]},
    "steer":     {L: [f"fic_fs_steer_l{L}", f"fic_fs_steer_l{L}_amp"]
                  for L in (4, 12, 20)},
    "axbench":   {L: [f"fic_bl_l{L}", f"fic_bl_l{L}_amp"] for L in (4, 12, 20)},
}
W = 0.5


def eff(pt, pb, kind):
    if math.isnan(pt) or math.isnan(pb):
        return float("nan")
    if kind == "abl":
        return (pt - pb) / pt if pt > 0 else float("nan")
    return (pt - pb) / (1.0 - pb) if pb < 1.0 else float("nan")


def penal(e):
    return e if e >= 0 else W * abs(e)


def fic(ee, ea):
    if math.isnan(ee) or math.isnan(ea):
        return float("nan")
    pe, pa = penal(ee), penal(ea)
    return 2 * pe * pa / (pe + pa) if (pe + pa) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tables/fic_fs_integrated.md")
    args = ap.parse_args()

    lines = ["# integrated FIC (feature-spec protocol; harmonic mean of "
             f"w={W}-penalized E_enh & E_abl, LinguaLens App. E.2)", "",
             "| arm | layer | mean E_enh | mean E_abl | **mean FIC** | "
             "features (both dirs) |", "|---|---|---|---|---|---|"]
    for arm, per_layer in SOURCES.items():
        for L, dirs in sorted(per_layer.items()):
            bucket = defaultdict(list)
            missing = [d for d in dirs if not (P / d / CACHE).exists()]
            if missing:
                lines.append(f"| {arm} | L{L} | — | — | — | "
                             f"(cache missing: {','.join(missing)}) |")
                continue
            for d in dirs:
                for line in open(P / d / CACHE):
                    try:
                        c = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fr, feat, uid, dr, a, cond = c["key"].split("|")
                    if a != arm:
                        continue
                    bucket[(feat, dr, cond)].append(c["rel"])
            feats = sorted({k[0] for k in bucket})
            ees, eas, fics = [], [], []
            for f in feats:
                def rate(dr, cond):
                    rels = bucket.get((f, dr, cond))
                    if not rels:
                        return float("nan")
                    want = "MORE" if dr == "enh" else "LESS"
                    return sum(r == want for r in rels) / len(rels)
                ee = eff(rate("enh", "targeted"), rate("enh", "random"),
                         "enh")
                ea = eff(rate("abl", "targeted"), rate("abl", "random"),
                         "abl")
                fc = fic(ee, ea)
                if not math.isnan(ee):
                    ees.append(ee)
                if not math.isnan(ea):
                    eas.append(ea)
                if not math.isnan(fc):
                    fics.append(fc)
            def m(v):
                return f"{sum(v) / len(v):.3f}" if v else "—"
            lines.append(f"| {arm} | L{L} | {m(ees)} | {m(eas)} | "
                         f"**{m(fics)}** | {len(fics)} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("COMBINE-FIC-DONE")


if __name__ == "__main__":
    main()
