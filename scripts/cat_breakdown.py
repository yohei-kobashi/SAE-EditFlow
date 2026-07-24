"""4-category (morphology/syntax/semantics/pragmatics) breakdown of the
FINAL-protocol results (user 2026-07-24): exact for the main and
adaptation rows, and FIC (E components + integrated) for the main row.

Multi-category features are assigned to their FIRST listed category
(runs/tables/feature_categories_en.json).

Usage (miyabi login is fine — stdlib only):
    python scripts/cat_breakdown.py
Writes runs/tables/cat_breakdown_final.md
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

P = Path("runs/prod_gemma_v6")
CATS = ("morphology", "syntax", "semantics", "pragmatics")
CATMAP = json.loads(Path("runs/tables/feature_categories_en.json").read_text())
W = 0.5


def cat_of(feat):
    return CATMAP.get(feat, "?")


def exact_by_cat(base):
    out = {}
    for suf, d in (("", "abl"), ("_amp", "enh")):
        agg = defaultdict(lambda: [0, 0, 0])          # cat -> [n, true, rand]
        for line in open(P / f"{base}{suf}" / "records.jsonl"):
            r = json.loads(line)
            c = cat_of(r.get("feature") or "?")
            t = r["outputs"].get("true", {}).get("ef", {}).get("text")
            rd = r["outputs"].get("random", {}).get("ef", {}).get("text")
            if t is None or rd is None:
                continue
            agg[c][0] += 1
            agg[c][1] += t.strip() == r["tgt"].strip()
            agg[c][2] += rd.strip() == r["tgt"].strip()
        out[d] = {c: (n, tr / n, rn / n, (tr - rn) / n)
                  for c, (n, tr, rn) in agg.items() if n}
    return out


def fic_by_cat(cache_dirs):
    """cache_dirs = [sup_dir, amp_dir]; returns per-cat E_abl/E_enh/FIC."""
    rel = {}
    for d in cache_dirs:
        for line in open(P / d / "judge_cache_gpt-4o.jsonl"):
            c = json.loads(line)
            rel[c["key"]] = c["rel"]
    # per (feature, dir, cond): success counts
    bucket = defaultdict(lambda: [0, 0])
    for key, r in rel.items():
        frame, feat, uid, dirn, arm, cond = key.split("|")
        if arm != "ef":
            continue
        ok = (r == "MORE") if dirn == "enh" else (r == "LESS")
        b = bucket[(feat, dirn, cond)]
        b[0] += 1
        b[1] += ok
    def pt(feat, dirn, cond):
        n, k = bucket.get((feat, dirn, cond), (0, 0))
        return (k / n) if n else float("nan")
    feats = sorted({k[0] for k in bucket})
    per_cat = defaultdict(lambda: defaultdict(list))
    for f in feats:
        ea = ee = float("nan")
        pt_a, pb_a = pt(f, "abl", "targeted"), pt(f, "abl", "random")
        if not math.isnan(pt_a) and not math.isnan(pb_a) and pt_a > 0:
            ea = (pt_a - pb_a) / pt_a
        pt_e, pb_e = pt(f, "enh", "targeted"), pt(f, "enh", "random")
        if not math.isnan(pt_e) and not math.isnan(pb_e) and pb_e < 1:
            ee = (pt_e - pb_e) / (1 - pb_e)
        c = cat_of(f)
        if not math.isnan(ea):
            per_cat[c]["ea"].append(ea)
        if not math.isnan(ee):
            per_cat[c]["ee"].append(ee)
        if not math.isnan(ea) and not math.isnan(ee):
            pa = ea if ea >= 0 else W * abs(ea)
            pe = ee if ee >= 0 else W * abs(ee)
            fic = 2 * pa * pe / (pa + pe) if pa + pe > 0 else 0.0
            per_cat[c]["fic"].append(fic)
    return per_cat


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    lines = ["# 4-category breakdown — final protocol (L12)", ""]
    for label, base in (("main row (zero-shot T2+(7))", "fs_v6t2_l12"),
                        ("adaptation row (blend a=0.3)", "fs_v3a_final_l12")):
        ex = exact_by_cat(base)
        lines += [f"## exact — {label}", "",
                  "| category | n | abl true | abl rand | abl net | "
                  "enh true | enh rand | enh net |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in CATS:
            a = ex["abl"].get(c)
            e = ex["enh"].get(c)
            if not a and not e:
                continue
            f = lambda v: "—" if v is None else f"{v:.3f}"
            lines.append(
                f"| {c} | {a[0] if a else 0} | "
                + (f"{a[1]:.3f} | {a[2]:.3f} | **{a[3]:.3f}** | " if a
                   else "— | — | — | ")
                + (f"{e[1]:.3f} | {e[2]:.3f} | **{e[3]:.3f}** |" if e
                   else "— | — | — |"))
        lines.append("")
    pc = fic_by_cat(["fic_ad_l12", "fic_ad_l12_amp"])
    lines += ["## FIC — main row (zero-shot T2+(7))", "",
              "| category | features | E_enh | E_abl | integrated FIC |",
              "|---|---|---|---|---|"]
    for c in CATS:
        d = pc.get(c, {})
        lines.append(f"| {c} | {len(d.get('fic', []))} | "
                     f"{mean(d.get('ee', [])):.3f} | "
                     f"{mean(d.get('ea', [])):.3f} | "
                     f"**{mean(d.get('fic', [])):.3f}** |")
    out = Path("runs/tables/cat_breakdown_final.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
