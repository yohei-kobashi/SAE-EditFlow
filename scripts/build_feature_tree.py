#!/usr/bin/env python3
"""C: the editable/uneditable decision tree — per-phenomenon disentanglement
of WHY intervention editing fails where it fails (🔵 2026-07-17 reframing).

Joins five measurements per phenomenon and classifies each into a suggested
failure locus (all thresholds explicit, all raw columns printed — the class
is a SUGGESTION for the Limitations section, not a proof):

  detection   AUROC of the top-1 latent (select_features_auroc output)
  WHERE       P-I causal signal: mean fires true - random, and >/< counts
              (readout records, same parsing as analyze_readout_where.py)
  intervention editing   steer exact (per-feature CSV) + clamp exact
              (records + mode, joined to phenomena via dataset idx)
  conditioning editing   ef32 / routed exact (per-feature CSV)
  prompting   B2 exact (records + mode) — the SAE-free reference

Classes:
  A 介入で編集可能        steer>0 or clamp>0 — the causal proof holds
  D LM/タスク側           nobody edits it (B2=0 and EF=0 and routed=0)
  B SAE側の示唆           text-editable by someone, but the identified
                          latents carry no causal WHERE signal
  C 効果器側(WHAT問題)  WHERE positive + text-editable, yet intervention
                          cannot execute the edit
  ? データ不足            a needed column is missing

CPU-only, reads existing artifacts. Run on prepost (records live on miyabi):
    python scripts/build_feature_tree.py \
        --out runs/tables/feature_tree
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-feature-csv",
                   default="runs/tables/main_metrics_997_per_feature.csv")
    p.add_argument("--auroc-json",
                   default="runs/auroc/identified_l12_16k_r1.json",
                   help="{phenomenon: [[latent, auroc], ...]} — top-1 score")
    p.add_argument("--where-records",
                   default="runs/prod_gemma_v6/clamp_readout500_v2/"
                           "delta_local/records.jsonl")
    p.add_argument("--where-mode", default="delta0.5")
    p.add_argument("--clamp-records",
                   default="runs/prod_gemma_v6/clamp_baseline500/"
                           "records.jsonl")
    p.add_argument("--clamp-mode", default="clamp10")
    p.add_argument("--b2-records",
                   default="runs/prod_gemma_v6/prompt_baseline500/"
                           "records.jsonl")
    p.add_argument("--b2-mode", default="prompt8")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--min-n", type=int, default=4)
    p.add_argument("--detect-thr", type=float, default=0.9)
    p.add_argument("--paper-mode", action="store_true",
                   help="⚫ EF除外版(2026-07-17決定): ef32/routed列を出さず、"
                        "C類の判定を『WHERE陽性+介入不可』に再定義する — "
                        "『活性に情報が在る』の証拠はEFでなくP-I WHEREが担う。"
                        "B2は補助列(LM側の切り分け)に残る")
    p.add_argument("--out", default="runs/tables/feature_tree")
    return p.parse_args()


def per_feature_exact(records_path, mode, idx2ph):
    """mean exact of outputs.true.<mode> per phenomenon; None if missing."""
    if not Path(records_path).exists():
        return None
    acc = defaultdict(list)
    with open(records_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            node = (r.get("outputs", {}).get("true") or {}).get(mode)
            if not isinstance(node, dict):
                continue
            ph = idx2ph.get(int(r["idx"]))
            if ph:
                acc[ph].append(float(node.get("exact", 0)))
    return {ph: sum(v) / len(v) for ph, v in acc.items()} if acc else None


def where_stats(records_path, mode, idx2ph):
    """per-phenomenon (mean true fires - random fires, n_gt, n_lt)."""
    if not Path(records_path).exists():
        return None
    acc = defaultdict(list)
    with open(records_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            o = r.get("outputs", {})
            t = o.get("true", {}).get(mode)
            c = o.get("random", {}).get(mode)
            if t is None or c is None:
                continue
            ph = idx2ph.get(int(r["idx"]))
            if ph:
                acc[ph].append((int(t["n_fire"]), int(c["n_fire"])))
    out = {}
    for ph, v in acc.items():
        d = sum(a - b for a, b in v) / len(v)
        gt = sum(1 for a, b in v if a > b)
        lt = sum(1 for a, b in v if a < b)
        out[ph] = (d, gt, lt)
    return out or None


def main():
    args = parse_args()
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    idx2ph = {i: (ds[i].get("feature") or "?") for i in range(len(ds))}

    # per-feature CSV: ef32/steer/routed exact
    base = {}
    with open(args.per_feature_csv) as f:
        for row in csv.DictReader(f):
            base[row["feature"]] = {
                "n": int(row["n"]),
                "ef": float(row["ef32_exact"]),
                "steer": float(row["steer_exact"]),
                "routed": float(row["routed_exact"]),
            }

    auroc = None
    if Path(args.auroc_json).exists():
        raw = json.loads(Path(args.auroc_json).read_text())
        auroc = {ph: float(lst[0][1]) for ph, lst in raw.items() if lst}

    where = where_stats(args.where_records, args.where_mode, idx2ph)
    clamp = per_feature_exact(args.clamp_records, args.clamp_mode, idx2ph)
    b2 = per_feature_exact(args.b2_records, args.b2_mode, idx2ph)
    for name, d in (("auroc", auroc), ("where", where), ("clamp", clamp),
                    ("b2", b2)):
        print(f"[tree] {name}: "
              f"{'MISSING (column will be —)' if d is None else len(d)}")

    def classify(ph, row):
        st, cl = row["steer"], row.get("clamp")
        ef, rt, b = row["ef"], row["routed"], row.get("b2")
        w = row.get("where")
        if st > 0 or (cl is not None and cl > 0):
            return "A 介入で編集可能"
        if w is None:
            return "? データ不足"
        d, gt, lt = w
        where_pos = d > 0 and gt > lt
        if args.paper_mode:
            # ⚫ EF除外版: 「活性に情報が在る」はP-I WHEREのみが担う。
            # C = WHERE陽性なのに介入が編集を実行できない(効果器側)。
            # B = WHERE無しなのにB2は編集できる(同定/SAE側の示唆)。
            # D = WHERE無しかつB2も不可(LM/タスク側 or SAE側、不定)。
            if where_pos:
                return "C 効果器側(WHERE有・介入実行不可)"
            if b is not None and b > 0:
                return "B SAE側の示唆(因果WHERE無し・B2可)"
            return "D LM/タスク側 or SAE側(WHERE無し・B2不可、不定)"
        if b is None:
            return "? データ不足"
        text_editable = (b > 0) or (ef > 0) or (rt > 0)
        if not text_editable:
            return "D LM/タスク側"
        if not where_pos:
            return "B SAE側の示唆(因果WHERE無し)"
        return "C 効果器側(WHERE有・実行不可=WHAT問題)"

    rows = []
    for ph, b_ in sorted(base.items()):
        if b_["n"] < args.min_n:
            continue
        row = dict(feature=ph, **b_)
        row["auroc"] = auroc.get(ph) if auroc else None
        row["where"] = where.get(ph) if where else None
        row["clamp"] = clamp.get(ph) if clamp else None
        row["b2"] = b2.get(ph) if b2 else None
        row["class"] = classify(ph, row)
        rows.append(row)

    fmt = lambda v, s="{:.3f}": "—" if v is None else s.format(v)  # noqa
    order = {"A": 0, "C": 1, "B": 2, "D": 3, "?": 4}
    rows.sort(key=lambda r: (order.get(r["class"][0], 9), -r["steer"],
                             -r["ef"]))
    mode_note = ("⚫ paper版(EF列なし — C類はWHERE基準)" if args.paper_mode
                 else "内部診断版(EF列あり)")
    L = ["# 編集可能性の判別木(現象別)— SAE側/効果器側/LM側の切り分け示唆",
         "",
         f"{mode_note}; detection thr {args.detect_thr}; where = mean fires "
         f"true−random ({args.where_mode}); clamp = {args.clamp_mode}; b2 = "
         f"{args.b2_mode}; 貪欲最小化・閾値は全て表に出す(classは示唆)",
         ""]
    if args.paper_mode:
        L += ["| feature | n | AUROC | WHERE Δ (>/< ) | steer | clamp | B2 "
              "| class |",
              "|---|---|---|---|---|---|---|---|"]
    else:
        L += ["| feature | n | AUROC | WHERE Δ (>/< ) | steer | clamp | "
              "ef32 | routed | B2 | class |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        w = r["where"]
        wtxt = "—" if w is None else f"{w[0]:+.2f} ({w[1]}/{w[2]})"
        if args.paper_mode:
            L.append(
                f"| {r['feature']} | {r['n']} | {fmt(r['auroc'])} | {wtxt} "
                f"| {r['steer']:.3f} | {fmt(r['clamp'])} | {fmt(r['b2'])} "
                f"| {r['class']} |")
        else:
            L.append(
                f"| {r['feature']} | {r['n']} | {fmt(r['auroc'])} | {wtxt} | "
                f"{r['steer']:.3f} | {fmt(r['clamp'])} | {r['ef']:.3f} | "
                f"{r['routed']:.3f} | {fmt(r['b2'])} | {r['class']} |")
    counts = defaultdict(int)
    for r in rows:
        counts[r["class"]] += 1
    L += ["", "## 集計", ""]
    L += [f"- {c}: {n}" for c, n in sorted(counts.items())]
    if args.paper_mode:
        L += ["",
              "読み(⚫paper版): 証拠は3種のみ — (1) do-介入編集(steer/"
              "clamp)= 因果証明、(2) P-I WHERE = 活性が因果信号を運ぶかの"
              "学習ゼロ証拠、(3) B2 = SAE不使用のタスク実行可能性フロア。"
              "A = 介入編集が成立(因果証明)。C = WHERE陽性なのに介入が"
              "編集を実行できない = 効果器側(WHAT問題)。B = WHERE無しなの"
              "にB2は編集できる = 同定/SAE側の示唆(交絡latent疑い)。D = "
              "WHERE無しかつB2不可 = LM/タスク側かSAE側か不定。編集器"
              "(EF/パイプライン/B2)はどれも条件付けで因果的に等価なため"
              "判別には使わない。切り分けは示唆(Limitation本文)。"]
    else:
        L += ["",
              "読み(内部診断版): A = 介入編集が因果証明として成立。C = "
              "情報は活性に在る(WHERE陽性・EF/B2可)が介入が実行できない "
              "= 効果器側(WHAT問題)。B = 誰かは編集できるのに同定活性に"
              "因果信号が無い = 同定/SAE側の示唆。D = B2すら編集できない = "
              "LM/タスク側。切り分けは示唆に留まる。"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["feature", "n", "auroc", "where_delta", "where_gt",
                       "where_lt", "steer", "clamp", "ef32", "routed", "b2",
                       "class"])
        for r in rows:
            w = r["where"] or (None, None, None)
            wcsv.writerow([r["feature"], r["n"], r["auroc"], w[0], w[1],
                           w[2], r["steer"], r["clamp"], r["ef"],
                           r["routed"], r["b2"], r["class"]])
    print("\n".join(L[:20]))
    print(f"...\n[tree] wrote {out.with_suffix('.md')} and .csv "
          f"({len(rows)} phenomena)")


if __name__ == "__main__":
    main()
