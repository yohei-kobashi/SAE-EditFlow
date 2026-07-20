#!/usr/bin/env python3
"""FIC judge + metrics over BOTH frames (CPU, stdlib + datasets only).

User decision 2026-07-21: the FIC table gets all four arms in BOTH frames.
  * bare frame  — eval_fic_gen.py records (LinguaLens-verbatim protocol:
    free continuation, temp 1.0, per-feature n_pairs x 2 directions,
    shared raw control). Arms present there: steer / clamp / prompting,
    plus ef once the bare ef generation lands (resume-safe: rerun me).
  * repeat frame — REUSE of the exact-probe generations (greedy, 499-pair
    seed-42 sample): ef+steer from eflm_l12_v5f/probe500, clamp10 from
    clamp_baseline500, prompting_edit from a3prime_edit; reference =
    probe500's raw repeat (outputs.empty.raw). Directions recovered by
    joining tgt against the dataset (sentence1 = feature-present side).
    Disclosed deviations from the bare protocol: greedy decode, single
    sample, direction balance follows the probe sample.

Judge follows judge_ll_repro.py (paper §3.3/§4.2.2/App. E.2): GPT-4o
MORE/LESS/SAME vs the control text, order randomized; success = MORE for
enhancement, LESS for ablation;
  E_abl = (PT - PB) / PT ;  E_enh = (PT - PB) / (1 - PB)
  FIC = harmonic mean of penalized effects (E' = E if E>=0 else w*|E|).

Usage (prepost/short-c, needs OPENAI_API_KEY):
  python scripts/eval_fic_judge.py \
      --bare-dir runs/prod_gemma_v6/fic_l12 \
      --repeat-probe500 runs/prod_gemma_v6/eflm_l12_v5f/probe500/records.jsonl \
      --repeat-clamp runs/prod_gemma_v6/clamp_baseline500/records.jsonl \
      --repeat-a3 runs/prod_gemma_v6/a3prime_edit/records.jsonl \
      --output-dir runs/prod_gemma_v6/fic_judge_l12
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_paper_metrics import Judge                           # noqa: E402

PROMPT = """You are evaluating the prominence of a linguistic feature in \
generated text.

Feature: {name}

TEXT A:
{a}

TEXT B:
{b}

Compared with TEXT A, does TEXT B exhibit this feature MORE prominently, \
LESS prominently, or about the SAME? Answer with exactly one word: MORE, \
LESS, or SAME."""

BARE_ARMS = ("ef", "steer", "clamp", "prompting")
REPEAT_ARMS = ("ef", "steer", "clamp", "prompting")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bare-dir", default="")
    p.add_argument("--repeat-probe500", default="")
    p.add_argument("--repeat-clamp", default="")
    p.add_argument("--repeat-a3", default="")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--dir-map", default="runs/tables/lingualens_dirmap_en.json",
                   help="feature -> {s1:[...], s2:[...]} JSON (avoids the "
                        "datasets dependency on CPU nodes); falls back to "
                        "loading the dataset if the file is absent")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--judge-model", default="gpt-4o")
    p.add_argument("--order-seed", type=int, default=1234)
    p.add_argument("--penalty-w", type=float, default=0.5)
    p.add_argument("--max-calls", type=int, default=0,
                   help="stop after N new judge calls (0 = no cap); the "
                        "cache makes reruns resume")
    return p.parse_args()


def parse_rel(text):
    t = (text or "").strip().upper()
    for w in ("MORE", "LESS", "SAME"):
        if w in t:
            return w
    return None


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def collect_trials(args):
    """Yield dicts: frame, feature, uid, dir, arm, cond, output, ref."""
    trials = []
    if args.bare_dir:
        recs = load_jsonl(Path(args.bare_dir) / "records.jsonl")
        ctl = {}
        for r in recs:
            if r["cond"] == "control" and r["arm"] == "_":
                ctl[(r["feature"], r["pair"], r["dir"])] = r["output"]
        n_skip = 0
        for r in recs:
            if r["cond"] not in ("targeted", "random"):
                continue        # control / clamp-recon reference rows
            ref = ctl.get((r["feature"], r["pair"], r["dir"]))
            if ref is None:
                n_skip += 1
                continue
            trials.append(dict(
                frame="bare", feature=r["feature"],
                uid=f'{r["pair"]}', dir=r["dir"], arm=r["arm"],
                cond=r["cond"], output=r["output"], ref=ref))
        print(f"[ficjudge] bare: {len(trials)} trials "
              f"({n_skip} skipped, no control)")
    if args.repeat_probe500:
        base = load_jsonl(args.repeat_probe500)
        # direction join: dataset sentence1 = feature-present side
        side = {}
        if args.dir_map and Path(args.dir_map).exists():
            dm = json.loads(Path(args.dir_map).read_text())
            for feat, d in dm.items():
                for s in d["s1"]:
                    side[(feat, s)] = "enh"
                for s in d["s2"]:
                    side[(feat, s)] = "abl"
        else:
            from datasets import load_dataset
            ds = load_dataset(args.dataset, split="train")
            ds = ds.filter(lambda r: r["language"] == args.language)
            for i in range(len(ds)):
                row = ds[i]
                side[(row["feature"], row["sentence1"])] = "enh"
                side[(row["feature"], row["sentence2"])] = "abl"
        clamp = ({int(r["idx"]): r for r in load_jsonl(args.repeat_clamp)}
                 if args.repeat_clamp else {})
        a3 = ({int(r["idx"]): r for r in load_jsonl(args.repeat_a3)}
              if args.repeat_a3 else {})
        n0 = len(trials)
        n_nodir = n_noref = 0
        for r in base:
            feat, idx = r["feature"], int(r["idx"])
            d = side.get((feat, r["tgt"]))
            if d is None:
                n_nodir += 1
                continue
            ref = (r["outputs"].get("empty", {}).get("raw", {})
                   .get("text"))
            if not ref:
                n_noref += 1
                continue
            for cond_rec, cond in (("true", "targeted"),
                                   ("random", "random")):
                for arm, src_rec, key in (
                        ("ef", r, "ef"), ("steer", r, "steer"),
                        ("clamp", clamp.get(idx), "clamp10"),
                        ("prompting", a3.get(idx), "prompting_edit")):
                    if src_rec is None:
                        continue
                    txt = (src_rec["outputs"].get(cond_rec, {})
                           .get(key, {}).get("text"))
                    if txt is None:
                        continue
                    trials.append(dict(
                        frame="repeat", feature=feat, uid=str(idx),
                        dir=d, arm=arm, cond=cond, output=txt, ref=ref))
        print(f"[ficjudge] repeat: {len(trials) - n0} trials "
              f"(no-direction {n_nodir}, no-raw-ref {n_noref})")
    return trials


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials = collect_trials(args)

    cache_path = out / f"judge_cache_{args.judge_model}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            try:
                c = json.loads(line)
                cache[c["key"]] = c["rel"]
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"[ficjudge] RESUME: {len(cache)} judgments cached")
    cfh = open(cache_path, "a")
    judge = Judge(args.judge_model, max_tokens=6)

    n_new = 0
    for t in trials:
        key = "|".join((t["frame"], t["feature"], t["uid"], t["dir"],
                        t["arm"], t["cond"]))
        if key in cache:
            continue
        if args.max_calls and n_new >= args.max_calls:
            break
        flip = random.Random(f"{args.order_seed}|{key}").random() < 0.5
        a, b = (t["output"], t["ref"]) if flip else (t["ref"], t["output"])
        ans = parse_rel(judge(PROMPT.format(name=t["feature"], a=a, b=b)))
        if ans is None or ans == "SAME":
            rel = "SAME"
        else:
            rel = ans if not flip else ("LESS" if ans == "MORE" else "MORE")
        cache[key] = rel
        cfh.write(json.dumps({"key": key, "rel": rel}) + "\n")
        n_new += 1
        if n_new % 50 == 0:
            cfh.flush()
            print(f"[ficjudge] +{n_new}", flush=True)
    cfh.close()
    print(f"[ficjudge] {n_new} new judgments, {len(cache)} total")

    # ---- metrics --------------------------------------------------------
    # bucket[(frame, arm, feature, dir, cond)] = [rel, ...]
    bucket = defaultdict(list)
    n_missing = 0
    for t in trials:
        key = "|".join((t["frame"], t["feature"], t["uid"], t["dir"],
                        t["arm"], t["cond"]))
        rel = cache.get(key)
        if rel is None:
            n_missing += 1
            continue
        bucket[(t["frame"], t["arm"], t["feature"], t["dir"],
                t["cond"])].append(rel)

    def rate(frame, arm, feat, d, cond):
        rels = bucket.get((frame, arm, feat, d, cond))
        if not rels:
            return float("nan"), 0
        want = "MORE" if d == "enh" else "LESS"
        return sum(r == want for r in rels) / len(rels), len(rels)

    def eff(pt, pb, kind):
        if math.isnan(pt) or math.isnan(pb):
            return float("nan")
        if kind == "abl":
            return (pt - pb) / pt if pt > 0 else float("nan")
        return (pt - pb) / (1.0 - pb) if pb < 1.0 else float("nan")

    def penal(e):
        return e if e >= 0 else args.penalty_w * abs(e)

    def fic(ee, ea):
        if math.isnan(ee) or math.isnan(ea):
            return float("nan")
        pe, pa = penal(ee), penal(ea)
        return 2 * pe * pa / (pe + pa) if (pe + pa) > 0 else 0.0

    feats = sorted({t["feature"] for t in trials})
    frames = sorted({t["frame"] for t in trials})
    lines = ["# FIC — four arms x both frames",
             "",
             f"judge: {args.judge_model} | penalty w={args.penalty_w} | "
             f"success: enh=MORE, abl=LESS vs frame control | "
             f"{n_missing} trials unjudged (budget cap; rerun resumes)",
             ""]
    for frame in frames:
        arms = BARE_ARMS if frame == "bare" else REPEAT_ARMS
        lines += [f"## frame = {frame}", "",
                  "| arm | mean E_enh | mean E_abl | mean FIC | "
                  "features (enh/abl/both) |",
                  "|---|---|---|---|---|"]
        per_feat_lines = ["| feature | arm | PT_enh | PB_enh | PT_abl | "
                          "PB_abl | E_enh | E_abl | FIC |",
                          "|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            ees, eas, fics = [], [], []
            n_e = n_a = n_b = 0
            for f in feats:
                pt_e, ne = rate(frame, arm, f, "enh", "targeted")
                pb_e, _ = rate(frame, arm, f, "enh", "random")
                pt_a, na = rate(frame, arm, f, "abl", "targeted")
                pb_a, _ = rate(frame, arm, f, "abl", "random")
                ee = eff(pt_e, pb_e, "enh")
                ea = eff(pt_a, pb_a, "abl")
                fc = fic(ee, ea)
                if ne:
                    n_e += 1
                if na:
                    n_a += 1
                if ne and na:
                    n_b += 1
                if not math.isnan(ee):
                    ees.append(ee)
                if not math.isnan(ea):
                    eas.append(ea)
                if not math.isnan(fc):
                    fics.append(fc)
                if ne or na:
                    def fmt(v):
                        return "—" if math.isnan(v) else f"{v:.3f}"
                    per_feat_lines.append(
                        f"| {f} | {arm} | {fmt(pt_e)} | {fmt(pb_e)} | "
                        f"{fmt(pt_a)} | {fmt(pb_a)} | {fmt(ee)} | "
                        f"{fmt(ea)} | {fmt(fc)} |")

            def m(v):
                return f"{sum(v)/len(v):.3f}" if v else "—"
            lines.append(f"| {arm} | {m(ees)} | {m(eas)} | {m(fics)} | "
                         f"{n_e}/{n_a}/{n_b} |")
        lines.append("")
        (out / f"perfeature_{frame}.md").write_text(
            "\n".join(per_feat_lines) + "\n")
    lines += ["notes:",
              "- repeat frame reuses the greedy exact-probe generations "
              "(single sample, probe-sample direction balance) — "
              "disclosed deviation from the bare temp-1.0 protocol.",
              "- FIC needs both directions; arms/features with one "
              "direction report E only.",
              ""]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    all_arms_bare = (not args.bare_dir) or all(
        any(t["frame"] == "bare" and t["arm"] == a for t in trials)
        for a in BARE_ARMS)
    if n_missing == 0 and all_arms_bare:
        print("==================== FIC-JUDGE-DONE ====================")
    else:
        print(f"[ficjudge] PARTIAL: missing={n_missing} "
              f"bare-arms-complete={all_arms_bare}")


if __name__ == "__main__":
    main()
