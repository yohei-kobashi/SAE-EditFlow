"""
P-N: validate the B1/B3 implementations on the PAPERS' OWN metrics.

exact match is OUR metric. If the clamp and steer implementations are
faithful, they should register on the metrics their papers designed them
for — judged over the outputs the runs already saved (no regeneration):

LinguaLens (judge: gpt-4o, their choice):
    "We use GPT-4o to assess the prominence of the targeted linguistic
    feature in the generated outputs."  Ablation success = the feature is
    ABSENT (Y=0). Our task is removal (s1->s2), i.e. their ablation arm.
        E_abl = (P(Y=0 | targeted) - P(Y=0 | random)) / P(Y=0 | targeted)
    (their normalized ablation effect; the random-features intervention is
    their baseline arm). Positive E_abl = the implementation reproduces
    their qualitative result. FIC needs the enhancement arm too — out of
    scope here (our eval direction is removal-only); reported as E_abl.

AxBench (judge: gpt-4o-mini, their choice — "we use OpenAI's
gpt-4o-mini-2024-07-18 ... throughout"):
    steering score = harmonic mean of three 0/1/2 subscores (their §4):
    concept realization / instruction-following / fluency. Adapted to the
    rewrite task: concept = the commanded shift (feature removal) is
    realized; instruction = the output is a rewrite of the source (content
    preserved); fluency = grammatical, natural text. Harmonic mean
    "heavily penalises poor performance on any of these three subscores"
    (their wording) — overall 0-2.

Calibration rows judged once and shared:
    src   : P(Y=1 | source) should be HIGH (the dataset guarantees the
            feature is present in sentence1) — judge sanity.
    raw   : the uninformed rewriter (no intervention) — both metrics'
            no-intervention reference.

Usage (CPU + API only):
    python scripts/judge_paper_metrics.py \
        --run b1_ours=runs/prod_gemma_v6/clamp_baseline500/records.jsonl:clamp10 \
        --run ll_protocol=runs/prod_gemma_v6/protocol_e2e/lingualens_frc_r3_clamp/records.jsonl:clamp10 \
        --run b3_ours=runs/prod_gemma_v6/steer_baseline500/records.jsonl:steer0.5 \
        --run ax_protocol=runs/prod_gemma_v6/protocol_e2e/axbench_auroc_r1_steer/records.jsonl:steer1 \
        --out-dir runs/paper_metrics
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LL_PROMPT = """Linguistic feature: {feature}

Sentence: {text}

Does the sentence exhibit the linguistic feature above? Answer with exactly one word: YES or NO."""

AX_PROMPT = """You are evaluating a text-editing system. It was asked to rewrite a source sentence so that it NO LONGER exhibits a linguistic feature.

Linguistic feature to remove: {feature}
Source sentence: {src}
System output: {text}

Rate the output on three aspects, each 0, 1, or 2:
- concept: the commanded change is realized (2 = the feature is clearly no longer exhibited; 1 = partially; 0 = still exhibited or output unrelated to the goal)
- instruct: the output is a rewrite of the source (2 = same content apart from the commanded change; 1 = partially preserved; 0 = unrelated or empty)
- fluency: the output is grammatical, natural text (2 = fluent; 1 = minor issues; 0 = broken)

Answer with exactly this format and nothing else:
concept=X instruct=Y fluency=Z"""


class Judge:
    """Minimal chat-completions caller. temperature=0; max_tokens sized per
    metric (the shared OpenAIJudge caps legacy models at 4 tokens, too small
    for the AxBench rubric line)."""

    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise SystemExit("needs OPENAI_API_KEY")

    def __call__(self, prompt: str) -> str:
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": self.max_tokens}
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.load(r)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 400 and "temperature" in str(e.read())[:500]:
                    body.pop("temperature", None)      # modern-model fallback
                    body["max_completion_tokens"] = body.pop(
                        "max_tokens", self.max_tokens)
                    continue
                if e.code in (429, 500, 502, 503) and attempt < 2:
                    import time
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("judge retries exhausted")


def parse_yesno(text: str):
    t = text.strip().upper()
    if re.match(r"^YES\b", t):
        return 1
    if re.match(r"^NO\b", t):
        return 0
    m = re.search(r"\b(YES|NO)\b", t)
    return None if m is None else (1 if m.group(1) == "YES" else 0)


def parse_rubric(text: str):
    m = re.search(r"concept\s*=\s*([0-2]).*?instruct\s*=\s*([0-2]).*?"
                  r"fluency\s*=\s*([0-2])", text, re.S | re.I)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def hmean3(a, b, c):
    return 0.0 if min(a, b, c) == 0 else 3.0 / (1 / a + 1 / b + 1 / c)


def load_run(path, mode):
    rows = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out = {}
            for cond, modes in (r.get("outputs") or {}).items():
                if not isinstance(modes, dict):
                    continue
                node = modes.get(mode)
                if node is None and len(modes) == 1:   # random often stores
                    node = next(iter(modes.values()))  # a single ctrl mode
                if isinstance(node, dict) and "text" in node:
                    out[cond] = node["text"]
            rows[int(r["idx"])] = {"src": r.get("src") or r.get("source"),
                                   "out": out}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   help="name=records.jsonl:mode")
    p.add_argument("--conditions", default="true,random")
    p.add_argument("--ll-judge", default="gpt-4o",
                   help="LinguaLens used GPT-4o")
    p.add_argument("--ax-judge", default="gpt-4o-mini",
                   help="AxBench used gpt-4o-mini-2024-07-18")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--limit", type=int, default=0,
                   help="judge only the first N pairs (0 = all)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ll = Judge(args.ll_judge, 8)
    ax = Judge(args.ax_judge, 40)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    runs = {}
    for spec in args.run:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        runs[name] = load_run(path, mode)
        print(f"[pm] {name}: {len(runs[name])} pairs (mode {mode})")

    def cached_loop(tag, items, call, parse):
        """items: list of (key, prompt). Appends {'k','raw'} rows; returns
        {key: parsed}."""
        path = out_dir / f"{tag}.jsonl"
        done = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        j = json.loads(line)
                        done[j["k"]] = j["raw"]
        f = open(path, "a")
        n_new = 0
        for k, prompt in items:
            if k in done:
                continue
            done[k] = call(prompt)
            f.write(json.dumps({"k": k, "raw": done[k]},
                               ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % 50 == 0:
                print(f"[pm] {tag}: +{n_new}")
        f.close()
        return {k: parse(v) for k, v in done.items()}

    # ---- assemble judge items -------------------------------------------
    first = next(iter(runs.values()))
    idxs = sorted(first)
    if args.limit:
        idxs = idxs[:args.limit]
    feat = {k: (ds[k].get("feature") or "?") for k in idxs}

    ll_items, ax_items = [], []
    for k in idxs:
        src = first[k]["src"]
        ll_items.append((f"src|{k}", LL_PROMPT.format(
            feature=feat[k], text=src)))
        raw = first[k]["out"].get("raw") or first[k]["out"].get("empty")
        if raw:
            ll_items.append((f"raw|{k}", LL_PROMPT.format(
                feature=feat[k], text=raw)))
            ax_items.append((f"raw|{k}", AX_PROMPT.format(
                feature=feat[k], src=src, text=raw)))
    for name, rows in runs.items():
        for k in idxs:
            if k not in rows:
                continue
            for c in conds:
                t = rows[k]["out"].get(c)
                if not t:
                    continue
                key = f"{name}|{c}|{k}"
                ll_items.append((key, LL_PROMPT.format(
                    feature=feat[k], text=t)))
                ax_items.append((key, AX_PROMPT.format(
                    feature=feat[k], src=rows[k]["src"], text=t)))

    print(f"[pm] LinguaLens metric: {len(ll_items)} judgments "
          f"({args.ll_judge}); AxBench metric: {len(ax_items)} "
          f"({args.ax_judge})")
    ll_res = cached_loop("ll_presence", ll_items, ll, parse_yesno)
    ax_res = cached_loop("ax_rubric", ax_items, ax, parse_rubric)

    # ---- aggregate --------------------------------------------------------
    def p_absent(prefix):
        vals = [v for k, v in ll_res.items()
                if k.startswith(prefix) and v is not None]
        return (sum(1 for v in vals if v == 0) / len(vals), len(vals)) \
            if vals else (float("nan"), 0)

    def ax_score(prefix):
        vals = [hmean3(*v) for k, v in ax_res.items()
                if k.startswith(prefix) and v is not None]
        sub = [v for k, v in ax_res.items()
               if k.startswith(prefix) and v is not None]
        if not vals:
            return (float("nan"),) * 4 + (0,)
        n = len(vals)
        return (sum(vals) / n,
                sum(s[0] for s in sub) / n, sum(s[1] for s in sub) / n,
                sum(s[2] for s in sub) / n, n)

    L = ["# Paper-native metrics over saved outputs (P-N)", "",
         f"LinguaLens judge: {args.ll_judge} (theirs). AxBench judge: "
         f"{args.ax_judge} (theirs). Task direction = removal (their "
         "ablation arm).", "",
         "## Calibration", ""]
    pa_src, n_src = p_absent("src|")
    L.append(f"- P(feature ABSENT | source) = {pa_src:.3f} (n={n_src}) — "
             "should be LOW; 1-it is the judge's hit rate on the dataset's "
             "own positives.")
    pa_raw, n_raw = p_absent("raw|")
    L.append(f"- P(feature ABSENT | uninformed rewrite) = {pa_raw:.3f} "
             f"(n={n_raw}) — the no-intervention reference.")
    axr = ax_score("raw|")
    L.append(f"- AxBench score (uninformed rewrite) = {axr[0]:.3f} "
             f"[concept {axr[1]:.2f} / instruct {axr[2]:.2f} / fluency "
             f"{axr[3]:.2f}] (n={axr[4]})")
    L += ["", "## LinguaLens metric — ablation success and E_abl", "",
          "| run | P(Y=0\\|targeted) | P(Y=0\\|random) | E_abl | n |",
          "|---|---|---|---|---|"]
    for name in runs:
        pt, nt = p_absent(f"{name}|true|")
        pr, _ = p_absent(f"{name}|random|")
        e = (pt - pr) / pt if pt and pt == pt and pt > 0 else float("nan")
        L.append(f"| {name} | {pt:.3f} | {pr:.3f} | **{e:+.3f}** | {nt} |")
    L += ["", "## AxBench metric — harmonic mean of 0-2 subscores", "",
          "| run | cond | overall | concept | instruct | fluency | n |",
          "|---|---|---|---|---|---|---|"]
    for name in runs:
        for c in conds:
            s = ax_score(f"{name}|{c}|")
            L.append(f"| {name} | {c} | **{s[0]:.3f}** | {s[1]:.2f} | "
                     f"{s[2]:.2f} | {s[3]:.2f} | {s[4]} |")
    L += ["", "Reading: a faithful clamp implementation should show "
          "E_abl > 0 on the LinguaLens metric (their qualitative claim), "
          "and a faithful steer implementation should beat the uninformed "
          "rewrite on the AxBench concept subscore with fluency held up. "
          "exact and these metrics may rank the runs differently — that is "
          "the paper's point, not a contradiction."]
    report = "\n".join(L)
    print("\n" + report)
    (out_dir / "report.md").write_text(report + "\n")
    print(f"\n[pm] wrote {out_dir}/report.md")


if __name__ == "__main__":
    main()
