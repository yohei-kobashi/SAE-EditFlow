"""
Sample LinguaLens minimal pairs and report token-level diff distribution.

For each (sentence1, sentence2) pair we tokenize with the Gemma tokenizer
used by the SAE-LEWIS pipeline (google/gemma-2-2b by default) and compute
three related but distinct measures of edit complexity at TOKEN
granularity (so the numbers stay directly tied to what the SAE-LEWIS
editor would have to emit):

  * `tok_edit` -- classical Levenshtein, insert/delete/substitute cost 1
    each. The minimum number of single-token edit operations needed to
    transform sentence1 into sentence2. Suffers from two over-counts when
    interpreted as an SAE-LEWIS op count:
      (a) a contiguous block insertion of K tokens scores K, whereas the
          editor emits 1 INS marker + length-K via the length head.
      (b) a single multi-subword word substitution scores #subtokens,
          whereas the corruption pipeline would call it 1 REPL op.

  * `n_hunks` -- number of contiguous non-equal blocks in the
    difflib.SequenceMatcher diff. Corresponds 1-to-1 to SAE-LEWIS's
    `N_total` (one REPL/INS/DEL hunk = one op), so this is the right
    metric for "does the LinguaLens N distribution match the
    corruption pipeline's bucket weights?".

  * `editor_ops` -- weighted hunk count that reflects the editor's
    inference cost:
        REPL hunk of K tokens   -> K ops  (one marker per token)
        DEL  hunk of K tokens   -> K ops  (one [DEL] marker per token)
        INS  gap of any size    -> 1 op   (one [INS] marker;
                                            length head emits the rest)
    `editor_ops` answers "how many emit-operations does the editor
    actually have to make at inference time?". By construction
    `tok_edit >= editor_ops >= n_hunks`.

Also recorded:
  * `len(tok1)`, `len(tok2)`, `|Δlen|`
  * `set_diff = |set(tok1) △ set(tok2)|` (order-insensitive, dedup-counted)
  * `op_types` -- per-hunk REPL/INS/DEL counts so the report can compare
    against the corruption pipeline's `(0.55, 0.25, 0.20)` op weights.

Output: a Markdown report with per-language and overall percentiles plus
per-integer histograms for `tok_edit`, `n_hunks`, and `editor_ops`.
Designed to be checked into the repo without going into README.md.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


def token_edit_distance(a: List[int], b: List[int]) -> int:
    """Classical Levenshtein with unit costs (insert/delete/substitute)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def diff_metrics(a: List[int], b: List[int]) -> Dict:
    """Return (n_hunks, editor_ops, op_types) from a difflib diff.

    Maps difflib opcodes onto SAE-LEWIS op types as follows:
      * 'replace' -> REPL hunk (1 hunk; editor cost = (i2-i1) per-token markers)
      * 'insert'  -> DEL hunk  (corruption side INSERTED tokens that the
                                 editor must DELETE; editor cost =
                                 (j2-j1) per-token markers)
      * 'delete'  -> INS hunk  (corruption side DELETED tokens that the
                                 editor must INSERT back; editor cost = 1
                                 marker + length-K from the length head)

    The 'insert' vs 'delete' inversion intentionally matches the
    SAE-LEWIS naming convention: op names describe what the editor must
    do, not what the corruption did.
    """
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    opcodes = sm.get_opcodes()
    n_hunks = 0
    editor_ops = 0
    counts = {"REPL": 0, "INS": 0, "DEL": 0}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        n_hunks += 1
        if tag == "replace":
            # Same-token-count substitution at the editor: one REPL marker
            # per token.
            counts["REPL"] += 1
            editor_ops += (i2 - i1)
        elif tag == "delete":
            # Tokens present in `a` but not in `b` → editor must INSERT
            # them back (SAE-LEWIS naming). One INS marker, length-K from
            # the length head.
            counts["INS"] += 1
            editor_ops += 1
        elif tag == "insert":
            # Tokens present in `b` but not in `a` → editor must DELETE
            # them. One [DEL] marker per inserted token.
            counts["DEL"] += 1
            editor_ops += (j2 - j1)
    return {
        "n_hunks": n_hunks,
        "editor_ops": editor_ops,
        "op_types": counts,
    }


def percentiles(values: List[float], qs: List[float]) -> List[float]:
    if not values:
        return [float("nan")] * len(qs)
    s = sorted(values)
    out: List[float] = []
    for q in qs:
        if q <= 0:
            out.append(float(s[0]))
            continue
        if q >= 1:
            out.append(float(s[-1]))
            continue
        pos = q * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        out.append(s[lo] * (1 - frac) + s[hi] * frac)
    return out


def histogram(values: List[int], buckets: List[Tuple[int, int, str]]) -> Counter:
    out: Counter = Counter()
    for v in values:
        placed = False
        for lo, hi, label in buckets:
            if lo <= v <= hi:
                out[label] += 1
                placed = True
                break
        if not placed:
            out["other"] += 1
    return out


def _per_value_hist(values: List[int], max_exact: int = 20) -> Tuple[Dict[str, int], List[str]]:
    """Histogram with one bucket per integer up to `max_exact`, plus tail
    buckets. Returns (counts, ordered_label_list)."""
    buckets: List[Tuple[int, int, str]] = (
        [(v, v, str(v)) for v in range(0, max_exact + 1)]
        + [(max_exact + 1, 50, f"{max_exact + 1}-50"), (51, 10_000, "51+")]
    )
    counts = histogram(values, buckets)
    labels = [lab for _, _, lab in buckets] + ["other"]
    return dict(counts), labels


def summarise(records: List[Dict]) -> Dict:
    if not records:
        return {"n": 0}
    len1 = [r["len1"] for r in records]
    len2 = [r["len2"] for r in records]
    dlen = [abs(r["len1"] - r["len2"]) for r in records]
    edit = [r["tok_edit"] for r in records]
    setd = [r["set_diff"] for r in records]
    nhk  = [r["n_hunks"]   for r in records]
    eops = [r["editor_ops"] for r in records]
    qs = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    p_len1 = percentiles(len1, qs)
    p_len2 = percentiles(len2, qs)
    p_dlen = percentiles(dlen, qs)
    p_edit = percentiles(edit, qs)
    p_setd = percentiles(setd, qs)
    p_nhk  = percentiles(nhk,  qs)
    p_eops = percentiles(eops, qs)
    edit_hist, edit_labels = _per_value_hist(edit, max_exact=20)
    set_diff_hist, set_diff_labels = _per_value_hist(setd, max_exact=20)
    nhk_hist, nhk_labels = _per_value_hist(nhk, max_exact=10)
    eops_hist, eops_labels = _per_value_hist(eops, max_exact=20)
    # Per-op-type totals (sums across the corpus). Lets us compute the
    # observed REPL : INS : DEL ratio and compare to the corruption
    # pipeline's (0.55, 0.25, 0.20) default.
    op_totals = {"REPL": 0, "INS": 0, "DEL": 0}
    for r in records:
        for t, c in r["op_types"].items():
            op_totals[t] += c
    op_total_sum = sum(op_totals.values())
    op_ratio = {
        t: (op_totals[t] / op_total_sum) if op_total_sum else 0.0
        for t in op_totals
    }
    return {
        "n": len(records),
        "len1_mean": sum(len1) / len(len1),
        "len2_mean": sum(len2) / len(len2),
        "tok_edit_mean":   sum(edit) / len(edit),
        "n_hunks_mean":    sum(nhk)  / len(nhk),
        "editor_ops_mean": sum(eops) / len(eops),
        "set_diff_mean":   sum(setd) / len(setd),
        "len1_pcts": p_len1,
        "len2_pcts": p_len2,
        "delta_len_pcts": p_dlen,
        "tok_edit_pcts":   p_edit,
        "n_hunks_pcts":    p_nhk,
        "editor_ops_pcts": p_eops,
        "set_diff_pcts": p_setd,
        # Per-integer histograms (0..max_exact exact, then tail).
        "tok_edit_hist": edit_hist,
        "tok_edit_hist_labels": edit_labels,
        "n_hunks_hist": nhk_hist,
        "n_hunks_hist_labels": nhk_labels,
        "editor_ops_hist": eops_hist,
        "editor_ops_hist_labels": eops_labels,
        "set_diff_hist": set_diff_hist,
        "set_diff_hist_labels": set_diff_labels,
        # Op-type histogram + ratio
        "op_totals": op_totals,
        "op_ratio":  op_ratio,
    }


def render_md(
    args, by_lang: Dict[str, Dict], overall: Dict,
    feature_summary: List[Tuple[str, Dict]],
) -> str:
    qs_labels = ["p05", "p25", "p50", "p75", "p90", "p95", "p99"]
    out: List[str] = []
    out.append("# LinguaLens minimal-pair token-diff distribution\n")
    out.append(
        f"Dataset: `THU-KEG/LinguaLens-Data` (split=train, total 7251 pairs).  "
        f"Tokenizer: `{args.tokenizer}`.  Sample size: {args.sample_size} "
        f"(seed={args.seed}).\n"
    )
    out.append(
        "Three token-level edit metrics (all `tok_edit ≥ editor_ops ≥ "
        "n_hunks`):\n"
        "* **`tok_edit`** — classical Levenshtein, insert/delete/substitute "
        "cost 1. The minimum number of single-token edits. Over-counts "
        "INS (a contiguous block insertion of K tokens scores K, not 1).\n"
        "* **`editor_ops`** — editor inference cost: REPL/DEL hunks "
        "contribute one marker per token, INS hunks contribute one marker "
        "per gap (length head emits the rest). What the editor actually "
        "has to emit at inference time.\n"
        "* **`n_hunks`** — number of contiguous non-equal diff blocks. "
        "Corresponds 1-to-1 to SAE-LEWIS's `N_total` (one hunk = one op). "
        "Use this to compare against the corruption pipeline's bucket "
        "weights.\n"
        "* `set_diff` = `|set(tok1) △ set(tok2)|` — order-insensitive, "
        "dedup-counted token set symmetric difference.\n"
    )

    # Overall
    out.append("\n## Overall\n")
    out.append(_render_summary_table(overall, qs_labels))
    out.append(_render_op_ratio_table("Op-type histogram (overall)", overall))
    out.append(_render_hist_table(
        "tok_edit histogram (overall)",
        overall["tok_edit_hist"], overall["tok_edit_hist_labels"], overall["n"],
    ))
    out.append(_render_hist_table(
        "editor_ops histogram (overall)",
        overall["editor_ops_hist"], overall["editor_ops_hist_labels"], overall["n"],
    ))
    out.append(_render_hist_table(
        "n_hunks histogram (overall) -- compare to SAE-LEWIS bucket weights",
        overall["n_hunks_hist"], overall["n_hunks_hist_labels"], overall["n"],
    ))
    out.append(_render_hist_table(
        "set_diff histogram (overall)",
        overall["set_diff_hist"], overall["set_diff_hist_labels"], overall["n"],
    ))

    # Per language
    out.append("\n## Per language\n")
    for lang, summ in sorted(by_lang.items()):
        out.append(f"\n### {lang}  (n = {summ['n']})\n")
        out.append(_render_summary_table(summ, qs_labels))
        out.append(_render_op_ratio_table(f"{lang}: Op-type histogram", summ))
        out.append(_render_hist_table(
            f"{lang}: tok_edit histogram",
            summ["tok_edit_hist"], summ["tok_edit_hist_labels"], summ["n"],
        ))
        out.append(_render_hist_table(
            f"{lang}: editor_ops histogram",
            summ["editor_ops_hist"], summ["editor_ops_hist_labels"], summ["n"],
        ))
        out.append(_render_hist_table(
            f"{lang}: n_hunks histogram",
            summ["n_hunks_hist"], summ["n_hunks_hist_labels"], summ["n"],
        ))

    # Top features (by n_hunks median, since that's the SAE-LEWIS-relevant
    # signal). Falls back to tok_edit for ties.
    def _feature_sort_key(kv):
        s = kv[1]
        return (s["n_hunks_pcts"][2], s["tok_edit_pcts"][2])

    out.append("\n## Top features by n_hunks p50 (largest first)\n")
    out.append(
        "| feature | n | n_hunks p50 | editor_ops p50 | tok_edit p50 | "
        "REPL% | INS% | DEL% | mean len1 | mean len2 |\n"
    )
    out.append(
        "|---------|---|-------------|----------------|--------------|"
        "-------|------|------|-----------|-----------|\n"
    )
    rows = sorted(feature_summary, key=_feature_sort_key, reverse=True)
    for name, s in rows[:25]:
        out.append(_render_feature_row(name, s))
    out.append("\n## Bottom features by n_hunks p50 (smallest first)\n")
    out.append(
        "| feature | n | n_hunks p50 | editor_ops p50 | tok_edit p50 | "
        "REPL% | INS% | DEL% | mean len1 | mean len2 |\n"
    )
    out.append(
        "|---------|---|-------------|----------------|--------------|"
        "-------|------|------|-----------|-----------|\n"
    )
    for name, s in rows[-25:][::-1]:
        out.append(_render_feature_row(name, s))

    out.append(
        "\n## Notes\n\n"
        "* Gemma is a subword (SentencePiece) tokenizer.  For Chinese the "
        "Gemma tokenizer falls through to roughly character-level "
        "segmentation, which inflates token counts and edit distance "
        "relative to a Chinese-native tokenizer.  Treat the English and "
        "Chinese rows separately when comparing to the SAE-LEWIS "
        "compound-corruption N distribution (which is currently calibrated "
        "on English Dolma only).\n"
        "* The edit distance is computed over the FULL tokenization "
        "including BOS but excluding nothing else — i.e. exactly the "
        "number of token-level operations the SAE-LEWIS editor would have "
        "to emit if `sentence1` were corrupted into `sentence2`.\n"
        "* Re-run with `--sample-size 7251` to use the full dataset, or "
        "`--language English` / `--language Chinese` to slice.\n"
    )
    return "".join(out)


def _render_hist_table(
    title: str, hist: Dict[str, int], labels: List[str], n_total: int,
) -> str:
    """Render a one-per-value histogram with count, %, and cumulative %."""
    out: List[str] = []
    out.append(f"\n**{title}:**\n\n")
    out.append("| value | count | % | cum % |\n")
    out.append("|-------|-------|---|-------|\n")
    cum = 0
    for label in labels:
        c = hist.get(label, 0)
        if c == 0 and label in ("21-50", "51+", "other"):
            # Skip empty tail rows to keep the table readable
            continue
        cum += c
        pct = (100.0 * c / n_total) if n_total else 0.0
        cum_pct = (100.0 * cum / n_total) if n_total else 0.0
        out.append(f"| {label} | {c} | {pct:.1f}% | {cum_pct:.1f}% |\n")
    return "".join(out)


def _render_summary_table(summ: Dict, qs_labels: List[str]) -> str:
    out: List[str] = []
    out.append("\n| metric | mean | " + " | ".join(qs_labels) + " |\n")
    out.append("|--------|------|" + "|".join("-----" for _ in qs_labels) + "|\n")
    out.append(
        f"| `len(tok1)`     | {summ['len1_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["len1_pcts"])
        + " |\n"
    )
    out.append(
        f"| `len(tok2)`     | {summ['len2_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["len2_pcts"])
        + " |\n"
    )
    out.append(
        f"| `|Δlen|`        | -    | "
        + " | ".join(f"{v:.1f}" for v in summ["delta_len_pcts"])
        + " |\n"
    )
    out.append(
        f"| `tok_edit`      | {summ['tok_edit_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["tok_edit_pcts"])
        + " |\n"
    )
    out.append(
        f"| `editor_ops`    | {summ['editor_ops_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["editor_ops_pcts"])
        + " |\n"
    )
    out.append(
        f"| `n_hunks`       | {summ['n_hunks_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["n_hunks_pcts"])
        + " |\n"
    )
    out.append(
        f"| `set_diff`      | {summ['set_diff_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["set_diff_pcts"])
        + " |\n"
    )
    return "".join(out)


def _render_op_ratio_table(title: str, summ: Dict) -> str:
    out: List[str] = []
    out.append(f"\n**{title}** (hunk counts across all pairs in the slice; "
               "compare ratio against the corruption pipeline's default "
               "`(REPL=0.55, INS=0.25, DEL=0.20)`):\n\n")
    out.append("| op | total hunks | ratio |\n")
    out.append("|----|-------------|-------|\n")
    for t in ("REPL", "INS", "DEL"):
        c = summ["op_totals"][t]
        r = summ["op_ratio"][t]
        out.append(f"| {t} | {c} | {r:.3f} |\n")
    return "".join(out)


def _render_feature_row(name: str, s: Dict) -> str:
    return (
        f"| {name} | {s['n']} | "
        f"{s['n_hunks_pcts'][2]:.1f} | {s['editor_ops_pcts'][2]:.1f} | "
        f"{s['tok_edit_pcts'][2]:.1f} | "
        f"{100*s['op_ratio']['REPL']:.0f}% | "
        f"{100*s['op_ratio']['INS']:.0f}% | "
        f"{100*s['op_ratio']['DEL']:.0f}% | "
        f"{s['len1_mean']:.1f} | {s['len2_mean']:.1f} |\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="google/gemma-2-2b",
                   help="HF id of the tokenizer to use (matches the LLM "
                        "in the SAE-LEWIS pipeline).")
    p.add_argument("--sample-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--language", default="English",
                   help="Filter to a single language (English | Chinese | "
                        "'all'). Default is English — matches the SAE-LEWIS "
                        "pipeline's current scope. Pass 'all' to keep both "
                        "languages.")
    p.add_argument("--out-md", default="runs/lingualens_token_diff.md")
    p.add_argument("--out-jsonl", default=None,
                   help="Optional path to write per-pair raw records as JSONL.")
    args = p.parse_args()

    print(f"[lingualens] loading dataset")
    ds = load_dataset("THU-KEG/LinguaLens-Data", split="train")
    print(f"[lingualens] total examples: {len(ds)}")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
        print(f"[lingualens] after language={args.language} filter: {len(ds)}")

    rng = random.Random(args.seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    sample_n = min(args.sample_size, len(idx))
    chosen = idx[:sample_n]
    print(f"[lingualens] sampling {sample_n} pairs (seed={args.seed})")

    print(f"[lingualens] loading tokenizer {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    records: List[Dict] = []
    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    by_feat: Dict[str, List[Dict]] = defaultdict(list)
    for i, k in enumerate(chosen):
        ex = ds[int(k)]
        ids1 = tok(ex["sentence1"], add_special_tokens=True)["input_ids"]
        ids2 = tok(ex["sentence2"], add_special_tokens=True)["input_ids"]
        ed = token_edit_distance(ids1, ids2)
        sd = len(set(ids1) ^ set(ids2))
        dm = diff_metrics(ids1, ids2)
        rec = {
            "language":    ex["language"],
            "feature":     ex["feature"],
            "len1":        len(ids1),
            "len2":        len(ids2),
            "tok_edit":    ed,
            "n_hunks":     dm["n_hunks"],
            "editor_ops":  dm["editor_ops"],
            "op_types":    dm["op_types"],
            "set_diff":    sd,
        }
        records.append(rec)
        by_lang[ex["language"]].append(rec)
        by_feat[ex["feature"]].append(rec)
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1} / {sample_n}")

    overall = summarise(records)
    by_lang_summ = {lang: summarise(rs) for lang, rs in by_lang.items()}
    feature_summary = [
        (name, summarise(rs)) for name, rs in by_feat.items() if len(rs) >= 5
    ]

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(args, by_lang_summ, overall, feature_summary))
    print(f"[lingualens] wrote {out_md}")

    if args.out_jsonl:
        out_jl = Path(args.out_jsonl)
        out_jl.parent.mkdir(parents=True, exist_ok=True)
        with out_jl.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[lingualens] wrote {out_jl}")

    # Brief console summary
    print()
    print(f"Overall (n={overall['n']}):")
    for label, key in (
        ("tok_edit  ", "tok_edit_pcts"),
        ("editor_ops", "editor_ops_pcts"),
        ("n_hunks   ", "n_hunks_pcts"),
        ("set_diff  ", "set_diff_pcts"),
    ):
        p = overall[key]
        print(f"  {label} p50={p[2]:.1f}  p75={p[3]:.1f}  "
              f"p90={p[4]:.1f}  p95={p[5]:.1f}")
    print(f"  len1       p50={overall['len1_pcts'][2]:.1f}  "
          f"len2 p50={overall['len2_pcts'][2]:.1f}")
    print(f"  op-type totals: REPL={overall['op_totals']['REPL']} "
          f"({100*overall['op_ratio']['REPL']:.1f}%)  "
          f"INS={overall['op_totals']['INS']} "
          f"({100*overall['op_ratio']['INS']:.1f}%)  "
          f"DEL={overall['op_totals']['DEL']} "
          f"({100*overall['op_ratio']['DEL']:.1f}%)")

    def _print_hist(title: str, hist_key: str, labels_key: str):
        print(f"  {title}:")
        n_overall = overall["n"]
        cum = 0
        for label in overall[labels_key]:
            c = overall[hist_key].get(label, 0)
            if c == 0 and label in ("21-50", "51+", "other"):
                continue
            cum += c
            print(f"    {label:>5} : {c:>5}  ({100.0*c/n_overall:5.1f}%, "
                  f"cum {100.0*cum/n_overall:5.1f}%)")

    _print_hist("tok_edit hist (overall)",
                "tok_edit_hist", "tok_edit_hist_labels")
    _print_hist("editor_ops hist (overall)",
                "editor_ops_hist", "editor_ops_hist_labels")
    _print_hist("n_hunks hist (overall)  -- compare vs SAE-LEWIS buckets",
                "n_hunks_hist", "n_hunks_hist_labels")


if __name__ == "__main__":
    main()
