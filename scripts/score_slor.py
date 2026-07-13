"""
Grammaticality quantification: SLOR over the outputs stored in any
records.jsonl (probe / B1 / B2 / B3 / pipeline formats).

SLOR(s) = (log p_M(s) − log p_unigram(s)) / n_pred — the corruption
gate's own definition (Pauls & Klein 2012 / Lau+ 2017 / Kann+ 2018),
frozen causal Gemma-2-2B + the cache's unigram table. Reported per
mode: mean SLOR of outputs, mean ΔSLOR vs the source sentence
(negative = the system degrades grammaticality), and the same for the
edited (non-copy) subset only — copies have ΔSLOR 0 by construction and
would mask the damage.

Usage (miyabi):
    python scripts/score_slor.py \
        --records runs/prod_gemma_v6/clamp_baseline500/records.jsonl \
        --modes clamp10,clampZ --condition true \
        --unigram runs/prod_gemma_v4/corruption/unigram.json \
        --out runs/slor/b1.json
    # pipeline records: --modes "" (text sits under the condition)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--modes", default="",
                   help="comma list of decode-mode keys; empty = "
                        "pipeline-format records")
    p.add_argument("--condition", default="true")
    p.add_argument("--unigram", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    tok = AutoTokenizer.from_pretrained(args.llm)
    lm = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=dtype).to(args.device).eval()
    raw = json.loads(Path(args.unigram).read_text())
    unig = {int(k): float(v) for k, v in raw["table"].items()}
    unk = float(raw["unk_log_prob"])
    print(f"[slor] unigram table: {len(unig)} tokens")

    cache: dict = {}

    @torch.no_grad()
    def slor(text: str):
        if text in cache:
            return cache[text]
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=256).to(args.device)
        T = int(enc.input_ids.shape[1])
        if T < 2:
            cache[text] = None
            return None
        out = lm(input_ids=enc.input_ids, labels=enc.input_ids,
                 use_cache=False)
        n_pred = T - 1
        sum_m = -float(out.loss.item()) * n_pred
        sum_u = sum(unig.get(int(t), unk)
                    for t in enc.input_ids[0, 1:].tolist())
        val = (sum_m - sum_u) / n_pred
        cache[text] = val
        return val

    with open(args.records) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    modes = [m for m in args.modes.split(",")] if args.modes else [""]

    def norm(s):
        return " ".join(s.split())

    result = {}
    for mode in modes:
        rows = []
        for r in recs:
            o = r["outputs"].get(args.condition)
            if o is None:
                continue
            node = o if not mode else o.get(mode)
            if not isinstance(node, dict) or "text" not in node:
                continue
            src = r.get("src") or r.get("source")
            s_out = slor(node["text"])
            s_src = slor(src)
            if s_out is None or s_src is None:
                continue
            rows.append({"idx": r["idx"], "slor": s_out,
                         "delta": s_out - s_src,
                         "copy": float(norm(node["text"]) == norm(src))})
        if not rows:
            continue
        edited = [r for r in rows if not r["copy"]]
        result[mode or "(pipeline)"] = {
            "n": len(rows),
            "mean_slor": sum(r["slor"] for r in rows) / len(rows),
            "mean_delta": sum(r["delta"] for r in rows) / len(rows),
            "n_edited": len(edited),
            "mean_delta_edited": (sum(r["delta"] for r in edited)
                                  / len(edited)) if edited else 0.0,
        }
        m = result[mode or "(pipeline)"]
        print(f"[slor] {mode or '(pipeline)':12s} n={m['n']} "
              f"SLOR={m['mean_slor']:.4f} dSLOR={m['mean_delta']:+.4f} "
              f"dSLOR(edited n={m['n_edited']})="
              f"{m['mean_delta_edited']:+.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"records": args.records, "condition": args.condition,
         "modes": result}, indent=1))
    print(f"[slor] wrote {out_path}")


if __name__ == "__main__":
    main()
