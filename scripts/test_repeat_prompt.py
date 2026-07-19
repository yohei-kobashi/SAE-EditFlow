"""v5 frame prep (user decision 2026-07-19): find a repeat-instruction
prompt on which PLAIN gemma-2-2b-it (no SAE, no editor) reproduces the
input sentence ~100% of the time. The chosen prompt becomes the exact-
metric frame; the intervention then decides WHAT gets reproduced.

Tests several English wordings x {chat-template, bare-text} on LinguaLens
src sentences, greedy decode, and reports the verbatim-copy rate.

    python scripts/test_repeat_prompt.py --sample-size 100 --device cuda
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

CANDIDATES = {
    "chat_A": ("chat", "Repeat the input sentence exactly. Never output "
                       "anything else.\n\nInput: {src}"),
    "chat_B": ("chat", "Repeat the following sentence exactly, and output "
                       "nothing else.\n\n{src}"),
    "chat_C": ("chat", "Output the following sentence verbatim, with no "
                       "additions, explanations, or changes.\n\n{src}"),
    "bare_D": ("bare", "Repeat the input sentence exactly. Never output "
                       "anything else.\nInput: {src}\nOutput: "),
    "bare_E": ("bare", "Input: {src}\nRepeat the input sentence exactly, "
                       "and output nothing else.\nOutput: "),
    # src-first variants (diag7 follow-up: instruction AFTER src keeps
    # src-position states near the bare identification context)
    "chat_F": ("chat", "Input: {src}\n\nRepeat the input sentence "
                       "exactly. Never output anything else."),
    "chat_G": ("chat", "{src}\n\nRepeat the sentence above exactly. "
                       "Never output anything else."),
}


def norm(s: str) -> str:
    s = s.strip().strip('"').strip("'").strip()
    return " ".join(s.split()).casefold()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    srcs = [ds[int(k)]["sentence1"] for k in
            order[:args.sample_size]]

    tok = AutoTokenizer.from_pretrained(args.it_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    @torch.no_grad()
    def gen(prompt_ids):
        ids = torch.tensor([prompt_ids], device=args.device)
        out = model.generate(
            input_ids=ids, max_new_tokens=len(prompt_ids) + 32,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    print(f"[repeat-test] {len(srcs)} sentences, {len(CANDIDATES)} prompts")
    results = {}
    for name, (kind, tpl) in CANDIDATES.items():
        hits, fails = 0, []
        for src in srcs:
            text = tpl.format(src=src)
            if kind == "chat":
                pids = tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=True)
                # tokenizers version drift (same guards as the fixed
                # AxBench chat_wrap): BatchEncoding / Encoding / nested
                # list -> flat id list
                if hasattr(pids, "input_ids"):
                    pids = pids.input_ids
                if hasattr(pids, "ids"):
                    pids = pids.ids
                if pids and isinstance(pids[0], list):
                    pids = pids[0]
                pids = [int(x) for x in pids]
            else:
                pids = tok(text, add_special_tokens=True).input_ids
            out = gen(pids).split("\n")[0]
            if norm(out) == norm(src):
                hits += 1
            elif len(fails) < 3:
                fails.append((src, out))
        rate = hits / len(srcs)
        results[name] = rate
        print(f"[repeat-test] {name}: copy {rate:.2%}")
        for s, o in fails:
            print(f"    FAIL src={s[:60]!r}")
            print(f"         out={o[:60]!r}")
    best = max(results, key=results.get)
    print(f"\n[repeat-test] BEST: {best} ({results[best]:.2%})")
    print(f"template: {CANDIDATES[best][1]!r} ({CANDIDATES[best][0]})")


if __name__ == "__main__":
    main()
