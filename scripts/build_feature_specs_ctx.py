"""Improvement ⑦ (user-approved 2026-07-22): IN-CONTEXT feature specs.

The bare-sentence specs are measured at a different operating point from
the intervention (diag 7: only ~0.59 of the spec survives the bare->prompt
context shift). Here every pool pair's sentences are embedded in the SAME
repeat-prompt chat context the frozen gemma-2-2b-it sees at intervention
time; per-layer deltas are pooled over the sentence's EDIT-SPAN tokens at
their positions inside the prompt. Aggregation (per-feature signed mean,
top-store, norm-median, split-half cosine) mirrors build_feature_specs.py;
output l{L}_specctx.json drops into eval_ef_bare --feature-spec unchanged.

Usage (GPU):
    python scripts/build_feature_specs_ctx.py \
        --out-dir runs/feature_specs --split runs/tables/eval_split.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

from eval_lingualens import edit_char_ranges                   # noqa: E402
from intervener import (REPEAT_PROMPT, chat_prompt_ids,        # noqa: E402
                        find_subseq)
from model import load_sae                                     # noqa: E402

LAYER_CFG = {
    4:  ("layer_4/width_16k/average_l0_60/params.npz",
         "runs/blocklist_l4/blocklist.npy"),
    12: ("layer_12/width_16k/average_l0_82/params.npz",
         "runs/blocklist/blocklist.npy"),
    20: ("layer_20/width_16k/average_l0_71/params.npz",
         "runs/blocklist_l20/blocklist.npy"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/feature_specs")
    p.add_argument("--split", default="runs/tables/eval_split.json")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--layers", default="4,12,20")
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--top-store", type=int, default=128)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    eval_idx = set(split["eval_idx"])

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    pool = [k for k in range(len(ds)) if k not in eval_idx]
    print(f"[ctx-specs] pool {len(pool)} pairs, layers {layers}, "
          f"context = repeat prompt on {args.it_model}")

    tok = AutoTokenizer.from_pretrained(args.it_model)
    llm = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=torch.bfloat16).to(args.device).eval()
    saes, blks = {}, {}
    for L in layers:
        sae_path, blk_path = LAYER_CFG[L]
        saes[L] = load_sae("jumprelu", args.sae_repo, sae_path
                           ).to(args.device).eval()
        blks[L] = torch.as_tensor(
            np.asarray(np.load(blk_path), dtype=np.int64))

    side, done = {}, {}
    for L in layers:
        sp = out_dir / f"l{L}_pairs_ctx.jsonl"
        done[L] = set()
        if sp.exists():
            with open(sp) as f:
                for line in f:
                    if line.strip():
                        done[L].add(int(json.loads(line)["idx"]))
            print(f"[ctx-specs] L{L} RESUME: {len(done[L])} pairs")
        side[L] = open(sp, "a")

    @torch.no_grad()
    def ctx_span_z(text, span_tok_idx):
        """{layer: pooled top-k vector over the sentence's edit-span
        tokens at their positions INSIDE the repeat prompt}."""
        pids = chat_prompt_ids(tok, REPEAT_PROMPT.format(src=text))
        needle = tok(text, add_special_tokens=False).input_ids
        off = 0
        lo = find_subseq(pids, needle)
        if lo is None and len(needle) > 1:
            lo = find_subseq(pids, needle[1:])
            if lo is not None:
                off = 1
        if lo is None:
            return None
        pos = [lo + (i - off) for i in span_tok_idx if i - off >= 0]
        pos = [p_ for p_ in pos if lo <= p_ < lo + len(needle) - off + 1]
        out = llm(input_ids=torch.tensor([pids], device=args.device),
                  output_hidden_states=True, use_cache=False)
        res = {}
        for L in layers:
            h = out.hidden_states[L + 1][0]
            z = saes[L].encode(h.to(saes[L].W_enc.dtype))
            zp = (z[pos] if pos else z[lo:lo + len(needle)]
                  ).max(dim=0).values.float().cpu()
            zp[blks[L]] = 0.0
            keep = torch.zeros_like(zp)
            v, i = zp.topk(min(args.pool_topk, zp.numel()))
            m = v > 0
            keep[i[m]] = v[m]
            res[L] = keep
        return res

    def span_token_indices(text, ranges):
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        idx = [ti for ti, (ts, te) in enumerate(enc["offset_mapping"])
               if any(ts < ce and te > cs for cs, ce in ranges)]
        return idx

    todo = [k for k in pool if any(k not in done[L] for L in layers)]
    print(f"[ctx-specs] encoding {len(todo)} pairs")
    for n_done, k in enumerate(todo):
        ex = ds[int(k)]
        s, t = ex["sentence1"], ex["sentence2"]
        feat = ex.get("feature") or "?"
        s_ids = tok(s, add_special_tokens=True).input_ids
        t_ids = tok(t, add_special_tokens=True).input_ids
        opcodes = difflib.SequenceMatcher(
            None, s_ids, t_ids, autojunk=False).get_opcodes()
        om_s = [tuple(o) for o in tok(
            s, add_special_tokens=True,
            return_offsets_mapping=True)["offset_mapping"]]
        om_t = [tuple(o) for o in tok(
            t, add_special_tokens=True,
            return_offsets_mapping=True)["offset_mapping"]]
        sr, tr = edit_char_ranges(opcodes, om_s, om_t)
        zs = ctx_span_z(s, span_token_indices(s, sr if sr else
                                              [(0, len(s))]))
        zt = ctx_span_z(t, span_token_indices(t, tr if tr else
                                              [(0, len(t))]))
        if zs is None or zt is None:
            continue
        for L in layers:
            if k in done[L]:
                continue
            delta = zt[L] - zs[L]
            nz = torch.nonzero(delta).flatten()
            rec = {"idx": int(k), "feature": feat,
                   "delta": {int(i): round(float(delta[i]), 5)
                             for i in nz},
                   "dnorm": round(float(delta.norm()), 5)}
            side[L].write(json.dumps(rec) + "\n")
            side[L].flush()
            done[L].add(k)
        if (n_done + 1) % 200 == 0:
            print(f"[ctx-specs] {n_done + 1}/{len(todo)}")
    for L in layers:
        side[L].close()

    for L in layers:
        by_ph = defaultdict(list)
        with open(out_dir / f"l{L}_pairs_ctx.jsonl") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if int(r["idx"]) not in eval_idx:
                        by_ph[r["feature"]].append(r)
        d_sae = int(saes[L].W_enc.shape[1])
        spec_out = {}
        cs = []
        for ph, rows in sorted(by_ph.items()):
            rows = sorted(rows, key=lambda r: int(r["idx"]))
            n = len(rows)
            acc = torch.zeros(d_sae, dtype=torch.float64)
            accA = torch.zeros(d_sae, dtype=torch.float64)
            accB = torch.zeros(d_sae, dtype=torch.float64)
            dn = []
            for j, r in enumerate(rows):
                for i, v in r["delta"].items():
                    acc[int(i)] += v
                    (accA if j % 2 == 0 else accB)[int(i)] += v
                dn.append(r["dnorm"])
            mean = acc / n
            cos = float(torch.nn.functional.cosine_similarity(
                accA, accB, dim=0)) if accA.norm() > 0 and \
                accB.norm() > 0 else 0.0
            cs.append(cos)
            order = torch.argsort(mean.abs(), descending=True)
            keep = [int(i) for i in order[:args.top_store]
                    if abs(float(mean[i])) > 0]
            spec_out[ph] = {
                "n": n,
                "spec": {int(i): round(float(mean[i]), 5) for i in keep},
                "mean_norm": round(float(mean.norm()), 5),
                "norm_median": round(float(np.median(dn)), 5),
                "splithalf_cos": round(cos, 4),
            }
        (out_dir / f"l{L}_specctx.json").write_text(json.dumps(spec_out))
        print(f"[ctx-specs] L{L}: {len(spec_out)} features, "
              f"mean sh-cos {sum(cs) / len(cs):.4f}")
    print("==================== CTX-SPECS-DONE ====================")


if __name__ == "__main__":
    main()
