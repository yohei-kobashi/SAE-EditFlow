"""
B2 baseline (PAPER_OUTLINE §5): instruction-prompted rewrite at the
MATCHED backbone — does SAE-native conditioning beat the same
information rendered as text?

The baseline receives exactly the information the EF model gets — the
commanded feature deltas (z_amp, z_sup), derived by the probe's own
conditioning path (local scope, blocklist, diff_intervention k64/k64) on
the same 500-pair sample (same seed shuffle) — rendered as natural
language via Neuronpedia auto-interp labels, and hands it to
google/gemma-2-2b-it (the instruction-tuned sibling of the frozen
backbone the EF model is built on). Greedy decode, minimal-edit
instruction, same exact/sim/copy metrics, same true/empty/random
controls. Records are written in the probe's records.jsonl format, so
compare_ef_pipeline.py joins them against the pipeline (or any EF run)
unchanged.

Fairness notes recorded for the paper: (a) the baseline sees the top
--n-desc features per side (16k-dim vectors don't fit a prompt) — we
sweep n_desc on the true condition; (b) auto-interp label quality is a
shared confounder, cutting both ways.

Usage (miyabi):
    python scripts/eval_prompt_baseline.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --explanations runs/np_explanations/gemma-2-2b_12-res-16k.json \
        --blocklist runs/blocklist/blocklist.npy \
        --output-dir runs/prod_gemma_v6/prompt_baseline500 \
        --sample-size 500 --device cuda
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

from editflow_ops import align_pair, slot_ops                  # noqa: E402
from eval_lingualens import (                                  # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk, pair_metrics,
    randomize_intervention, sae_z_with_offsets,
)
from model import SAEFeatureExtractor                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True,
                   help="tokenizer for the n_ops alignment (bucket "
                        "comparability with the EF probes)")
    p.add_argument("--explanations", required=True,
                   help="{feature_index: description} JSON "
                        "(scripts/fetch_sae_explanations.py)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")

    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"],
                   default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--blocklist", default="")
    p.add_argument("--conditions", default="true,empty,random")
    p.add_argument("--n-desc-list", default="8,16",
                   help="how many top features per side to describe; "
                        "all values run on `true`, the FIRST on "
                        "empty/random")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


PROMPT = """You are a precise text editor. Rewrite the sentence below by \
making the SMALLEST possible edit that accomplishes all of the requested \
changes. Keep every other word exactly unchanged. If no changes are \
requested, output the sentence unchanged. Output ONLY the rewritten \
sentence, nothing else.

Requested changes:
{changes}

Sentence: {src}"""


def render_changes(za, zs, expl, n_desc: int) -> str:
    lines = []
    for z, verb in ((za, "MORE"), (zs, "LESS")):
        nz = torch.nonzero(z > 0).flatten().tolist()
        nz.sort(key=lambda i: -float(z[i]))
        for fid in nz[:n_desc]:
            desc = expl.get(str(int(fid)),
                            f"SAE feature #{int(fid)}").strip()
            lines.append(f"- express {verb} of: {desc}")
    if not lines:
        return "(none — output the sentence unchanged)"
    return "\n".join(lines)


def extract_sentence(text: str, src: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip().strip('"').strip()
        if not line:
            continue
        for pre in ("Rewritten sentence:", "Sentence:", "Output:"):
            if line.lower().startswith(pre.lower()):
                line = line[len(pre):].strip().strip('"').strip()
        if line:
            return line
    return src                                # empty generation → copy


def bname(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4-8" if n <= 8 else "9+"


def main():
    args = parse_args()
    conditions = [c for c in args.conditions.split(",") if c]
    n_desc_list = [int(x) for x in args.n_desc_list.split(",") if x]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    expl = json.loads(Path(args.explanations).read_text())
    print(f"[b2] {len(expl)} feature explanations")

    import random
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen = order[:min(args.sample_size, len(order))]
    print(f"[b2] {len(ds)} pairs, sampling {len(chosen)}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    blk = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        blk = torch.as_tensor(np.asarray(_bl, dtype=np.int64))
        print(f"[b2] blocklist: {len(_bl)} features masked")

    it_tok = AutoTokenizer.from_pretrained(args.it_model)
    it_model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype).to(args.device).eval()
    print(f"[b2] rewriter: {args.it_model}")

    @torch.no_grad()
    def rewrite(changes: str, src: str) -> str:
        prompt = PROMPT.format(changes=changes, src=src)
        # tokenize=False + plain tokenizer call: version-proof (newer
        # transformers returns a BatchEncoding from apply_chat_template,
        # which generate() rejects as a positional input). The rendered
        # template already contains <bos>, so add_special_tokens=False.
        text_in = it_tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
        enc = it_tok(text_in, return_tensors="pt",
                     add_special_tokens=False).to(args.device)
        gen = it_model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=it_tok.pad_token_id or it_tok.eos_token_id)
        text = it_tok.decode(gen[0, enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
        return extract_sentence(text, src)

    partial_path = out_dir / "records.partial.jsonl"
    records, done_idx = [], set()
    if partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(r)
                done_idx.add(int(r["idx"]))
        print(f"[b2] RESUME: {len(records)} pairs")
    pf = open(partial_path, "a")

    for step_i, k in enumerate(chosen):
        if int(k) in done_idx:
            continue
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids
        slots = align_pair(src_ids, tgt_ids)
        n_ops = len(slot_ops(slots))
        if n_ops == 0:
            continue                          # probe skips these too
        prng = np.random.default_rng(args.seed * 1000003 + int(k))

        # conditioning — identical to the EF probe's local path
        with torch.no_grad():
            s_off, z_s = sae_z_with_offsets(extractor, src, args.device)
            t_off, z_t = sae_z_with_offsets(extractor, tgt, args.device)
            om_s = [tuple(o) for o in tokenizer(
                src, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            om_t = [tuple(o) for o in tokenizer(
                tgt, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            opcodes = difflib.SequenceMatcher(
                None, src_ids, tgt_ids, autojunk=False).get_opcodes()
            sr, tr = edit_char_ranges(opcodes, om_s, om_t)
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
        za_t, zs_t = diff_intervention(z_src, z_tgt, args.k_amp,
                                       args.k_sup)
        zvar = {"true": (za_t, zs_t),
                "empty": (torch.zeros_like(za_t),
                          torch.zeros_like(zs_t)),
                "random": (randomize_intervention(za_t, prng),
                           randomize_intervention(zs_t, prng))}

        rec = {"idx": int(k), "src": src, "tgt": tgt, "n_ops": n_ops,
               "outputs": {}}
        for c in conditions:
            za, zs = zvar[c]
            rec["outputs"][c] = {}
            descs = n_desc_list if c == "true" else n_desc_list[:1]
            for nd in descs:
                out_text = rewrite(render_changes(za, zs, expl, nd), src)
                pm = pair_metrics(out_text, src, tgt)
                rec["outputs"][c][f"prompt{nd}"] = {
                    "text": out_text, "exact": pm["exact_match"],
                    "sim_target": pm["sim_target"],
                    "copy": pm["copy_rate"],
                    "no_edit": pm["copy_rate"]}
        records.append(rec)
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        if (step_i + 1) % 10 == 0:
            print(f"[b2] {step_i + 1}/{len(chosen)} pairs "
                  f"({len(records)} scored)")
    pf.close()

    # ---- report (rebuilt from records — resume-safe) --------------------
    lines = ["# B2 prompt-rewrite baseline (LinguaLens)", ""]
    lines.append(f"pairs scored: {len(records)}; rewriter {args.it_model}; "
                 f"n_desc {n_desc_list}; conditioning identical to the EF "
                 f"probe (local, blocklist, k={args.k_amp}/{args.k_sup})")
    lines += ["", "| condition | mode | exact | sim_target | copy |",
              "|---|---|---|---|---|"]
    for c in conditions:
        modes = sorted({m for r in records
                        for m in r["outputs"].get(c, {})})
        for m in modes:
            rows = [r["outputs"][c][m] for r in records
                    if m in r["outputs"].get(c, {})]
            if not rows:
                continue
            lines.append(
                f"| {c} | {m} | "
                f"{np.mean([r['exact'] for r in rows]):.4f} | "
                f"{np.mean([r['sim_target'] for r in rows]):.4f} | "
                f"{np.mean([r['copy'] for r in rows]):.4f} |")
    lines += ["", "## Multi-site breakdown (condition = true)", ""]
    modes = sorted({m for r in records for m in r["outputs"]["true"]})
    lines.append("| n_ops | pairs | " +
                 " | ".join(f"{m} exact" for m in modes) + " | " +
                 " | ".join(f"{m} sim" for m in modes) + " |")
    lines.append("|---" * (2 + 2 * len(modes)) + "|")
    byb = defaultdict(list)
    for r in records:
        byb[bname(r["n_ops"])].append(r)
    for b in ("1", "2-3", "4-8", "9+"):
        rs = byb.get(b, [])
        if not rs:
            continue
        cells = [f"{np.mean([r['outputs']['true'][m]['exact'] for r in rs]):.4f}"
                 for m in modes]
        cells += [f"{np.mean([r['outputs']['true'][m]['sim_target'] for r in rs]):.4f}"
                  for m in modes]
        lines.append(f"| {b} | {len(rs)} | " + " | ".join(cells) + " |")

    report = "\n".join(lines)
    print(report)
    (out_dir / "report.md").write_text(report + "\n")
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)
    print(f"[b2] wrote {out_dir}/report.md, records.jsonl")


if __name__ == "__main__":
    main()
