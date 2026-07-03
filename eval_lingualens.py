"""
End-to-end SAE-LEWIS evaluation on LinguaLens minimal pairs.

Randomly samples pairs (sentence1 → sentence2) from THU-KEG/LinguaLens-Data,
derives the intervention from the SAE pool-max difference between the two
sentences (the same semantics as training-time conditioning, §6.3.3:
`z_amp = top-k of max(z(s2) − z(s1), 0)` — "make the input more like the
target"), runs the full tagger → template-enumeration → ranker pipeline on
sentence1, and scores the output against sentence2.

This is the LEWIS-comparable end-to-end benchmark of §13.1: the reference
edit is a real human minimal pair, not a synthetic corruption, and the
conditioning is exactly what a user would supply if they knew which SAE
features should move.

Conditions (per pair, --conditions):
  true   : diff-derived (z_amp, z_sup)
  empty  : zeros — measures how much of the output is conditioning-driven
           end-to-end (the ranker's sae_align term also degrades to noise)
  random : same k / same magnitudes at random feature indices

Metrics per condition (report.md + records.jsonl under --output-dir):
  exact_match     output == sentence2 (casefold/strip)
  copy_rate       output == sentence1 (the editor did nothing)
  sim_target      word-level SequenceMatcher ratio vs sentence2
  sim_source      same vs sentence1 (content preservation)
  edit_loc_iou    IoU between gold-edited and predicted-edited source word
                  positions (did it edit the right place?)
  sae_shift       cos(z(out), z(s2)) − cos(z(s1), z(s2)) on pooled SAE
                  features (did the output move toward the target in SAE
                  space?)
  BLEU / chrF     corpus-level via sacrebleu when installed
  + an input-copy baseline row (LEWIS Table 2's "Input Copy").

Usage:
    python eval_lingualens.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --tagger-ckpt runs/prod_gemma/tagger/tagger-final.pt \
        --editor-ckpt runs/prod_gemma/editor/editor-final.pt \
        --output-dir runs/prod_gemma/eval_lingualens \
        --sample-size 100
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from editor import load_editor_from_checkpoint
from evaluate_intervention import TemplateBudgetExceeded, edit_once
from model import BidirectionalLLM, SAEFeatureExtractor, load_causal_gemma
from ranker import Ranker, RankerWeights
from tagger import load_tagger_from_checkpoint

CONDITIONS_ALL = ("true", "empty", "random")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--tagger-ckpt", required=True)
    p.add_argument("--editor-ckpt", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    # NOTE: pipeline default (l0_82), unlike evaluate_intervention.py's l0_71.
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"], default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k-amp", type=int, default=4,
                   help="Top positive diff features for z_amp (training used "
                        "k ∈ {1..4}; take the max).")
    p.add_argument("--k-sup", type=int, default=4)
    p.add_argument("--pool-topk", type=int, default=64,
                   help="Sentence pool-max top-K before diffing (= K_train).")
    p.add_argument("--conditions", default="true,empty",
                   help=f"Comma list from {CONDITIONS_ALL}.")

    p.add_argument("--l-max", type=int, default=3)
    p.add_argument("--ins-threshold", type=float, default=0.8)
    p.add_argument("--max-templates", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Diff-based intervention (training semantics, §6.3.3)
# ---------------------------------------------------------------------------
def diff_intervention(
    z_src: torch.Tensor, z_tgt: torch.Tensor, k_amp: int, k_sup: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = z_tgt - z_src
    z_amp = torch.zeros_like(delta)
    z_sup = torch.zeros_like(delta)
    pos = torch.clamp(delta, min=0.0)
    neg = torch.clamp(-delta, min=0.0)
    k = min(k_amp, int((pos > 0).sum()))
    if k > 0:
        v, i = pos.topk(k)
        z_amp[i] = v
    k = min(k_sup, int((neg > 0).sum()))
    if k > 0:
        v, i = neg.topk(k)
        z_sup[i] = v
    return z_amp, z_sup


def randomize_intervention(
    z: torch.Tensor, rng: np.random.Generator,
) -> torch.Tensor:
    """Same nonzero count and (permuted) magnitudes at random indices."""
    out = torch.zeros_like(z)
    nz = (z > 0).nonzero(as_tuple=True)[0]
    if len(nz) == 0:
        return out
    vals = z[nz].cpu().numpy().copy()
    rng.shuffle(vals)
    idx = rng.choice(z.numel(), size=len(nz), replace=False)
    for i, v in zip(idx, vals):
        out[int(i)] = float(v)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _words(s: str) -> List[str]:
    return s.strip().split()


def _norm(s: str) -> str:
    return " ".join(s.strip().split()).casefold()


def edited_positions(src_words: List[str], other_words: List[str]) -> set:
    """Source word positions touched by the src→other diff (insertions mark
    the boundary position i1)."""
    pos = set()
    sm = difflib.SequenceMatcher(None, src_words, other_words, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        pos.update(range(i1, max(i2, i1 + 1)))
    return pos


def pair_metrics(out_text: str, src: str, tgt: str) -> Dict[str, float]:
    ow, sw, tw = _words(out_text), _words(src), _words(tgt)
    gold_pos = edited_positions(sw, tw)
    pred_pos = edited_positions(sw, ow)
    union = gold_pos | pred_pos
    return {
        "exact_match": float(_norm(out_text) == _norm(tgt)),
        "copy_rate": float(_norm(out_text) == _norm(src)),
        "sim_target": difflib.SequenceMatcher(None, ow, tw, autojunk=False).ratio(),
        "sim_source": difflib.SequenceMatcher(None, ow, sw, autojunk=False).ratio(),
        "edit_loc_iou": (len(gold_pos & pred_pos) / len(union)) if union else 1.0,
    }


def corpus_bleu_chrf(outputs: List[str], targets: List[str]) -> Dict[str, Optional[float]]:
    try:
        import sacrebleu
    except ImportError:
        return {"bleu": None, "chrf": None}
    bleu = sacrebleu.corpus_bleu(outputs, [targets]).score
    chrf = sacrebleu.corpus_chrf(outputs, [targets]).score
    return {"bleu": float(bleu), "chrf": float(chrf)}


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS_ALL:
            raise SystemExit(f"unknown condition {c!r}; pick from {CONDITIONS_ALL}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    # ---- data -------------------------------------------------------------
    from datasets import load_dataset
    print(f"[lingua-eval] loading {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    print(f"[lingua-eval] {len(ds)} pairs after language={args.language}")
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen = order[:min(args.sample_size, len(order))]

    # ---- models -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    print("[lingua-eval] loading tagger / editor")
    tagger = load_tagger_from_checkpoint(
        args.llm2vec_dir, args.tagger_ckpt,
        d_sae=_peek_d_sae(args.tagger_ckpt), dtype=dtype,
    ).to(args.device).eval()
    editor = load_editor_from_checkpoint(
        args.llm2vec_dir, args.editor_ckpt,
        d_sae=_peek_d_sae(args.editor_ckpt), dtype=dtype,
    ).to(args.device).eval()

    print("[lingua-eval] loading SAE extractor / ranker components")
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    causal, _ = load_causal_gemma(args.llm2vec_dir)
    bid = BidirectionalLLM(args.llm2vec_dir, dtype=dtype)
    ranker = Ranker(extractor, causal, bid, RankerWeights(), device=args.device)

    rng = np.random.default_rng(args.seed)
    records: List[Dict] = []
    agg = {c: defaultdict(list) for c in conditions}
    texts = {c: {"out": [], "tgt": []} for c in conditions}
    baseline = defaultdict(list)
    base_texts = {"out": [], "tgt": []}
    skipped = 0

    for step, k in enumerate(chosen):
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]

        with torch.no_grad():
            z_src = extractor.pool_max_topk(extractor.encode_text(src),
                                            args.pool_topk).float().cpu()
            z_tgt = extractor.pool_max_topk(extractor.encode_text(tgt),
                                            args.pool_topk).float().cpu()
        z_amp_t, z_sup_t = diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)
        variants = {}
        if "true" in conditions:
            variants["true"] = (z_amp_t, z_sup_t)
        if "empty" in conditions:
            variants["empty"] = (torch.zeros_like(z_amp_t),
                                 torch.zeros_like(z_sup_t))
        if "random" in conditions:
            variants["random"] = (randomize_intervention(z_amp_t, rng),
                                  randomize_intervention(z_sup_t, rng))

        rec = {"idx": int(k), "feature": ex.get("feature"),
               "source": src, "target": tgt,
               "amp_features": (z_amp_t > 0).nonzero(as_tuple=True)[0].tolist(),
               "sup_features": (z_sup_t > 0).nonzero(as_tuple=True)[0].tolist(),
               "outputs": {}}
        try:
            for c, (za, zs) in variants.items():
                out_text = edit_once(
                    text=src, z_amp_full=za, z_sup_full=zs,
                    tagger=tagger, editor=editor, ranker=ranker,
                    tokenizer=tokenizer, l_max=args.l_max, device=args.device,
                    ins_threshold=args.ins_threshold,
                    max_templates=args.max_templates,
                    verbose=False,
                )
                m = pair_metrics(out_text, src, tgt)
                with torch.no_grad():
                    z_out = extractor.pool_max_topk(
                        extractor.encode_text(out_text), args.pool_topk,
                    ).float().cpu()
                eps = 1e-8
                cos_out = float(torch.dot(z_out, z_tgt)
                                / (z_out.norm() * z_tgt.norm() + eps))
                cos_src = float(torch.dot(z_src, z_tgt)
                                / (z_src.norm() * z_tgt.norm() + eps))
                m["sae_shift"] = cos_out - cos_src
                for key, v in m.items():
                    agg[c][key].append(v)
                texts[c]["out"].append(out_text)
                texts[c]["tgt"].append(tgt)
                rec["outputs"][c] = {"text": out_text, **m}
        except TemplateBudgetExceeded as e:
            skipped += 1
            rec["skipped"] = str(e)
            records.append(rec)
            continue

        bm = pair_metrics(src, src, tgt)      # input-copy baseline
        bm["sae_shift"] = 0.0
        for key, v in bm.items():
            baseline[key].append(v)
        base_texts["out"].append(src)
        base_texts["tgt"].append(tgt)
        records.append(rec)
        if (step + 1) % 10 == 0:
            print(f"[lingua-eval] {step + 1}/{len(chosen)} pairs "
                  f"(skipped={skipped})")

    # ---- aggregate + report -------------------------------------------------
    metric_keys = ["exact_match", "copy_rate", "sim_target", "sim_source",
                   "edit_loc_iou", "sae_shift"]
    summary = {"n_pairs": len(chosen), "n_scored": len(baseline["exact_match"]),
               "n_skipped_enumeration": skipped,
               "conditions": {}, "input_copy_baseline": {}}
    for c in conditions:
        row = {k2: float(np.mean(agg[c][k2])) if agg[c][k2] else None
               for k2 in metric_keys}
        row.update(corpus_bleu_chrf(texts[c]["out"], texts[c]["tgt"]))
        summary["conditions"][c] = row
    brow = {k2: float(np.mean(baseline[k2])) if baseline[k2] else None
            for k2 in metric_keys}
    brow.update(corpus_bleu_chrf(base_texts["out"], base_texts["tgt"]))
    summary["input_copy_baseline"] = brow

    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cols = metric_keys + ["bleu", "chrf"]
    lines = ["# LinguaLens end-to-end evaluation", "",
             f"dataset: `{args.dataset}` (language={args.language}), "
             f"sample={len(chosen)} (seed={args.seed}), scored="
             f"{summary['n_scored']}, enumeration-skipped={skipped}", "",
             f"intervention: diff-based, k_amp={args.k_amp} k_sup={args.k_sup} "
             f"pool_topk={args.pool_topk}; l_max={args.l_max} "
             f"ins_threshold={args.ins_threshold}", "",
             "| condition | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]

    def _fmt(v):
        return "—" if v is None else f"{v:.4f}"

    for c in conditions:
        row = summary["conditions"][c]
        lines.append(f"| {c} |" + "".join(f" {_fmt(row[k2])} |" for k2 in cols))
    lines.append("| input-copy baseline |"
                 + "".join(f" {_fmt(brow[k2])} |" for k2 in cols))
    lines += ["",
              "Reading guide:",
              "- `sim_target` must beat the input-copy baseline for the "
              "system to be editing toward the reference at all.",
              "- `Δ(true − empty)` on sim_target / sae_shift is the "
              "end-to-end conditioning-causality signal (the ranker's "
              "sae_align term participates here, unlike §13.5's probes).",
              "- high `copy_rate` = the tagger proposed nothing or the "
              "ranker preferred the unedited candidate.", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps(
        {"args": vars(args), **summary}, indent=2))

    print(f"[lingua-eval] wrote {out_dir}/report.md, summary.json, records.jsonl")
    for c in conditions:
        row = summary["conditions"][c]
        print(f"[lingua-eval] {c}: sim_target={_fmt(row['sim_target'])} "
              f"exact={_fmt(row['exact_match'])} copy={_fmt(row['copy_rate'])} "
              f"iou={_fmt(row['edit_loc_iou'])} sae_shift={_fmt(row['sae_shift'])}")
    print(f"[lingua-eval] input-copy: sim_target={_fmt(brow['sim_target'])}")


def _peek_d_sae(ckpt_path: str) -> int:
    blob = torch.load(ckpt_path, map_location="cpu")
    return int(blob["d_sae"])


if __name__ == "__main__":
    main()
