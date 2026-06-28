"""
Evaluate a trained LLM2Vec checkpoint.

Run as a standalone post-mortem on a finished MNTP run. Four evals:

  (1) MNTP held-out loss / perplexity. We re-apply the canonical LLM2Vec
      objective (DataCollatorForLanguageModeling 15% / 80-10-10 +
      GemmaForCausalLM.forward(labels=...) with its built-in +1 shift) to
      a held-out slice of Dolma. Lower = better. A self-consistency check
      that MNTP training has actually learned something — random init
      would give loss ≈ log(V) ≈ 12-13.

  (2) Causal PPL drift vs the base LLM. Both models loaded WITHOUT the
      bidirectional patch (standard causal forward) and scored on the same
      held-out sentences. This is the metric that matters for the PPL
      scorer in `corruption.py`, which loads the same checkpoint in causal
      mode. A large drift signals MNTP has shifted the causal LM behavior
      meaningfully; a small drift means PPL ratios in stage 2 will track
      base Gemma's closely.

  (3) Bidirectional vs causal hidden-state divergence on THIS checkpoint.
      Same weights, two forwards: with and without `_patch_attention_bidirectional`.
      Cosine sim per token position; mean/std/percentiles. Confirms the
      bidir patch is active. A mean cosine close to 1.0 means the patch
      didn't take effect (e.g., SDPA path overrode it).

  (4) Special token embedding sanity. For [MASK], [INS], [DEL]: row norm,
      ratio-to-median, and nearest neighbors by cosine. Used to spot
      undertrained or anomalous embeddings before they propagate into
      editor / tagger training.

Outputs:
  $OUTPUT_DIR/eval_metrics.json   (machine-readable)
  $OUTPUT_DIR/eval_report.md      (human-readable)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)

from data import download_dolma_shards, iter_dolma_texts, iter_sentences
from model import _patch_attention_bidirectional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True,
                   help="Path to a trained LLM2Vec checkpoint (output of "
                        "train_llm2vec.py).")
    p.add_argument("--baseline-llm", default=None,
                   help="HF id of the base LLM for causal-PPL comparison. "
                        "Defaults to llm2vec_meta.json's base_llm field; pass "
                        "'none' to skip.")
    p.add_argument("--data-cache-dir", default="./dolma_cache")
    p.add_argument("--max-files", type=int, default=1,
                   help="# of Dolma shards to stream.")
    p.add_argument("--n-sentences", type=int, default=500,
                   help="Held-out sentence count for evals (1) and (2).")
    p.add_argument("--n-bidir-causal", type=int, default=100,
                   help="Subset for eval (3) (each item does 2 forwards).")
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--mlm-probability", type=float, default=0.15,
                   help="Masking rate for MNTP eval. Should match the value "
                        "used at training time for a meaningful comparison.")
    p.add_argument("--sentence-splitter",
                   choices=["pysbd", "nltk"], default="pysbd")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[s]


# --------------------------------------------------------------------------- #
# Model loading helpers
# --------------------------------------------------------------------------- #
def load_model(
    path: str,
    dtype: torch.dtype,
    device: str,
    bidir_patch: bool,
) -> torch.nn.Module:
    """Load a HF causal LM, optionally with the LLM2Vec bidirectional patch.

    Matches the canonical setup: `attn_implementation="eager"` so the
    explicit 4D mask from `_update_causal_mask` is consumed by the
    attention kernel directly (no SDPA `is_causal` override).
    """
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="eager",
    )
    if bidir_patch:
        _patch_attention_bidirectional(model.model)
    model.config.use_cache = False
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def collect_sentences(args, tokenizer) -> List[str]:
    """Stream Dolma until we have N quality-filtered sentences."""
    shards = download_dolma_shards(args.data_cache_dir, max_files=args.max_files)
    sents: List[str] = []
    for s in iter_sentences(
        iter_dolma_texts(shards, min_chars=64),
        splitter=args.sentence_splitter,
        min_chars=16,
        max_chars=2000,
        max_sentences_per_text=None,
        sample_strategy="head",
        # Distinct seed so held-out sentences don't overlap with any
        # training-time stream that used --seed args.seed.
        seed=args.seed + 9999,
        quality_filter=True,
    ):
        sents.append(s)
        if len(sents) >= args.n_sentences:
            break
    return sents


# --------------------------------------------------------------------------- #
# Eval (1): MNTP held-out loss
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_mntp_loss(model, tokenizer, sents, args) -> dict:
    """Average MNTP cross-entropy over a held-out slice."""
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    total_loss_weighted = 0.0
    total_tokens = 0
    n_seqs = 0
    for s in sents:
        enc = tokenizer(
            s,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True,
        )
        if len(enc["input_ids"]) < 4:
            continue
        batch = collator([{
            "input_ids": enc["input_ids"],
            "special_tokens_mask": enc["special_tokens_mask"],
        }])
        n_labels = int((batch["labels"] != -100).sum().item())
        if n_labels == 0:
            continue
        batch = {k: v.to(args.device) for k, v in batch.items()}
        out = model(**batch)
        # HF returns mean CE over non-ignored labels; un-mean to aggregate.
        total_loss_weighted += float(out.loss.item()) * n_labels
        total_tokens += n_labels
        n_seqs += 1
    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"),
                "n_tokens": 0, "n_sequences": 0}
    mean_loss = total_loss_weighted / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": float(np.exp(mean_loss)),
        "n_tokens": total_tokens,
        "n_sequences": n_seqs,
    }


# --------------------------------------------------------------------------- #
# Eval (2): Causal PPL drift
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_causal_ppl(model, tokenizer, sents, args) -> dict:
    """Standard causal-LM perplexity on held-out Dolma sentences.

    Each sentence is forwarded as `labels = input_ids`; the model's
    forward applies the +1 shift internally, so this is the standard
    next-token PPL (same as `corruption.causal_perplexity_text`).
    """
    total_loss_weighted = 0.0
    total_tokens = 0
    n_seqs = 0
    for s in sents:
        enc = tokenizer(
            s, return_tensors="pt",
            truncation=True, max_length=args.max_seq_length,
        )
        if enc.input_ids.shape[1] < 2:
            continue
        enc = {k: v.to(args.device) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"], labels=enc["input_ids"],
                    use_cache=False)
        # +1 shift means n labels predicted = T - 1 (no <-100> involved here).
        n_pred = enc["input_ids"].shape[1] - 1
        total_loss_weighted += float(out.loss.item()) * n_pred
        total_tokens += n_pred
        n_seqs += 1
    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"),
                "n_tokens": 0, "n_sequences": 0}
    mean_loss = total_loss_weighted / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": float(np.exp(mean_loss)),
        "n_tokens": total_tokens,
        "n_sequences": n_seqs,
    }


# --------------------------------------------------------------------------- #
# Eval (3): bidirectional vs causal hidden-state divergence
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_bidir_vs_causal(
    model_bidir, model_causal, tokenizer, sents, args,
) -> dict:
    """Per-position cosine sim between bidir and causal hidden states.

    Same weights, same input — only the attention mask differs. If the
    bidirectional patch is doing its job, h_bidir at non-final positions
    incorporates future context that h_causal cannot see, so the cosine
    should drop materially below 1.0 (typically 0.5-0.9 range after
    MNTP). A mean cosine ≥ 0.999 means the patch silently failed.
    """
    cosines: List[np.ndarray] = []
    first_pos_cos: List[float] = []
    last_pos_cos: List[float] = []
    n_seq_used = 0
    target = args.n_bidir_causal
    for s in sents[:target * 3]:
        enc = tokenizer(
            s, return_tensors="pt",
            truncation=True, max_length=args.max_seq_length,
        ).to(args.device)
        if enc.input_ids.shape[1] < 4:
            continue
        out_b = model_bidir(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        out_c = model_causal(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        h_b = out_b.hidden_states[-1][0].float()  # (T, d)
        h_c = out_c.hidden_states[-1][0].float()  # (T, d)
        cos = F.cosine_similarity(h_b, h_c, dim=-1).cpu().numpy()  # (T,)
        cosines.append(cos)
        first_pos_cos.append(float(cos[0]))
        last_pos_cos.append(float(cos[-1]))
        n_seq_used += 1
        if n_seq_used >= target:
            break

    if not cosines:
        return {"mean_cos": float("nan"), "n_positions": 0}

    flat = np.concatenate(cosines)
    return {
        "mean_cos": float(flat.mean()),
        "std_cos": float(flat.std()),
        "min_cos": float(flat.min()),
        "max_cos": float(flat.max()),
        "p25_cos": float(np.percentile(flat, 25)),
        "p50_cos": float(np.percentile(flat, 50)),
        "p75_cos": float(np.percentile(flat, 75)),
        "p95_cos": float(np.percentile(flat, 95)),
        "first_pos_mean": float(np.mean(first_pos_cos)),
        # The last position can see no future tokens even with bidir
        # attention (there is no right context), so its cosine should be
        # close to 1.0. Useful as a sanity check that the comparison is set
        # up correctly: if last_pos_mean isn't ~1.0, something is wrong.
        "last_pos_mean": float(np.mean(last_pos_cos)),
        "n_positions": int(flat.size),
        "n_sequences": int(n_seq_used),
    }


# --------------------------------------------------------------------------- #
# Eval (4): special-token embedding sanity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_special_token_embeddings(model, tokenizer) -> dict:
    """Inspect embedding rows for [MASK], [INS], [DEL]."""
    emb = model.get_input_embeddings().weight  # (V, d)
    norms = emb.float().norm(dim=-1)
    median_norm = float(norms.median().item())
    mean_norm = float(norms.mean().item())

    tokens_report: dict = {}
    for name in ["[MASK]", "[INS]", "[DEL]"]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is None or tid == tokenizer.unk_token_id:
            tokens_report[name] = {"found": False}
            continue
        row = emb[tid].float()
        row_norm = float(row.norm().item())
        # Cosine against all rows
        all_f = emb.float()
        cos = F.cosine_similarity(row.unsqueeze(0), all_f, dim=-1)
        cos[tid] = -2.0  # exclude self
        # Filter out PAD / BOS / EOS from neighbors to keep output readable
        top_v, top_i = cos.topk(8)
        nn = []
        for v, i in zip(top_v.tolist(), top_i.tolist()):
            tok_str = tokenizer.decode([int(i)])
            nn.append({"token": tok_str, "id": int(i), "cosine": float(v)})
        tokens_report[name] = {
            "found": True,
            "id": int(tid),
            "norm": row_norm,
            "norm_ratio_median": row_norm / median_norm if median_norm > 0 else float("nan"),
            "norm_ratio_mean": row_norm / mean_norm if mean_norm > 0 else float("nan"),
            "top8_neighbors": nn,
        }
    return {
        "median_norm_all": median_norm,
        "mean_norm_all": mean_norm,
        "vocab_size": int(emb.shape[0]),
        "embed_dim": int(emb.shape[1]),
        "tokens": tokens_report,
    }


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def render_report(args, metrics: dict, out_path: Path):
    L: List[str] = []
    L.append("# LLM2Vec evaluation report")
    L.append("")
    L.append(f"- LLM2Vec checkpoint : `{args.llm2vec_dir}`")
    L.append(f"- Baseline LLM       : `{metrics.get('baseline_llm') or 'n/a'}`")
    L.append(f"- Held-out sentences : {args.n_sentences}")
    L.append(f"- mlm_probability    : {args.mlm_probability}")
    L.append("")

    # ---- Eval 1: MNTP loss ---------------------------------------------------
    L.append("## 1. MNTP held-out loss / perplexity")
    L.append("")
    m1 = metrics["mntp"]
    L.append("Loss is averaged over non-ignored labels; ppl = exp(loss).")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| MNTP loss     | {m1['loss']:.4f} |")
    L.append(f"| MNTP ppl      | {m1['perplexity']:.3f} |")
    L.append(f"| # tokens      | {m1['n_tokens']} |")
    L.append(f"| # sequences   | {m1['n_sequences']} |")
    L.append("")
    L.append("Interpretation: random-init loss ≈ log(V) ≈ 12. Healthy MNTP")
    L.append("typically lands in the 2-6 range on Dolma after several thousand")
    L.append("steps. A value > 8 suggests the model barely trained.")
    L.append("")

    # ---- Eval 2: causal PPL drift -------------------------------------------
    L.append("## 2. Causal PPL drift vs base LLM")
    L.append("")
    m2 = metrics["causal_ppl"]
    L.append("Both models forwarded WITHOUT the bidir patch — same causal next-")
    L.append("token PPL used by `corruption.causal_perplexity_text`.")
    L.append("")
    L.append("| Model | Loss | PPL | # tokens |")
    L.append("|---|---|---|---|")
    L.append(f"| LLM2Vec (this ckpt) | {m2['llm2vec']['loss']:.4f} | "
             f"{m2['llm2vec']['perplexity']:.3f} | {m2['llm2vec']['n_tokens']} |")
    if "baseline" in m2:
        b = m2["baseline"]
        L.append(f"| Base LLM            | {b['loss']:.4f} | "
                 f"{b['perplexity']:.3f} | {b['n_tokens']} |")
        if m2['llm2vec']['loss'] == m2['llm2vec']['loss']:
            drift = (m2['llm2vec']['loss'] - b['loss']) / b['loss'] * 100
            sign = "+" if drift >= 0 else ""
            L.append("")
            L.append(f"Causal-PPL drift (LLM2Vec relative to base): **{sign}{drift:.2f}%**")
            L.append("")
            L.append("MNTP fine-tunes the model under bidirectional attention,")
            L.append("then we re-evaluate with causal attention. A small positive")
            L.append("drift (a few %) is expected; large drift means causal PPL")
            L.append("ratios in stage 2 will diverge from base Gemma's.")
    L.append("")

    # ---- Eval 3: bidir vs causal divergence --------------------------------
    L.append("## 3. Bidirectional vs causal hidden-state divergence")
    L.append("")
    m3 = metrics["bidir_vs_causal"]
    L.append("Per-position cosine sim between bidir and causal forwards of the")
    L.append("SAME model. Lower = bidir is using right context more.")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| mean cosine   | {m3['mean_cos']:.4f} |")
    L.append(f"| std cosine    | {m3['std_cos']:.4f} |")
    L.append(f"| p25 / p50 / p75 | {m3['p25_cos']:.4f} / "
             f"{m3['p50_cos']:.4f} / {m3['p75_cos']:.4f} |")
    L.append(f"| min / max     | {m3['min_cos']:.4f} / {m3['max_cos']:.4f} |")
    L.append(f"| first-pos mean | {m3['first_pos_mean']:.4f} "
             f"(position 0 gains the most right context — should be lowest) |")
    L.append(f"| last-pos mean  | {m3['last_pos_mean']:.4f} "
             f"(no right context to gain — should be ≈ 1.000) |")
    L.append(f"| # positions    | {m3['n_positions']} |")
    L.append("")
    if m3['last_pos_mean'] < 0.99:
        L.append("⚠️  Last-position cosine is < 0.99 even though no future context")
        L.append("   exists. Something other than the attention mask is differing —")
        L.append("   double-check the patch + eager attention setup.")
        L.append("")
    if m3['mean_cos'] > 0.999:
        L.append("⚠️  Mean cosine ≈ 1.0 — the bidirectional patch is likely NOT")
        L.append("   active. Confirm `attn_implementation=\"eager\"` and that the")
        L.append("   patched `_update_causal_mask` is being called.")
        L.append("")
    elif m3['mean_cos'] > 0.95:
        L.append("📝  Mean cosine > 0.95: bidir effect is subtle. Either MNTP")
        L.append("   training was very short (so weights barely moved) or the")
        L.append("   bidirectional context isn't being used heavily.")
        L.append("")

    # ---- Eval 4: special token embeddings ----------------------------------
    L.append("## 4. Special token embedding sanity")
    L.append("")
    m4 = metrics["special_tokens"]
    L.append(f"Vocab size = {m4['vocab_size']}, embed dim = {m4['embed_dim']}.")
    L.append(f"All-token row norm: median = {m4['median_norm_all']:.4f}, "
             f"mean = {m4['mean_norm_all']:.4f}.")
    L.append("")
    for name, info in m4["tokens"].items():
        L.append(f"### `{name}`")
        if not info.get("found"):
            L.append("- ❌ Not found in vocab.")
            L.append("")
            continue
        ratio = info["norm_ratio_median"]
        flag = ""
        if ratio < 0.5 or ratio > 2.0:
            flag = "  ⚠️  outside [0.5, 2.0]×median — possibly under/over-trained"
        L.append(f"- id = {info['id']}")
        L.append(f"- norm = {info['norm']:.4f} (×{ratio:.2f} median){flag}")
        L.append("- top neighbours (cosine):")
        for nn in info["top8_neighbors"]:
            L.append(f"    - `{nn['token']}` (id={nn['id']}): {nn['cosine']:.4f}")
        L.append("")

    out_path.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)
    dtype = _dtype_from_str(args.llm_dtype)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve baseline LLM from meta if not supplied.
    baseline_llm: Optional[str] = args.baseline_llm
    meta_path = Path(args.llm2vec_dir) / "llm2vec_meta.json"
    if baseline_llm is None and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        baseline_llm = meta.get("base_llm")
    if baseline_llm == "none":
        baseline_llm = None

    print(f"[eval] LLM2Vec checkpoint : {args.llm2vec_dir}")
    print(f"[eval] Baseline LLM       : {baseline_llm}")
    print(f"[eval] Output dir         : {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Two copies of the same MNTP'd weights: with and without bidir patch.
    print("[eval] loading LLM2Vec (bidir)...")
    model_bidir = load_model(args.llm2vec_dir, dtype, args.device, bidir_patch=True)
    print("[eval] loading LLM2Vec (causal — same weights, no patch)...")
    model_causal = load_model(args.llm2vec_dir, dtype, args.device, bidir_patch=False)

    print(f"[eval] collecting {args.n_sentences} held-out Dolma sentences...")
    sents = collect_sentences(args, tokenizer)
    print(f"[eval] collected {len(sents)} sentences")

    metrics: dict = {"baseline_llm": baseline_llm, "config": vars(args)}

    # ---- Eval 1: MNTP held-out loss ----------------------------------------
    print("[eval] (1/4) MNTP held-out loss / perplexity")
    set_seed(args.seed)  # deterministic masking across model swaps
    metrics["mntp"] = eval_mntp_loss(model_bidir, tokenizer, sents, args)
    print(f"        loss = {metrics['mntp']['loss']:.4f}  "
          f"ppl = {metrics['mntp']['perplexity']:.2f}  "
          f"({metrics['mntp']['n_tokens']} tokens)")

    # ---- Eval 2: Causal PPL drift ------------------------------------------
    print("[eval] (2/4) Causal PPL (this ckpt, no patch)")
    metrics["causal_ppl"] = {
        "llm2vec": eval_causal_ppl(model_causal, tokenizer, sents, args),
    }
    print(f"        loss = {metrics['causal_ppl']['llm2vec']['loss']:.4f}  "
          f"ppl = {metrics['causal_ppl']['llm2vec']['perplexity']:.2f}")

    if baseline_llm:
        print(f"[eval] (2/4) Causal PPL (baseline {baseline_llm}, no patch)")
        base_model = load_model(baseline_llm, dtype, args.device, bidir_patch=False)
        base_tok = AutoTokenizer.from_pretrained(baseline_llm)
        if base_tok.pad_token is None:
            base_tok.pad_token = base_tok.eos_token
        # Use base's own tokenizer to score — for Dolma sentences (no
        # [INS]/[DEL]/[MASK]), this tokenization matches LLM2Vec's, so the
        # numbers are directly comparable.
        metrics["causal_ppl"]["baseline"] = eval_causal_ppl(
            base_model, base_tok, sents, args,
        )
        print(f"        loss = {metrics['causal_ppl']['baseline']['loss']:.4f}  "
              f"ppl = {metrics['causal_ppl']['baseline']['perplexity']:.2f}")
        del base_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- Eval 3: bidir vs causal divergence --------------------------------
    print("[eval] (3/4) Bidir vs causal hidden-state divergence")
    metrics["bidir_vs_causal"] = eval_bidir_vs_causal(
        model_bidir, model_causal, tokenizer, sents, args,
    )
    print(f"        mean cosine = {metrics['bidir_vs_causal']['mean_cos']:.4f}  "
          f"(first-pos {metrics['bidir_vs_causal']['first_pos_mean']:.4f}, "
          f"last-pos {metrics['bidir_vs_causal']['last_pos_mean']:.4f})")

    # ---- Eval 4: special token embedding sanity ----------------------------
    print("[eval] (4/4) Special token embedding sanity")
    metrics["special_tokens"] = eval_special_token_embeddings(model_bidir, tokenizer)
    for name, info in metrics["special_tokens"]["tokens"].items():
        if info.get("found"):
            print(f"        {name}: norm = {info['norm']:.3f}  "
                  f"(×{info['norm_ratio_median']:.2f} median)")
        else:
            print(f"        {name}: NOT FOUND")

    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    render_report(args, metrics, out_dir / "eval_report.md")
    print(f"[eval] wrote {out_dir / 'eval_metrics.json'}")
    print(f"[eval] wrote {out_dir / 'eval_report.md'}")


if __name__ == "__main__":
    main()
