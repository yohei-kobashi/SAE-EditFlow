"""
NER probing eval for an LLM2Vec / base-LLM encoder.

Mirrors LLM2Vec paper §4 Table 2: freeze the encoder, train a single
linear layer on top of per-token hidden states, report entity-level F1
on CoNLL-2003 test.

Why this matters for SAE-LEWIS more than STS:
  - The tagger and editor consume per-token hidden states, NOT pooled
    sentence vectors. NER probing measures exactly that.
  - The paper's Mistral-7B NER F1 jumps from ~60 (base) to ~80 (Bi+MNTP+
    SimCSE) — a 20-point delta, much sharper than STS-B's 0.45 → 0.79.
  - The linear probe is structurally identical to SAE-LEWIS's `tagger.py`,
    so the score is a direct upper-bound proxy for tagger quality.

What you'll typically see:
  - Base Gemma-2B (causal):              entity F1 ~55-68
  - Base Gemma-2B (bidir patch only):    entity F1 ~60-72  (free win)
  - Bi+MNTP+SimCSE Gemma-2B (LoRA):      entity F1 ~75-85
  Δ ≥ 10 from base+bidir → recipe is doing useful work.

Usage:
    # Single run
    python scripts/eval_ner_probe.py \\
        --encoder ./runs/llm2vec_lora/llm2vec_simcse \\
        --output-json ./runs/llm2vec_lora/eval_ner.json

    # Base Gemma comparison (no bidir patch — pure causal LM)
    python scripts/eval_ner_probe.py \\
        --encoder google/gemma-2-2b --no-bidir \\
        --output-json ./runs/baseline_gemma_ner_causal.json

    # Base Gemma + bidir patch (free-attention bidir, no training)
    python scripts/eval_ner_probe.py \\
        --encoder google/gemma-2-2b \\
        --output-json ./runs/baseline_gemma_ner_bidir.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to path so we can import the bidir patch helper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import _patch_attention_bidirectional  # type: ignore  # noqa: E402


# CoNLL-2003 label set used as fallback when the dataset doesn't expose
# `features["ner_tags"].feature.names`. Modern parquet mirrors do expose
# them; we auto-detect at load time.
CONLL2003_LABELS_FALLBACK = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                              "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL_PAD = -100   # CE ignore_index


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True,
                   help="HF id or local path of the encoder. AutoModelForCausalLM "
                        "is used internally; we forward through model.model (the "
                        "inner backbone) to get `last_hidden_state` per token.")
    p.add_argument("--bidir", dest="bidir", action="store_true", default=True,
                   help="Apply `_patch_attention_bidirectional` to the inner "
                        "backbone before encoding (default). For a fair LLM2Vec "
                        "comparison, leave on for both LLM2Vec and base — the "
                        "delta then measures the contribution of training "
                        "rather than the patch.")
    p.add_argument("--no-bidir", dest="bidir", action="store_false",
                   help="Plain causal forward — useful for measuring the "
                        "base LLM in its native mode.")
    p.add_argument("--dataset", default="eriktks/conll2003",
                   help="HF dataset id. Default `eriktks/conll2003` is a "
                        "parquet-format mirror of CoNLL-2003 (the legacy "
                        "`conll2003` uses a Python script that newer "
                        "datasets versions refuse to load). Alternatives: "
                        "`tomaarsen/conll2003`, `conllpp`, `tner/wikiann` "
                        "with --dataset-config en.")
    p.add_argument("--dataset-config", default=None,
                   help="Optional dataset config (e.g. 'en' for wikiann).")
    p.add_argument("--tags-field", default="ner_tags",
                   help="Field name holding the per-token tag ids. "
                        "Some mirrors use `tags` instead of `ner_tags`.")

    # Per-token probe hyperparameters. Paper-style defaults (small linear
    # head over a 2-3K-d hidden state on a few-K-example dataset converges
    # in 3-5 epochs regardless of which encoder is underneath).
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Linear-probe convention: no weight decay.")
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


# --------------------------------------------------------------------------- #
# Data: tokenize word lists and align word-level NER labels to subwords.
# Standard HF NER recipe: the first subword of each word inherits the
# word's NER tag; subsequent subwords + all special tokens are set to -100
# (LABEL_PAD) so they don't contribute to the CE loss.
# --------------------------------------------------------------------------- #
def _tokenize_and_align(examples, tokenizer, max_length: int, tags_field: str):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,                # padded at collate time
    )
    labels = []
    for i, word_labels in enumerate(examples[tags_field]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        seq = []
        for word_id in word_ids:
            if word_id is None:
                seq.append(LABEL_PAD)        # BOS / EOS / pad
            elif word_id != prev:
                seq.append(int(word_labels[word_id]))
            else:
                seq.append(LABEL_PAD)        # later subwords of a word
            prev = word_id
        labels.append(seq)
    tokenized["labels"] = labels
    return tokenized


def _to_example_list(tokenized) -> List[Dict[str, List[int]]]:
    """Transpose dict-of-lists → list-of-dicts for the DataLoader."""
    n = len(tokenized["input_ids"])
    out = []
    for i in range(n):
        out.append({
            "input_ids": tokenized["input_ids"][i],
            "attention_mask": tokenized["attention_mask"][i],
            "labels": tokenized["labels"][i],
        })
    return out


def _make_collate(pad_id: int):
    def _collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), LABEL_PAD, dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
            attention_mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long)
            labels[i, :L] = torch.tensor(b["labels"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return _collate


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
def _load_encoder(encoder_path: str, dtype: torch.dtype, device: str, bidir: bool):
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        encoder_path, torch_dtype=dtype, attn_implementation="sdpa",
    )
    if bidir:
        _patch_attention_bidirectional(model.model)
        print(f"[ner] bidirectional patch applied to {encoder_path}")
    else:
        print(f"[ner] causal forward (no patch) on {encoder_path}")
    model.config.use_cache = False
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _token_macro_f1(all_preds: List[List[int]], all_labels: List[List[int]],
                    n_labels: int) -> float:
    """Macro F1 averaged over non-O classes (token-level). Works without
    seqeval — useful as a fallback / sanity check vs entity-level F1."""
    flat_pred, flat_lab = [], []
    for p_seq, l_seq in zip(all_preds, all_labels):
        for p, l in zip(p_seq, l_seq):
            flat_pred.append(p)
            flat_lab.append(l)
    flat_pred = np.asarray(flat_pred)
    flat_lab = np.asarray(flat_lab)
    f1s = []
    for c in range(1, n_labels):    # skip class 0 (O)
        tp = int(((flat_pred == c) & (flat_lab == c)).sum())
        fp = int(((flat_pred == c) & (flat_lab != c)).sum())
        fn = int(((flat_pred != c) & (flat_lab == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def _entity_f1(all_pred_seqs: List[List[str]],
               all_label_seqs: List[List[str]]) -> Tuple[float, str]:
    """BIO-aware entity-level F1 via seqeval. Returns (f1, classification_report)."""
    try:
        from seqeval.metrics import f1_score, classification_report
    except ImportError:
        return float("nan"), ("seqeval not installed; `pip install seqeval` for "
                              "entity-level F1.")
    f1 = float(f1_score(all_label_seqs, all_pred_seqs))
    report = classification_report(all_label_seqs, all_pred_seqs, digits=4)
    return f1, report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = _dtype(args.dtype)

    # ---- Dataset ----------------------------------------------------------
    # Try the requested dataset first, then fall back through a chain of
    # parquet mirrors so we don't fail just because one mirror went down or
    # got renamed.
    from datasets import load_dataset  # local import to keep startup snappy
    candidates = [args.dataset]
    fallbacks = ["eriktks/conll2003", "tomaarsen/conll2003", "conllpp"]
    for f in fallbacks:
        if f not in candidates:
            candidates.append(f)
    ds = None
    chosen = None
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            print(f"[ner] loading dataset {cand}"
                  + (f" (config={args.dataset_config})" if args.dataset_config else ""))
            if args.dataset_config:
                ds = load_dataset(cand, args.dataset_config)
            else:
                ds = load_dataset(cand)
            chosen = cand
            break
        except Exception as e:
            print(f"[ner]   {cand} failed: {type(e).__name__}: {e}")
            last_err = e
    if ds is None:
        raise RuntimeError(
            f"All dataset candidates failed: {candidates}. Last error: {last_err}"
        )
    print(f"[ner] using {chosen}: splits "
          f"train={len(ds['train'])} val={len(ds['validation'])} test={len(ds['test'])}")

    # Auto-detect label names from the dataset features when possible —
    # different mirrors may use slightly different orderings.
    label_names = list(CONLL2003_LABELS_FALLBACK)
    try:
        feat = ds["train"].features[args.tags_field]
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            label_names = list(feat.feature.names)
        elif hasattr(feat, "names"):
            label_names = list(feat.names)
        print(f"[ner] detected label set ({len(label_names)}): {label_names}")
    except (KeyError, AttributeError) as e:
        print(f"[ner] could not auto-detect labels ({e}); "
              f"falling back to CoNLL-2003 fixed set")
    n_labels = len(label_names)

    # ---- Encoder + tokenizer ---------------------------------------------
    model, tokenizer = _load_encoder(args.encoder, dtype, args.device, args.bidir)
    hidden_size = model.config.hidden_size
    inner = model.model
    print(f"[ner] encoder hidden_size={hidden_size}, vocab={len(tokenizer)}")

    # ---- Tokenize + align labels for each split --------------------------
    print("[ner] tokenizing + aligning subword labels...")
    tokenized = {}
    for split in ("train", "validation", "test"):
        # Bring the HF Dataset into a dict-of-lists in one shot.
        as_dict = ds[split][:]
        tokenized[split] = _tokenize_and_align(
            as_dict, tokenizer, args.max_seq_length, args.tags_field,
        )
        n_tokens_with_label = sum(
            sum(1 for x in seq if x != LABEL_PAD)
            for seq in tokenized[split]["labels"]
        )
        print(f"[ner]   {split}: {len(tokenized[split]['input_ids'])} seqs, "
              f"{n_tokens_with_label} labeled tokens")

    train_examples = _to_example_list(tokenized["train"])
    test_examples = _to_example_list(tokenized["test"])
    collate = _make_collate(tokenizer.pad_token_id)

    # ---- Linear probe ----------------------------------------------------
    classifier = nn.Linear(hidden_size, n_labels).to(args.device)
    optim = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- Training --------------------------------------------------------
    print(f"[ner] training linear classifier for {args.epochs} epochs, "
          f"lr={args.lr}, bs={args.batch_size}")
    t0 = time.time()
    for epoch in range(args.epochs):
        classifier.train()
        loader = DataLoader(
            train_examples, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate, drop_last=False,
        )
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            attn = batch["attention_mask"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)

            with torch.no_grad():
                out = inner(input_ids=input_ids, attention_mask=attn, use_cache=False)
                h = out.last_hidden_state.float()           # (B, T, d) fp32

            logits = classifier(h)                         # (B, T, n_labels)
            loss = F.cross_entropy(
                logits.reshape(-1, n_labels),
                labels.reshape(-1),
                ignore_index=LABEL_PAD,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()

            n_tokens_batch = int((labels != LABEL_PAD).sum().item())
            total_loss += float(loss.item()) * n_tokens_batch
            total_tokens += n_tokens_batch
            n_batches += 1
        avg_loss = total_loss / max(1, total_tokens)
        elapsed = time.time() - t0
        print(f"[ner]   epoch {epoch + 1}/{args.epochs}: "
              f"loss={avg_loss:.4f} tokens={total_tokens} "
              f"batches={n_batches} elapsed={elapsed:.1f}s")

    # ---- Eval on test ----------------------------------------------------
    print("[ner] evaluating on test set...")
    classifier.eval()
    test_loader = DataLoader(
        test_examples, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, drop_last=False,
    )
    all_pred_ids: List[List[int]] = []
    all_label_ids: List[List[int]] = []
    all_pred_seqs: List[List[str]] = []
    all_label_seqs: List[List[str]] = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            attn = batch["attention_mask"].to(args.device, non_blocking=True)
            labels = batch["labels"]                       # keep on CPU

            out = inner(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = classifier(out.last_hidden_state.float())
            preds = logits.argmax(dim=-1).cpu()

            for pred_row, lab_row in zip(preds.tolist(), labels.tolist()):
                p_ids, l_ids, p_str, l_str = [], [], [], []
                for p, l in zip(pred_row, lab_row):
                    if l == LABEL_PAD:
                        continue
                    p_ids.append(p)
                    l_ids.append(l)
                    p_str.append(label_names[p])
                    l_str.append(label_names[l])
                all_pred_ids.append(p_ids)
                all_label_ids.append(l_ids)
                all_pred_seqs.append(p_str)
                all_label_seqs.append(l_str)

    # ---- Metrics ---------------------------------------------------------
    entity_f1, report = _entity_f1(all_pred_seqs, all_label_seqs)
    token_macro_f1 = _token_macro_f1(all_pred_ids, all_label_ids, n_labels)
    elapsed_total = time.time() - t0

    print()
    print("=" * 66)
    print(f" NER probing results — {args.encoder}")
    print("=" * 66)
    print(f"  encoder              : {args.encoder}")
    print(f"  bidirectional patch  : {args.bidir}")
    print(f"  dataset              : {args.dataset}"
          + (f"/{args.dataset_config}" if args.dataset_config else ""))
    print(f"  entity-level F1      : {entity_f1:.4f}")
    print(f"  token macro-F1 (non-O classes): {token_macro_f1:.4f}")
    print(f"  total elapsed        : {elapsed_total:.1f}s")
    print()
    print(report)

    results = {
        "encoder": args.encoder,
        "bidirectional_patch": bool(args.bidir),
        "dataset": chosen,
        "dataset_requested": args.dataset,
        "dataset_config": args.dataset_config,
        "tags_field": args.tags_field,
        "labels": label_names,
        "entity_f1": float(entity_f1) if not np.isnan(entity_f1) else None,
        "token_macro_f1_non_o": float(token_macro_f1),
        "n_train": int(len(train_examples)),
        "n_test": int(len(test_examples)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "elapsed_sec": float(elapsed_total),
        "hidden_size": int(hidden_size),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[ner] wrote {out_path}")


if __name__ == "__main__":
    main()
