"""
Stage 2: corruption data generation for SAE-LEWIS.

Produces a sharded JSON-lines corruption cache (see README §8.2) by applying
fluency-preserving, MLM-based corruption to sentence-segmented Dolma:

  REPL : substitute words with MLM-predicted alternatives
  INS  : delete words whose context can recover them
  DEL  : insert MLM-predicted words that the editor must drop

Corruption operates at the WORD / TEXT level. The MLM (any HF AutoModelFor-
MaskedLM via `model.MLMProvider`) is decoupled from the downstream editor /
tagger encoder: the MLM's tokenizer is encapsulated and the final training
sample is re-tokenized with the downstream Gemma tokenizer at the end. The
two systems communicate only through text.

Per-sample filters (rejection sampling):
  - MLM recoverability for INS (text-level word equality against the MLM's
    top-K predictions)
  - Perplexity ratio under the frozen causal Gemma
  - SAE-shift L2 between clean and corrupted text (via Gemma + Gemma Scope)

The SAE forward and the perplexity scorer keep using Gemma; only the MLM
that proposes corruption candidates is swappable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, set_seed

from data import download_dolma_shards, iter_dolma_texts, iter_sentences
from lewis_ops import OP_DEL, OP_INS, OP_KEEP, OP_REPL
from model import MLMProvider, SAEFeatureExtractor, load_causal_gemma


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-cache-dir", default="./dolma_cache")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out-dir", required=True)

    # Downstream tokenizer (Gemma family). The MNTP'd Gemma checkpoint
    # already has [INS] / [DEL] in its vocabulary; we reuse it as the
    # tokenizer source.
    p.add_argument("--llm2vec-dir", required=True,
                   help="MNTP'd Gemma checkpoint (output of train_llm2vec.py). "
                        "Used for the downstream tokenizer ([INS]/[DEL] aware) "
                        "and as the causal LM for perplexity scoring.")
    p.add_argument("--llm", default="google/gemma-2-2b",
                   help="Base Gemma used by SAE extractor.")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", default="layer_12/width_16k/average_l0_82/params.npz",
                   help="Path inside --sae-repo. Available L0 values for "
                        "google/gemma-scope-2b-pt-res layer_12/width_16k are "
                        "{22, 41, 82, 176, 445}.")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"], default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    # Corruption MLM (pluggable)
    p.add_argument("--mlm-model", default="modernbert-base",
                   help="MLM key (modernbert-base, deberta-v3-base, ...) or any "
                        "HF model id. See model.MLMProvider.PRESETS.")
    p.add_argument("--mlm-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    # Sentence segmentation
    p.add_argument("--sentence-splitter", choices=["pysbd", "nltk"], default="pysbd")
    p.add_argument("--sent-min-tokens", type=int, default=5)
    p.add_argument("--sent-max-tokens", type=int, default=256)
    p.add_argument("--max-sentences-per-text", type=int, default=None,
                   help="Cap on qualifying sentences kept per source document. "
                        "None = use every sentence.")
    p.add_argument("--sentence-sample-strategy",
                   choices=["head", "random", "stride"], default="head")
    p.add_argument("--no-quality-filter", action="store_true")
    p.add_argument("--quality-min-words", type=int, default=3)
    p.add_argument("--quality-min-alpha-ratio", type=float, default=0.5)
    p.add_argument("--quality-require-terminal-punct", action="store_true", default=True)
    p.add_argument("--quality-require-initial-capital", action="store_true", default=False)

    p.add_argument("--target-samples", type=int, default=100000)
    p.add_argument("--samples-per-shard", type=int, default=10000)
    p.add_argument("--reject-budget", type=int, default=5)

    p.add_argument("--p-identity", type=float, default=0.05)
    p.add_argument("--p-repl", type=float, default=0.30)
    p.add_argument("--p-ins", type=float, default=0.30)
    p.add_argument("--p-del", type=float, default=0.22)
    p.add_argument("--p-mixed-repl-ins", type=float, default=0.03)
    p.add_argument("--p-mixed-repl-del", type=float, default=0.03)

    p.add_argument("--repl-words-max", type=int, default=4,
                   help="Max # of words to substitute per REPL sample.")
    p.add_argument("--repl-mlm-topk", type=int, default=8)

    p.add_argument("--ins-word-span-max", type=int, default=3,
                   help="Max # of consecutive words to delete per INS sample.")

    p.add_argument("--del-word-span-max", type=int, default=2,
                   help="Max # of consecutive words to insert per DEL sample.")
    p.add_argument("--del-mlm-topk", type=int, default=8)
    p.add_argument("--del-top1-prob", type=float, default=0.5)

    p.add_argument("--sae-shift-threshold", type=float, default=0.3)
    p.add_argument("--ppl-max-ratio", type=float, default=2.0)
    p.add_argument("--k-train", type=int, default=64)

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    extractor: SAEFeatureExtractor
    causal_llm: torch.nn.Module
    mlm: MLMProvider
    gemma_tok: any
    mask_id: int           # downstream (Gemma) mask id
    ins_id: int
    del_id: int
    pad_id: int
    device: str
    k_train: int


# ---------------------------------------------------------------------------
# Text / word utilities
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\S+")


def words_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def gemma_tokenize(gemma_tok, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    enc = gemma_tok(text, add_special_tokens=True,
                    return_offsets_mapping=True, truncation=True)
    return list(enc["input_ids"]), list(enc["offset_mapping"])


def find_token_range(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Tuple[Optional[int], Optional[int]]:
    """Return [tok_start, tok_end) covering chars [char_start, char_end)."""
    tok_start: Optional[int] = None
    tok_end: Optional[int] = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:                  # added special tokens (BOS/EOS)
            continue
        if tok_start is None and e > char_start:
            tok_start = i
        if e >= char_end:
            tok_end = i + 1
            break
    if tok_start is not None and tok_end is None:
        tok_end = len(offsets)
    return tok_start, tok_end


def looks_like_word(s: str) -> bool:
    s = s.strip()
    return bool(s) and any(c.isalpha() for c in s) and not s.startswith(("##", "▁"))


# ---------------------------------------------------------------------------
# SAE pool-max top-K helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def sae_pool_max_topk_from_text(
    stage: Stage, text: str,
) -> Tuple[List[int], List[float]]:
    enc = stage.extractor.llm_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256,
    ).to(stage.device)
    out = stage.extractor.llm(
        **enc, output_hidden_states=True, use_cache=False,
    )
    h = out.hidden_states[stage.extractor.layer_idx][0]
    z = stage.extractor.sae.encode(h.to(stage.extractor.sae.W_enc.dtype))
    sparse = stage.extractor.pool_max_topk(z, stage.k_train)
    nz = (sparse > 0).nonzero(as_tuple=True)[0]
    return nz.tolist(), sparse[nz].cpu().float().tolist()


def topk_records(feature_ids: List[int], values: List[float]) -> List[Dict]:
    return [{"f": int(f), "v": float(v)} for f, v in zip(feature_ids, values)]


def sae_l2_shift(
    a_feats: List[int], a_vals: List[float],
    b_feats: List[int], b_vals: List[float],
) -> float:
    a_map = dict(zip(a_feats, a_vals))
    b_map = dict(zip(b_feats, b_vals))
    union = set(a_map) | set(b_map)
    return math.sqrt(sum((a_map.get(f, 0.0) - b_map.get(f, 0.0)) ** 2 for f in union))


@torch.no_grad()
def causal_perplexity_text(stage: Stage, text: str) -> float:
    enc = stage.gemma_tok(
        text, return_tensors="pt", truncation=True, max_length=256,
    ).to(stage.device)
    if enc.input_ids.shape[1] < 2:
        return float("nan")
    out = stage.causal_llm(input_ids=enc.input_ids, labels=enc.input_ids, use_cache=False)
    return float(math.exp(float(out.loss.item())))


# ---------------------------------------------------------------------------
# Word-level corruption operations
# ---------------------------------------------------------------------------
def _norm(w: str) -> str:
    return w.strip().lower()


def make_repl_sample(
    stage: Stage, text: str, rng: random.Random,
    repl_words_max: int, mlm_topk: int,
) -> Optional[Dict]:
    """Replace each chosen word with an MLM-predicted alternative.

    Same-Gemma-token-count constraint: each substitution must use the same
    number of Gemma tokens as the original word. This keeps editor input
    and target aligned by position without an INS/DEL spillover.
    """
    words = words_with_offsets(text)
    if len(words) < 5:
        return None

    n_repl = rng.randint(1, max(1, min(repl_words_max, len(words) // 3)))
    word_indices = sorted(rng.sample(range(len(words)), n_repl))

    x_ids, x_offsets = gemma_tokenize(stage.gemma_tok, text)
    current_text = text
    char_shift = 0
    repl_char_ranges: List[Tuple[int, int]] = []
    orig_token_ranges_in_x: List[Tuple[int, int]] = []
    for wi in word_indices:
        orig_word, ws, we = words[wi]
        new_ws = ws + char_shift
        new_we = we + char_shift
        # Build masked text with a SINGLE [MASK] for this word
        masked_text = current_text[:new_ws] + stage.mlm.mask_token + current_text[new_we:]
        preds = stage.mlm.predict_at_masks(masked_text, top_k=mlm_topk)
        if not preds or not preds[0]:
            return None
        cands = [c for c in preds[0]
                 if looks_like_word(c) and _norm(c) != _norm(orig_word)
                 and len(c) <= max(20, len(orig_word) * 3)]
        if not cands:
            return None
        replacement = cands[rng.randrange(min(len(cands), max(1, mlm_topk // 2)))]
        # Substitute
        current_text = current_text[:new_ws] + replacement + current_text[new_we:]
        repl_char_ranges.append((new_ws, new_ws + len(replacement)))
        char_shift += len(replacement) - len(orig_word)
        # Track the original word's Gemma token range for the same-count check
        os, oe = find_token_range(x_offsets, ws, we)
        if os is None or oe is None:
            return None
        orig_token_ranges_in_x.append((os, oe))

    xp_ids, xp_offsets = gemma_tokenize(stage.gemma_tok, current_text)
    if len(xp_ids) != len(x_ids):
        return None

    # Same-token-count check for each substitution, and capture REPL positions
    repl_positions_in_xp: List[int] = []
    for (cs, ce), (os, oe) in zip(repl_char_ranges, orig_token_ranges_in_x):
        ns, ne = find_token_range(xp_offsets, cs, ce)
        if ns is None or ne is None:
            return None
        if (ne - ns) != (oe - os):
            return None
        repl_positions_in_xp.extend(range(ns, ne))
    if not repl_positions_in_xp:
        return None

    tagger_gold = [OP_KEEP] * len(xp_ids)
    editor_input = list(xp_ids)
    for i in repl_positions_in_xp:
        tagger_gold[i] = OP_REPL
        editor_input[i] = stage.mask_id
    editor_target = list(x_ids)            # aligned 1:1 by construction

    return {
        "corruption_type": "repl",
        "x_token_ids": x_ids,
        "x_prime_token_ids": xp_ids,
        "tagger_gold": tagger_gold,
        "editor_input_token_ids": editor_input,
        "editor_target_token_ids": editor_target,
        "ins_span_length": 0,
        "del_span_length": 0,
        "x_text": text,
        "x_prime_text": current_text,
    }


INS_REJECT_REASONS = (
    "too_short_words",     # too few words to fit `word_span_max` with margin
    "too_short_xprime",    # corrupted text would be < 8 non-whitespace chars
    "ins_span_nonpos",     # token-count diff is <= 0 (STRUCTURAL)
    "suffix_mismatch",     # prefix matched but suffix did not (STRUCTURAL)
    "length_mismatch",     # final editor_input vs target length mismatch (STRUCTURAL)
)


def make_ins_sample(
    stage: Stage, text: str, rng: random.Random,
    word_span_max: int,
    *,
    reject_counter: Optional[Dict[str, int]] = None,
) -> Optional[Dict]:
    """Delete a span of consecutive words.

    No MLM-based recovery check: the editor recovers the deleted words from
    context + z-conditioning, the same way REPL recovers the original word.
    A recoverability gate on the MLM was over-conservative (an MLM rarely
    predicts the EXACT original word from context the way our SAE-conditioned
    editor can), and broke symmetry with REPL/DEL. PPL and SAE-shift gates in
    `finalize_sample` still apply.

    When `reject_counter` is provided, increments the matching reason key on
    every early return so the caller can see which gate is dominating the
    yield. Reasons are listed in `INS_REJECT_REASONS`.
    """
    def _rej(reason: str):
        if reject_counter is not None:
            reject_counter[reason] = reject_counter.get(reason, 0) + 1
        return None

    words = words_with_offsets(text)
    if len(words) < 5:
        return _rej("too_short_words")
    # Require at least 1 word on each side of the deletion span:
    #   start_wi >= 1, end_wi <= len(words) - 1
    # which means n_words <= len(words) - 2.
    max_span = min(word_span_max, len(words) - 2)
    if max_span < 1:
        return _rej("too_short_words")
    n_words = rng.randint(1, max_span)
    start_wi = rng.randint(1, len(words) - n_words - 1)
    end_wi = start_wi + n_words

    delete_start = words[start_wi][1]
    delete_end = words[end_wi - 1][2]
    # Absorb one trailing whitespace if any (avoids "the  cat")
    if delete_end < len(text) and text[delete_end].isspace():
        delete_end += 1

    xprime_text = text[:delete_start] + text[delete_end:]
    if len(xprime_text.strip()) < 8:
        return _rej("too_short_xprime")

    x_ids, _ = gemma_tokenize(stage.gemma_tok, text)
    xp_ids, _ = gemma_tokenize(stage.gemma_tok, xprime_text)

    # ins_span_length is the number of Gemma tokens removed by the deletion.
    # Derive it directly from the token-count diff — this is robust to the
    # SentencePiece-style leading-space encoding (▁cat's char range starts
    # at the space BEFORE "cat", so an offset-based scan over the eaten
    # delete_end character range would over-reach into the next token and
    # break `len(editor_input) == len(editor_target)` for every INS sample).
    ins_span_length = len(x_ids) - len(xp_ids)
    if ins_span_length <= 0:
        return _rej("ins_span_nonpos")

    # Find the gap position in xp_ids by walking the matching prefix
    # between x_ids and xp_ids; the first divergence is where the deletion
    # happened. This is token-level — no offset arithmetic involved.
    gap_xp = 0
    upper = min(len(x_ids), len(xp_ids))
    while gap_xp < upper and x_ids[gap_xp] == xp_ids[gap_xp]:
        gap_xp += 1
    # Verify the suffix matches after skipping the deleted span — otherwise
    # the deletion did not produce a clean token-level transposition (rare;
    # punctuation-adjacent deletions or SentencePiece boundary shifts).
    if list(xp_ids[gap_xp:]) != list(x_ids[gap_xp + ins_span_length:]):
        return _rej("suffix_mismatch")

    # Construct training tuple
    tagger_gold = [OP_KEEP] * len(xp_ids)
    if gap_xp < len(tagger_gold):
        tagger_gold[gap_xp] = OP_INS
    elif tagger_gold:
        tagger_gold[-1] = OP_INS

    editor_input = list(xp_ids[:gap_xp]) + [stage.ins_id] * ins_span_length + list(xp_ids[gap_xp:])
    editor_target = list(x_ids)
    if len(editor_input) != len(editor_target):
        return _rej("length_mismatch")

    return {
        "corruption_type": "ins",
        "x_token_ids": x_ids,
        "x_prime_token_ids": xp_ids,
        "tagger_gold": tagger_gold,
        "editor_input_token_ids": editor_input,
        "editor_target_token_ids": editor_target,
        "ins_span_length": int(ins_span_length),
        "del_span_length": 0,
        "x_text": text,
        "x_prime_text": xprime_text,
    }


def make_del_sample(
    stage: Stage, text: str, rng: random.Random,
    word_span_max: int, mlm_topk: int, top1_prob: float,
) -> Optional[Dict]:
    """Insert MLM-predicted words after a chosen position; editor learns to drop them."""
    words = words_with_offsets(text)
    if len(words) < 5:
        return None
    insert_after_wi = rng.randint(0, len(words) - 2)
    insert_char = words[insert_after_wi][2]
    n_words = rng.randint(1, max(1, word_span_max))

    current_text = text
    char_shift = 0
    inserted_chars_start = insert_char
    for _ in range(n_words):
        pos = insert_char + char_shift
        masked_text = current_text[:pos] + " " + stage.mlm.mask_token + current_text[pos:]
        preds = stage.mlm.predict_at_masks(masked_text, top_k=mlm_topk)
        if not preds or not preds[0]:
            return None
        cands = [c for c in preds[0] if looks_like_word(c)]
        if not cands:
            return None
        w = cands[0] if rng.random() < top1_prob else cands[
            rng.randrange(min(len(cands), max(1, mlm_topk // 2)))
        ]
        ins_text = " " + w
        current_text = current_text[:pos] + ins_text + current_text[pos:]
        char_shift += len(ins_text)

    inserted_chars_end = insert_char + char_shift

    x_ids, x_offsets = gemma_tokenize(stage.gemma_tok, text)
    xp_ids, xp_offsets = gemma_tokenize(stage.gemma_tok, current_text)

    is_, ie_ = find_token_range(xp_offsets, inserted_chars_start, inserted_chars_end)
    if is_ is None or ie_ is None or ie_ == is_:
        return None
    del_span_length = ie_ - is_

    tagger_gold = [OP_KEEP] * len(xp_ids)
    editor_input = list(xp_ids)
    editor_target = list(xp_ids)
    for i in range(is_, ie_):
        if i < len(tagger_gold):
            tagger_gold[i] = OP_DEL
            editor_target[i] = stage.del_id

    return {
        "corruption_type": "del",
        "x_token_ids": x_ids,
        "x_prime_token_ids": xp_ids,
        "tagger_gold": tagger_gold,
        "editor_input_token_ids": editor_input,
        "editor_target_token_ids": editor_target,
        "ins_span_length": 0,
        "del_span_length": int(del_span_length),
        "x_text": text,
        "x_prime_text": current_text,
    }


def make_identity_sample(stage: Stage, text: str) -> Optional[Dict]:
    x_ids, _ = gemma_tokenize(stage.gemma_tok, text)
    if len(x_ids) < 3:
        return None
    return {
        "corruption_type": "identity",
        "x_token_ids": x_ids,
        "x_prime_token_ids": list(x_ids),
        "tagger_gold": [OP_KEEP] * len(x_ids),
        "editor_input_token_ids": list(x_ids),
        "editor_target_token_ids": list(x_ids),
        "ins_span_length": 0,
        "del_span_length": 0,
        "x_text": text,
        "x_prime_text": text,
    }


# ---------------------------------------------------------------------------
# Finalisation (PPL + SAE-shift filters + conditioning precompute)
# ---------------------------------------------------------------------------
def finalize_sample(
    stage: Stage, sample: Dict, source_id: str,
    ppl_max_ratio: float, sae_shift_threshold: float,
) -> Optional[Dict]:
    x_text = sample.get("x_text", "")
    xp_text = sample.get("x_prime_text", "")
    if not xp_text:
        return None

    if sample["corruption_type"] != "identity":
        ppl_clean = causal_perplexity_text(stage, x_text)
        ppl_corr = causal_perplexity_text(stage, xp_text)
        if not (math.isfinite(ppl_clean) and math.isfinite(ppl_corr)):
            return None
        if ppl_corr > ppl_max_ratio * ppl_clean:
            return None
    else:
        ppl_clean = ppl_corr = float("nan")

    fX, vX = sae_pool_max_topk_from_text(stage, x_text)
    fXp, vXp = sae_pool_max_topk_from_text(stage, xp_text)
    shift = sae_l2_shift(fX, vX, fXp, vXp)
    if sample["corruption_type"] != "identity" and shift < sae_shift_threshold:
        return None

    sample.pop("x_text", None)
    sample.pop("x_prime_text", None)
    sample.update({
        "source_sent_id": source_id,
        "z_X_topk": topk_records(fX, vX),
        "z_X_prime_topk": topk_records(fXp, vXp),
        "filter_telemetry": {
            "ppl_clean": ppl_clean,
            "ppl_ratio": (ppl_corr / ppl_clean) if math.isfinite(ppl_clean) else None,
            "sae_shift_l2": shift,
        },
    })
    return sample


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _normalize_probs(args) -> List[Tuple[str, float]]:
    weights = [
        ("identity", args.p_identity),
        ("repl", args.p_repl),
        ("ins", args.p_ins),
        ("del", args.p_del),
        ("mixed_repl_ins", args.p_mixed_repl_ins),
        ("mixed_repl_del", args.p_mixed_repl_del),
    ]
    total = sum(w for _, w in weights)
    if total <= 0:
        raise ValueError("all bucket probabilities are zero")
    return [(k, w / total) for k, w in weights if w > 0]


def pick_bucket(buckets: List[Tuple[str, float]], rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for name, p in buckets:
        cum += p
        if r < cum:
            return name
    return buckets[-1][0]


def _str_dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corruption] loading downstream tokenizer + causal Gemma from {args.llm2vec_dir}")
    causal_llm, gemma_tok = load_causal_gemma(args.llm2vec_dir)
    causal_llm = causal_llm.to(args.device)

    print(f"[corruption] loading SAE extractor (frozen Gemma + Gemma Scope)")
    extractor = SAEFeatureExtractor(
        llm_name=args.llm,
        sae_repo=args.sae_repo,
        sae_path=args.sae_path,
        sae_layer=args.sae_layer,
        sae_type=args.sae_type,
        sae_k=args.sae_k,
    ).to(args.device)
    extractor.eval()

    print(f"[corruption] loading corruption MLM: {args.mlm_model}")
    mlm = MLMProvider(args.mlm_model, dtype=_str_dtype(args.mlm_dtype)).to(args.device)
    print(f"[corruption] resolved MLM: {mlm.resolved_name}  mask={mlm.mask_token!r}")

    stage = Stage(
        extractor=extractor, causal_llm=causal_llm, mlm=mlm,
        gemma_tok=gemma_tok,
        mask_id=int(gemma_tok.mask_token_id),
        ins_id=int(gemma_tok.convert_tokens_to_ids("[INS]")),
        del_id=int(gemma_tok.convert_tokens_to_ids("[DEL]")),
        pad_id=int(gemma_tok.pad_token_id),
        device=args.device, k_train=int(args.k_train),
    )

    buckets = _normalize_probs(args)
    print(f"[corruption] sample buckets: {buckets}")

    shard_paths = download_dolma_shards(args.data_cache_dir, max_files=args.max_files)
    text_iter = iter_dolma_texts(shard_paths, min_chars=64)
    quality_kwargs = {
        "min_words": args.quality_min_words,
        "min_alpha_ratio": args.quality_min_alpha_ratio,
        "require_terminal_punct": args.quality_require_terminal_punct,
        "require_initial_capital": args.quality_require_initial_capital,
    }
    sent_iter = iter_sentences(
        text_iter, splitter=args.sentence_splitter,
        min_chars=16, max_chars=2000,
        max_sentences_per_text=args.max_sentences_per_text,
        sample_strategy=args.sentence_sample_strategy,
        seed=args.seed,
        quality_filter=not args.no_quality_filter,
        quality_kwargs=quality_kwargs,
    )

    written = 0
    shard_idx = 0
    sent_idx = 0
    attempted = 0
    bucket_attempts: Dict[str, int] = defaultdict(int)
    bucket_accepts: Dict[str, int] = defaultdict(int)
    ins_reject_counter: Dict[str, int] = defaultdict(int)
    cur_shard_file = None

    def open_shard():
        nonlocal cur_shard_file, shard_idx
        path = out_dir / f"shard-{shard_idx:05d}.jsonl.gz"
        cur_shard_file = gzip.open(path, "wt", encoding="utf-8")
        shard_idx += 1

    open_shard()

    for sent in sent_iter:
        if written >= args.target_samples:
            break
        gemma_token_count = len(gemma_tok(sent, add_special_tokens=False).input_ids)
        if not (args.sent_min_tokens <= gemma_token_count <= args.sent_max_tokens):
            continue
        sent_idx += 1
        for _ in range(args.reject_budget):
            attempted += 1
            bucket = pick_bucket(buckets, rng)
            bucket_attempts[bucket] += 1
            sample: Optional[Dict] = None
            if bucket == "identity":
                sample = make_identity_sample(stage, sent)
            elif bucket == "repl":
                sample = make_repl_sample(
                    stage, sent, rng, args.repl_words_max, args.repl_mlm_topk,
                )
            elif bucket == "ins":
                sample = make_ins_sample(
                    stage, sent, rng,
                    args.ins_word_span_max,
                    reject_counter=ins_reject_counter,
                )
            elif bucket == "del":
                sample = make_del_sample(
                    stage, sent, rng,
                    args.del_word_span_max, args.del_mlm_topk, args.del_top1_prob,
                )
            elif bucket == "mixed_repl_ins":
                mid = make_repl_sample(stage, sent, rng,
                                       args.repl_words_max, args.repl_mlm_topk)
                if mid is None:
                    continue
                # Use the corrupted text as the new clean for INS
                sample = make_ins_sample(
                    stage, mid["x_prime_text"], rng,
                    args.ins_word_span_max,
                    reject_counter=ins_reject_counter,
                )
                if sample is not None:
                    sample["corruption_type"] = "mixed_repl_ins"
            elif bucket == "mixed_repl_del":
                mid = make_repl_sample(stage, sent, rng,
                                       args.repl_words_max, args.repl_mlm_topk)
                if mid is None:
                    continue
                sample = make_del_sample(
                    stage, mid["x_prime_text"], rng,
                    args.del_word_span_max, args.del_mlm_topk, args.del_top1_prob,
                )
                if sample is not None:
                    sample["corruption_type"] = "mixed_repl_del"

            if sample is None:
                continue
            final = finalize_sample(
                stage, sample, source_id=f"dolma:s{sent_idx}",
                ppl_max_ratio=args.ppl_max_ratio,
                sae_shift_threshold=args.sae_shift_threshold,
            )
            if final is None:
                continue
            cur_shard_file.write(json.dumps(final, ensure_ascii=False) + "\n")
            written += 1
            bucket_accepts[bucket] += 1
            if written % args.samples_per_shard == 0 and written < args.target_samples:
                cur_shard_file.close()
                open_shard()
                yield_ = written / max(1, attempted)
                per_bucket = ", ".join(
                    f"{name}={bucket_accepts[name]}/"
                    f"{bucket_attempts[name]}="
                    f"{bucket_accepts[name] / max(1, bucket_attempts[name]):.2f}"
                    for name, _ in buckets
                )
                print(
                    f"[corruption] written={written} sents={sent_idx} "
                    f"attempts={attempted} yield={yield_:.3f}  ({per_bucket})"
                )
                ins_total = sum(ins_reject_counter.values())
                if ins_total:
                    ins_breakdown = ", ".join(
                        f"{r}={ins_reject_counter[r]}"
                        f"({100.0 * ins_reject_counter[r] / ins_total:.0f}%)"
                        for r in INS_REJECT_REASONS
                        if ins_reject_counter[r] > 0
                    )
                    print(f"[corruption] INS reject reasons: {ins_breakdown}")
            break

    if cur_shard_file is not None:
        cur_shard_file.close()

    meta = {
        "samples_written": int(written),
        "sentences_seen": int(sent_idx),
        "attempts": int(attempted),
        "yield": float(written / max(1, attempted)),
        "bucket_attempts": {k: int(v) for k, v in bucket_attempts.items()},
        "bucket_accepts": {k: int(v) for k, v in bucket_accepts.items()},
        "bucket_yields": {
            k: float(bucket_accepts[k] / max(1, bucket_attempts[k]))
            for k in bucket_attempts
        },
        "ins_reject_reasons": {k: int(ins_reject_counter[k]) for k in INS_REJECT_REASONS},
        "d_sae": int(extractor.d_sae),
        "k_train": int(args.k_train),
        "mask_id": int(stage.mask_id),
        "ins_id": int(stage.ins_id),
        "del_id": int(stage.del_id),
        "pad_id": int(stage.pad_id),
        "llm2vec_dir": args.llm2vec_dir,
        "mlm_model": args.mlm_model,
        "mlm_resolved": mlm.resolved_name,
        "sae_repo": args.sae_repo,
        "sae_path": args.sae_path,
        "sae_layer": int(args.sae_layer),
        "seed": int(args.seed),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[corruption] done: {written} samples in {shard_idx} shards → {out_dir}")


if __name__ == "__main__":
    main()
