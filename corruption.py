"""
Stage 2: corruption data generation for SAE-LEWIS.

Produces a sharded JSON-lines corruption cache (see README §8.2) by applying
fluency-preserving, MLM-based corruption to sentence-segmented Dolma.

**Compound corruption (§6.2.5).** Each accepted sample combines N ops drawn
from {REPL, INS, DEL} on a single clean sentence:

  REPL : substitute words with MLM-predicted alternatives
  INS  : delete words whose context can recover them (POS-priority biased)
  DEL  : insert MLM-predicted words that the editor must drop

N is drawn from a truncated geometric, capped at N_MAX. Ops have
non-overlapping word ranges on the original X. First-accept rejection
within K_BUDGET attempts per source sentence preserves the natural
post-gate distribution (no PPL-ranking bias).

**N-dependent gates (§6.2.6).** PPL ratio and SAE-shift thresholds scale
with sqrt(N):

  ppl_max(N) = ppl_per_op_factor ** sqrt(N)
  sae_min(N) = sae_per_op_min     * sqrt(N)
  sae_max(N) = sae_per_op_max     * sqrt(N)   (minimality upper bound)

The scale constants are calibrated against the empirical Dolma + MLM
compound distribution via `--calibration-mode`, which records every
attempt's (N, ppl_ratio, sae_shift) without applying the gate.

**Position selection (§6.2.3).** INS deletion positions are biased toward
syntactic categories where removal is least disruptive: spaCy Universal
POS tags ADJ, ADV, DET, ADP, CCONJ, SCONJ, AUX, PART, plus the sentence-
initial word. REPL position selection is unbiased (top-K MLM is the
implicit quality gate). DEL position is MLM-driven.

Corruption operates at the WORD / TEXT level. The MLM (any HF AutoModel-
ForMaskedLM via `model.MLMProvider`) is decoupled from the downstream
editor / tagger encoder: the MLM's tokenizer is encapsulated and the
final training sample is re-tokenised with the downstream Gemma
tokenizer at the end.

The SAE forward and the perplexity scorer use Gemma; only the MLM that
proposes corruption candidates is swappable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

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

    # spaCy POS tagger (UPOS)
    p.add_argument("--spacy-model", default="en_core_web_sm",
                   help="spaCy model for POS tagging. UPOS tags are language-"
                        "agnostic, so switching to ja_core_news_sm / "
                        "de_core_news_sm / ... is enough for other languages.")

    # Sentence segmentation
    p.add_argument("--sentence-splitter", choices=["pysbd", "nltk"], default="pysbd")
    p.add_argument("--sent-min-tokens", type=int, default=5)
    p.add_argument("--sent-max-tokens", type=int, default=256)
    p.add_argument("--max-sentences-per-text", type=int, default=None)
    p.add_argument("--sentence-sample-strategy",
                   choices=["head", "random", "stride"], default="head")
    p.add_argument("--no-quality-filter", action="store_true")
    p.add_argument("--quality-min-words", type=int, default=3)
    p.add_argument("--quality-min-alpha-ratio", type=float, default=0.5)
    p.add_argument("--quality-require-terminal-punct", action="store_true", default=True)
    p.add_argument("--quality-require-initial-capital", action="store_true", default=False)

    # Output
    p.add_argument("--target-samples", type=int, default=100000)
    p.add_argument("--samples-per-shard", type=int, default=10000)
    p.add_argument("--k-budget", type=int, default=6,
                   help="First-accept rejection attempts per source sentence "
                        "(§6.2.5).")

    # Sample bucket weights (§6.3.1). Buckets are determined post-hoc by
    # realised N from the compound generator. We rejection-sample on N to
    # match the target weights.
    p.add_argument("--p-identity", type=float, default=0.10)
    p.add_argument("--p-single-op", type=float, default=0.15)
    p.add_argument("--p-compound-2-3", type=float, default=0.45)
    p.add_argument("--p-compound-4-plus", type=float, default=0.30)

    # Compound op sampling (§6.2.5)
    p.add_argument("--n-max", type=int, default=5,
                   help="Cap on op count per compound sample.")
    p.add_argument("--n-distribution-p", type=float, default=0.4,
                   help="Truncated geometric parameter for N over {0..N_MAX}.")
    p.add_argument("--op-weight-repl", type=float, default=0.55)
    p.add_argument("--op-weight-ins", type=float, default=0.25)
    p.add_argument("--op-weight-del", type=float, default=0.20)
    p.add_argument("--op-position-max-retries", type=int, default=20,
                   help="Per-op attempts to find a non-conflicting position "
                        "before giving up on the compound.")

    # Per-op span / MLM knobs
    p.add_argument("--repl-mlm-topk", type=int, default=8)
    p.add_argument("--ins-word-span-max", type=int, default=3,
                   help="Max # of consecutive words to delete per INS op.")
    p.add_argument("--ins-p-high", type=float, default=0.85,
                   help="Probability of sampling INS span from HIGH-priority "
                        "POS positions (§6.2.3).")
    p.add_argument("--del-word-span-max", type=int, default=2,
                   help="Max # of consecutive words to insert per DEL op.")
    p.add_argument("--del-mlm-topk", type=int, default=8)
    p.add_argument("--del-top1-prob", type=float, default=0.5)

    # N-dependent gates (§6.2.6)
    p.add_argument("--ppl-per-op-factor", type=float, default=1.8,
                   help="ppl_max(N) = ppl_per_op_factor ** sqrt(N).")
    p.add_argument("--sae-per-op-min", type=float, default=0.30,
                   help="sae_min(N) = sae_per_op_min * sqrt(N).")
    p.add_argument("--sae-per-op-max", type=float, default=2.50,
                   help="sae_max(N) = sae_per_op_max * sqrt(N) (minimality "
                        "upper bound).")
    p.add_argument("--calibration-mode", action="store_true",
                   help="Skip the PPL/SAE-shift gate; record every attempt's "
                        "(N, ppl_ratio, sae_shift) to a JSONL file for "
                        "percentile fitting.")
    p.add_argument("--calibration-out",
                   help="Path to JSONL of (N, ppl_ratio, sae_shift) records "
                        "in --calibration-mode. Defaults to "
                        "<out-dir>/calibration.jsonl.")

    p.add_argument("--k-train", type=int, default=64,
                   help="Top-K for SAE pool-max conditioning.")

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
    spacy_nlp: any
    mask_id: int           # downstream (Gemma) mask id
    ins_id: int
    del_id: int
    pad_id: int
    device: str
    k_train: int


# ---------------------------------------------------------------------------
# Text / word utilities (kept from earlier version)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\S+")


def words_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Whitespace-tokenized words with (token_str, char_start, char_end)."""
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


def find_token_at_char(
    offsets: List[Tuple[int, int]],
    char_pos: int,
) -> Optional[int]:
    """Return the index of the token whose char range starts at or after `char_pos`.

    Used for finding the INS gap marker position (zero-width gap in xp_text).
    """
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if s >= char_pos:
            return i
    return None  # gap at end of sequence


def looks_like_word(s: str) -> bool:
    s = s.strip()
    return bool(s) and any(c.isalpha() for c in s) and not s.startswith(("##", "▁"))


# ---------------------------------------------------------------------------
# spaCy POS tagger (UPOS)
# ---------------------------------------------------------------------------
HIGH_PRIORITY_UPOS: Set[str] = {
    "ADJ", "ADV", "DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART",
}


def load_spacy(model_name: str):
    """Load spaCy with POS-only pipeline (parser/ner/lemmatizer disabled).

    Auto-downloads the model if not present.
    """
    import spacy
    try:
        nlp = spacy.load(
            model_name,
            # `attribute_ruler` must stay enabled — it's what maps tagger
            # output (token.tag_, PTB style) to UPOS (token.pos_), which is
            # what we read for the HIGH/LOW priority decision.
            disable=["parser", "ner", "lemmatizer"],
        )
    except OSError:
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(
            model_name,
            # `attribute_ruler` must stay enabled — it's what maps tagger
            # output (token.tag_, PTB style) to UPOS (token.pos_), which is
            # what we read for the HIGH/LOW priority decision.
            disable=["parser", "ner", "lemmatizer"],
        )
    return nlp


def upos_priority_for_words(
    nlp,
    text: str,
    words: List[Tuple[str, int, int]],
) -> List[str]:
    """Return ["HIGH" or "LOW"] per entry in `words`.

    A word is HIGH if any spaCy token overlapping its char span has a
    HIGH-priority UPOS tag, or if it is the sentence-initial word.
    """
    doc = nlp(text)
    # spaCy tokens with their UPOS; doc.token.idx is the char start.
    spacy_tokens: List[Tuple[int, int, str]] = []
    for t in doc:
        if t.is_space:
            continue
        cs = t.idx
        ce = t.idx + len(t.text)
        spacy_tokens.append((cs, ce, t.pos_))

    priorities: List[str] = []
    for wi, (_, ws, we) in enumerate(words):
        if wi == 0:
            priorities.append("HIGH")              # sentence-initial
            continue
        upos_set = set()
        for cs, ce, pos in spacy_tokens:
            if cs >= we:
                break
            if ce <= ws:
                continue
            upos_set.add(pos)
        prio = "HIGH" if upos_set & HIGH_PRIORITY_UPOS else "LOW"
        priorities.append(prio)
    return priorities


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
# OpSpec + non-overlap helpers
# ---------------------------------------------------------------------------
@dataclass
class OpSpec:
    """A single op within a compound corruption.

    Coordinates are word indices on the ORIGINAL X. Final char positions in
    xp_text are computed at apply time via cumulative shifts.
    """
    op_type: str                   # "repl" | "ins" | "del"
    word_start: int                # for REPL/INS: start of span; for DEL: word AFTER which to insert (== insertion point)
    word_end: int                  # for REPL/INS: exclusive end of span; for DEL: == word_start (zero-width)
    payload: Optional[str] = None  # REPL: replacement text; DEL: insertion text; INS: None


def claim_slots(op: OpSpec, claims: Set[Tuple[str, int]]) -> None:
    """Add `op`'s exclusion claims to the running `claims` set."""
    if op.op_type == "repl":
        claims.add(("word", op.word_start))
    elif op.op_type == "ins":
        for w in range(op.word_start, op.word_end):
            claims.add(("word", w))
        claims.add(("ins_gap_marker", op.word_end))
        # No DEL may insert at any word position in [word_start, word_end]:
        for w in range(op.word_start, op.word_end + 1):
            claims.add(("ins_forbid_del", w))
    elif op.op_type == "del":
        claims.add(("del_boundary", op.word_start))
        claims.add(("word", op.word_start))             # forbid REPL/INS on that word
        claims.add(("ins_gap_marker", op.word_start))   # forbid INS marker landing here


def can_add_op(op: OpSpec, claims: Set[Tuple[str, int]]) -> bool:
    """Check if `op` can be added without conflicting with existing `claims`."""
    if op.op_type == "repl":
        for w in range(op.word_start, op.word_end):
            if ("word", w) in claims:
                return False
            if ("ins_gap_marker", w) in claims:
                return False
        return True
    elif op.op_type == "ins":
        for w in range(op.word_start, op.word_end):
            if ("word", w) in claims:
                return False
            if ("ins_gap_marker", w) in claims:
                return False
        if ("word", op.word_end) in claims:
            return False
        if ("ins_gap_marker", op.word_end) in claims:
            return False
        for w in range(op.word_start, op.word_end + 1):
            if ("ins_forbid_del", w) in claims:
                return False
        return True
    elif op.op_type == "del":
        if ("del_boundary", op.word_start) in claims:
            return False
        if ("word", op.word_start) in claims:
            return False
        if ("ins_gap_marker", op.word_start) in claims:
            return False
        if ("ins_forbid_del", op.word_start) in claims:
            return False
        return True
    raise ValueError(f"unknown op_type {op.op_type!r}")


# ---------------------------------------------------------------------------
# Per-op proposers
# ---------------------------------------------------------------------------
def _norm(w: str) -> str:
    return w.strip().lower()


def propose_repl_op(
    stage: Stage, text: str, words: List[Tuple[str, int, int]],
    rng: random.Random, claims: Set[Tuple[str, int]],
    mlm_topk: int, max_retries: int,
) -> Optional[OpSpec]:
    """Sample one REPL op on a random unclaimed word, with MLM-generated replacement."""
    for _ in range(max_retries):
        candidates = [
            wi for wi in range(len(words))
            if can_add_op(OpSpec("repl", wi, wi + 1), claims)
        ]
        if not candidates:
            return None
        wi = rng.choice(candidates)
        orig_word, ws, we = words[wi]
        masked_text = text[:ws] + stage.mlm.mask_token + text[we:]
        preds = stage.mlm.predict_at_masks(masked_text, top_k=mlm_topk)
        if not preds or not preds[0]:
            continue
        cands = [c for c in preds[0]
                 if looks_like_word(c) and _norm(c) != _norm(orig_word)
                 and len(c) <= max(20, len(orig_word) * 3)]
        if not cands:
            continue
        replacement = cands[rng.randrange(min(len(cands), max(1, mlm_topk // 2)))]
        return OpSpec("repl", wi, wi + 1, payload=replacement)
    return None


def propose_ins_op(
    stage: Stage, text: str, words: List[Tuple[str, int, int]],
    priorities: List[str], rng: random.Random, claims: Set[Tuple[str, int]],
    word_span_max: int, p_high: float, max_retries: int,
) -> Optional[OpSpec]:
    """Sample one INS op with priority-biased word selection (§6.2.3)."""
    # Enumerate candidate (start, span) pairs that satisfy the placement rules.
    # Leave at least 1 word on each side: start >= 1 AND start + span <= len-1.
    high_cands: List[Tuple[int, int]] = []
    low_cands: List[Tuple[int, int]] = []
    for start in range(1, len(words) - 1):
        for span in range(1, min(word_span_max, len(words) - start - 1) + 1):
            op = OpSpec("ins", start, start + span)
            if not can_add_op(op, claims):
                continue
            # Priority is based on the FIRST word of the span (sentence-initial
            # is always HIGH and only applies to wi==0, which we exclude).
            prio = priorities[start]
            if prio == "HIGH":
                high_cands.append((start, span))
            else:
                low_cands.append((start, span))
    pool: Optional[List[Tuple[int, int]]] = None
    if rng.random() < p_high and high_cands:
        pool = high_cands
    elif low_cands:
        pool = low_cands
    elif high_cands:
        pool = high_cands
    if not pool:
        return None
    start, span = rng.choice(pool)
    return OpSpec("ins", start, start + span, payload=None)


def propose_del_op(
    stage: Stage, text: str, words: List[Tuple[str, int, int]],
    rng: random.Random, claims: Set[Tuple[str, int]],
    word_span_max: int, mlm_topk: int, top1_prob: float, max_retries: int,
) -> Optional[OpSpec]:
    """Sample one DEL op: pick an insertion point and grow a MLM-driven payload."""
    for _ in range(max_retries):
        # DEL inserts BEFORE words[insert_at]; require insert_at >= 1 (skip
        # sentence-initial slot for now) and < len(words).
        candidates = [
            wi for wi in range(1, len(words))
            if can_add_op(OpSpec("del", wi, wi), claims)
        ]
        if not candidates:
            return None
        insert_at = rng.choice(candidates)
        n_words = rng.randint(1, max(1, word_span_max))
        # Build payload iteratively by masking BEFORE words[insert_at].
        # We do this on a copy of `text` (independent of other ops, which
        # are applied at compound-apply time).
        current = text
        char_anchor = words[insert_at][1]
        # Inserted text grows: each iteration prepends "{w} " at char_anchor.
        for _step in range(n_words):
            masked = current[:char_anchor] + stage.mlm.mask_token + " " + current[char_anchor:]
            preds = stage.mlm.predict_at_masks(masked, top_k=mlm_topk)
            if not preds or not preds[0]:
                break
            cands = [c for c in preds[0] if looks_like_word(c)]
            if not cands:
                break
            w = cands[0] if rng.random() < top1_prob else cands[
                rng.randrange(min(len(cands), max(1, mlm_topk // 2)))
            ]
            # Insert the new word + space at char_anchor; subsequent iteration
            # will mask before the same anchor → words accumulate in order of
            # appearance from left.
            current = current[:char_anchor] + w + " " + current[char_anchor:]
            char_anchor += len(w) + 1
        # The payload is the joined sequence of words (without trailing space);
        # apply_compound adds the trailing space.
        payload = current[words[insert_at][1]:char_anchor].rstrip()
        if not payload:
            continue
        return OpSpec("del", insert_at, insert_at, payload=payload)
    return None


# ---------------------------------------------------------------------------
# Text-level compound application
# ---------------------------------------------------------------------------
def length_change(op: OpSpec, text: str, words: List[Tuple[str, int, int]]) -> int:
    """Net char length contribution of `op` to xp_text."""
    if op.op_type == "repl":
        cs = words[op.word_start][1]
        ce = words[op.word_end - 1][2]
        return len(op.payload) - (ce - cs)
    elif op.op_type == "ins":
        cs = words[op.word_start][1]
        ce = words[op.word_end - 1][2]
        # Trailing whitespace absorption (matches apply logic).
        if ce < len(text) and text[ce].isspace():
            ce += 1
        return -(ce - cs)
    elif op.op_type == "del":
        return len(op.payload) + 1   # payload + trailing space
    raise ValueError(f"unknown op_type {op.op_type!r}")


def apply_compound_to_text(
    text: str,
    words: List[Tuple[str, int, int]],
    ops: List[OpSpec],
) -> Tuple[str, Dict[int, int]]:
    """Apply `ops` to `text` and return (xp_text, final_pos_by_op_index).

    `final_pos_by_op_index[i]` is the char position in `xp_text` where op `i`
    in the input list starts contributing its content (REPL/DEL) or where
    its gap lives (INS, zero-width).

    Implementation: build xp_text left-to-right by interleaving unchanged
    `text` slices with each op's content/skip, sorted by original char
    position. Same-position tiebreaker not needed here because the
    non-overlap rules forbid two ops at the same char anchor.
    """
    # Tag each op with its ORIGINAL anchor char position and the index for
    # the result map. For DEL/INS the anchor is the start of word_start;
    # for REPL the anchor is the start of word_start as well.
    indexed = list(enumerate(ops))
    indexed.sort(key=lambda iop: words[iop[1].word_start][1])

    pieces: List[str] = []
    cursor = 0
    cum_shift = 0
    final_pos: Dict[int, int] = {}

    for idx, op in indexed:
        orig_anchor = words[op.word_start][1]
        final_pos[idx] = orig_anchor + cum_shift
        if op.op_type == "repl":
            cs = words[op.word_start][1]
            ce = words[op.word_end - 1][2]
            pieces.append(text[cursor:cs])
            pieces.append(op.payload)
            cursor = ce
            cum_shift += len(op.payload) - (ce - cs)
        elif op.op_type == "ins":
            cs = words[op.word_start][1]
            ce = words[op.word_end - 1][2]
            # Match length_change(): absorb one trailing whitespace if any.
            if ce < len(text) and text[ce].isspace():
                ce += 1
            pieces.append(text[cursor:cs])
            cursor = ce          # skip deleted chars
            cum_shift -= (ce - cs)
        elif op.op_type == "del":
            pos = words[op.word_start][1]
            pieces.append(text[cursor:pos])
            ins_text = op.payload + " "
            pieces.append(ins_text)
            cursor = pos          # don't consume any original char
            cum_shift += len(ins_text)
        else:
            raise ValueError(f"unknown op_type {op.op_type!r}")
    pieces.append(text[cursor:])
    return "".join(pieces), final_pos


# ---------------------------------------------------------------------------
# Token-level gold construction from compound
# ---------------------------------------------------------------------------
@dataclass
class OpTokenRanges:
    """Token-level coordinates for one op after compound application."""
    op_type: str
    # For REPL/INS: token range covering the original word span in x_ids
    x_tok_start: Optional[int] = None
    x_tok_end: Optional[int] = None
    # For REPL/DEL: token range covering the op's content in xp_ids
    xp_tok_start: Optional[int] = None
    xp_tok_end: Optional[int] = None
    # For INS: gap position in xp_ids (token index where marker lands)
    xp_gap_pos: Optional[int] = None


def resolve_token_ranges(
    ops: List[OpSpec],
    final_pos: Dict[int, int],
    text: str,
    xp_text: str,
    words: List[Tuple[str, int, int]],
    x_offsets: List[Tuple[int, int]],
    xp_offsets: List[Tuple[int, int]],
) -> Optional[List[OpTokenRanges]]:
    """Map each op's char spans to token ranges in x_ids / xp_ids.

    Returns `None` if any op fails to resolve (e.g. REPL token count mismatch,
    INS span boundary not found cleanly, etc.) — caller will reject the
    compound sample.
    """
    out: List[OpTokenRanges] = []
    for idx, op in enumerate(ops):
        rec = OpTokenRanges(op_type=op.op_type)
        if op.op_type == "repl":
            # Original word span in X
            orig_cs = words[op.word_start][1]
            orig_ce = words[op.word_end - 1][2]
            xs, xe = find_token_range(x_offsets, orig_cs, orig_ce)
            if xs is None or xe is None:
                return None
            # Final span in xp_text
            fcs = final_pos[idx]
            fce = fcs + len(op.payload)
            ps, pe = find_token_range(xp_offsets, fcs, fce)
            if ps is None or pe is None:
                return None
            if (pe - ps) != (xe - xs):
                return None  # same-token-count constraint violated
            rec.x_tok_start, rec.x_tok_end = xs, xe
            rec.xp_tok_start, rec.xp_tok_end = ps, pe
        elif op.op_type == "ins":
            orig_cs = words[op.word_start][1]
            orig_ce = words[op.word_end - 1][2]
            # Note: don't include the absorbed trailing space in the x_ids
            # token range; it belongs to the NEXT word's token anyway under
            # SentencePiece.
            xs, xe = find_token_range(x_offsets, orig_cs, orig_ce)
            if xs is None or xe is None:
                return None
            rec.x_tok_start, rec.x_tok_end = xs, xe
            # Gap position in xp_text: at the char where the deletion landed.
            gap_char = final_pos[idx]
            gap_tok = find_token_at_char(xp_offsets, gap_char)
            if gap_tok is None:
                # Gap at the very end of xp_ids — this shouldn't happen since
                # we leave at least one word after the deletion, but if it
                # does, we fail rather than silently corrupting the gold.
                return None
            rec.xp_gap_pos = gap_tok
        elif op.op_type == "del":
            fcs = final_pos[idx]
            # DEL payload (without trailing space) sits at [fcs, fcs + len)
            fce = fcs + len(op.payload)
            ps, pe = find_token_range(xp_offsets, fcs, fce)
            if ps is None or pe is None:
                return None
            rec.xp_tok_start, rec.xp_tok_end = ps, pe
        else:
            return None
        out.append(rec)
    return out


def build_compound_gold(
    x_ids: List[int],
    xp_ids: List[int],
    ops: List[OpSpec],
    token_ranges: List[OpTokenRanges],
    mask_id: int,
    ins_id: int,
    del_id: int,
) -> Optional[Dict]:
    """Build tagger_gold, editor_input, editor_target from the compound.

    Returns None if alignment fails (an asserted invariant breaks); the
    sample is then rejected.
    """
    # Sort ops by their xp anchor: REPL/DEL use xp_tok_start; INS uses xp_gap_pos.
    def xp_anchor(rec: OpTokenRanges) -> int:
        if rec.op_type == "ins":
            return rec.xp_gap_pos
        return rec.xp_tok_start

    paired = sorted(
        enumerate(token_ranges),
        key=lambda iop: xp_anchor(iop[1]),
    )

    tagger_gold: List[int] = []
    editor_input: List[int] = []
    editor_target: List[int] = []
    ins_span_lengths: List[int] = []
    del_span_lengths: List[int] = []

    i_x = 0
    i_xp = 0
    pending_ins = False        # next emitted xp position should carry OP_INS

    for _, rec in paired:
        target_xp = xp_anchor(rec)
        # Emit KEEP segment up to target_xp.
        while i_xp < target_xp:
            if i_x >= len(x_ids):
                return None
            tagger_gold.append(OP_INS if pending_ins else OP_KEEP)
            editor_input.append(xp_ids[i_xp])
            editor_target.append(x_ids[i_x])
            if pending_ins:
                pending_ins = False
            i_xp += 1
            i_x += 1

        if rec.op_type == "repl":
            assert rec.xp_tok_start is not None and rec.xp_tok_end is not None
            assert rec.x_tok_start is not None and rec.x_tok_end is not None
            L = rec.xp_tok_end - rec.xp_tok_start
            if pending_ins:
                # An INS gap immediately before a REPL violates non-overlap
                # rules; treat as a hard failure for safety.
                return None
            for k in range(L):
                tagger_gold.append(OP_REPL)
                editor_input.append(mask_id)
                editor_target.append(x_ids[rec.x_tok_start + k])
            i_xp = rec.xp_tok_end
            i_x = rec.x_tok_end
        elif rec.op_type == "ins":
            assert rec.x_tok_start is not None and rec.x_tok_end is not None
            L = rec.x_tok_end - rec.x_tok_start
            ins_span_lengths.append(L)
            for k in range(L):
                editor_input.append(ins_id)
                editor_target.append(x_ids[rec.x_tok_start + k])
            i_x = rec.x_tok_end
            pending_ins = True
        elif rec.op_type == "del":
            assert rec.xp_tok_start is not None and rec.xp_tok_end is not None
            L = rec.xp_tok_end - rec.xp_tok_start
            if pending_ins:
                return None
            del_span_lengths.append(L)
            for k in range(L):
                tagger_gold.append(OP_DEL)
                editor_input.append(xp_ids[rec.xp_tok_start + k])
                editor_target.append(del_id)
            i_xp = rec.xp_tok_end
            # i_x doesn't advance (DEL doesn't consume x_ids)

    # Trailing KEEP segment.
    while i_xp < len(xp_ids):
        if i_x >= len(x_ids):
            return None
        tagger_gold.append(OP_INS if pending_ins else OP_KEEP)
        editor_input.append(xp_ids[i_xp])
        editor_target.append(x_ids[i_x])
        if pending_ins:
            pending_ins = False
        i_xp += 1
        i_x += 1

    if pending_ins:
        # INS gap at the end of xp_ids — should not happen with our placement
        # rules (we require at least one word after every INS). Fail safely.
        return None

    # Final consistency checks.
    if i_x != len(x_ids):
        return None
    if len(tagger_gold) != len(xp_ids):
        return None
    if len(editor_input) != len(editor_target):
        return None

    return {
        "tagger_gold": tagger_gold,
        "editor_input_token_ids": editor_input,
        "editor_target_token_ids": editor_target,
        "ins_span_lengths": ins_span_lengths,
        "del_span_lengths": del_span_lengths,
    }


# ---------------------------------------------------------------------------
# Compound sample builder
# ---------------------------------------------------------------------------
COMPOUND_REJECT_REASONS = (
    "too_short_sentence",     # < threshold word count
    "ops_unsuitable",         # could not propose N non-conflicting ops
    "apply_failed",           # text-level application failed (rare)
    "token_align_failed",     # find_token_range failed for some op
    "repl_token_count_mismatch",
    "gold_build_failed",      # tagger/editor gold construction failed
    "ppl_inf",
    "ppl_too_high",
    "sae_shift_too_small",
    "sae_shift_too_large",
    "empty_xprime_text",
)


def sample_op_types(
    rng: random.Random, N: int,
    w_repl: float, w_ins: float, w_del: float,
) -> List[str]:
    types = ("repl", "ins", "del")
    weights = (w_repl, w_ins, w_del)
    total = sum(weights)
    norm = [w / total for w in weights]
    out: List[str] = []
    for _ in range(N):
        r = rng.random()
        cum = 0.0
        choice = types[-1]
        for t, w in zip(types, norm):
            cum += w
            if r < cum:
                choice = t
                break
        out.append(choice)
    return out


def sample_compound(
    stage: Stage, text: str, rng: random.Random,
    N: int, args,
) -> Tuple[Optional[List[OpSpec]], str]:
    """Propose N non-overlapping ops on `text` with MLM-generated payloads.

    Returns (ops, "") on success or (None, reason) on failure.
    """
    words = words_with_offsets(text)
    if len(words) < 3:
        return None, "too_short_sentence"
    priorities = upos_priority_for_words(stage.spacy_nlp, text, words)

    op_types = sample_op_types(
        rng, N, args.op_weight_repl, args.op_weight_ins, args.op_weight_del,
    )
    claims: Set[Tuple[str, int]] = set()
    ops: List[OpSpec] = []
    for t in op_types:
        op: Optional[OpSpec] = None
        if t == "repl":
            op = propose_repl_op(
                stage, text, words, rng, claims,
                mlm_topk=args.repl_mlm_topk,
                max_retries=args.op_position_max_retries,
            )
        elif t == "ins":
            op = propose_ins_op(
                stage, text, words, priorities, rng, claims,
                word_span_max=args.ins_word_span_max,
                p_high=args.ins_p_high,
                max_retries=args.op_position_max_retries,
            )
        elif t == "del":
            op = propose_del_op(
                stage, text, words, rng, claims,
                word_span_max=args.del_word_span_max,
                mlm_topk=args.del_mlm_topk,
                top1_prob=args.del_top1_prob,
                max_retries=args.op_position_max_retries,
            )
        if op is None:
            return None, "ops_unsuitable"
        claim_slots(op, claims)
        ops.append(op)
    return ops, ""


def build_compound_sample(
    stage: Stage, text: str, ops: List[OpSpec],
) -> Tuple[Optional[Dict], str]:
    """Apply `ops` to `text`, build the training sample. Returns (sample, "")
    or (None, reason)."""
    words = words_with_offsets(text)
    try:
        xp_text, final_pos = apply_compound_to_text(text, words, ops)
    except Exception:
        return None, "apply_failed"
    if not xp_text:
        return None, "empty_xprime_text"

    x_ids, x_offsets = gemma_tokenize(stage.gemma_tok, text)
    xp_ids, xp_offsets = gemma_tokenize(stage.gemma_tok, xp_text)

    token_ranges = resolve_token_ranges(
        ops, final_pos, text, xp_text, words, x_offsets, xp_offsets,
    )
    if token_ranges is None:
        # Distinguish REPL token-count mismatch from other alignment failures.
        # (Cheap re-check: any REPL whose payload tokenization length differs
        # from its original?)
        for op in ops:
            if op.op_type == "repl":
                orig_cs = words[op.word_start][1]
                orig_ce = words[op.word_end - 1][2]
                xs, xe = find_token_range(x_offsets, orig_cs, orig_ce)
                if xs is None:
                    return None, "token_align_failed"
        return None, "token_align_failed"

    gold = build_compound_gold(
        x_ids, xp_ids, ops, token_ranges,
        mask_id=stage.mask_id, ins_id=stage.ins_id, del_id=stage.del_id,
    )
    if gold is None:
        return None, "gold_build_failed"

    sample = {
        "x_token_ids": x_ids,
        "x_prime_token_ids": xp_ids,
        "tagger_gold": gold["tagger_gold"],
        "editor_input_token_ids": gold["editor_input_token_ids"],
        "editor_target_token_ids": gold["editor_target_token_ids"],
        "ins_span_lengths": gold["ins_span_lengths"],
        "del_span_lengths": gold["del_span_lengths"],
        "op_types": [op.op_type.upper() for op in ops],
        "N_total": len(ops),
        "x_text": text,
        "x_prime_text": xp_text,
    }
    return sample, ""


def build_identity_sample(stage: Stage, text: str) -> Tuple[Optional[Dict], str]:
    x_ids, _ = gemma_tokenize(stage.gemma_tok, text)
    if len(x_ids) < 3:
        return None, "too_short_sentence"
    sample = {
        "x_token_ids": x_ids,
        "x_prime_token_ids": list(x_ids),
        "tagger_gold": [OP_KEEP] * len(x_ids),
        "editor_input_token_ids": list(x_ids),
        "editor_target_token_ids": list(x_ids),
        "ins_span_lengths": [],
        "del_span_lengths": [],
        "op_types": [],
        "N_total": 0,
        "x_text": text,
        "x_prime_text": text,
    }
    return sample, ""


# ---------------------------------------------------------------------------
# N-dependent gates (§6.2.6)
# ---------------------------------------------------------------------------
def gate_thresholds(N: int, args) -> Tuple[float, float, float]:
    """Return (ppl_max_ratio, sae_min, sae_max) for a compound of N ops."""
    if N <= 0:
        return (float("inf"), 0.0, float("inf"))
    s = math.sqrt(N)
    ppl_max = args.ppl_per_op_factor ** s
    sae_min = args.sae_per_op_min * s
    sae_max = args.sae_per_op_max * s
    return ppl_max, sae_min, sae_max


def finalize_sample(
    stage: Stage, sample: Dict, source_id: str, args,
    calibration_writer: Optional[any] = None,
) -> Tuple[Optional[Dict], str]:
    """Apply N-dependent PPL / SAE-shift gates and attach conditioning topk.

    In --calibration-mode, the gate is skipped and the metrics are written
    to `calibration_writer` (a JSONL handle). The sample is then still
    finalised and returned so the calibration run still emits a usable
    cache (useful for end-to-end shape testing).
    """
    x_text = sample.get("x_text", "")
    xp_text = sample.get("x_prime_text", "")
    if not xp_text:
        return None, "empty_xprime_text"

    N = int(sample.get("N_total", 0))
    is_identity = (N == 0)

    if is_identity:
        ppl_clean = ppl_corr = float("nan")
        ppl_ratio: Optional[float] = None
    else:
        ppl_clean = causal_perplexity_text(stage, x_text)
        ppl_corr = causal_perplexity_text(stage, xp_text)
        if not (math.isfinite(ppl_clean) and math.isfinite(ppl_corr)):
            ppl_ratio = None
        else:
            ppl_ratio = ppl_corr / ppl_clean

    fX, vX = sae_pool_max_topk_from_text(stage, x_text)
    fXp, vXp = sae_pool_max_topk_from_text(stage, xp_text)
    shift = sae_l2_shift(fX, vX, fXp, vXp)

    ppl_max, sae_min, sae_max = gate_thresholds(N, args)

    if calibration_writer is not None:
        calibration_writer.write(json.dumps({
            "N": N,
            "source_sent_id": source_id,
            "op_types": sample["op_types"],
            "ppl_clean": ppl_clean if math.isfinite(ppl_clean) else None,
            "ppl_corr": ppl_corr if math.isfinite(ppl_corr) else None,
            "ppl_ratio": ppl_ratio,
            "sae_shift": shift,
            "ppl_max_at_N": ppl_max if math.isfinite(ppl_max) else None,
            "sae_min_at_N": sae_min,
            "sae_max_at_N": sae_max,
        }) + "\n")

    if not args.calibration_mode and not is_identity:
        if ppl_ratio is None:
            return None, "ppl_inf"
        if ppl_ratio > ppl_max:
            return None, "ppl_too_high"
        if shift < sae_min:
            return None, "sae_shift_too_small"
        if shift > sae_max:
            return None, "sae_shift_too_large"

    sample.pop("x_text", None)
    sample.pop("x_prime_text", None)
    sample.update({
        "source_sent_id": source_id,
        "z_X_topk": topk_records(fX, vX),
        "z_X_prime_topk": topk_records(fXp, vXp),
        "filter_telemetry": {
            "ppl_clean": ppl_clean if math.isfinite(ppl_clean) else None,
            "ppl_ratio": ppl_ratio,
            "ppl_max_at_N": ppl_max if math.isfinite(ppl_max) else None,
            "sae_shift_l2": shift,
            "sae_min_at_N": sae_min,
            "sae_max_at_N": sae_max,
        },
    })
    return sample, ""


# ---------------------------------------------------------------------------
# Bucket logic (§6.3.1)
# ---------------------------------------------------------------------------
BUCKETS = ("identity", "single_op", "compound_2_3", "compound_4_plus")


def bucket_for_N(N: int) -> str:
    if N <= 0:
        return "identity"
    if N == 1:
        return "single_op"
    if N <= 3:
        return "compound_2_3"
    return "compound_4_plus"


def normalize_bucket_weights(args) -> Dict[str, float]:
    weights = {
        "identity": args.p_identity,
        "single_op": args.p_single_op,
        "compound_2_3": args.p_compound_2_3,
        "compound_4_plus": args.p_compound_4_plus,
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("all bucket weights are zero")
    return {k: v / total for k, v in weights.items()}


def sample_N_for_bucket(bucket: str, rng: random.Random, args) -> int:
    """Sample N from the truncated geometric, conditioned on the bucket."""
    if bucket == "identity":
        return 0
    if bucket == "single_op":
        return 1
    # Truncated geometric over {2..N_MAX} or {4..N_MAX}
    p = args.n_distribution_p
    lo = 2 if bucket == "compound_2_3" else 4
    hi = 3 if bucket == "compound_2_3" else args.n_max
    if hi < lo:
        return lo
    # Sample a geometric variate truncated to [lo, hi].
    weights = [(1 - p) ** (k - lo) * p for k in range(lo, hi + 1)]
    s = sum(weights)
    if s <= 0:
        return lo
    r = rng.random() * s
    cum = 0.0
    for k, w in zip(range(lo, hi + 1), weights):
        cum += w
        if r < cum:
            return k
    return hi


def pick_bucket(probs: Dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for b in BUCKETS:
        cum += probs[b]
        if r < cum:
            return b
    return BUCKETS[-1]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
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

    print(f"[corruption] loading spaCy POS tagger: {args.spacy_model}")
    spacy_nlp = load_spacy(args.spacy_model)

    stage = Stage(
        extractor=extractor, causal_llm=causal_llm, mlm=mlm,
        gemma_tok=gemma_tok,
        spacy_nlp=spacy_nlp,
        mask_id=int(gemma_tok.mask_token_id),
        ins_id=int(gemma_tok.convert_tokens_to_ids("[INS]")),
        del_id=int(gemma_tok.convert_tokens_to_ids("[DEL]")),
        pad_id=int(gemma_tok.pad_token_id),
        device=args.device, k_train=int(args.k_train),
    )

    bucket_probs = normalize_bucket_weights(args)
    print(f"[corruption] bucket weights: {bucket_probs}")
    print(f"[corruption] compound op weights: "
          f"REPL={args.op_weight_repl}, INS={args.op_weight_ins}, DEL={args.op_weight_del}")
    if args.calibration_mode:
        print(f"[corruption] CALIBRATION MODE: gate disabled; metrics → "
              f"{args.calibration_out or (out_dir / 'calibration.jsonl')}")

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
    n_attempts: Dict[int, int] = defaultdict(int)
    n_accepts: Dict[int, int] = defaultdict(int)
    reject_reasons: Dict[str, int] = defaultdict(int)
    cur_shard_file = None

    calibration_writer = None
    if args.calibration_mode:
        cal_path = Path(args.calibration_out) if args.calibration_out else (out_dir / "calibration.jsonl")
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_writer = open(cal_path, "wt", encoding="utf-8")

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

        bucket = pick_bucket(bucket_probs, rng)
        for _attempt in range(args.k_budget):
            attempted += 1
            bucket_attempts[bucket] += 1
            N = sample_N_for_bucket(bucket, rng, args)
            n_attempts[N] += 1
            sample: Optional[Dict] = None
            reason = ""
            if N == 0:
                sample, reason = build_identity_sample(stage, sent)
            else:
                ops, reason = sample_compound(stage, sent, rng, N, args)
                if ops is None:
                    reject_reasons[reason] += 1
                else:
                    sample, reason = build_compound_sample(stage, sent, ops)
                    if sample is None:
                        reject_reasons[reason] += 1
            if sample is None:
                continue
            final, finalize_reason = finalize_sample(
                stage, sample,
                source_id=f"dolma:s{sent_idx}",
                args=args,
                calibration_writer=calibration_writer,
            )
            if final is None:
                reject_reasons[finalize_reason] += 1
                continue
            final["bucket"] = bucket
            cur_shard_file.write(json.dumps(final, ensure_ascii=False) + "\n")
            written += 1
            bucket_accepts[bucket] += 1
            n_accepts[N] += 1
            if written % args.samples_per_shard == 0 and written < args.target_samples:
                cur_shard_file.close()
                open_shard()
                yield_pct = written / max(1, attempted)
                per_bucket = ", ".join(
                    f"{b}={bucket_accepts[b]}/{bucket_attempts[b]}="
                    f"{bucket_accepts[b] / max(1, bucket_attempts[b]):.2f}"
                    for b in BUCKETS
                )
                per_N = ", ".join(
                    f"N={k}:{n_accepts[k]}/{n_attempts[k]}="
                    f"{n_accepts[k] / max(1, n_attempts[k]):.2f}"
                    for k in sorted(n_attempts)
                )
                print(
                    f"[corruption] written={written} sents={sent_idx} "
                    f"attempts={attempted} yield={yield_pct:.3f}  "
                    f"buckets({per_bucket})  N({per_N})"
                )
                if reject_reasons:
                    top = sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:8]
                    print(f"[corruption] reject reasons: " +
                          ", ".join(f"{k}={v}" for k, v in top))
            break

    if cur_shard_file is not None:
        cur_shard_file.close()
    if calibration_writer is not None:
        calibration_writer.close()

    # Final summary
    yield_pct = written / max(1, attempted)
    per_bucket = ", ".join(
        f"{b}={bucket_accepts[b]}/{bucket_attempts[b]}="
        f"{bucket_accepts[b] / max(1, bucket_attempts[b]):.2f}"
        for b in BUCKETS
    )
    per_N = ", ".join(
        f"N={k}:{n_accepts[k]}/{n_attempts[k]}="
        f"{n_accepts[k] / max(1, n_attempts[k]):.2f}"
        for k in sorted(n_attempts)
    )
    print(
        f"[corruption] FINAL written={written} sents={sent_idx} "
        f"attempts={attempted} yield={yield_pct:.3f}  "
        f"buckets({per_bucket})  N({per_N})"
    )
    if reject_reasons:
        all_rej = sorted(reject_reasons.items(), key=lambda kv: -kv[1])
        print(f"[corruption] FINAL reject reasons: " +
              ", ".join(f"{k}={v}" for k, v in all_rej))

    meta = {
        "samples_written": int(written),
        "sentences_seen": int(sent_idx),
        "attempts": int(attempted),
        "yield": float(yield_pct),
        "bucket_attempts": {k: int(v) for k, v in bucket_attempts.items()},
        "bucket_accepts": {k: int(v) for k, v in bucket_accepts.items()},
        "bucket_yields": {
            k: float(bucket_accepts[k] / max(1, bucket_attempts[k]))
            for k in bucket_attempts
        },
        "n_attempts": {str(k): int(v) for k, v in n_attempts.items()},
        "n_accepts": {str(k): int(v) for k, v in n_accepts.items()},
        "reject_reasons": {k: int(v) for k, v in reject_reasons.items()},
        "calibration_mode": bool(args.calibration_mode),
        "gates": {
            "ppl_per_op_factor": float(args.ppl_per_op_factor),
            "sae_per_op_min": float(args.sae_per_op_min),
            "sae_per_op_max": float(args.sae_per_op_max),
        },
        "compound": {
            "n_max": int(args.n_max),
            "n_distribution_p": float(args.n_distribution_p),
            "op_weight_repl": float(args.op_weight_repl),
            "op_weight_ins": float(args.op_weight_ins),
            "op_weight_del": float(args.op_weight_del),
            "k_budget": int(args.k_budget),
        },
        "ins_position": {
            "p_high": float(args.ins_p_high),
            "high_priority_upos": sorted(HIGH_PRIORITY_UPOS),
        },
        "d_sae": int(extractor.d_sae),
        "k_train": int(args.k_train),
        "mask_id": int(stage.mask_id),
        "ins_id": int(stage.ins_id),
        "del_id": int(stage.del_id),
        "pad_id": int(stage.pad_id),
        "llm2vec_dir": args.llm2vec_dir,
        "mlm_model": args.mlm_model,
        "mlm_resolved": mlm.resolved_name,
        "spacy_model": args.spacy_model,
        "sae_repo": args.sae_repo,
        "sae_path": args.sae_path,
        "sae_layer": int(args.sae_layer),
        "seed": int(args.seed),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[corruption] done: {written} samples in {shard_idx} shards → {out_dir}")


if __name__ == "__main__":
    main()
