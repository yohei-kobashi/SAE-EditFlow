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

**N-dependent gates (§6.2.6).** Two gates with independent shape:

  (i) Fluency — SLOR drop (per-token, unigram-normalised log-likelihood;
      Pauls&Klein 2012 / Lau+ 2017 / Kann+ 2018). LINEAR-in-N budget
      because per-token Δlog-likelihood per op is empirically constant.

         slor_drop_max(N) = slor_drop_per_op * N

  (ii) SAE side — top-K identity change (BINARY by default). Reject if
       fewer than `sae_min_topk_change` features differ between
       top_K(X) and top_K(X'). N-independent.

         require:  |top_K(X) \\ top_K(X')|  >=  sae_min_topk_change

The previous L2 upper bound on SAE shift (sae_max) has been removed:
top-K identity change already captures whether the representation moved
meaningfully, and an L2 cap on top of it would also reject corruptions
that flip a small number of top-K features with high magnitude — those
are exactly the cases where the SAE conditioning has a clear, focused
interpretation that the editor can learn from. `sae_shift` is still
recorded in calibration / telemetry for offline analysis.

SLOR(s) = (1/|s|) * [log p_M(s) - log p_unigram(s)]. The unigram baseline
is built once from a slice of Dolma at startup (or loaded from
`--unigram-cache`) and reused for every (X, X') pair.

The scale constants are calibrated against the empirical Dolma + MLM
compound distribution via `--calibration-mode`, which records every
attempt's (N, slor_drop, sae_shift, sae_topk_change, plus legacy
ppl_ratio) without applying the gate.

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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

import difflib

from data import download_dolma_shards, iter_dolma_texts, iter_sentences
from transforms import FAMILIES as TRANSFORM_FAMILIES
from transforms import propose_transforms, roundtrip_ok
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
                   choices=["head", "random", "stride"], default="random",
                   help="Per-document sampling when --max-sentences-per-text "
                        "is set. Default 'random' to avoid lead-bias. No "
                        "effect when --max-sentences-per-text is None.")
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
    # Bucket weights calibrated to the LinguaLens English minimal-pair N
    # distribution (n_hunks histogram from scripts/lingualens_token_diff.py):
    # 66.6% N=1, 28.9% N=2, 3.6% N=3, 0.1% N>=4. We bias slightly more
    # toward N=2-3 than the empirical 32.5% (35%) and keep a small
    # N>=4 long-tail bucket (5%) so the editor still sees the
    # occasional harder compound.
    p.add_argument("--p-identity",        type=float, default=0.05)
    p.add_argument("--p-single-op",       type=float, default=0.55)
    p.add_argument("--p-compound-2-3",    type=float, default=0.35)
    p.add_argument("--p-compound-4-plus", type=float, default=0.05)

    # Compound op sampling (§6.2.5)
    p.add_argument("--n-max", type=int, default=5,
                   help="Cap on op count per compound sample.")
    p.add_argument("--n-distribution-p", type=float, default=0.4,
                   help="Truncated geometric parameter for N over {0..N_MAX}.")
    # PROPOSAL weights (pre-gate), calibrated so the ACCEPTED op-type mix
    # matches the LinguaLens English histogram (REPL 70.0% / INS 18.2% /
    # DEL 11.8%). The gates (§6.2.6) are not op-type-neutral: the SLOR
    # fluency bound rejects INS's natural-word deletions hardest and passes
    # DEL's MLM-plausible insertions easiest, so proposing at the target
    # ratio undershoots INS by ~3x and overshoots DEL by ~2x. The weights
    # below are the damped inverse-acceptance reweighting of the target
    # (README §6.2.7); re-verify the accepted mix on a pilot shard if the
    # gate / MLM / SAE / encoder changes.
    p.add_argument("--op-weight-repl", type=float, default=0.60)
    p.add_argument("--op-weight-ins",  type=float, default=0.34)
    p.add_argument("--op-weight-del",  type=float, default=0.06)
    p.add_argument("--op-position-max-retries", type=int, default=20,
                   help="Per-op attempts to find a non-conflicting position "
                        "before giving up on the compound.")

    # Condition-selective emission (v3, README §6.2.8). From every accepted
    # compound, besides the full-revert record, also emit:
    #   partial : a random proper nonempty op-subset S is reverted; the
    #             record's "clean" side becomes X_S and the conditioning
    #             becomes diff(SAE(X_S), SAE(X')). The same X' with a
    #             different S has DIFFERENT gold, so the conditioning is
    #             the only disambiguator — the loss cannot be minimised
    #             without reading it. (N >= 2 only.)
    #   null    : S = ∅ — the record's clean side IS X' (all-KEEP gold,
    #             zero SAE diff). Teaches "corrupted-looking text + no
    #             instruction → edit nothing", which the v1/v2 cache never
    #             contained and which replaces the harmful empty-cond
    #             training hack (z = 0 while the target stayed the full
    #             restore).
    p.add_argument("--emit-full", type=float, default=1.0,
                   help="Probability of emitting the full-revert record per "
                        "accepted compound.")
    p.add_argument("--emit-partial", type=float, default=1.0,
                   help="Probability of emitting one random proper-subset "
                        "record per accepted compound (N >= 2 only).")
    p.add_argument("--emit-null", type=float, default=0.5,
                   help="Probability of emitting the S=∅ (all-KEEP) record "
                        "per accepted compound.")

    # v4 linguistically-typed transformation ops (transforms.py, README
    # §6.2.10). A fraction of sentence attempts goes to rule-based
    # grammatical↔grammatical transformations (tense/aspect/modality/
    # number/degree/negation+NPI/det-quant/anaphor/voice/questions/
    # existential/valency/tough/ellipsis/cleft/inversion/split). Both
    # directions are emitted, so the same surface form appears with
    # opposite conditioning and opposite gold — the direction is decidable
    # only through z.
    p.add_argument("--sentence-regex", default="",
                   help="fast pre-filter: skip sentences NOT matching this "
                        "regex BEFORE tokenization/parsing. For rare-pattern "
                        "harvests (e.g. SPLITINF needs both a to-infinitive "
                        "and an -ly adverb), this multiplies effective scan "
                        "throughput by the regex's rejection rate.")
    p.add_argument("--transform-prob", type=float, default=0.35,
                   help="Fraction of sentence attempts routed to "
                        "transformation ops (0 disables v4).")
    p.add_argument("--transform-families", default="all",
                   help="'all' or comma list of transforms.FAMILIES keys.")
    p.add_argument("--family-priority-pick", action="store_true",
                   help="Weight the per-sentence family pick by inverse "
                        "running acceptance count (1/(1+accepted)) instead "
                        "of uniform-over-available. Rare structural "
                        "families (their predicates seldom fire) get "
                        "picked whenever they ARE available, and the "
                        "high-availability v6 word-class families "
                        "self-throttle. Fixes the v5 rare-family "
                        "starvation (PARTICLE 0.25%, QUOTINV absent — "
                        "README §13.6).")
    p.add_argument("--transform-compose-prob", type=float, default=0.15,
                   help="Given an accepted transform T1(X), probability of "
                        "chaining a SECOND family T2 (re-parsed, per-step "
                        "round-trip) so the pair becomes (X, T2(T1(X))) "
                        "with t_type 'A+B'. Teaches compositional feature "
                        "semantics — unseen edits are often unseen "
                        "COMBINATIONS of seen operations.")
    p.add_argument("--transform-slor-delta", type=float, default=1.5,
                   help="Symmetric fluency gate for transform pairs: "
                        "|SLOR(X) − SLOR(T(X))| must not exceed this "
                        "(both sides are supposed to be grammatical).")
    p.add_argument("--blocklist", default="",
                   help="Path to blocklist.npy (generic grammaticality "
                        "features, scripts/build_grammaticality_blocklist"
                        ".py); masked before conditioning top-k.")
    p.add_argument("--cond-scope", choices=["local", "global"],
                   default="local",
                   help="Conditioning top-k source: pool over edit-local "
                        "tokens (default; higher information content about "
                        "the edit) or the whole sentence (pre-v4).")

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
    # Fluency gate: SLOR drop per op (linear N scaling). SLOR(s) =
    # (log p_M(s) - log p_unigram(s)) / |s|. ΔSLOR = SLOR(X) - SLOR(X')
    # is the per-token fluency degradation; budget is linear in N
    # (each op contributes ~constant Δ on average, matching the
    # empirical factor^N fit to p50 of raw PPL ratio).
    p.add_argument("--slor-drop-per-op", type=float, default=0.10,
                   help="Per-op SLOR drop budget. Gate: reject if "
                        "(SLOR(X) - SLOR(X')) > slor_drop_per_op * N. "
                        "0.10 nats/token/op is an initial guess; "
                        "re-fit from calibration data.")
    p.add_argument("--unigram-cache",
                   help="Path to a JSON {token_id_str: log_prob} unigram "
                        "table for SLOR. Defaults to <out-dir>/unigram.json. "
                        "Built once from Dolma if missing.")
    p.add_argument("--unigram-sample-size", type=int, default=5000,
                   help="# of Dolma sentences to scan when building the "
                        "unigram table (only used when no cache exists).")
    p.add_argument("--unigram-smoothing", type=float, default=1.0,
                   help="Add-k Laplace smoothing for unigram log-probs.")
    # Lower bound — top-K identity change.
    # `--sae-min-topk-size` is K (default 10). `--sae-min-topk-change` is the
    # minimum number of features that must DIFFER between the top-K sets of
    # X and X' (default 1 = "any change" — strictly binary "changed vs not").
    p.add_argument("--sae-min-topk-size", type=int, default=10,
                   help="Size of the top-K SAE feature set used for the "
                        "lower-bound 'did the activation pattern change' "
                        "check. K must be <= --k-train (default 64).")
    p.add_argument("--sae-min-topk-change", type=int, default=1,
                   help="Reject if fewer than this many features differ "
                        "between top-K(X) and top-K(X'). Default 1 = at "
                        "least one feature must change identity.")
    # Deprecated SAE knobs — kept so old scripts that pass them do not
    # error out, but ignored at runtime.
    #   sae_per_op_max — L2 upper-bound on shift. Removed because the
    #     top-K identity check already captures whether the
    #     representation moved meaningfully; an additional L2 cap
    #     would conflate "wide L2 spread" with "non-minimal pair",
    #     and a corruption that flips top-K features by definition
    #     changes the meaning regardless of L2 magnitude.
    #   sae_per_op_min — L2 lower bound. Replaced by top-K identity
    #     change (--sae-min-topk-size / --sae-min-topk-change).
    p.add_argument("--sae-per-op-max", type=float, default=2.50,
                   help="DEPRECATED. The L2 upper-bound gate has been "
                        "removed; top-K identity change is the sole "
                        "SAE-side gate. Flag kept for back-compat.")
    p.add_argument("--sae-per-op-min", type=float, default=0.30,
                   help="DEPRECATED. Replaced by top-K identity change "
                        "(--sae-min-topk-size / --sae-min-topk-change). "
                        "Kept so old scripts do not break.")
    # Deprecated — kept for backwards compatibility with old run scripts;
    # ignored at runtime. The PPL ratio is still recorded in calibration
    # JSONL alongside SLOR so historical analysis still works.
    p.add_argument("--ppl-per-op-factor", type=float, default=1.8,
                   help="DEPRECATED. PPL ratio is no longer the gate; "
                        "SLOR drop is used instead. Kept so old scripts "
                        "do not break.")
    p.add_argument("--force-n", type=int, default=None,
                   help="Override bucket sampling and force every attempt to "
                        "use this N. Useful for measuring per-N yield / "
                        "distribution shifts without bucket-sample noise. "
                        "When set, --p-identity / --p-single-op / "
                        "--p-compound-* and --n-distribution-p are ignored.")
    p.add_argument("--calibration-mode", action="store_true",
                   help="Skip the SLOR / SAE-shift gate; record every attempt's "
                        "(N, slor_drop, sae_shift, ppl_ratio) to a JSONL file "
                        "for percentile fitting.")
    p.add_argument("--calibration-out",
                   help="Path to JSONL of calibration records (slor_drop, "
                        "sae_shift, plus legacy ppl_ratio). Defaults to "
                        "<out-dir>/calibration.jsonl.")

    p.add_argument("--k-train", type=int, default=64,
                   help="Top-K for SAE pool-max conditioning.")

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-sentences", type=int, default=0,
                   help="Skip the first N length-eligible Dolma sentences "
                        "before sampling. Use this to generate a held-out "
                        "DEV cache disjoint from a training cache: pass the "
                        "training run's `sentences_seen` (from its meta.json) "
                        "with the SAME --seed so the sentence stream order "
                        "matches and the dev sources start where training "
                        "stopped.")
    # Parallel generation: N processes on one GPU, each taking sentences
    # where sent_idx % stride == offset (disjoint, complete partition).
    # Driven by scripts/corruption_parallel.sh.
    p.add_argument("--sentence-stride", type=int, default=1,
                   help="Process only sentences whose index satisfies "
                        "idx %% stride == offset (for parallel workers).")
    p.add_argument("--sentence-offset", type=int, default=0,
                   help="This worker's residue class (< --sentence-stride).")
    # Resume: default ON. Counts samples in existing shards under --out-dir,
    # advances the Dolma stream past the source_sent_id of the last written
    # sample, and continues writing until target_samples is reached. RNG
    # state is NOT exactly restored (re-seeded), so the post-resume
    # sample-by-sample sequence may differ from a fresh run, but the
    # cumulative distribution is preserved.
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Default. Resume from existing shards under --out-dir.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore existing shards and start fresh "
                        "(overwrites previous output).")
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
    # SLOR unigram baseline: {token_id (int): log p_unigram (float)}.
    # `unigram_log_unk` is the log-prob assigned to any token id not in
    # the table (smoothed). Built once at startup; see build_unigram(...).
    unigram_log: Dict[int, float]
    unigram_log_unk: float
    # Full spaCy pipeline (parser + lemmatizer; ner off) — required by the
    # v4 transformation proposers (dependency patterns, doc.sents, lemma_).
    # spacy_nlp above stays POS-only for the fast INS-priority path.
    spacy_full: any = None


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
    """Return the index of the first token whose char range extends past
    `char_pos` — the token covering `char_pos`, or the first one after it.

    Used for finding the INS gap marker position (zero-width gap in xp_text).
    SentencePiece tokens absorb the preceding space, so the token following
    a deletion gap typically STARTS one char before the gap (at the space);
    matching on `s >= char_pos` would skip it and return the token one
    position too far right, misaligning the INS gold.
    """
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if e > char_pos:
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


def load_spacy_full(model_name: str):
    """Full pipeline (only ner disabled) for the v4 transformation
    proposers: they need the dependency parse (doc.sents, token.dep_) and
    the lemmatizer (token.lemma_), both of which the fast POS-only
    pipeline below strips — with it, every proposer's guard except-ed out
    and the transform branch was silently empty."""
    import spacy
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download(model_name)
        return spacy.load(model_name, disable=["ner"])


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
# SAE pool-max top-K helpers
#
# We split the work into:
#   (1) sae_encode_with_offsets: tokenize + forward + SAE encode once, return
#       (offsets, z[seq_len, d_sae]). This is the expensive step.
#   (2) sae_global_pool_topk: max-pool z over ALL positions → top-K (used
#       for the downstream SAE conditioning passed to the editor and for the
#       global L2 shift telemetry).
#   (3) sae_local_topk_at_char_ranges: max-pool z over ONLY tokens that
#       overlap a given list of (char_start, char_end) ranges → top-K set.
#       This is the gate signal: "did the SAE features at edited positions
#       change?" The empirical issue we hit at K=10 was that global top-K
#       is dominated by content-independent features (BOS/EOS/punctuation/
#       language id) whose pool-max is taken at unedited positions, so
#       local edits do not flip global top-K. Restricting the pool to
#       edited positions makes the gate sensitive to those edits.
# ---------------------------------------------------------------------------
@torch.no_grad()
def sae_encode_with_offsets(
    stage: Stage, text: str,
) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """Tokenize `text`, run the SAE backbone, and return (offsets, z).

    `offsets[i] = (char_start, char_end)` for token i, computed by the
    SAE's base-Gemma tokenizer (NOT the LLM2Vec tokenizer). Used to
    locate edited regions by char range. `z` has shape [seq_len, d_sae].
    """
    enc = stage.extractor.llm_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256,
        return_offsets_mapping=True, add_special_tokens=True,
    )
    offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
    enc_input = {k: v.to(stage.device) for k, v in enc.items()
                 if k in ("input_ids", "attention_mask")}
    out = stage.extractor.llm(
        **enc_input, output_hidden_states=True, use_cache=False,
    )
    h = out.hidden_states[stage.extractor.layer_idx][0]
    z = stage.extractor.sae.encode(h.to(stage.extractor.sae.W_enc.dtype))
    return offsets, z


def sae_global_pool_topk(
    stage: Stage, z: torch.Tensor,
) -> Tuple[List[int], List[float]]:
    """Global max-pool top-K (matches the historical behavior used for the
    z_X_topk conditioning fed into the editor)."""
    sparse = stage.extractor.pool_max_topk(z, stage.k_train)
    nz = (sparse > 0).nonzero(as_tuple=True)[0]
    return nz.tolist(), sparse[nz].cpu().float().tolist()


def _char_ranges_to_token_positions(
    offsets: List[Tuple[int, int]], char_ranges: List[Tuple[int, int]],
) -> List[int]:
    """Return the list of token indices that overlap any (cs, ce) range.

    Special tokens whose offset is (0, 0) are skipped (BOS/EOS pad).
    """
    pos: Set[int] = set()
    for (cs, ce) in char_ranges:
        if ce <= cs:
            continue
        for ti, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:
                continue
            if te <= cs or ts >= ce:
                continue
            pos.add(ti)
    return sorted(pos)


def sae_local_topk_at_char_ranges(
    z: torch.Tensor, offsets: List[Tuple[int, int]],
    char_ranges: List[Tuple[int, int]], k: int,
) -> Set[int]:
    """Pool z over the tokens that overlap `char_ranges`, return top-K
    feature ids by that pooled activation. Empty if no positions match.
    """
    if k <= 0 or not char_ranges:
        return set()
    positions = _char_ranges_to_token_positions(offsets, char_ranges)
    if not positions:
        return set()
    pos_idx = torch.tensor(positions, device=z.device, dtype=torch.long)
    z_restricted = z.index_select(0, pos_idx)
    z_pooled, _ = z_restricted.max(dim=0)
    k_eff = min(k, z_pooled.numel())
    vals, ids = z_pooled.topk(k_eff)
    return {int(f) for v, f in zip(vals.tolist(), ids.tolist()) if v > 0}


@torch.no_grad()
def sae_pool_max_topk_from_text(
    stage: Stage, text: str,
) -> Tuple[List[int], List[float]]:
    """Backwards-compatible wrapper used outside finalize_sample
    (e.g. precompute_sae paths)."""
    _, z = sae_encode_with_offsets(stage, text)
    return sae_global_pool_topk(stage, z)


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


def topk_feature_set(feats: List[int], vals: List[float], k: int) -> set:
    """Return the set of feature ids with the K largest values.

    `feats` / `vals` are NOT necessarily sorted (pool_max_topk returns
    non-zero positions in feature-id order, not value order). We sort
    descending by value and take the first K.
    """
    if not feats or k <= 0:
        return set()
    pairs = sorted(zip(vals, feats), reverse=True)
    return {int(f) for _, f in pairs[:k]}


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
# SLOR (Syntactic Log-Odds Ratio) — referenceless fluency score
#
#   SLOR(s) = (1/|s|) * [log p_M(s) - log p_unigram(s)]
#
# Conventional definition from Pauls & Klein (2012) / Lau et al. (2017) /
# Kann et al. (NAACL 2018). Per-token normalisation removes the
# raw-PPL length bias (Wang et al. 2022); the unigram baseline absorbs
# rare-word penalties so that semantically equivalent but lexically
# distinct paraphrases score similarly (Kann+ 2018 reports SLOR Pearson
# 0.454 vs raw PPL 0.325 on referenceless fluency evaluation).
#
# Implementation alignment with `+1` shift:
#   * `causal_llm(input_ids, labels=input_ids)` averages CE over T-1
#     positions, predicting tokens at index 1..T-1 from hidden states
#     at index 0..T-2. So sum log p_M(s) = -loss * (T-1) and the
#     "tokens" SLOR averages over are the same content positions
#     (index 1..T-1), excluding the BOS.
# ---------------------------------------------------------------------------
@torch.no_grad()
def causal_log_prob_and_token_ids(stage: Stage, text: str):
    """Return (sum_log_p_M, label_token_ids).

    sum_log_p_M  = -loss * n_pred  where n_pred = T - 1.
    label_token_ids = the (T-1) token ids being predicted (positions 1..T-1).

    Returns (None, None) when the sentence is too short for the +1 shift.
    """
    enc = stage.gemma_tok(
        text, return_tensors="pt", truncation=True, max_length=256,
    ).to(stage.device)
    T = int(enc.input_ids.shape[1])
    if T < 2:
        return None, None
    out = stage.causal_llm(
        input_ids=enc.input_ids, labels=enc.input_ids, use_cache=False,
    )
    n_pred = T - 1
    sum_log_p_M = -float(out.loss.item()) * n_pred
    label_ids = enc.input_ids[0, 1:].tolist()  # positions 1..T-1
    return sum_log_p_M, label_ids


def unigram_sum_log_prob(token_ids, unigram_log: Dict[int, float],
                         unigram_log_unk: float) -> float:
    """Sum_t log p_unigram(t) over a list of token ids."""
    s = 0.0
    for tid in token_ids:
        s += unigram_log.get(int(tid), unigram_log_unk)
    return s


def slor_text(stage: Stage, text: str):
    """Return (slor, n_pred, sum_log_p_M, sum_log_p_unig).

    SLOR = (sum_log_p_M - sum_log_p_unig) / n_pred.

    Returns (None, 0, None, None) on too-short input. All four scalars
    are useful for debugging / calibration recording.
    """
    sum_log_p_M, label_ids = causal_log_prob_and_token_ids(stage, text)
    if sum_log_p_M is None:
        return None, 0, None, None
    sum_log_p_unig = unigram_sum_log_prob(
        label_ids, stage.unigram_log, stage.unigram_log_unk,
    )
    n_pred = len(label_ids)
    slor = (sum_log_p_M - sum_log_p_unig) / max(n_pred, 1)
    return slor, n_pred, sum_log_p_M, sum_log_p_unig


# ---------------------------------------------------------------------------
# Unigram table — built once at startup from a slice of Dolma using the
# downstream Gemma tokenizer. Skip special tokens (BOS / EOS / PAD /
# [MASK] / [INS] / [DEL]) so the baseline reflects content vocabulary.
# Cached as JSON so re-runs do not re-tokenize.
# ---------------------------------------------------------------------------
def build_unigram(
    sentences,
    gemma_tok,
    sample_size: int,
    smoothing: float,
    skip_special_ids: set,
):
    """Scan up to `sample_size` sentences and return (log_prob_dict, log_unk).

    Add-`smoothing` Laplace over the observed vocabulary plus one UNK
    bucket.
    """
    counts: Dict[int, int] = {}
    seen = 0
    for sent in sentences:
        if seen >= sample_size:
            break
        enc = gemma_tok(sent, add_special_tokens=False, truncation=True,
                        max_length=256)
        for tid in enc["input_ids"]:
            tid = int(tid)
            if tid in skip_special_ids:
                continue
            counts[tid] = counts.get(tid, 0) + 1
        seen += 1
    if not counts:
        # Defensive fallback so the pipeline keeps going (we will produce
        # SLOR ~ 0 which makes the gate inactive).
        return {}, math.log(1e-9)
    total = sum(counts.values())
    V = len(counts) + 1  # +1 for the UNK bucket
    denom = total + smoothing * V
    log_prob: Dict[int, float] = {}
    for tid, c in counts.items():
        log_prob[tid] = math.log((c + smoothing) / denom)
    log_unk = math.log(smoothing / denom)
    return log_prob, log_unk


def load_unigram(path: Path):
    """Load {token_id: log_prob} JSON. Returns (dict, log_unk)."""
    raw = json.loads(Path(path).read_text())
    table = {int(k): float(v) for k, v in raw["table"].items()}
    return table, float(raw["unk_log_prob"])


def save_unigram(path: Path, table: Dict[int, float], log_unk: float):
    """Persist {token_id: log_prob} to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "table": {str(k): v for k, v in table.items()},
        "unk_log_prob": log_unk,
        "vocab_observed": len(table),
    }))


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
    "slor_undefined",         # SLOR could not be computed (e.g. T<2 tokens)
    "slor_drop_too_high",     # corruption exceeds the per-N SLOR budget
    "sae_topk_unchanged",     # top-K SAE feature set did not change enough
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


def derive_edit_char_ranges(
    ops: List[OpSpec], final_pos: Dict[int, int],
    text: str, xp_text: str, words: List[Tuple[str, int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (x_edit_char_ranges, xp_edit_char_ranges).

    Used to restrict the SAE gate signal to "the SAE features at edited
    positions". For each op:
      REPL: X = original-word span;  X' = replacement payload span.
      INS:  X = original-word span;  X' = [INS] marker span.
      DEL:  X = 1-char window around the insertion anchor (zero-width edit
            on X side); X' = inserted payload span.

    The X side for DEL ops is a small window because the deletion has no
    extent in X — but anchoring at the gap captures the immediate
    syntactic context where the edit happens.
    """
    x_ranges: List[Tuple[int, int]] = []
    xp_ranges: List[Tuple[int, int]] = []
    for idx, op in enumerate(ops):
        if op.op_type == "repl":
            cs = words[op.word_start][1]
            ce = words[op.word_end - 1][2]
            x_ranges.append((cs, ce))
            ps = final_pos[idx]
            pe = ps + len(op.payload or "")
            xp_ranges.append((ps, pe))
        elif op.op_type == "ins":
            cs = words[op.word_start][1]
            ce = words[op.word_end - 1][2]
            x_ranges.append((cs, ce))
            ps = final_pos[idx]
            # The [INS] marker text in xp_text starts at final_pos[idx]; the
            # marker length is fixed ("[INS]"). Hard-code rather than depending
            # on tokenizer specials to avoid a circular dep.
            xp_ranges.append((ps, ps + len("[INS]")))
        elif op.op_type == "del":
            # X-side anchor: 1-char window centered on the gap. Clamp to text.
            if op.word_start < len(words):
                anchor = words[op.word_start][1]
            else:
                anchor = len(text)
            cs_win = max(0, anchor - 1)
            ce_win = min(len(text), anchor + 1)
            if cs_win < ce_win:
                x_ranges.append((cs_win, ce_win))
            ps = final_pos[idx]
            pe = ps + len(op.payload or "")
            if pe > ps:
                xp_ranges.append((ps, pe))
    return x_ranges, xp_ranges


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

    x_edit_char_ranges, xp_edit_char_ranges = derive_edit_char_ranges(
        ops, final_pos, text, xp_text, words,
    )

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
        "_x_edit_char_ranges": x_edit_char_ranges,
        "_xp_edit_char_ranges": xp_edit_char_ranges,
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
# Arbitrary-pair gold (v4): token-level diff → v1 cache format. Handles
# reordering/length changes (VOICE etc.) that the OpSpec machinery cannot.
# ---------------------------------------------------------------------------
def pair_to_gold(x_ids, xp_ids, mask_id, ins_id, del_id):
    """Build (tagger_gold, editor pair, span lens, op hunks) for any token
    pair, mirroring build_compound_gold's v1 record conventions (INS tag on
    the gap-adjacent kept token; gap adjacent to REPL/DEL or at the end →
    None)."""
    sm = difflib.SequenceMatcher(None, x_ids, xp_ids, autojunk=False)
    tagger_gold, ei, et = [], [], []
    ins_lens, del_lens, op_types = [], [], []
    pend = False
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                tagger_gold.append(OP_INS if pend else OP_KEEP)
                pend = False
                ei.append(xp_ids[j1 + k])
                et.append(x_ids[i1 + k])
        elif tag == "replace":
            if pend:
                return None
            lx, lxp = i2 - i1, j2 - j1
            m = min(lx, lxp)
            for k in range(m):
                tagger_gold.append(OP_REPL)
                ei.append(mask_id)
                et.append(x_ids[i1 + k])
            if lxp > m:                       # extra corrupted tokens → DEL
                del_lens.append(lxp - m)
                for k in range(m, lxp):
                    tagger_gold.append(OP_DEL)
                    ei.append(xp_ids[j1 + k])
                    et.append(del_id)
            elif lx > m:                      # extra clean tokens → INS gap
                ins_lens.append(lx - m)
                for k in range(m, lx):
                    ei.append(ins_id)
                    et.append(x_ids[i1 + k])
                pend = True
            op_types.append("REPL")
        elif tag == "delete":                 # clean-only tokens → INS gap
            if pend:
                ins_lens[-1] += i2 - i1
            else:
                ins_lens.append(i2 - i1)
                op_types.append("INS")
            for k in range(i1, i2):
                ei.append(ins_id)
                et.append(x_ids[k])
            pend = True
        elif tag == "insert":                 # corrupted-only tokens → DEL
            if pend:
                return None
            del_lens.append(j2 - j1)
            op_types.append("DEL")
            for k in range(j1, j2):
                tagger_gold.append(OP_DEL)
                ei.append(xp_ids[k])
                et.append(del_id)
    if pend or len(tagger_gold) != len(xp_ids) or len(ei) != len(et):
        return None
    return {
        "tagger_gold": tagger_gold,
        "editor_input_token_ids": ei,
        "editor_target_token_ids": et,
        "ins_span_lengths": ins_lens,
        "del_span_lengths": del_lens,
        "op_types": op_types,
        "opcodes": sm.get_opcodes(),
    }


def build_pair_sample(stage: Stage, clean_text: str, corrupted_text: str,
                      ) -> Tuple[Optional[Dict], str]:
    """Record whose clean side / corrupted side are ARBITRARY texts (v4
    transformations; also usable for external pair corpora)."""
    x_ids, x_off = gemma_tokenize(stage.gemma_tok, clean_text)
    xp_ids, xp_off = gemma_tokenize(stage.gemma_tok, corrupted_text)
    if len(x_ids) < 3 or len(xp_ids) < 3:
        return None, "too_short_sentence"
    gold = pair_to_gold(x_ids, xp_ids, mask_id=stage.mask_id,
                        ins_id=stage.ins_id, del_id=stage.del_id)
    if gold is None:
        return None, "pair_gold_failed"
    x_ranges, xp_ranges = [], []
    for tag, i1, i2, j1, j2 in gold.pop("opcodes"):
        if tag == "equal":
            continue
        if i2 > i1:
            spans = [x_off[i] for i in range(i1, i2) if x_off[i] != (0, 0)]
            if spans:
                x_ranges.append((spans[0][0], spans[-1][1]))
        if j2 > j1:
            spans = [xp_off[j] for j in range(j1, j2) if xp_off[j] != (0, 0)]
            if spans:
                xp_ranges.append((spans[0][0], spans[-1][1]))
    sample = {
        "x_token_ids": x_ids,
        "x_prime_token_ids": xp_ids,
        **gold,
        "N_total": len(gold["op_types"]),
        "x_text": clean_text,
        "x_prime_text": corrupted_text,
        "_x_edit_char_ranges": x_ranges,
        "_xp_edit_char_ranges": xp_ranges,
    }
    return sample, ""


# ---------------------------------------------------------------------------
# Condition-selective subsets (v3, README §6.2.8)
# ---------------------------------------------------------------------------
def rebase_subset_ops(
    text: str,
    words: List[Tuple[str, int, int]],
    ops: List[OpSpec],
    subset_idx: Set[int],
) -> Optional[Tuple[str, List[OpSpec]]]:
    """Rebase the ops in `subset_idx` from X onto X_S.

    X_S = X with the COMPLEMENT ops applied (i.e. X' with the subset
    reverted). The subset ops' own char spans are untouched by the
    complement (the claim rules of `can_add_op` guarantee non-overlap), so
    each span survives verbatim in X_S at a shifted position; we map the
    positions through the complement's cumulative char shifts, verify the
    substring is identical, and re-derive word indices on X_S.

    Returns (xs_text, rebased_subset_ops) or None if any mapping step
    fails (caller counts the reason and skips the partial record).
    """
    comp = [op for i, op in enumerate(ops) if i not in subset_idx]
    sub = [op for i, op in enumerate(ops) if i in subset_idx]
    if not sub:
        return None
    try:
        xs_text, _ = apply_compound_to_text(text, words, comp)
    except Exception:
        return None
    if not xs_text:
        return None

    # Char-shift events induced by the complement ops on X.
    events: List[Tuple[int, int, int]] = []   # (cs, ce, delta)
    for op in comp:
        if op.op_type == "repl":
            cs = words[op.word_start][1]
            ce = words[op.word_end - 1][2]
            events.append((cs, ce, len(op.payload or "") - (ce - cs)))
        elif op.op_type == "ins":
            cs = words[op.word_start][1]
            ce = words[op.word_end - 1][2]
            if ce < len(text) and text[ce].isspace():
                ce += 1
            events.append((cs, ce, -(ce - cs)))
        elif op.op_type == "del":
            pos = words[op.word_start][1]
            events.append((pos, pos, len(op.payload or "") + 1))
    events.sort()

    def map_pos(p: int) -> Optional[int]:
        shift = 0
        for cs, ce, d in events:
            if ce <= p:
                shift += d
            elif cs >= p:
                break
            else:
                return None   # inside a complement-edited region
        return p + shift

    xs_words = words_with_offsets(xs_text)
    starts = {w[1]: i for i, w in enumerate(xs_words)}
    ends = {w[2]: i for i, w in enumerate(xs_words)}

    new_ops: List[OpSpec] = []
    for op in sub:
        if op.op_type == "del":
            pos = map_pos(words[op.word_start][1])
            if pos is None or pos not in starts:
                return None
            wi = starts[pos]
            new_ops.append(OpSpec("del", wi, wi, payload=op.payload))
        else:
            cs = words[op.word_start][1]
            ce = words[op.word_end - 1][2]
            cs2 = map_pos(cs)
            if cs2 is None:
                return None
            ce2 = cs2 + (ce - cs)
            if xs_text[cs2:ce2] != text[cs:ce]:
                return None            # self-check: span must survive verbatim
            if cs2 not in starts or ce2 not in ends:
                return None
            new_ops.append(OpSpec(op.op_type, starts[cs2], ends[ce2] + 1,
                                  payload=op.payload))
    return xs_text, new_ops


def build_partial_sample(
    stage: Stage, text: str, words: List[Tuple[str, int, int]],
    ops: List[OpSpec], subset_idx: Set[int],
) -> Tuple[Optional[Dict], str]:
    """Build the record whose clean side is X_S (subset reverted).

    The pair is self-consistent by construction: build_compound_sample
    re-derives its own corrupted text from (X_S, rebased ops), so the
    record's gold is exact even if whitespace details differ marginally
    from the parent X'.
    """
    rb = rebase_subset_ops(text, words, ops, subset_idx)
    if rb is None:
        return None, "partial_rebase_failed"
    xs_text, sub_ops = rb
    return build_compound_sample(stage, xs_text, sub_ops)


# ---------------------------------------------------------------------------
# N-dependent gates (§6.2.6)
# ---------------------------------------------------------------------------
def gate_thresholds(N: int, args) -> Tuple[float, int]:
    """Return (slor_drop_max, sae_min_topk_change) for a compound of N ops.

    Fluency gate: SLOR drop budget is LINEAR in N (one "drop" per op).
    Each independent edit, on average, lowers per-token log-likelihood
    by a constant Δ; the gate budget tracks that with no √N correction.
    See Kann+ 2018 (SLOR), Wang+ 2022 (length bias), and the per-N
    empirical fit (factor^N matches p50 of raw PPL ratio for N ∈ 1..5).

    SAE gate: top-K identity check (binary). Reject if fewer than
    `sae_min_topk_change` features differ between top-K(X) and
    top-K(X'). This catches corruptions that move SAE feature
    magnitudes a bit but do not change which features are most active —
    which is the case the editor cannot learn from. N-independent by
    design: even at N=1 we require at least 1 change.

    The previous L2 upper bound (sae_max) was removed: top-K identity
    change already captures whether the representation moved
    meaningfully, and an L2 cap on top of it would penalise
    corruptions that change a small number of top-K features but with
    high magnitude — exactly the cases where the SAE conditioning has
    a clear, focused interpretation.
    """
    if N <= 0:
        return (float("inf"), 0)
    slor_drop_max = args.slor_drop_per_op * N
    sae_min_topk_change = int(args.sae_min_topk_change)
    return slor_drop_max, sae_min_topk_change


def finalize_sample(
    stage: Stage, sample: Dict, source_id: str, args,
    calibration_writer: Optional[any] = None,
    light: bool = False,
    transform: bool = False,
) -> Tuple[Optional[Dict], str]:
    """Apply N-dependent PPL / SAE-shift gates and attach conditioning topk.

    In --calibration-mode, the gate is skipped and the metrics are written
    to `calibration_writer` (a JSONL handle). The sample is then still
    finalised and returned so the calibration run still emits a usable
    cache (useful for end-to-end shape testing).

    `light=True` (v3 partial/null records): skip SLOR/PPL scoring and all
    gates — the record derives from a compound that already passed them —
    and compute only the SAE conditioning. Cuts the per-extra-record cost
    to two SAE forwards (one when the two texts are identical).
    """
    x_text = sample.get("x_text", "")
    xp_text = sample.get("x_prime_text", "")
    if not xp_text:
        return None, "empty_xprime_text"

    N = int(sample.get("N_total", 0))
    is_identity = (N == 0) or light

    if is_identity:
        slor_X = slor_Xp = None
        slor_drop: Optional[float] = None
        ppl_clean = ppl_corr = float("nan")
        ppl_ratio: Optional[float] = None
    else:
        # SLOR is the primary fluency signal. The PPL ratio is recorded
        # alongside for backwards-compat analysis but is no longer the gate.
        slor_X, n_X, logp_M_X, logp_unig_X = slor_text(stage, x_text)
        slor_Xp, n_Xp, logp_M_Xp, logp_unig_Xp = slor_text(stage, xp_text)
        if slor_X is None or slor_Xp is None:
            slor_drop = None
        else:
            slor_drop = slor_X - slor_Xp   # positive = corruption hurt fluency

        ppl_clean = causal_perplexity_text(stage, x_text)
        ppl_corr = causal_perplexity_text(stage, xp_text)
        if not (math.isfinite(ppl_clean) and math.isfinite(ppl_corr)):
            ppl_ratio = None
        else:
            ppl_ratio = ppl_corr / ppl_clean

    # One SAE forward per text, then derive both global and local pools
    # from the same z. Global is used for the downstream conditioning and
    # L2 shift telemetry; local (= pool-max over tokens that overlap an
    # edited char range) is what the SAE gate decides on.
    offsets_X, z_X = sae_encode_with_offsets(stage, x_text)
    if xp_text == x_text:
        offsets_Xp, z_Xp = offsets_X, z_X    # null records: identical texts
    else:
        offsets_Xp, z_Xp = sae_encode_with_offsets(stage, xp_text)
    fX, vX = sae_global_pool_topk(stage, z_X)
    fXp, vXp = sae_global_pool_topk(stage, z_Xp)
    shift = sae_l2_shift(fX, vX, fXp, vXp)

    # Top-K SAE feature identity diff (lower bound).
    topk_size = int(args.sae_min_topk_size)

    # Global top-K change (for telemetry / back-compat with old reports).
    topk_X_global = topk_feature_set(fX, vX, topk_size)
    topk_Xp_global = topk_feature_set(fXp, vXp, topk_size)
    topk_overlap_global = len(topk_X_global & topk_Xp_global)
    topk_change_global = (
        max(len(topk_X_global), len(topk_Xp_global)) - topk_overlap_global
    )

    # Local top-K change (the gate). Falls back to global when no edit
    # char ranges are available (identity sample, or compound build that
    # did not populate the ranges).
    x_ranges = sample.get("_x_edit_char_ranges", [])
    xp_ranges = sample.get("_xp_edit_char_ranges", [])
    if x_ranges or xp_ranges:
        topk_X_local = sae_local_topk_at_char_ranges(
            z_X, offsets_X, x_ranges, topk_size,
        )
        topk_Xp_local = sae_local_topk_at_char_ranges(
            z_Xp, offsets_Xp, xp_ranges, topk_size,
        )
        topk_overlap_local = len(topk_X_local & topk_Xp_local)
        topk_change_local = (
            max(len(topk_X_local), len(topk_Xp_local)) - topk_overlap_local
        )
        used_local = True
    else:
        topk_change_local = topk_change_global
        used_local = False

    # The gate uses the local change. `topk_change` (the value referenced
    # by downstream tools / the JSONL schema) reflects the gate signal.
    topk_change = topk_change_local

    slor_drop_max, sae_min_topk_change = gate_thresholds(N, args)

    if calibration_writer is not None and not light:
        calibration_writer.write(json.dumps({
            "N": N,
            "source_sent_id": source_id,
            "op_types": sample["op_types"],
            # Primary fluency signal (SLOR)
            "slor_clean": slor_X,
            "slor_corr":  slor_Xp,
            "slor_drop":  slor_drop,
            "slor_drop_max_at_N": (
                slor_drop_max if math.isfinite(slor_drop_max) else None
            ),
            # Legacy PPL ratio recorded for back-compat with analyze tools
            "ppl_clean": ppl_clean if math.isfinite(ppl_clean) else None,
            "ppl_corr":  ppl_corr  if math.isfinite(ppl_corr)  else None,
            "ppl_ratio": ppl_ratio,
            # SAE
            "sae_shift": shift,                           # telemetry only
            "sae_topk_size": topk_size,
            # The gate-relevant signal: pool-max restricted to tokens
            # overlapping edited char ranges.
            "sae_topk_change": topk_change_local,
            "sae_topk_change_local": topk_change_local,
            # Global pool-max (all positions). Kept so the legacy K=10
            # comparison in old runs is still reproducible.
            "sae_topk_change_global": topk_change_global,
            "sae_topk_used_local": used_local,
            "sae_min_topk_change_at_N": sae_min_topk_change,
        }) + "\n")

    if not args.calibration_mode and not is_identity:
        if slor_drop is None:
            return None, "slor_undefined"
        if transform:
            # v4 transformation pairs: BOTH sides are supposed to be
            # grammatical, so gate on the symmetric fluency difference
            # instead of the one-sided corruption drop.
            if abs(slor_drop) > float(args.transform_slor_delta):
                return None, "transform_slor_delta"
        elif slor_drop > slor_drop_max:
            return None, "slor_drop_too_high"
        if topk_change < sae_min_topk_change:
            return None, "sae_topk_unchanged"

    # ---- conditioning extraction (v4: edit-local scope + blocklist) ---- #
    def _pooled_dense(z, offsets, ranges):
        if ranges and getattr(args, "cond_scope", "local") == "local":
            pos = _char_ranges_to_token_positions(offsets, ranges)
            if pos:
                return z[pos].max(dim=0).values.float()
        return z.max(dim=0).values.float()

    dense_x = _pooled_dense(z_X, offsets_X, x_ranges)
    dense_xp = _pooled_dense(z_Xp, offsets_Xp, xp_ranges)
    blk = getattr(args, "_blocklist", None)
    if blk is not None:
        blk = blk.to(dense_x.device)
    cond_blocked_mass = None
    if blk is not None:
        diff_abs = (dense_x - dense_xp).abs()
        k8 = min(8, diff_abs.numel())
        vals, idx = diff_abs.topk(k8)
        tot = float(vals.sum())
        if tot > 0:
            bset = set(int(i) for i in blk.tolist())
            cond_blocked_mass = float(sum(
                float(v) for v, i in zip(vals.tolist(), idx.tolist())
                if i in bset) / tot)
        dense_x[blk] = 0.0
        dense_xp[blk] = 0.0

    def _cond_topk(dense, k=64):
        k = min(k, dense.numel())
        vals, idx = dense.topk(k)
        keep = vals > 0
        return idx[keep].tolist(), vals[keep].tolist()

    cfX, cvX = _cond_topk(dense_x)
    cfXp, cvXp = _cond_topk(dense_xp)

    sample.pop("x_text", None)
    sample.pop("x_prime_text", None)
    sample.pop("_x_edit_char_ranges", None)
    sample.pop("_xp_edit_char_ranges", None)
    sample.update({
        "source_sent_id": source_id,
        "z_X_topk": topk_records(cfX, cvX),
        "z_X_prime_topk": topk_records(cfXp, cvXp),
        "filter_telemetry": {
            "slor_clean": slor_X,
            "slor_corr":  slor_Xp,
            "slor_drop":  slor_drop,
            "slor_drop_max_at_N": (
                slor_drop_max if math.isfinite(slor_drop_max) else None
            ),
            "ppl_clean":   ppl_clean if math.isfinite(ppl_clean) else None,
            "ppl_ratio":   ppl_ratio,
            "sae_shift_l2":  shift,
            "sae_topk_size": topk_size,
            "sae_topk_change": topk_change_local,
            "sae_topk_change_local":  topk_change_local,
            "sae_topk_change_global": topk_change_global,
            "sae_topk_used_local": used_local,
            "sae_min_topk_change_at_N": sae_min_topk_change,
            "cond_scope": ("local" if x_ranges or xp_ranges else "global"),
            "cond_blocked_mass": cond_blocked_mass,
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
    if args.sentence_stride < 1:
        raise SystemExit("[corruption] --sentence-stride must be >= 1")
    if not (0 <= args.sentence_offset < args.sentence_stride):
        raise SystemExit("[corruption] --sentence-offset must be in "
                         "[0, --sentence-stride)")
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

    # The SAE reads hidden_states[layer_idx] — the raw residual ENTERING
    # layer `layer_idx`, recorded before that layer runs. Layers at or above
    # layer_idx contribute nothing to it, so drop them: the kept prefix
    # computes bit-identical activations while skipping ~half the forward
    # (and freeing the dropped layers' weights on the GPU). The extractor's
    # LLM is used solely for this hidden state (sae_encode_with_offsets);
    # SLOR / PPL scoring runs on the separate full `causal_llm`.
    _inner = extractor.llm
    if hasattr(_inner, "layers"):
        _n_full = len(_inner.layers)
        # hidden_states[i] is recorded as the INPUT to layer i just before it
        # runs, so layer `layer_idx` itself must stay in the loop for index
        # `layer_idx` to hold the raw (pre-norm) residual.
        _n_keep = int(extractor.layer_idx) + 1
        if _n_keep < _n_full:
            _inner.layers = _inner.layers[:_n_keep]
            _inner.config.num_hidden_layers = _n_keep
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"[corruption] SAE forward truncated to {_n_keep}/{_n_full} "
                  f"layers (hidden_states[{extractor.layer_idx}] is unchanged)")

    print(f"[corruption] loading corruption MLM: {args.mlm_model}")
    mlm = MLMProvider(args.mlm_model, dtype=_str_dtype(args.mlm_dtype)).to(args.device)
    print(f"[corruption] resolved MLM: {mlm.resolved_name}  mask={mlm.mask_token!r}")

    print(f"[corruption] loading spaCy POS tagger: {args.spacy_model}")
    args._blocklist = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        args._blocklist = torch.as_tensor(np.asarray(_bl, dtype=np.int64))
        print(f"[corruption] blocklist: {len(_bl)} generic grammaticality "
              f"features masked from conditioning ({args.blocklist})")
    args._families = (None if args.transform_families.strip() == "all"
                      else [x.strip() for x in
                            args.transform_families.split(",") if x.strip()])

    spacy_nlp = load_spacy(args.spacy_model)
    spacy_full = (load_spacy_full(args.spacy_model)
                  if args.transform_prob > 0 else None)

    # ----- Build / load the SLOR unigram baseline -------------------------
    skip_special_ids = {
        int(gemma_tok.bos_token_id) if gemma_tok.bos_token_id is not None else -1,
        int(gemma_tok.eos_token_id) if gemma_tok.eos_token_id is not None else -1,
        int(gemma_tok.pad_token_id) if gemma_tok.pad_token_id is not None else -1,
        int(gemma_tok.mask_token_id) if gemma_tok.mask_token_id is not None else -1,
        int(gemma_tok.convert_tokens_to_ids("[INS]")),
        int(gemma_tok.convert_tokens_to_ids("[DEL]")),
    }
    skip_special_ids.discard(-1)
    unigram_path = Path(args.unigram_cache) if args.unigram_cache else (out_dir / "unigram.json")
    if unigram_path.exists():
        print(f"[corruption] loading unigram baseline from {unigram_path}")
        unigram_log, unigram_log_unk = load_unigram(unigram_path)
    else:
        print(f"[corruption] building unigram baseline from "
              f"{args.unigram_sample_size} Dolma sentences "
              f"(smoothing={args.unigram_smoothing})")
        shard_paths_for_unig = download_dolma_shards(
            args.data_cache_dir, max_files=args.max_files,
        )
        unig_text_iter = iter_dolma_texts(shard_paths_for_unig, min_chars=64)
        unig_sent_iter = iter_sentences(
            unig_text_iter,
            splitter=args.sentence_splitter,
            min_chars=16, max_chars=2000,
            max_sentences_per_text=args.max_sentences_per_text,
            sample_strategy=args.sentence_sample_strategy,
            seed=args.seed + 7777,   # distinct stream
            quality_filter=True,
        )
        unigram_log, unigram_log_unk = build_unigram(
            unig_sent_iter, gemma_tok,
            sample_size=args.unigram_sample_size,
            smoothing=args.unigram_smoothing,
            skip_special_ids=skip_special_ids,
        )
        save_unigram(unigram_path, unigram_log, unigram_log_unk)
        print(f"[corruption] unigram: V_obs={len(unigram_log)}  "
              f"log_unk={unigram_log_unk:.3f}  saved={unigram_path}")

    stage = Stage(
        extractor=extractor, causal_llm=causal_llm, mlm=mlm,
        gemma_tok=gemma_tok,
        spacy_nlp=spacy_nlp,
        spacy_full=spacy_full,
        mask_id=int(gemma_tok.mask_token_id),
        ins_id=int(gemma_tok.convert_tokens_to_ids("[INS]")),
        del_id=int(gemma_tok.convert_tokens_to_ids("[DEL]")),
        pad_id=int(gemma_tok.pad_token_id),
        device=args.device, k_train=int(args.k_train),
        unigram_log=unigram_log, unigram_log_unk=unigram_log_unk,
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

    # Resume: count samples in existing shards and skip Dolma sentences to
    # the last written sample's source_sent_id. New samples land in a
    # FRESH shard (we never append to a partial shard — keeps gzip
    # streams atomic).
    skip_n_sentences = 0
    if args.resume:
        existing_shards = sorted(out_dir.glob("shard-*.jsonl.gz"))
        last_sent_idx = 0
        for p in existing_shards:
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        written += 1
                        try:
                            obj = json.loads(line)
                            sid = obj.get("source_sent_id", "")
                            if sid.startswith("dolma:s"):
                                last_sent_idx = max(
                                    last_sent_idx, int(sid.split("s", 1)[1])
                                )
                        except (json.JSONDecodeError, ValueError):
                            pass
            except (OSError, EOFError):
                # Last shard may have been interrupted mid-gzip. Skip it
                # silently — we will overwrite it with a fresh shard.
                break
        shard_idx = len(existing_shards)
        skip_n_sentences = last_sent_idx
        if written > 0:
            print(f"[corruption] RESUME: {written} samples in "
                  f"{shard_idx} shards under {out_dir}; skipping the "
                  f"first {skip_n_sentences} Dolma sentences and "
                  f"continuing to shard-{shard_idx:05d}")
        if written >= args.target_samples:
            print(f"[corruption] RESUME: already have "
                  f"{written} >= target {args.target_samples}; nothing to do")
            return
        sent_idx = last_sent_idx

    # Held-out (dev) generation: start the stream past the sentences a
    # training cache consumed. Resume skip takes precedence when larger.
    if args.skip_sentences > skip_n_sentences:
        skip_n_sentences = args.skip_sentences
        sent_idx = args.skip_sentences
        print(f"[corruption] --skip-sentences: starting at Dolma sentence "
              f"{args.skip_sentences}")

    calibration_writer_mode = "at" if args.resume and written > 0 else "wt"
    calibration_writer = None
    if args.calibration_mode:
        cal_path = Path(args.calibration_out) if args.calibration_out else (out_dir / "calibration.jsonl")
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_writer = open(cal_path, calibration_writer_mode, encoding="utf-8")

    def open_shard():
        nonlocal cur_shard_file, shard_idx
        path = out_dir / f"shard-{shard_idx:05d}.jsonl.gz"
        cur_shard_file = gzip.open(path, "wt", encoding="utf-8")
        shard_idx += 1

    open_shard()

    emit_counts: Counter = Counter()
    ttype_counts: Counter = Counter()
    fam_accepts: Counter = Counter()   # accepted records per family
                                       # (drives --family-priority-pick)

    def pick_family(by_fam: Dict[str, list]) -> str:
        """Pick a family among those available on this sentence: uniform,
        or inverse-acceptance-weighted under --family-priority-pick."""
        fams = sorted(by_fam)
        if not args.family_priority_pick or len(fams) == 1:
            return fams[rng.randrange(len(fams))]
        weights = [1.0 / (1.0 + fam_accepts[f]) for f in fams]
        r = rng.random() * sum(weights)
        for f, w in zip(fams, weights):
            r -= w
            if r <= 0:
                return f
        return fams[-1]

    def pick_proposal(by_fam: Dict[str, list]):
        """Family pick, then t_type-balanced pick within the family, then
        uniform among that t_type's proposals. The inner balance stops
        prolific sub-rules from crowding out their siblings (FS:DET emits
        |class|−1 alternatives per determiner and would otherwise dominate
        FUNCSWAP)."""
        fam_props = by_fam[pick_family(by_fam)]
        by_tt: Dict[str, list] = {}
        for pr in fam_props:
            by_tt.setdefault(pr.t_type, []).append(pr)
        tts = sorted(by_tt)
        tt_props = by_tt[tts[rng.randrange(len(tts))]]
        return tt_props[rng.randrange(len(tt_props))]

    def write_record(rec: Dict):
        """Write one record; rotate shards and log at shard boundaries."""
        nonlocal written
        cur_shard_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
        pbar.update(1)
        emit_counts[rec.get("subset_kind", "full")] += 1
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
            per_kind = ", ".join(f"{k}={v}" for k, v in emit_counts.items())
            print(
                f"[corruption] written={written} sents={sent_idx} "
                f"attempts={attempted} yield={yield_pct:.3f}  "
                f"buckets({per_bucket})  N({per_N})  kinds({per_kind})"
            )
            if reject_reasons:
                top = sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:8]
                print(f"[corruption] reject reasons: " +
                      ", ".join(f"{k}={v}" for k, v in top))

    # tqdm progress bar over written/target_samples. initial=written lets
    # the bar start at the resumed offset.
    pbar = tqdm(
        total=args.target_samples, initial=written,
        desc="[corruption]", unit="sample", dynamic_ncols=True,
        smoothing=0.05,
    )

    # Advance the Dolma stream past sentences already represented in the
    # existing shards (resume) before entering the main loop. Count only
    # LENGTH-ELIGIBLE sentences while skipping — `sent_idx` (and therefore
    # `source_sent_id` / `sentences_seen`) only increments for eligible
    # sentences, so skipping raw sentences would land short of the intended
    # position and re-corrupt sources already in the cache.
    if skip_n_sentences > 0:
        _skipped = 0
        for _sent in sent_iter:
            _tc = len(gemma_tok(_sent, add_special_tokens=False).input_ids)
            if not (args.sent_min_tokens <= _tc <= args.sent_max_tokens):
                continue
            _skipped += 1
            if _skipped >= skip_n_sentences:
                break

    sent_re = re.compile(args.sentence_regex) if args.sentence_regex else None
    n_regex_skip = 0
    for sent in sent_iter:
        if written >= args.target_samples:
            break
        if sent_re is not None and not sent_re.search(sent):
            n_regex_skip += 1          # cheap reject: no tokenize, no parse
            if n_regex_skip % 100000 == 0:
                print(f"[corruption] regex pre-filter skipped "
                      f"{n_regex_skip} sentences")
            continue
        gemma_token_count = len(gemma_tok(sent, add_special_tokens=False).input_ids)
        if not (args.sent_min_tokens <= gemma_token_count <= args.sent_max_tokens):
            continue
        sent_idx += 1

        # Parallel-worker partition: sent_idx increments for EVERY eligible
        # sentence (all workers see the same stream and count identically),
        # so residue classes are disjoint and complete across workers.
        if args.sentence_stride > 1 and \
                (sent_idx % args.sentence_stride) != args.sentence_offset:
            continue

        # ---- v4: transformation-op attempt (README §6.2.10) ---------- #
        if (args.transform_prob > 0 and args.force_n is None
                and rng.random() < args.transform_prob):
            wrote_t = 0
            props = propose_transforms(stage.spacy_full, sent, rng,
                                       families=args._families)
            if not props:
                ttype_counts["no_proposal"] += 1
            if props:
                # Family-balanced sampling: pooled-uniform lets
                # high-applicability families (DEGREE:+very fires on any
                # adjective) crowd out the rare structural ones (pilot:
                # 20% of transform records were DEG:+very). Pick a family
                # uniformly first, then a proposal within it.
                by_fam: Dict[str, list] = {}
                for pr in props:
                    by_fam.setdefault(pr.family, []).append(pr)
                prop = pick_proposal(by_fam)
                ttype_counts[f"prop:{prop.t_type}"] += 1
                if not roundtrip_ok(stage.spacy_full, prop, sent, rng):
                    ttype_counts[f"rtfail:{prop.family}"] += 1
                    reject_reasons["transform_roundtrip"] += 1
                else:
                    out_text = prop.out_text
                    t_type = prop.t_type
                    t_family = prop.family
                    # Optional composition: chain a SECOND family on T1(X).
                    # Each step is round-trip checked individually; the
                    # symmetric SLOR gate below sees the composed pair.
                    if rng.random() < float(args.transform_compose_prob):
                        ttype_counts["compose_attempt"] += 1
                        fams2 = [f for f in (args._families
                                             or list(TRANSFORM_FAMILIES))
                                 if f != prop.family]
                        props2 = propose_transforms(
                            stage.spacy_full, out_text, rng, families=fams2)
                        by_fam2: Dict[str, list] = {}
                        for pr2 in props2:
                            if pr2.out_text != sent:
                                by_fam2.setdefault(pr2.family, []).append(pr2)
                        if by_fam2:
                            prop2 = pick_proposal(by_fam2)
                            if roundtrip_ok(stage.spacy_full, prop2,
                                            out_text, rng):
                                out_text = prop2.out_text
                                t_type = f"{prop.t_type}+{prop2.t_type}"
                                t_family = f"{prop.family}+{prop2.family}"
                                ttype_counts["compose_accept"] += 1
                    source_id_t = f"dolma:s{sent_idx}"
                    # both directions: apply (X→T) and revert (T→X)
                    for clean, corr, tt in (
                            (out_text, sent, t_type),
                            (sent, out_text, t_type + "/rev")):
                        ps, pr = build_pair_sample(stage, clean, corr)
                        if ps is None:
                            reject_reasons[f"transform_{pr}"] += 1
                            continue
                        pf, fr = finalize_sample(
                            stage, ps, source_id=source_id_t, args=args,
                            calibration_writer=None, transform=True)
                        if pf is None:
                            reject_reasons[f"transform_{fr}"] += 1
                            continue
                        pf["bucket"] = "transform"
                        pf["subset_kind"] = "full"
                        pf["n_parent"] = int(pf.get("N_total", 1))
                        pf["t_type"] = tt
                        pf["t_family"] = t_family
                        if written < args.target_samples:
                            write_record(pf)
                            ttype_counts[f"acc:{tt}"] += 1
                            wrote_t += 1
                            for f in t_family.split("+"):
                                fam_accepts[f] += 1
                    # null record: corrupted side, all-KEEP, zero diff
                    if wrote_t and rng.random() < args.emit_null                             and written < args.target_samples:
                        ns, nr = build_identity_sample(stage, out_text)
                        if ns is not None:
                            nf, _ = finalize_sample(
                                stage, ns, source_id=source_id_t, args=args,
                                calibration_writer=None, light=True)
                            if nf is not None:
                                nf["bucket"] = "transform"
                                nf["subset_kind"] = "null"
                                nf["n_parent"] = 0
                                nf["t_type"] = "null"
                                nf["t_family"] = t_family
                                write_record(nf)
            if wrote_t:
                continue          # sentence consumed by the transform path
            # else fall through to the lexical corruption path

        if args.force_n is not None:
            N_forced = max(0, min(int(args.force_n), int(args.n_max)))
            bucket = bucket_for_N(N_forced)
        else:
            N_forced = None
            bucket = pick_bucket(bucket_probs, rng)
        for _attempt in range(args.k_budget):
            attempted += 1
            bucket_attempts[bucket] += 1
            N = N_forced if N_forced is not None else sample_N_for_bucket(bucket, rng, args)
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
            # v3: keep the parent's corrupted text for the S=∅ record
            # (finalize_sample strips the text fields from the dict).
            parent_xp_text = sample.get("x_prime_text", "")
            source_id = f"dolma:s{sent_idx}"
            final, finalize_reason = finalize_sample(
                stage, sample,
                source_id=source_id,
                args=args,
                calibration_writer=calibration_writer,
            )
            if final is None:
                reject_reasons[finalize_reason] += 1
                continue
            bucket_accepts[bucket] += 1
            n_accepts[N] += 1

            # ---- condition-selective emission (v3, README §6.2.8) ------- #
            to_write: List[Dict] = []
            final["bucket"] = bucket
            final["n_parent"] = N
            if N == 0:
                final["subset_kind"] = "identity"
                to_write.append(final)
            else:
                if rng.random() < args.emit_full:
                    final["subset_kind"] = "full"
                    to_write.append(final)
                if N >= 2 and ops is not None and rng.random() < args.emit_partial:
                    k_sub = rng.randint(1, N - 1)
                    subset = set(rng.sample(range(N), k_sub))
                    psample, preason = build_partial_sample(
                        stage, sent, words_with_offsets(sent), ops, subset,
                    )
                    if psample is None:
                        reject_reasons[f"partial_{preason}"] += 1
                    else:
                        pfinal, pr = finalize_sample(
                            stage, psample, source_id=source_id, args=args,
                            calibration_writer=calibration_writer, light=True,
                        )
                        if pfinal is None:
                            reject_reasons[f"partial_{pr}"] += 1
                        else:
                            pfinal["bucket"] = bucket
                            pfinal["subset_kind"] = "partial"
                            pfinal["n_parent"] = N
                            to_write.append(pfinal)
                if parent_xp_text and rng.random() < args.emit_null:
                    nsample, nreason = build_identity_sample(stage, parent_xp_text)
                    if nsample is None:
                        reject_reasons[f"null_{nreason}"] += 1
                    else:
                        nfinal, nr = finalize_sample(
                            stage, nsample, source_id=source_id, args=args,
                            calibration_writer=calibration_writer, light=True,
                        )
                        if nfinal is None:
                            reject_reasons[f"null_{nr}"] += 1
                        else:
                            nfinal["bucket"] = bucket
                            nfinal["subset_kind"] = "null"
                            nfinal["n_parent"] = N
                            to_write.append(nfinal)

            for rec in to_write:
                if written >= args.target_samples:
                    break
                write_record(rec)
            break

    if cur_shard_file is not None:
        cur_shard_file.close()
    if calibration_writer is not None:
        calibration_writer.close()
    pbar.close()

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
    per_kind = ", ".join(f"{k}={v}" for k, v in emit_counts.items())
    if ttype_counts:
        acc_t = {k[4:]: v for k, v in ttype_counts.items()
                 if k.startswith("acc:")}
        print("[corruption] transform accepts: " +
              ", ".join(f"{k}={v}" for k, v in sorted(acc_t.items())))
    print(
        f"[corruption] FINAL written={written} sents={sent_idx} "
        f"attempts={attempted} yield={yield_pct:.3f}  "
        f"buckets({per_bucket})  N({per_N})  kinds({per_kind})"
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
        # v4 transformation ops (README §6.2.10)
        "transform": {
            "transform_prob": float(args.transform_prob),
            "families": (args.transform_families),
            "compose_prob": float(args.transform_compose_prob),
            "slor_delta": float(args.transform_slor_delta),
            "blocklist": args.blocklist or None,
            "cond_scope": args.cond_scope,
            "counts": {k: int(v) for k, v in sorted(ttype_counts.items())},
        },
        # v3 condition-selective emission (README §6.2.8)
        "subset_emission": {
            "emit_full": float(args.emit_full),
            "emit_partial": float(args.emit_partial),
            "emit_null": float(args.emit_null),
            "counts": {k: int(v) for k, v in emit_counts.items()},
        },
        "calibration_mode": bool(args.calibration_mode),
        "gates": {
            # Fluency: SLOR drop, linear N
            "slor_drop_per_op": float(args.slor_drop_per_op),
            "slor_n_scaling": "linear",
            # SAE side: top-K identity change (binary by default).
            # No L2 upper bound — top-K change subsumes minimality
            # because a corruption that flips top-K features has by
            # definition altered the SAE-space interpretation.
            "sae_min_topk_size":   int(args.sae_min_topk_size),
            "sae_min_topk_change": int(args.sae_min_topk_change),
            "unigram_smoothing": float(args.unigram_smoothing),
            "unigram_sample_size": int(args.unigram_sample_size),
            # Deprecated, kept so old downstream tools can read meta.json.
            "ppl_per_op_factor": float(args.ppl_per_op_factor),
            "sae_per_op_min":    float(args.sae_per_op_min),
            "sae_per_op_max":    float(args.sae_per_op_max),
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
        "skip_sentences": int(args.skip_sentences),
        "sentence_stride": int(args.sentence_stride),
        "sentence_offset": int(args.sentence_offset),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[corruption] done: {written} samples in {shard_idx} shards → {out_dir}")


if __name__ == "__main__":
    main()
