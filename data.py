"""
Data loaders for SAE-LEWIS.

Two on-disk caches:

  Stage 0 (`precompute_sae.py`)  → SAE cache (per-sentence sparse top-L features)
  Stage 2 (`corruption.py`)      → Corruption cache (training samples)

This module exposes:
  * download_dolma_shards / iter_dolma_texts (raw corpus streaming)
  * iter_sentences (sentence-level streaming with PySBD)
  * CorruptionDataset (training-time loader)
  * CorruptionCollator
"""

from __future__ import annotations

import gzip
import io
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import requests
import torch
from torch.utils.data import IterableDataset, get_worker_info


DOLMA_URL_LIST = "https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_6-sample.txt"


# ---------------------------------------------------------------------------
# Dolma raw-shard streaming
# ---------------------------------------------------------------------------
def download_dolma_shards(cache_dir: str, max_files: Optional[int] = None) -> List[str]:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    url_list_path = cdir / "v1_6-sample.txt"
    if not url_list_path.exists():
        r = requests.get(DOLMA_URL_LIST, timeout=60)
        r.raise_for_status()
        url_list_path.write_text(r.text)
    urls = [u.strip() for u in url_list_path.read_text().splitlines() if u.strip()]
    if max_files is not None:
        urls = urls[:max_files]
    local_paths: List[str] = []
    for url in urls:
        fname = url.rsplit("/", 1)[-1]
        local = cdir / fname
        if not local.exists():
            print(f"[dolma] downloading {url}")
            with requests.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                tmp = local.with_suffix(local.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.rename(local)
        local_paths.append(str(local))
    return local_paths


def iter_dolma_texts(
    shard_paths: List[str],
    min_chars: int = 32,
    text_max_chars: Optional[int] = None,
) -> Iterator[str]:
    for shard in shard_paths:
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = doc.get("text") or ""
                    if len(text) < min_chars:
                        continue
                    if text_max_chars is not None and len(text) > text_max_chars:
                        text = text[:text_max_chars]
                    yield text
        except (OSError, EOFError) as e:
            print(f"[dolma] WARN: failed to read {shard}: {e}")


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------
# Pre-compiled patterns for the quality filter.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[*\-•]|\d{1,3}[.)]|[a-zA-Z][.)])\s+")
_URL_RE = re.compile(r"https?://|www\.[A-Za-z0-9]")
_LONG_PUNCT_RUN_RE = re.compile(r"[^\w\s　-ヿ一-鿿]{5,}")
_SENT_FINAL_CHARS = (".", "!", "?", "。", "！", "？", "…", "‼", "⁇", "⁈", "⁉")
_QUOTE_CLOSERS = ("\"", "'", "”", "’", ")", "]", "》")


def looks_like_sentence(
    s: str,
    min_words: int = 3,
    min_alpha_ratio: float = 0.5,
    require_terminal_punct: bool = True,
    reject_urls: bool = True,
    reject_list_items: bool = True,
    reject_punct_runs: bool = True,
    require_initial_capital: bool = False,
) -> bool:
    """Heuristic 'looks like prose, not junk' filter.

    Designed to weed out navigation menus, list items, code, URLs, repeated
    punctuation, headings, and other non-prose strings that sentence splitters
    sometimes yield from web text. The defaults are conservative — they pass
    casual prose but reject obviously non-sentential strings. All knobs are
    surfaced so callers can relax or tighten as needed.

    Parameters
    ----------
    s : str
        The candidate sentence.
    min_words : int
        Minimum whitespace-separated word count (default 3).
    min_alpha_ratio : float
        Minimum ratio of alphabetic characters over total characters.
    require_terminal_punct : bool
        Reject if the rightmost non-quote character isn't sentence-final
        punctuation (`. ! ? 。 ！ ？ …`).
    reject_urls : bool
        Reject if the string contains http(s):// or `www.`.
    reject_list_items : bool
        Reject leading bullet / numbered list markers ("* foo", "1) bar", ...).
    reject_punct_runs : bool
        Reject if there is a run of 5+ consecutive non-word characters
        (e.g. "----" separators, ASCII art).
    require_initial_capital : bool
        If True, require the first letter to be uppercase. Off by default
        because mid-document fragments and many languages don't follow that
        convention reliably.
    """
    s = s.strip()
    if not s:
        return False

    # Word count
    if len(s.split()) < min_words:
        return False

    # Alphabetic ratio
    n_alpha = sum(c.isalpha() for c in s)
    if n_alpha / len(s) < min_alpha_ratio:
        return False

    # Terminal sentence punctuation (allow trailing quote/bracket closers)
    if require_terminal_punct:
        tail = s.rstrip()
        while tail and tail[-1] in _QUOTE_CLOSERS:
            tail = tail[:-1]
        if not tail or not tail.endswith(_SENT_FINAL_CHARS):
            return False

    # URL-like content
    if reject_urls and _URL_RE.search(s):
        return False

    # List item start
    if reject_list_items and _LIST_ITEM_RE.match(s):
        return False

    # Long punctuation runs ("----", "####", "====", ...)
    if reject_punct_runs and _LONG_PUNCT_RUN_RE.search(s):
        return False

    # First letter casing
    if require_initial_capital:
        first = next((c for c in s if c.isalpha()), "")
        if first and not first.isupper():
            return False

    return True


def iter_sentences(
    texts: Iterable[str],
    splitter: str = "pysbd",
    min_chars: int = 16,
    max_chars: int = 2000,
    max_sentences_per_text: Optional[int] = None,
    sample_strategy: str = "random",
    seed: int = 42,
    quality_filter: bool = True,
    quality_kwargs: Optional[Dict] = None,
) -> Iterator[str]:
    """Stream sentences out of an iterable of raw documents.

    The splitter is the only language-dependent component in the data
    pipeline; the rest of the corruption stack is MLM-driven and language-
    agnostic.

    Parameters
    ----------
    max_sentences_per_text : Optional[int]
        Cap on the number of length-filtered sentences yielded per source
        document. `None` (default) yields all qualifying sentences.
    sample_strategy : str
        How to choose which sentences to keep when capping:
          * "head"    — first N qualifying sentences (deterministic, cheapest)
          * "random"  — uniform random N (per-doc RNG seeded by `seed` + index)
          * "stride"  — evenly spaced across the document
    seed : int
        Base RNG seed for the "random" / "stride" strategies.
    quality_filter : bool
        If True (default), drop strings that fail `looks_like_sentence`
        (list items, URLs, ASCII separators, sub-3-word fragments, missing
        terminal punctuation, mostly-non-letter content). Disable to inspect
        the raw splitter output or to relax for casual / multilingual data.
    quality_kwargs : Optional[Dict]
        Forwarded as keyword arguments to `looks_like_sentence` so callers
        can tighten or relax individual knobs without disabling the filter
        entirely.
    """
    qkw = quality_kwargs or {}
    if sample_strategy not in ("head", "random", "stride"):
        raise ValueError(f"unknown sample_strategy: {sample_strategy!r}")

    if splitter == "pysbd":
        try:
            import pysbd
        except ImportError as e:
            raise RuntimeError("pysbd is required for sentence splitting") from e
        seg = pysbd.Segmenter(language="en", clean=False)
        split_one = seg.segment
    elif splitter == "nltk":
        try:
            import nltk
            nltk.data.find("tokenizers/punkt_tab")
        except (ImportError, LookupError):
            import nltk  # type: ignore
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        split_one = sent_tokenize
    else:
        raise ValueError(f"unknown sentence splitter: {splitter!r}")

    doc_rng = random.Random(seed)
    for doc_idx, doc in enumerate(texts):
        try:
            sents = split_one(doc)
        except Exception:
            continue

        # Length-filter first so the cap applies to qualifying sentences only.
        filtered: List[str] = []
        for s in sents:
            s = s.strip()
            if not (min_chars <= len(s) <= max_chars):
                continue
            if quality_filter and not looks_like_sentence(s, **qkw):
                continue
            filtered.append(s)
        if not filtered:
            continue

        if max_sentences_per_text is not None and len(filtered) > max_sentences_per_text:
            n = int(max_sentences_per_text)
            if sample_strategy == "head":
                filtered = filtered[:n]
            elif sample_strategy == "random":
                rng = random.Random(seed + doc_idx)
                idx = rng.sample(range(len(filtered)), n)
                idx.sort()
                filtered = [filtered[i] for i in idx]
            else:  # stride
                step = len(filtered) / n
                idx = sorted({int(i * step) for i in range(n)})[:n]
                filtered = [filtered[i] for i in idx]

        for s in filtered:
            yield s


# ---------------------------------------------------------------------------
# Corruption cache loader
# ---------------------------------------------------------------------------
class CorruptionDataset(IterableDataset):
    """Streams training samples from a corruption cache (JSONL shards).

    Each record looks like (see README §8.2):
        {
          "source_sent_id": str,
          "bucket": str,                      # "identity" / "single_op" /
                                              # "compound_2_3" / "compound_4_plus"
          "N_total": int,                     # number of ops (0 for identity)
          "op_types": [...],                  # per-op type (REPL/INS/DEL) in
                                              # application order
          "x_token_ids": [...],
          "x_prime_token_ids": [...],
          "editor_input_token_ids": [...],
          "editor_target_token_ids": [...],
          "tagger_gold": [...],               # values in {KEEP, REPL, INS, DEL}
          "z_X_topk": [{"f": int, "v": float}, ...],
          "z_X_prime_topk": [{"f": int, "v": float}, ...],
          "ins_span_lengths": [int, ...],     # one entry per INS gap
          "del_span_lengths": [int, ...],     # one entry per DEL span
          "filter_telemetry": {...}
        }
    """

    def __init__(
        self,
        cache_dir: str,
        shuffle: bool = True,
        seed: int = 42,
        infinite: bool = True,
        glob_pattern: str = "shard-*.jsonl.gz",
    ):
        self.cache_dir = Path(cache_dir)
        self.shards = sorted(self.cache_dir.glob(glob_pattern))
        if not self.shards:
            raise FileNotFoundError(
                f"no corruption shards matching {glob_pattern} in {cache_dir}"
            )
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        meta_path = self.cache_dir / "meta.json"
        self.meta: Dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.d_sae = int(self.meta.get("d_sae", 0))

    def _iter_shard(self, path: Path) -> Iterator[Dict]:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def __iter__(self) -> Iterator[Dict]:
        worker = get_worker_info()
        epoch_seed = self.seed
        shards = list(self.shards)
        worker_label = (
            f"{worker.id}/{worker.num_workers}" if worker is not None else "main"
        )
        if worker is not None:
            my_shard_ids = list(range(worker.id, len(shards), worker.num_workers))
            if not my_shard_ids:
                # More workers than shards — this worker has nothing to read.
                # Return cleanly so the DataLoader marks the worker as done
                # rather than spinning the infinite loop with no yields and
                # blocking the round-robin fetch on this worker forever.
                print(
                    f"[CorruptionDataset] worker {worker_label} "
                    f"has 0 of {len(shards)} shards — exiting (raise num_workers "
                    f"only up to the shard count, or shard the cache finer)."
                )
                return
        else:
            my_shard_ids = list(range(len(shards)))
        while True:
            order = list(my_shard_ids)
            if self.shuffle:
                random.Random(epoch_seed).shuffle(order)
            n_yielded = 0
            for i in order:
                for rec in self._iter_shard(shards[i]):
                    n_yielded += 1
                    yield rec
            if n_yielded == 0:
                # All assigned shards exist but contain no records (e.g. the
                # trailing empty shard corruption.py opens just before exit).
                # Without this guard, infinite=True would spin reading empty
                # files forever and the DataLoader's round-robin fetch would
                # block on this worker.
                print(
                    f"[CorruptionDataset] worker {worker_label} read 0 records "
                    f"across {len(my_shard_ids)} assigned shards "
                    f"({[shards[i].name for i in my_shard_ids]}) — exiting."
                )
                return
            if not self.infinite:
                break
            epoch_seed += 1


def _dense_topk(records: List[Dict], d_sae: int) -> torch.Tensor:
    z = torch.zeros(d_sae, dtype=torch.float32)
    for r in records:
        z[int(r["f"])] = float(r["v"])
    return z


class CorruptionCollator:
    """Pads a batch of corruption-cache records into editor/tagger inputs.

    Output tensors:
      tagger_input_ids       (B, T_tag) long
      tagger_attention_mask  (B, T_tag) long
      tagger_gold            (B, T_tag) long       -100 outside text
      editor_input_ids       (B, T_ed) long
      editor_attention_mask  (B, T_ed) long
      editor_target_ids      (B, T_ed) long       -100 = ignore
      z_X                    (B, d_sae) float32   pool-max top-K_train
      z_X_prime              (B, d_sae) float32
      ins_span_length        (B,) long
    """

    def __init__(
        self,
        d_sae: int,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ):
        self.d_sae = int(d_sae)
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        # tagger side uses x_prime_token_ids as the visible input
        T_tag = max(len(r["x_prime_token_ids"]) for r in batch)
        T_ed = max(len(r["editor_input_token_ids"]) for r in batch)

        tag_ids = np.full((B, T_tag), self.pad_token_id, dtype=np.int64)
        tag_attn = np.zeros((B, T_tag), dtype=np.int64)
        tag_gold = np.full((B, T_tag), self.ignore_index, dtype=np.int64)

        ed_ids = np.full((B, T_ed), self.pad_token_id, dtype=np.int64)
        ed_attn = np.zeros((B, T_ed), dtype=np.int64)
        ed_tgt = np.full((B, T_ed), self.ignore_index, dtype=np.int64)

        z_X = torch.zeros(B, self.d_sae, dtype=torch.float32)
        z_X_prime = torch.zeros(B, self.d_sae, dtype=torch.float32)
        ins_span_length = torch.zeros(B, dtype=torch.long)

        for b, r in enumerate(batch):
            xp = r["x_prime_token_ids"]
            tag_ids[b, :len(xp)] = xp
            tag_attn[b, :len(xp)] = 1
            gold = r["tagger_gold"]
            tag_gold[b, :len(gold)] = gold

            ei = r["editor_input_token_ids"]
            ed_ids[b, :len(ei)] = ei
            ed_attn[b, :len(ei)] = 1
            et = r["editor_target_token_ids"]
            ed_tgt[b, :len(et)] = et

            z_X[b] = _dense_topk(r["z_X_topk"], self.d_sae)
            z_X_prime[b] = _dense_topk(r["z_X_prime_topk"], self.d_sae)
            # ins_span_length is a single scalar for the length-head trainer.
            # For compound samples with multiple INS gaps we take the first
            # gap's length as a temporary simplification; the length head
            # will need refactoring to handle multi-gap supervision.
            spans = r.get("ins_span_lengths") or ([r["ins_span_length"]] if "ins_span_length" in r else [])
            ins_span_length[b] = int(spans[0]) if spans else 0

        return {
            "tagger_input_ids": torch.from_numpy(tag_ids),
            "tagger_attention_mask": torch.from_numpy(tag_attn),
            "tagger_gold": torch.from_numpy(tag_gold),
            "editor_input_ids": torch.from_numpy(ed_ids),
            "editor_attention_mask": torch.from_numpy(ed_attn),
            "editor_target_ids": torch.from_numpy(ed_tgt),
            "z_X": z_X,
            "z_X_prime": z_X_prime,
            "ins_span_length": ins_span_length,
        }
