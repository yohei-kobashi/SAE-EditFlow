"""
Stage 0: precompute Gemma-2 + Gemma Scope SAE per-LLM-token sparse features
over sentence-segmented Dolma.

Layout (see data.py for the reader):

    <out_dir>/meta.json
    <out_dir>/texts.jsonl              one {"text": ...} per line, sentence i ↔ line i
    <out_dir>/text_offsets.npy         int64 (N+1,) byte offsets into texts.jsonl

    <out_dir>/input_ids.bin            int32   (total_tokens,)        per-LLM-token vocab ID
    <out_dir>/sae_indices.bin          int32   (total_tokens, top_l)  top-L SAE indices
    <out_dir>/sae_values.bin           float16 (total_tokens, top_l)  top-L SAE values
    <out_dir>/sae_doc_offsets.npy      int64   (N+1,)                 row range per sentence

    <out_dir>/threshold.npy            float32 (d_sae,)               JumpReLU threshold
    <out_dir>/mu.npy                   float32 (d_sae,)               per-feature mean act

The unit of segmentation is the SENTENCE: documents are split by PySBD (or
NLTK) before any further processing. The corruption pipeline at §6.2 operates
on individual sentences as well, so the SAE cache and corruption cache share
granularity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import set_seed

from data import download_dolma_shards, iter_dolma_texts, iter_sentences
from model import SAEFeatureExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-cache-dir", default="./dolma_cache")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", default="layer_12/width_16k/average_l0_82/params.npz",
                   help="Path inside --sae-repo. Available L0 values for "
                        "google/gemma-scope-2b-pt-res layer_12/width_16k are "
                        "{22, 41, 82, 176, 445}.")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"], default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--llm-max-length", type=int, default=256)
    p.add_argument("--doc-max-chars", type=int, default=20000)
    p.add_argument("--sent-min-chars", type=int, default=16)
    p.add_argument("--sent-max-chars", type=int, default=2000)
    p.add_argument("--sentence-splitter", choices=["pysbd", "nltk"], default="pysbd")
    p.add_argument("--max-sentences-per-text", type=int, default=None,
                   help="Cap on qualifying sentences kept per source document. "
                        "None = use every sentence.")
    p.add_argument("--sentence-sample-strategy",
                   choices=["head", "random", "stride"], default="random",
                   help="Per-document sampling when --max-sentences-per-text "
                        "is set. Default 'random' to avoid the lead-bias of "
                        "'head' (document-initial sentences are often titles / "
                        "lead-ins). Strategy has no effect when "
                        "--max-sentences-per-text is None.")
    p.add_argument("--no-quality-filter", action="store_true",
                   help="Disable the looks-like-sentence heuristic filter.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-l", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--max-sentences", type=int, default=None,
                   help="Optional cap on the number of cached sentences.")
    p.add_argument("--seed", type=int, default=42)
    # Resume is all-or-nothing here: SAE caches use length-prefix binary
    # files that are awkward to truncate mid-write, so we just skip the
    # whole stage if meta.json (the last thing written) exists.
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Default. Skip the entire stage if a previous "
                        "complete run (meta.json) already exists under "
                        "--out-dir.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore any existing cache files and start fresh "
                        "(overwrites previous output).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and (out_dir / "meta.json").exists():
        print(f"[stage0] RESUME: {out_dir / 'meta.json'} already exists; nothing to do")
        return

    shard_paths = download_dolma_shards(args.data_cache_dir, max_files=args.max_files)
    print(f"[stage0] {len(shard_paths)} Dolma shards")

    extractor = SAEFeatureExtractor(
        llm_name=args.llm,
        sae_repo=args.sae_repo,
        sae_path=args.sae_path,
        sae_layer=args.sae_layer,
        sae_type=args.sae_type,
        sae_k=args.sae_k,
    ).to(args.device)
    extractor.eval()
    d_sae = extractor.d_sae
    top_l = int(args.top_l)
    print(f"[stage0] d_sae={d_sae} top_l={top_l}")

    paths = {
        "sae_indices": out_dir / "sae_indices.bin",
        "sae_values": out_dir / "sae_values.bin",
        "input_ids": out_dir / "input_ids.bin",
        "texts": out_dir / "texts.jsonl",
        "text_off": out_dir / "text_offsets.npy",
        "sae_doc_off": out_dir / "sae_doc_offsets.npy",
        "threshold": out_dir / "threshold.npy",
        "mu": out_dir / "mu.npy",
        "meta": out_dir / "meta.json",
    }

    n_sents = 0
    text_offsets: List[int] = [0]
    sae_doc_offsets: List[int] = [0]
    cur_tok = 0
    t0 = time.time()
    batch: List[str] = []
    mu_accum = np.zeros(d_sae, dtype=np.float64)

    text_stream = iter_dolma_texts(
        shard_paths, min_chars=64, text_max_chars=args.doc_max_chars,
    )
    sent_stream = iter_sentences(
        text_stream,
        splitter=args.sentence_splitter,
        min_chars=args.sent_min_chars,
        max_chars=args.sent_max_chars,
        max_sentences_per_text=args.max_sentences_per_text,
        sample_strategy=args.sentence_sample_strategy,
        seed=args.seed,
        quality_filter=not args.no_quality_filter,
    )

    with open(paths["sae_indices"], "wb") as fi, \
         open(paths["sae_values"], "wb") as fv, \
         open(paths["input_ids"], "wb") as fid, \
         open(paths["texts"], "wb") as ft:

        def flush(batch_texts: List[str]):
            nonlocal n_sents, cur_tok
            if not batch_texts:
                return
            results = extractor.extract_per_token_sparse(
                batch_texts, max_length=args.llm_max_length, top_l=top_l,
            )
            for text, r in zip(batch_texts, results):
                line = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8") + b"\n"
                ft.write(line)
                text_offsets.append(text_offsets[-1] + len(line))

                fi.write(r["sae_indices"].astype(np.int32, copy=False).tobytes())
                fv.write(r["sae_values"].astype(np.float16, copy=False).tobytes())
                fid.write(r["input_ids"].astype(np.int32, copy=False).tobytes())

                idx_flat = r["sae_indices"].astype(np.int64, copy=False).ravel()
                val_flat = r["sae_values"].astype(np.float32, copy=False).ravel()
                np.add.at(mu_accum, idx_flat, val_flat)

                cur_tok += int(r["sae_indices"].shape[0])
                sae_doc_offsets.append(cur_tok)
                n_sents += 1

        pbar = tqdm(
            total=args.max_sentences, desc="[stage0]",
            unit="sent", dynamic_ncols=True, smoothing=0.05,
        )
        last_logged = 0
        for sent in sent_stream:
            batch.append(sent)
            if len(batch) >= args.batch_size:
                flush(batch)
                pbar.update(n_sents - last_logged)
                last_logged = n_sents
                batch = []
                if n_sents % args.log_every == 0:
                    rate = n_sents / max(1e-6, time.time() - t0)
                    print(f"[stage0] sents={n_sents} tokens={cur_tok} rate={rate:.1f} sents/s")
            if args.max_sentences is not None and n_sents >= args.max_sentences:
                break
        flush(batch)
        pbar.update(n_sents - last_logged)
        pbar.close()

    np.save(paths["text_off"], np.array(text_offsets, dtype=np.int64))
    np.save(paths["sae_doc_off"], np.array(sae_doc_offsets, dtype=np.int64))
    np.save(paths["threshold"],
            extractor.sae.threshold.detach().to(torch.float32).cpu().numpy())
    mu = (mu_accum / max(1, cur_tok)).astype(np.float32)
    np.save(paths["mu"], mu)

    meta = {
        "granularity": "sentence",
        "num_sentences": int(n_sents),
        "d_sae": int(d_sae),
        "top_l": int(top_l),
        "total_tokens": int(cur_tok),
        "sae_indices_dtype": "int32",
        "sae_values_dtype": "float16",
        "input_ids_dtype": "int32",
        "llm_max_length": int(args.llm_max_length),
        "llm": args.llm,
        "sae_repo": args.sae_repo,
        "sae_path": args.sae_path,
        "sae_layer": int(args.sae_layer),
        "sae_type": args.sae_type,
        "sae_k": (int(args.sae_k) if args.sae_k is not None else None),
        "sentence_splitter": args.sentence_splitter,
        "seed": int(args.seed),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    print(f"[stage0] done: N_sent={n_sents} tokens={cur_tok} d_sae={d_sae} cache={out_dir}")


if __name__ == "__main__":
    main()
