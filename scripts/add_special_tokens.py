"""
Idempotently add missing SAE-LEWIS special tokens to an ALREADY merged +
expanded LLM2Vec checkpoint (output of mcgill_merge_and_expand.py).

v2 of the editor needs a [SEP] token to separate the source text x' from
the edit template x'_c (LEWIS's `x SEP x_c` input). Checkpoints produced
before v2 lack it; this script appends any missing specials with HF's
mean-of-existing initialisation and saves in place (or to --output-dir).

Existing token ids are NEVER changed — new tokens are appended at the end
of the vocabulary — so corruption caches tokenized with the old checkpoint
remain valid.

Usage:
    python scripts/add_special_tokens.py --checkpoint runs/.../final
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Merged+expanded LLM2Vec checkpoint dir.")
    p.add_argument("--output-dir", default=None,
                   help="Where to save (default: in place, --checkpoint).")
    p.add_argument("--tokens", nargs="+",
                   default=["[MASK]", "[INS]", "[DEL]", "[SEP]"],
                   help="Specials that must exist; missing ones are added.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir or args.checkpoint)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    unk = tokenizer.unk_token_id
    missing = []
    for tok in args.tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tok == "[MASK]" and tokenizer.mask_token is not None:
            continue
        if tid is None or tid == unk:
            missing.append(tok)

    if not missing:
        print(f"[add-specials] all of {args.tokens} already present — "
              f"nothing to do.")
        return

    print(f"[add-specials] loading {args.checkpoint} (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=dtype, attn_implementation="sdpa",
    )
    old_vocab = model.config.vocab_size

    added = 0
    for tok in missing:
        if tok == "[MASK]":
            added += tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        else:
            added += tokenizer.add_special_tokens(
                {"additional_special_tokens": [tok]})
    model.resize_token_embeddings(len(tokenizer))
    print(f"[add-specials] added {missing}; vocab {old_vocab} → "
          f"{model.config.vocab_size} (new rows mean-of-existing init)")

    # NEVER save straight into the source directory: transformers mmap-loads
    # the .bin shards, and save_pretrained writing over the very files that
    # back the live tensors dies with SIGBUS mid-write, corrupting the
    # checkpoint. Save to a temp dir on the same filesystem, free the model
    # (and with it the mmaps), then swap the files in.
    tmp_dir = Path(tempfile.mkdtemp(prefix=".add_specials_tmp_",
                                    dir=str(out_dir.parent)))
    try:
        # Gemma ties lm_head to embed_tokens; safetensors refuses tied
        # tensors, so use the legacy format (same as mcgill_merge_and_expand).
        model.save_pretrained(tmp_dir, safe_serialization=False)
        tokenizer.save_pretrained(tmp_dir)
        print(f"[add-specials] wrote {tmp_dir}; swapping into {out_dir}")

        del model
        gc.collect()

        # Remove stale weight files in the destination (the shard layout may
        # differ from the source's), then move the fresh files in.
        if out_dir.exists():
            for pat in ("pytorch_model*.bin", "model*.safetensors",
                        "*.index.json"):
                for stale in out_dir.glob(pat):
                    stale.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in tmp_dir.iterdir():
            shutil.move(str(src), str(out_dir / src.name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    meta_p = out_dir / "llm2vec_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        meta["vocab_size"] = len(tokenizer)
        for tok in ("[MASK]", "[INS]", "[DEL]", "[SEP]"):
            key = tok.strip("[]").lower() + "_token_id"
            meta[key] = tokenizer.convert_tokens_to_ids(tok)
        meta_p.write_text(json.dumps(meta, indent=2))
        print(f"[add-specials] updated {meta_p}")

    ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in args.tokens}
    print(f"[add-specials] token ids: {ids}")
    print(f"[add-specials] saved to {out_dir}")


if __name__ == "__main__":
    main()
