"""
Post-training bridge for McGill's LLM2Vec pipeline → SAE-LEWIS downstream.

McGill's `run_mntp.py` / `run_simcse.py` save LoRA adapters (`adapter_config.json`
+ `adapter_model.safetensors`) on top of the base checkpoint. Our downstream
stages (corruption, tagger, editor, length_head, eval) expect a plain HF-format
`AutoModelForCausalLM` at `--llm2vec-dir` — with the LoRA changes already merged
in — plus a tokenizer that has `[INS]`, `[DEL]`, `[MASK]` in the vocabulary.

This script bridges the two:

  1. Load the base model.
  2. Stack the MNTP LoRA, merge_and_unload().
  3. Stack the SimCSE LoRA, merge_and_unload().
  4. Add `[INS]`, `[DEL]`, `[MASK]` to the tokenizer if missing.
  5. resize_token_embeddings() so the new rows are initialised via the
     mean-of-existing trick (same as `train_llm2vec.py`).
  6. Save the merged + expanded model + tokenizer as a drop-in
     `--llm2vec-dir` for downstream.

The expanded rows are NOT trained here (McGill's training didn't see them
either); they'll get trained in the editor/tagger stage. This matches our
previous full-FT setup's behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-2b",
                   help="HF id of the underlying base model.")
    p.add_argument("--mntp-adapter", required=True,
                   help="Path to McGill's MNTP training output (a directory "
                        "containing adapter_config.json + adapter_model.*).")
    p.add_argument("--simcse-adapter", default=None,
                   help="Path to McGill's SimCSE training output. Optional — "
                        "if omitted we save Bi+MNTP only.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the merged + expanded HF checkpoint.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--add-special-tokens", nargs="+",
                   default=["[INS]", "[DEL]", "[MASK]"],
                   help="Tokens to add to the tokenizer before saving. "
                        "[MASK] is only added if the tokenizer has no "
                        "mask_token already; [INS] / [DEL] are new either "
                        "way (SAE-LEWIS-specific).")
    return p.parse_args()


def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def _sample_weight(model) -> torch.Tensor:
    """Snapshot a q_proj row so we can verify each merge produced a real delta."""
    return model.model.layers[0].self_attn.q_proj.weight.detach().float().clone()


def main():
    args = parse_args()

    # peft is only needed here — the downstream stages don't touch it.
    from peft import PeftModel

    dtype = _dtype(args.dtype)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load base ----------------------------------------------------
    print(f"[bridge] loading base model {args.base_model} (dtype={args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[bridge] base vocab_size = {base.config.vocab_size}, "
          f"tokenizer len = {len(tokenizer)}")

    # ---- 2. Merge MNTP adapter -------------------------------------------
    print(f"[bridge] applying MNTP adapter {args.mntp_adapter}")
    pre_mntp = _sample_weight(base)
    peft_mntp = PeftModel.from_pretrained(base, args.mntp_adapter)
    print("[bridge]   merging MNTP → base")
    base = peft_mntp.merge_and_unload()
    delta_mntp = (_sample_weight(base) - pre_mntp).abs().mean().item()
    print(f"[bridge]   layer-0 q_proj delta (MNTP): {delta_mntp:.4e}")
    if delta_mntp < 1e-8:
        raise SystemExit("[bridge] FATAL: MNTP merge produced no weight change. "
                         "Adapter path may be wrong or weights zeroed.")

    # ---- 3. Merge SimCSE adapter (optional) ------------------------------
    if args.simcse_adapter is not None:
        print(f"[bridge] applying SimCSE adapter {args.simcse_adapter}")
        pre_simcse = _sample_weight(base)
        peft_simcse = PeftModel.from_pretrained(base, args.simcse_adapter)
        print("[bridge]   merging SimCSE → base")
        base = peft_simcse.merge_and_unload()
        delta_simcse = (_sample_weight(base) - pre_simcse).abs().mean().item()
        print(f"[bridge]   layer-0 q_proj delta (SimCSE): {delta_simcse:.4e}")
        if delta_simcse < 1e-8:
            print("[bridge] WARNING: SimCSE merge produced no weight change.")

    # ---- 4. Add SAE-LEWIS-specific tokens --------------------------------
    print(f"[bridge] adding special tokens: {args.add_special_tokens}")
    added_total = 0
    for tok in args.add_special_tokens:
        if tok == "[MASK]":
            if tokenizer.mask_token is None:
                added_total += tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        else:
            added_total += tokenizer.add_special_tokens(
                {"additional_special_tokens": [tok]}
            )
    print(f"[bridge]   added {added_total} new tokens; tokenizer len = {len(tokenizer)}")

    # ---- 5. Resize embeddings if we added anything -----------------------
    if added_total > 0 or base.config.vocab_size != len(tokenizer):
        old = base.config.vocab_size
        base.resize_token_embeddings(len(tokenizer))
        print(f"[bridge] resize_token_embeddings: {old} → {base.config.vocab_size}")
        print("[bridge]   new rows initialised via HF's mean-of-existing trick "
              "(same as our previous train_llm2vec.py)")
    ids = {tok: tokenizer.convert_tokens_to_ids(tok)
           for tok in ("[MASK]", "[INS]", "[DEL]")
           if tokenizer.convert_tokens_to_ids(tok) is not None}
    print(f"[bridge]   token ids: {ids}")

    # ---- 6. Save as drop-in --llm2vec-dir --------------------------------
    # Gemma ties lm_head.weight to embed_tokens.weight; safetensors refuses
    # tied tensors under the Trainer default save path, so use the legacy
    # binary format. Matches what our train_llm2vec.py has always done.
    print(f"[bridge] saving to {out_dir}")
    base.save_pretrained(out_dir, safe_serialization=False)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "base_llm": args.base_model,
        "mntp_source": str(args.mntp_adapter),
        "simcse_source": str(args.simcse_adapter) if args.simcse_adapter else None,
        "training_recipe": "McGill-NLP/llm2vec (vendored)",
        "vocab_size": len(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "ins_token_id": tokenizer.convert_tokens_to_ids("[INS]"),
        "del_token_id": tokenizer.convert_tokens_to_ids("[DEL]"),
        "lora": {
            "mntp":   {"merged": True, "source": str(args.mntp_adapter)},
            "simcse": ({"merged": True, "source": str(args.simcse_adapter)}
                       if args.simcse_adapter else None),
        },
        # Downstream shard-holdout eval expects this key. McGill trained on
        # Wikipedia so Dolma leakage isn't an issue — dolma_max_files=0.
        "dolma_max_files": 0,
        "simcse": ({
            "source": str(args.simcse_adapter),
            "merged": True,
        } if args.simcse_adapter else None),
    }
    (out_dir / "llm2vec_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bridge] wrote {out_dir}/llm2vec_meta.json")
    print("[bridge] done.")


if __name__ == "__main__":
    main()
