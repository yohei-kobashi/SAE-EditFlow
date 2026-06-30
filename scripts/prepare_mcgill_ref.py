"""
Assemble McGill-NLP's public LLM2Vec checkpoint into a self-contained
HF directory that `eval_llm2vec.py` can read.

McGill publishes the LLM2Vec recipe as a CHAIN of LoRA adapters on top
of Mistral-7B-Instruct-v0.2:

    base = mistralai/Mistral-7B-Instruct-v0.2
    + LoRA adapter: McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp
    + LoRA adapter: McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse

The adapter repos contain only adapter_config.json + adapter_model.safetensors
(no config.json), so `AutoTokenizer.from_pretrained(adapter_dir)` and our
eval script's `AutoModelForCausalLM.from_pretrained(...)` both fail.

This script:
  1. Downloads the base + both adapters into the HF cache.
  2. Loads base, applies MNTP adapter, merge_and_unload() bakes MNTP into base.
  3. Applies the SimCSE adapter on top, merge_and_unload() again.
  4. save_pretrained() the resulting GemmaForCausalLM (=> Mistral) as a
     plain HF dir.
  5. Writes a minimal `llm2vec_meta.json` so eval_llm2vec.py can detect
     the checkpoint as "complete" and pick its baseline LLM.

Idempotent: if --output-dir already has a `config.json` + `llm2vec_meta.json`
we skip everything.

Usage:
    python scripts/prepare_mcgill_ref.py \\
        --output-dir ./runs/mcgill_ref/llm2vec_simcse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="./runs/mcgill_ref/llm2vec_simcse",
        help="Where to write the merged HF checkpoint.",
    )
    p.add_argument(
        "--base-llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model for the LoRA adapters.",
    )
    p.add_argument(
        "--mntp-adapter",
        default="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    )
    p.add_argument(
        "--simcse-adapter",
        default="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
    )
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--force", action="store_true",
                   help="Re-merge even if output-dir looks complete.")
    return p.parse_args()


def _sample_weight(model) -> torch.Tensor:
    """Snapshot a representative weight for before/after comparison."""
    # Layer-0 q_proj is one of the canonical LoRA targets; if the adapter
    # is real it will be changed by merge_and_unload.
    return model.model.layers[0].self_attn.q_proj.weight.detach().clone()


def _peek_adapter_config(adapter_repo: str) -> dict:
    """Download just adapter_config.json (~1KB) to see target_modules etc."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return {}
    try:
        path = hf_hub_download(adapter_repo, filename="adapter_config.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def _load_adapter_with_remap(model, adapter_repo: str, adapter_name: str = "default"):
    """Manually load LoRA weights from `adapter_model.safetensors`, remapping
    legacy keys to the format current PEFT expects.

    Two known mismatches in McGill's checkpoints:
      1. Missing `.{adapter_name}.` infix (older PEFT save format).
      2. Missing one `.model.` level: McGill saved with LoRA attached to
         the inner MistralModel directly, so keys read
            base_model.model.layers.X.<sub>.lora_A.weight
         while PEFT-on-MistralForCausalLM expects
            base_model.model.model.layers.X.<sub>.lora_A.{name}.weight
         (one extra `model.` because the LoRA is wrapped inside the outer
         CausalLM's `.model` attribute).

    The helper auto-detects the expected prefix from the live model's
    state_dict, then transforms each safetensors key by computing the
    prefix difference. This avoids hard-coding the missing-`.model.`
    pattern (which would break for future adapters with other layouts).

    Returns the number of keys actually loaded.
    """
    import re
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download(adapter_repo, filename="adapter_model.safetensors")
    raw = load_file(weights_path)

    sample_keys = list(raw.items())[:2]
    print(f"[mcgill]   safetensors sample keys: {[k for k, _ in sample_keys]}")

    # Discover the expected key format from the live PEFT model.
    expected_lora_keys = [k for k in model.state_dict() if "lora_" in k]
    sample_expected = expected_lora_keys[:2] if expected_lora_keys else []
    print(f"[mcgill]   PEFT-expected sample keys: {sample_expected}")
    if not expected_lora_keys:
        print("[mcgill]   WARNING: live model has no lora_* keys — wrap step failed?")
        return 0

    # Use a representative pair to learn the structural transformation:
    # safetensors key → expected key. Pick a layer-0 q_proj entry from each.
    def _pick(keys, sub: str, suffix: str):
        for k in keys:
            if sub in k and k.endswith(suffix):
                return k
        return None

    src_key = _pick(list(raw.keys()), "layers.0", "q_proj.lora_A.weight")
    if src_key is None:
        # Fall back: any layer-0 lora_A.
        src_key = next((k for k in raw if "layers.0" in k and "lora_A" in k), None)
    dst_key = _pick(expected_lora_keys, "layers.0",
                    f"q_proj.lora_A.{adapter_name}.weight")
    if dst_key is None:
        dst_key = next(
            (k for k in expected_lora_keys
             if "layers.0" in k and "lora_A" in k
             and f".{adapter_name}." in k),
            None,
        )
    print(f"[mcgill]   sample mapping: {src_key}\n"
          f"                      -> {dst_key}")
    if src_key is None or dst_key is None:
        print("[mcgill]   could not derive remap; loading raw with no transform")
        remapped = raw
    else:
        # Compute the prefix and suffix transforms. We split each key into
        # (prefix, tail-after-`layers.0`) and (head, suffix-after-`lora_A`).
        # Then apply: new = dst_prefix + src_middle + dst_suffix.
        src_pre, src_mid = src_key.split("layers.0", 1)
        dst_pre, dst_mid = dst_key.split("layers.0", 1)
        # src_mid: ".self_attn.q_proj.lora_A.weight"
        # dst_mid: ".self_attn.q_proj.lora_A.default.weight"
        # The structural change is prefix-only AND lora_A/B suffix.
        src_lora_pat = re.compile(r"(lora_[AB])\.weight$")
        dst_lora_pat = f"\\1.{adapter_name}.weight"
        remapped: Dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            if not k.startswith(src_pre):
                # Unfamiliar key — keep as is (lets non-lora_* keys pass through).
                remapped[k] = v
                continue
            new_k = dst_pre + k[len(src_pre):]
            new_k = src_lora_pat.sub(dst_lora_pat, new_k)
            remapped[new_k] = v
        print(f"[mcgill]   remapped {len(remapped)} keys "
              f"(prefix {src_pre!r} → {dst_pre!r}, "
              f"+infix .{adapter_name}.)")

    # Load directly via state_dict; bypassing set_peft_model_state_dict
    # because it does its own (different) prefix stripping which conflicts
    # with our hand-computed remap.
    out = model.load_state_dict(remapped, strict=False)
    missing = len(out.missing_keys)
    unexpected = len(out.unexpected_keys)
    loaded = sum(1 for k in remapped if k in model.state_dict())
    print(f"[mcgill]   load_state_dict: missing={missing}, "
          f"unexpected={unexpected}, matched={loaded}")
    if loaded == 0:
        print("[mcgill]   first 3 missing : ", list(out.missing_keys)[:3])
        print("[mcgill]   first 3 unexpected: ", list(out.unexpected_keys)[:3])
    return loaded


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = out_dir / "config.json"
    meta = out_dir / "llm2vec_meta.json"
    if not args.force and cfg.exists() and meta.exists():
        print(f"[mcgill] {out_dir} already has config.json + "
              f"llm2vec_meta.json — skipping merge. Pass --force to redo.")
        return

    # PEFT is required for the LoRA stacking. Import lazily so the helpful
    # error message is the missing dep rather than a stack trace.
    try:
        from peft import PeftModel
    except ImportError as e:
        raise SystemExit(
            f"[mcgill] PEFT is required: `pip install peft`. Underlying "
            f"error: {e}"
        )

    # Inspect adapter configs before downloading the base — surfaces module-
    # naming or task-type mismatches early.
    for label, repo in [("MNTP", args.mntp_adapter),
                         ("SimCSE", args.simcse_adapter)]:
        cfg_blob = _peek_adapter_config(repo)
        target = cfg_blob.get("target_modules", "?")
        mts = cfg_blob.get("modules_to_save", None)
        task = cfg_blob.get("task_type", "?")
        rank = cfg_blob.get("r", "?")
        alpha = cfg_blob.get("lora_alpha", "?")
        print(f"[mcgill] {label} adapter ({repo}):")
        print(f"          target_modules    = {target}")
        print(f"          modules_to_save   = {mts}")
        print(f"          task_type r/alpha = {task} / {rank} / {alpha}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[mcgill] loading base {args.base_llm} (dtype={args.dtype})...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_llm,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    print(f"[mcgill] base loaded: vocab_size={base.config.vocab_size}")
    base_weight_orig = _sample_weight(base)

    print(f"[mcgill] applying MNTP adapter {args.mntp_adapter}...")
    mntp = PeftModel.from_pretrained(base, args.mntp_adapter)

    # Quick sanity: how many trainable params does the loaded adapter expose?
    n_lora = sum(p.numel() for n, p in mntp.named_parameters() if "lora" in n.lower())
    print(f"[mcgill]   PEFT-loaded LoRA params: {n_lora:,}")
    if n_lora == 0:
        raise SystemExit(
            "[mcgill] FATAL: PEFT loaded the adapter but found 0 LoRA params "
            "— target_modules in adapter_config.json don't match any module "
            "name in the base. Likely the McGill adapter targets a custom "
            "BiMistral subclass; use the llm2vec library instead "
            "(`pip install llm2vec`)."
        )

    # PEFT.from_pretrained may have silently left lora_A at Kaiming init +
    # lora_B at zero if the safetensors uses the legacy `lora_A.weight`
    # naming (no `.default.` infix). The helper re-loads the actual weights
    # with key remapping; harmless no-op when keys are already current.
    print("[mcgill]   hot-patching adapter weights via remap helper...")
    _load_adapter_with_remap(mntp, args.mntp_adapter, adapter_name="default")

    print("[mcgill] merging MNTP adapter into base...")
    base = mntp.merge_and_unload()
    delta = (_sample_weight(base) - base_weight_orig).abs().mean().item()
    print(f"[mcgill]   weight delta after MNTP merge: mean_abs = {delta:.2e}")
    if delta < 1e-8:
        raise SystemExit(
            "[mcgill] FATAL: MNTP merge produced zero weight change. The "
            "adapter loaded but its LoRA B matrices are all-zero "
            "(remapping didn't catch the key format). Inspect the "
            "safetensors keys printed above and update _load_adapter_with_remap."
        )

    base_weight_after_mntp = _sample_weight(base)
    print(f"[mcgill] applying SimCSE adapter {args.simcse_adapter}...")
    simcse = PeftModel.from_pretrained(base, args.simcse_adapter)
    n_lora2 = sum(p.numel() for n, p in simcse.named_parameters() if "lora" in n.lower())
    print(f"[mcgill]   PEFT-loaded LoRA params: {n_lora2:,}")
    print("[mcgill]   hot-patching adapter weights via remap helper...")
    _load_adapter_with_remap(simcse, args.simcse_adapter, adapter_name="default")
    print("[mcgill] merging SimCSE adapter...")
    final = simcse.merge_and_unload()
    delta2 = (_sample_weight(final) - base_weight_after_mntp).abs().mean().item()
    print(f"[mcgill]   weight delta after SimCSE merge: mean_abs = {delta2:.2e}")
    if delta2 < 1e-8:
        print("[mcgill] WARNING: SimCSE merge produced no weight change.")

    print(f"[mcgill] saving merged checkpoint to {out_dir}...")
    final.save_pretrained(out_dir, safe_serialization=False)

    # Use the base tokenizer — neither LoRA adapter changes the vocabulary
    # (canonical LLM2Vec uses the existing mask_token; no [INS]/[DEL]).
    tokenizer = AutoTokenizer.from_pretrained(args.base_llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(out_dir)

    # Minimal meta so eval_llm2vec.py treats this as a finished run.
    # Set dolma_max_files=0 because the public ckpt was trained on
    # Wikipedia subsets, not Dolma — no shard-level holdout applies and
    # we don't want eval to skip-ahead in our Dolma cache.
    meta_obj = {
        "base_llm": args.base_llm,
        "source_mntp_adapter": args.mntp_adapter,
        "source_simcse_adapter": args.simcse_adapter,
        "simcse": {"source": args.simcse_adapter},
        "lora": {"merged": True, "stacked_chain": ["mntp", "simcse"]},
        "vocab_size": len(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "dolma_max_files": 0,
    }
    meta.write_text(json.dumps(meta_obj, indent=2))
    print(f"[mcgill] wrote {meta}")
    print(f"[mcgill] done. Now run:\n"
          f"  SKIP_TRAIN=1 \\\n"
          f"  SIMCSE_DIR={out_dir} \\\n"
          f"  EVAL_DIR={out_dir.parent}/eval \\\n"
          f"  RUN_DIR={out_dir.parent} \\\n"
          f"  bash scripts/train_eval_llm2vec.sh")


if __name__ == "__main__":
    main()
