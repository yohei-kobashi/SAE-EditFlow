"""
Diagnose whether `_patch_attention_bidirectional` actually changes the
encoder's behaviour on Gemma-2.

The NER probe showed BIT-IDENTICAL training loss between base Gemma +
bidir patch and base Gemma + no patch (epoch-1 loss = 0.2247 both),
suggesting the patch is a no-op on Gemma-2 under the current transformers
version + attn_implementation. This script confirms layer-by-layer.

Method
------
1. Load the model TWICE (separate instances): causal and bidir-patched.
2. Forward the same short prompt through both inner backbones with
   `output_hidden_states=True`.
3. For each of the (n_layers + 1) hidden states, report:
     - max_abs and mean_abs diff
     - cosine similarity at positions 0 and T-1
4. Repeat for `attn_implementation` ∈ {sdpa, eager}.

Interpretation
--------------
- All-layers IDENTICAL on a given attn_impl → patch is fully bypassed
  there. The `is_causal` flag and `_update_causal_mask` override don't
  reach the actual SDPA/eager call.
- Some layers differ, others don't → likely a per-layer-type bypass
  (Gemma-2 alternates full and sliding-window layers).
- All-layers differ → patch is taking effect; the NER no-op must come
  from somewhere else (e.g., representations differ but linear probe is
  insensitive to the difference).

Recommended fix paths
---------------------
- patch_broken_sdpa && patch_works_eager  →  switch our LLM2Vec model
  loads to `attn_implementation="eager"`. Cost is ~3-5x compute, but
  that's the canonical LLM2Vec choice anyway.
- patch_broken_both                        →  the override site has
  moved in current transformers; we'll need to monkey-patch the SDPA
  call directly or the layer-level mask construction.

Usage
-----
    python scripts/diagnose_bidir_patch.py
    python scripts/diagnose_bidir_patch.py --model-id mistralai/Mistral-7B-Instruct-v0.2
    python scripts/diagnose_bidir_patch.py --prompt "Custom test sentence."
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `from model import ...` work when invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import _patch_attention_bidirectional  # type: ignore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="google/gemma-2-2b")
    p.add_argument("--prompt",
                   default="The capital of France is Paris and it is famous.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--threshold", type=float, default=1e-4,
                   help="max_abs diff above which a layer is considered "
                        "'different' between causal and bidir.")
    p.add_argument("--skip-eager", action="store_true",
                   help="Only test sdpa. eager is ~3-5x slower so this skips "
                        "the second loop.")
    return p.parse_args()


def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _verify_patch_state(model_b) -> None:
    """Confirm the patch actually mutated the live model."""
    # 1. _update_causal_mask should be a MethodType bound to the inner backbone
    inner = model_b.model
    method = getattr(inner, "_update_causal_mask", None)
    if method is None:
        print("    [patch-verify] no _update_causal_mask attribute "
              "(unexpected for Gemma2Model)")
    else:
        is_bound = hasattr(method, "__self__") and method.__self__ is inner
        print(f"    [patch-verify] _update_causal_mask: bound={is_bound}, "
              f"func={getattr(method, '__func__', method).__qualname__}")

    # 2. Every attention module's is_causal should be False
    n_total = 0
    n_false = 0
    sample = []
    for name, mod in model_b.named_modules():
        if hasattr(mod, "is_causal") and "layers." in name:
            n_total += 1
            if mod.is_causal is False:
                n_false += 1
            if len(sample) < 3:
                sample.append((name.split(".layers.")[-1], mod.is_causal))
    print(f"    [patch-verify] attention modules with is_causal: "
          f"{n_false}/{n_total} set to False")
    print(f"    [patch-verify] sample: {sample}")


def _run_pair(model_id: str, prompt: str, attn_impl: str, args) -> Tuple[int, int]:
    """Returns (n_layers_identical, n_layers_total) for causal vs bidir."""
    print()
    print("=" * 80)
    print(f"  attn_implementation = {attn_impl}")
    print("=" * 80)
    dtype = _dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    T = int(inputs.input_ids.shape[1])
    print(f"  prompt   : {prompt!r}")
    print(f"  T tokens : {T}")
    print(f"  tokens   : {tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}")

    # Two separate instances so the patch on one can't leak.
    print(f"\n  loading causal model ({attn_impl})...")
    model_c = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=attn_impl,
    ).to(args.device).eval()
    print(f"  loading bidir model ({attn_impl})...")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=attn_impl,
    ).to(args.device).eval()

    print("\n  applying _patch_attention_bidirectional to bidir model...")
    _patch_attention_bidirectional(model_b.model)
    _verify_patch_state(model_b)

    # Layer-by-layer comparison through the inner backbone (skips lm_head).
    with torch.no_grad():
        out_c = model_c.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        out_b = model_b.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    print(f"\n  Layer-by-layer hidden-state diff "
          f"(threshold for 'identical': max_abs ≤ {args.threshold:.0e}):")
    print(f"  {'layer':<6s} {'max_abs':>14s} {'mean_abs':>14s} "
          f"{'cos@0':>10s} {'cos@T-1':>10s}  status")
    n_total = len(out_c.hidden_states)
    n_identical = 0
    for layer_idx, (hc, hb) in enumerate(zip(out_c.hidden_states, out_b.hidden_states)):
        diff = (hc - hb).abs()
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        cos_first = float(F.cosine_similarity(
            hc[0, 0].float(), hb[0, 0].float(), dim=0,
        ))
        cos_last = float(F.cosine_similarity(
            hc[0, -1].float(), hb[0, -1].float(), dim=0,
        ))
        identical = max_d <= args.threshold
        status = "= IDENTICAL" if identical else "≠ differs"
        if identical:
            n_identical += 1
        print(f"  {layer_idx:<6d} {max_d:>14.4e} {mean_d:>14.4e} "
              f"{cos_first:>10.4f} {cos_last:>10.4f}  {status}")
    print(f"\n  → {n_identical}/{n_total} layers IDENTICAL between causal and bidir")

    del model_c, model_b
    _free()
    return n_identical, n_total


def _print_verdict(model_id: str, sdpa: Tuple[int, int], eager: Tuple[int, int] | None):
    print()
    print("=" * 80)
    print(f"  VERDICT for {model_id}")
    print("=" * 80)
    s_id, s_n = sdpa
    print(f"  sdpa : {s_id}/{s_n} layers identical → "
          f"{'PATCH BROKEN' if s_id == s_n else ('partially bypassed' if s_id > 0 else 'patch effective')}")
    if eager is not None:
        e_id, e_n = eager
        print(f"  eager: {e_id}/{e_n} layers identical → "
              f"{'PATCH BROKEN' if e_id == e_n else ('partially bypassed' if e_id > 0 else 'patch effective')}")

    print()
    print("  Recommended action:")
    if eager is None:
        print("    (skipped eager; rerun without --skip-eager to disambiguate)")
        return
    s_broken = s_id == s_n
    e_broken = e_id == e_n
    if s_broken and not e_broken:
        print("    → SDPA bypasses our patch on this model. Switch every load")
        print("      site to attn_implementation='eager' for the bidir path:")
        print("        - eval_llm2vec.load_model")
        print("        - eval_ner_probe._load_encoder")
        print("        - model.BidirectionalLLM")
        print("        - train_llm2vec / train_simcse (training forward)")
        print("    Performance cost ~3-5x; canonical LLM2Vec uses eager anyway.")
    elif e_broken and s_broken:
        print("    → Patch broken on BOTH backends. The override site has moved")
        print("      in current transformers; need to patch deeper (e.g. the")
        print("      SDPA function itself or each Attention forward).")
    elif (not s_broken) and (not e_broken):
        print("    → Patch works on BOTH. The NER identical-loss anomaly must")
        print("      come from elsewhere — investigate the NER probe pipeline.")
    else:
        print(f"    → Inconsistent: sdpa_broken={s_broken}, eager_broken={e_broken}")


def main():
    args = parse_args()
    print(f"\n[diag-bidir] model: {args.model_id}")
    print(f"[diag-bidir] threshold for 'identical': {args.threshold:.0e}")
    print(f"[diag-bidir] dtype: {args.dtype}, device: {args.device}")

    sdpa_stats = _run_pair(args.model_id, args.prompt, "sdpa", args)
    eager_stats = None
    if not args.skip_eager:
        eager_stats = _run_pair(args.model_id, args.prompt, "eager", args)
    _print_verdict(args.model_id, sdpa_stats, eager_stats)


if __name__ == "__main__":
    main()
