"""
Shared model classes for SAE-LEWIS.

Frozen components:
  * Gemma-2 (causal LM)             — used for SAE feature extraction (h_L)
                                       and as the fluency scorer in the ranker.
  * Gemma Scope SAE                  — JumpReLU or TopK.
  * LLM2Vec encoder                  — Gemma backbone with bidirectional
                                       attention; produced by `train_llm2vec.py`.
  * Gemma LM head                    — weight-tied to the input embedding;
                                       reused by the editor.

Trainable components (defined in tagger.py / editor.py):
  * Proj_A : ℝ^{d_sae} → ℝ^{d_model}
  * Tagger 6-class head
  * type_emb[0..2]
  * embedding rows for [INS] and [DEL]
"""

from __future__ import annotations

import types
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


# ---------------------------------------------------------------------------
# JumpReLU SAE — Gemma Scope convention (frozen)
# ---------------------------------------------------------------------------
class JumpReLUSAE(nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        super().__init__()
        self.register_buffer("W_enc", W_enc, persistent=False)
        self.register_buffer("W_dec", W_dec, persistent=False)
        self.register_buffer("b_enc", b_enc, persistent=False)
        self.register_buffer("b_dec", b_dec, persistent=False)
        self.register_buffer("threshold", threshold, persistent=False)

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[1])

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    @classmethod
    def from_gemma_scope(cls, repo_id: str, path_in_repo: str,
                         dtype: torch.dtype = torch.float32) -> "JumpReLUSAE":
        local = hf_hub_download(repo_id=repo_id, filename=path_in_repo)
        npz = np.load(local)
        return cls(
            W_enc=torch.tensor(npz["W_enc"], dtype=dtype),
            W_dec=torch.tensor(npz["W_dec"], dtype=dtype),
            b_enc=torch.tensor(npz["b_enc"], dtype=dtype),
            b_dec=torch.tensor(npz["b_dec"], dtype=dtype),
            threshold=torch.tensor(npz["threshold"], dtype=dtype),
        )


class TopKSAE(nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, k):
        super().__init__()
        self.register_buffer("W_enc", W_enc, persistent=False)
        self.register_buffer("W_dec", W_dec, persistent=False)
        self.register_buffer("b_enc", b_enc, persistent=False)
        self.register_buffer("b_dec", b_dec, persistent=False)
        self.register_buffer(
            "threshold", torch.ones(W_enc.shape[0], dtype=W_enc.dtype),
            persistent=False,
        )
        self.k = int(k)

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[0])

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc.T + self.b_enc
        k = min(self.k, pre.shape[-1])
        topk_vals, topk_idx = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    @classmethod
    def from_state(cls, repo_id: str, path_in_repo: str, k: int,
                   dtype: torch.dtype = torch.float32) -> "TopKSAE":
        local = hf_hub_download(repo_id=repo_id, filename=path_in_repo)
        state = torch.load(local, map_location="cpu")
        return cls(
            W_enc=state["W_enc"].to(dtype),
            W_dec=state["W_dec"].to(dtype),
            b_enc=state["b_enc"].to(dtype),
            b_dec=state["b_dec"].to(dtype),
            k=k,
        )


def load_sae(sae_type: str, sae_repo: str, sae_path: str,
             sae_k: Optional[int] = None, dtype: torch.dtype = torch.float32):
    if sae_type == "jumprelu":
        return JumpReLUSAE.from_gemma_scope(sae_repo, sae_path, dtype=dtype)
    if sae_type == "topk":
        if sae_k is None:
            raise ValueError("sae_type='topk' requires sae_k")
        return TopKSAE.from_state(sae_repo, sae_path, k=int(sae_k), dtype=dtype)
    raise ValueError(f"unknown sae_type: {sae_type!r}")


# ---------------------------------------------------------------------------
# Frozen LLM + SAE → per-LLM-token sparse features
# ---------------------------------------------------------------------------
class SAEFeatureExtractor(nn.Module):
    """Frozen Gemma + frozen SAE. Returns per-token sparse SAE features
    (top-L) for a batch of texts. Also exposes a `pool_max_topk` utility for
    sentence-level pooling used by diff-based conditioning."""

    def __init__(
        self,
        llm_name: str = "google/gemma-2-2b",
        sae_repo: str = "google/gemma-scope-2b-pt-res",
        sae_path: str = "layer_12/width_16k/average_l0_71/params.npz",
        sae_layer: int = 12,
        sae_type: str = "jumprelu",
        sae_k: Optional[int] = None,
        llm_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.llm = AutoModel.from_pretrained(llm_name, torch_dtype=llm_dtype)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.sae = load_sae(sae_type, sae_repo, sae_path, sae_k=sae_k)
        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad_(False)
        self.sae_type = sae_type
        self.sae_k = int(sae_k) if sae_k is not None else None

        # Gemma Scope `layer_L` indexes the residual stream AFTER layer L,
        # which in HF is hidden_states[L+1].
        self.layer_idx = sae_layer + 1

    @property
    def d_sae(self) -> int:
        return self.sae.d_sae

    @torch.no_grad()
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Returns (T, d_sae) dense SAE features for a single sentence."""
        device = next(self.llm.parameters()).device
        enc = self.llm_tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        out = self.llm(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.layer_idx][0]            # (T, d_llm)
        z = self.sae.encode(h.to(self.sae.W_enc.dtype))     # (T, d_sae)
        return z

    @torch.no_grad()
    def encode_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Returns (T, d_sae) dense SAE features for a single token id sequence."""
        device = next(self.llm.parameters()).device
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        attn = (token_ids != self.llm_tokenizer.pad_token_id).long()
        token_ids = token_ids.to(device)
        attn = attn.to(device)
        out = self.llm(
            input_ids=token_ids, attention_mask=attn,
            output_hidden_states=True, use_cache=False,
        )
        h = out.hidden_states[self.layer_idx][0]
        z = self.sae.encode(h.to(self.sae.W_enc.dtype))
        return z

    @torch.no_grad()
    def pool_max_topk(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """Sentence-level pool-max over the SAE-feature axis, then top-K.

        Returns a (k,) tuple-like via topk: this routine returns a sparse
        dense (d_sae,) vector with only the top-K entries filled.
        """
        pooled = z.max(dim=0).values        # (d_sae,)
        k = min(k, pooled.numel())
        vals, idx = pooled.topk(k)
        out = torch.zeros_like(pooled)
        out[idx] = vals
        return out

    @torch.no_grad()
    def extract_per_token_sparse(
        self,
        texts: List[str],
        max_length: int = 512,
        top_l: int = 128,
    ) -> List[dict]:
        """Batched sparse extraction used by `precompute_sae.py`."""
        device = next(self.llm.parameters()).device
        enc = self.llm_tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        out = self.llm(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.layer_idx]
        feats = self.sae.encode(h.to(self.sae.W_enc.dtype))

        k = min(top_l, feats.shape[-1])
        topk_vals, topk_idx = feats.topk(k, dim=-1)
        topk_vals_h = topk_vals.detach().to(torch.float16).cpu().numpy()
        topk_idx_h = topk_idx.detach().to(torch.int32).cpu().numpy()
        ids_h = enc["input_ids"].to(torch.int32).cpu().numpy()
        attn_h = enc["attention_mask"].cpu().numpy().astype(bool)

        results: List[dict] = []
        for b in range(feats.shape[0]):
            mask = attn_h[b]
            results.append({
                "input_ids": ids_h[b][mask].copy(),
                "sae_indices": topk_idx_h[b][mask].copy(),
                "sae_values": topk_vals_h[b][mask].copy(),
            })
        return results


# ---------------------------------------------------------------------------
# Bidirectional LLM (LLM2Vec encoder runtime)
# ---------------------------------------------------------------------------
def _patch_attention_bidirectional(model):
    """Make a HF decoder-only LM behave bidirectionally — the LLM2Vec recipe.

    Mirrors the class-level patch from the canonical LLM2Vec repo
    (`llm2vec/models/bidirectional_gemma.py`):

      1. Override `_update_causal_mask` so the returned 4D mask has NO
         triangular (causal) component — only padding positions are masked
         to `-inf`. This matches `GemmaBiModel._update_causal_mask` which
         comments out the `torch.triu(..., diagonal=1)` line.
      2. Set `is_causal=False` on every self-attention module so SDPA /
         eager kernels do NOT apply causal masking via the `is_causal`
         override path. (Canonical: `ModifiedGemmaAttention.is_causal=False`.)
      3. Tag `config.is_decoder = False` so any downstream HF logic that
         branches on it treats this as an encoder.

    Step (1) alone is sufficient on transformers ≥ 4.41 when SDPA receives
    a non-None attn_mask (it ignores `is_causal` in that case), but step
    (2) makes the patch robust across attention implementations.
    """
    def _padding_mask_only(self, attention_mask, input_tensor, *args, **kwargs):
        if attention_mask is None:
            return None
        dtype = input_tensor.dtype
        min_value = torch.finfo(dtype).min
        B, T = attention_mask.shape
        mask = torch.zeros(B, 1, T, T, dtype=dtype, device=attention_mask.device)
        pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
        mask = mask.masked_fill(pad, min_value)
        return mask

    model._update_causal_mask = types.MethodType(_padding_mask_only, model)
    # Force `is_causal=False` on every self-attention module. Equivalent to
    # subclassing GemmaAttention with `self.is_causal = False`.
    for module in model.modules():
        if hasattr(module, "is_causal"):
            try:
                module.is_causal = False
            except Exception:
                pass
    if hasattr(model, "config"):
        model.config.is_decoder = False


class BidirectionalLLM(nn.Module):
    """Bidirectional Gemma backbone. Used both during LLM2Vec MNTP training
    and at editor / tagger inference. Tokenizer is shared with the causal
    Gemma so [INS] / [DEL] embeddings added before MNTP are also valid here.

    We use `attn_implementation="sdpa"` for speed. The bidirectional patch
    is still respected because Gemma's SDPA path
    (`transformers.integrations.sdpa_attention.sdpa_attention_forward`)
    computes `is_causal = q_len > 1 and attention_mask is None and
    module.is_causal`. Our `_patch_attention_bidirectional` sets
    `module.is_causal = False` AND `_update_causal_mask` returns a
    non-None padding-only 4D mask, so SDPA falls through to
    `is_causal=False` and uses the explicit mask (= bidirectional).
    Eager attention is 3-5x slower with no semantic gain here.
    """

    def __init__(self, model_name_or_path: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        _patch_attention_bidirectional(self.backbone)

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        return self.backbone(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )


# ---------------------------------------------------------------------------
# Causal Gemma loader (for ranker fluency and MLM-style next-token scoring)
# ---------------------------------------------------------------------------
def load_causal_gemma(llm_name: str, dtype: torch.dtype = torch.bfloat16):
    """Load a frozen causal Gemma + tokenizer for fluency scoring / MLM use."""
    model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(llm_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# ---------------------------------------------------------------------------
# MLMProvider — pluggable text-level MLM for corruption data generation
# ---------------------------------------------------------------------------
class MLMProvider(nn.Module):
    """A frozen, text-level masked language model used by `corruption.py`.

    The MLM is independent of the downstream editor / tagger. Inputs and
    outputs are TEXT (raw strings); the tokenizer of the MLM is encapsulated
    and never leaks to callers, so the corruption pipeline can mix and match
    any HF MLM regardless of how the downstream tokenizer (Gemma) splits the
    same text.

    Provider keys
    -------------
        modernbert-base     answerdotai/ModernBERT-base       # fast, modern, 8K context
        modernbert-large    answerdotai/ModernBERT-large
        deberta-v3-base     microsoft/deberta-v3-base         # very strong cloze
        deberta-v3-large    microsoft/deberta-v3-large
        bert-base           bert-base-uncased
        roberta-base        FacebookAI/roberta-base
        xlm-roberta-base    FacebookAI/xlm-roberta-base       # multilingual
    Any other string is forwarded verbatim to AutoTokenizer / AutoModel.
    """

    PRESETS = {
        "modernbert-base":  "answerdotai/ModernBERT-base",
        "modernbert-large": "answerdotai/ModernBERT-large",
        "deberta-v3-base":  "microsoft/deberta-v3-base",
        "deberta-v3-large": "microsoft/deberta-v3-large",
        "bert-base":        "bert-base-uncased",
        "bert-large":       "bert-large-uncased",
        "roberta-base":     "FacebookAI/roberta-base",
        "roberta-large":    "FacebookAI/roberta-large",
        "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    }

    def __init__(self, model_key_or_name: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        resolved = self.PRESETS.get(model_key_or_name, model_key_or_name)
        self.resolved_name = resolved
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        # ModernBERT defaults `reference_compile=True`, which JIT-compiles a
        # CUDA helper through Triton. On systems without Python development
        # headers (typical of HPC nodes) this crashes with a gcc error on
        # `Python.h`. Inference-only here, so disable torch.compile across all
        # MLM backends. The flag is silently ignored by other architectures.
        from_pretrained_kwargs = {
            "torch_dtype": dtype,
            "reference_compile": False,
        }
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(
                resolved, **from_pretrained_kwargs,
            )
        except TypeError:
            # Backend doesn't know `reference_compile` — retry without it.
            from_pretrained_kwargs.pop("reference_compile", None)
            self.model = AutoModelForMaskedLM.from_pretrained(
                resolved, **from_pretrained_kwargs,
            )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError(
                f"MLM {resolved} has no mask token; cannot be used as a corruption MLM."
            )
        self.mask_token = self.tokenizer.mask_token
        self.mask_id = int(self.tokenizer.mask_token_id)
        # Soft cap on input length to keep memory bounded.
        max_pos = getattr(self.model.config, "max_position_embeddings", 512) or 512
        self.max_seq_length = min(max_pos, 1024)

    @torch.no_grad()
    def predict_at_masks(
        self,
        masked_text: str,
        top_k: int = 10,
    ) -> List[List[str]]:
        """Run MLM on a string that ALREADY CONTAINS `mask_token`s.

        Returns a list of length `n_masks`, each element a list of top-K
        candidate TEXTS (whitespace-trimmed, skip_special_tokens=True).
        The MLM's tokenizer is hidden — callers stay at the text level.
        """
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(device)
        logits = self.model(**enc).logits[0]               # (T, vocab)
        mask_pos = (enc.input_ids[0] == self.mask_id).nonzero(as_tuple=True)[0]
        out: List[List[str]] = []
        for mp in mask_pos:
            probs = F.softmax(logits[mp].float(), dim=-1)
            topk_v, topk_i = probs.topk(min(top_k, probs.shape[-1]))
            cands: List[str] = []
            for i in topk_i.tolist():
                t = self.tokenizer.decode([i], skip_special_tokens=True).strip()
                if t:
                    cands.append(t)
            out.append(cands)
        return out


def add_special_tokens(tokenizer, model_with_resize, names: List[str]) -> List[int]:
    """Add special tokens to tokenizer and resize the model's embedding table.

    Returns the list of new token ids in the same order as `names`. Tokens
    that already exist in the tokenizer are returned with their existing id
    and no resize is performed for them.
    """
    new = [n for n in names if tokenizer.convert_tokens_to_ids(n) == tokenizer.unk_token_id]
    if new:
        tokenizer.add_special_tokens({"additional_special_tokens": new})
        model_with_resize.resize_token_embeddings(len(tokenizer))
    return [tokenizer.convert_tokens_to_ids(n) for n in names]
