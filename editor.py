"""
SAE-LEWIS bidirectional editor (v2 — LEWIS-faithful `x' [SEP] x'_c` input).

Forward (see README §4.3):

    [INT_amp, INT_sup, e(x'_1..T), e([SEP]), e(x'_c_1..T_c)]
            │                                       │
            └── LLM2Vec'd Gemma (frozen) ──────────►│
                                                    │
                                       Gemma LM head (frozen)
                                                    │
                                            token logits per position

Following LEWIS (Reid & Zhong 2021, eq. 2), the editor sees BOTH the source
text x' and the edit template x'_c (REPL → [MASK], insertion gaps → [INS]
slots, DEL tokens removed). Seeing the pre-edit words turns REPL filling
from unconstrained recovery into conditioned choice — in v1 the [MASK] hid
the source word entirely and the conditioning probe came out IGNORED.
Deletion is the tagger's decision alone (as in LEWIS, where BART never sees
deleted tokens); the editor no longer emits a [DEL] marker.

Trainable: Proj_A (d_sae → d_model), type_emb[0..2], cond_scale,
embedding-delta rows for [MASK] / [INS] / [SEP], and — when lora_r > 0
(the LEWIS-faithful default in train_editor_phaseA.py) — LoRA adapters on
the backbone's attention/MLP projections (lora.py). Embedding table and
LM head are always frozen.

At inference, template enumeration sets [INS] slot counts per gap; the
ranker then scores each template's argmax output.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora import apply_lora, load_lora_state_dict, lora_state_dict
from model import BidirectionalLLM


class SAEEditor(nn.Module):
    def __init__(
        self,
        llm2vec_dir: str,
        d_sae: int,
        dtype: torch.dtype = torch.bfloat16,
        train_token_ids: Optional[Dict[str, int]] = None,
        lora_r: int = 0,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        proj_a_mode: str = "learned",
        proj_a_rank: int = 32,
        w_dec: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        llm2vec_dir : str
            Path to the MNTP'd Gemma checkpoint (output of train_llm2vec.py).
        d_sae : int
            SAE feature dimension.
        train_token_ids : Optional[Dict[str, int]]
            Optional dict of {token_name: id} for tokens whose embedding rows
            should be trained (v2: [MASK], [INS], [SEP] — the markers that
            appear in the editor input; the editor emits no special tokens).
        lora_r : int
            LoRA rank for backbone adaptation (attention + MLP projections;
            embeddings and LM head stay frozen). 0 disables LoRA — the
            frozen-backbone ablation. LEWIS fine-tunes its generator's
            backbone; see lora.py.
        proj_a_mode : str
            "learned"     — random-init trainable Proj_A (pre-v3.1);
            "wdec-init"   — Proj_A initialized from the SAE decoder W_dec
                            (feature f ↦ its residual-stream direction),
                            then trained;
            "wdec-frozen" — Proj_A fixed to W_dec plus a trainable rank-
                            `proj_a_rank` correction: feature identity is
                            inherited from the SAE geometry and can never
                            be washed out (the OPAQUE-FLAG fix, §4.1).
        w_dec : Optional[Tensor]
            (d_sae, d_model) SAE decoder (model.load_sae_w_dec). Required
            for the wdec modes at TRAINING time; may be None when the
            weights are about to be overwritten by load_trainable.
        """
        super().__init__()
        self.encoder = BidirectionalLLM(llm2vec_dir, dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.lora_cfg = None
        if lora_r > 0:
            n_wrapped = apply_lora(self.encoder.backbone, r=lora_r,
                                   alpha=lora_alpha, dropout=lora_dropout)
            self.lora_cfg = {"r": int(lora_r), "alpha": float(lora_alpha),
                             "dropout": float(lora_dropout)}
            print(f"[editor] LoRA r={lora_r} on {n_wrapped} backbone modules")

        # LM head is the causal Gemma's output projection. We load the
        # causal model only to grab its LM head module — frozen.
        causal = AutoModelForCausalLM.from_pretrained(llm2vec_dir, torch_dtype=dtype)
        self.lm_head = causal.get_output_embeddings()
        for p in self.lm_head.parameters():
            p.requires_grad_(False)
        # Drop the rest of the causal model to free memory
        del causal

        d_model = self.encoder.config.hidden_size
        self.d_model = int(d_model)
        self.d_sae = int(d_sae)

        # Proj_A: d_sae → d_model. Float32 for stability. See proj_a_mode.
        if proj_a_mode not in ("learned", "wdec-init", "wdec-frozen"):
            raise ValueError(f"unknown proj_a_mode {proj_a_mode!r}")
        self.proj_a_mode = proj_a_mode
        self.proj_a_rank = int(proj_a_rank)
        self.proj_a = nn.Linear(d_sae, d_model, bias=True)
        nn.init.normal_(self.proj_a.weight, std=0.02)
        nn.init.zeros_(self.proj_a.bias)
        if proj_a_mode != "learned" and w_dec is not None:
            if tuple(w_dec.shape) != (d_sae, d_model):
                raise ValueError(f"W_dec shape {tuple(w_dec.shape)} != "
                                 f"({d_sae}, {d_model})")
            self.proj_a.weight.data.copy_(w_dec.t().to(self.proj_a.weight.dtype))
            nn.init.zeros_(self.proj_a.bias)
        if proj_a_mode == "wdec-frozen":
            for p in self.proj_a.parameters():
                p.requires_grad_(False)
            # Low-rank correction on top of the frozen W_dec map: adapts the
            # layer-12-residual geometry to the encoder's embedding space
            # without ever losing per-feature identity. B = 0 → exact W_dec
            # at step 0.
            self.proj_a_corr_A = nn.Parameter(
                torch.empty(self.proj_a_rank, d_sae, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.proj_a_corr_A, a=math.sqrt(5))
            self.proj_a_corr_B = nn.Parameter(
                torch.zeros(d_model, self.proj_a_rank, dtype=torch.float32))
        else:
            self.proj_a_corr_A = None
            self.proj_a_corr_B = None

        # type_emb[0..2]: text / amp / sup
        self.type_emb = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)

        # Conditioning scale calibration. Gemma multiplies inputs_embeds by
        # sqrt(hidden_size) internally, and z values are raw SAE activation
        # deltas whose magnitude varies per feature by orders of magnitude.
        # Each cond vector is RMS-normalized to the median token-embedding
        # row RMS so the prefix neither vanishes against type_emb nor pushes
        # the frozen encoder off distribution. cond_scale is a learnable
        # global gain on top (init 1.0).
        with torch.no_grad():
            emb_w = self.encoder.get_input_embeddings().weight
            row_rms = emb_w.float().pow(2).mean(dim=-1).sqrt()
            target_rms = row_rms.median()
        self.register_buffer("cond_target_rms", target_rms.to(torch.float32))
        self.cond_scale = nn.Parameter(torch.ones(1))

        # Trainable rows for [INS] and [DEL] embeddings within the encoder's
        # input embedding. We carry a parameter slice that is added at the
        # right token-id rows during forward — keeps the encoder frozen
        # while letting the new tokens learn.
        self.train_token_ids = train_token_ids or {}
        if self.train_token_ids:
            self.delta_emb = nn.Parameter(
                torch.zeros(len(self.train_token_ids), d_model, dtype=torch.float32)
            )
            # Map token_id → slot in delta_emb
            self._delta_slots = {
                int(tid): slot for slot, tid in enumerate(self.train_token_ids.values())
            }
        else:
            self.delta_emb = nn.Parameter(torch.zeros(0, d_model))
            self._delta_slots = {}

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def encoder_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.encoder.get_input_embeddings()(input_ids)
        if self._delta_slots:
            # Add the trainable delta at positions whose id ∈ delta_slots
            slots = torch.full_like(input_ids, -1, dtype=torch.long)
            for tid, slot in self._delta_slots.items():
                slots = torch.where(input_ids == tid, torch.tensor(slot, device=input_ids.device), slots)
            mask = (slots >= 0)
            if mask.any():
                slot_idx = slots.clamp(min=0)
                # Add per-position
                d_add = self.delta_emb.to(emb.dtype)[slot_idx]
                emb = emb + d_add * mask.unsqueeze(-1).to(emb.dtype)
        return emb

    def _calibrate_cond(self, x: torch.Tensor) -> torch.Tensor:
        """RMS-normalize a (B, d_model) cond vector to the calibrated target."""
        # eps INSIDE the sqrt: empty-conditioning samples (z all-zero) give
        # x == 0, and sqrt'(0) is infinite. With eps added after the sqrt the
        # gradient still blows up the instant Proj_A is unfrozen, NaN-ing the
        # whole run. Folding eps under the sqrt keeps the gradient finite.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x / rms * (self.cond_target_rms * self.cond_scale)

    def _proj(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Proj_A (+ the low-rank correction in wdec-frozen mode)."""
        x = self.proj_a(z)
        if self.proj_a_corr_A is not None:
            x = x + (z @ self.proj_a_corr_A.t()) @ self.proj_a_corr_B.t()
        return x

    def proj_a_trainable_parameters(self):
        """The parameters the Proj_A freeze/unfreeze schedule governs:
        proj_a itself (learned / wdec-init) or the low-rank correction
        (wdec-frozen — the W_dec base stays frozen forever)."""
        if self.proj_a_mode == "wdec-frozen":
            return [self.proj_a_corr_A, self.proj_a_corr_B]
        return list(self.proj_a.parameters())

    def cond_embeds(self, z_amp: torch.Tensor, z_sup: torch.Tensor) -> torch.Tensor:
        """Build the (B, 2, d_model) prefix conditioning tensor."""
        # Proj_A in float32; cast back to encoder dtype at the boundary.
        amp = self._calibrate_cond(self._proj(z_amp.to(self.proj_a.weight.dtype)))  # (B, d_model)
        sup = self._calibrate_cond(self._proj(z_sup.to(self.proj_a.weight.dtype)))
        # Add type embeddings
        type_amp = self.type_emb(torch.full((amp.shape[0],), 1, device=amp.device, dtype=torch.long))
        type_sup = self.type_emb(torch.full((sup.shape[0],), 2, device=sup.device, dtype=torch.long))
        amp = amp + type_amp
        sup = sup + type_sup
        return torch.stack([amp, sup], dim=1)                  # (B, 2, d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T) editor input (with [MASK] / [INS] / γ / etc.)
        attention_mask: torch.Tensor,      # (B, T)
        z_amp: torch.Tensor,               # (B, d_sae)
        z_sup: torch.Tensor,               # (B, d_sae)
        labels: Optional[torch.Tensor] = None,   # (B, T) -100 = ignore
        keep_loss_weight: float = 1.0,     # CE weight on copy positions (label == input)
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device

        tok_embs = self.encoder_embed(input_ids)               # (B, T, d_model)
        cond = self.cond_embeds(z_amp, z_sup).to(tok_embs.dtype)  # (B, 2, d_model)
        full_embs = torch.cat([cond, tok_embs], dim=1)         # (B, T+2, d_model)

        full_mask = torch.cat([
            torch.ones(B, 2, dtype=attention_mask.dtype, device=device),
            attention_mask,
        ], dim=1)

        h = self.encoder(
            inputs_embeds=full_embs, attention_mask=full_mask,
        ).last_hidden_state                                     # (B, T+2, d_model)

        # Drop the 2 prefix positions before projecting to logits
        h_text = h[:, 2:, :]                                    # (B, T, d_model)
        logits = self.lm_head(h_text.to(self.lm_head.weight.dtype))
        # (No output-side logit correction: since v2 the editor never emits
        # a special token — [DEL] output is gone, deletion being the
        # tagger's decision — so the delta rows only shape the INPUT
        # embeddings of [MASK]/[INS]/[SEP].)

        loss = None
        if labels is not None:
            if keep_loss_weight != 1.0:
                # Copy positions (label == input token) dominate the target
                # ~20:1 over edit positions ([MASK]/[INS] slots), so uniform
                # CE mostly trains copying and starves Proj_A of the
                # gradient that maps SAE diff features to lexical identity.
                # Down-weight copies; edit positions stay at weight 1.
                # (The x' [SEP] prefix carries label -100 → weight 0.)
                per_tok = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1).long(),
                    ignore_index=-100,
                    reduction="none",
                )
                flat_labels = labels.reshape(-1)
                valid = flat_labels != -100
                copy_pos = valid & (flat_labels == input_ids.reshape(-1))
                w = torch.ones_like(per_tok)
                w[copy_pos] = keep_loss_weight
                w[~valid] = 0.0
                loss = (per_tok * w).sum() / w.sum().clamp_min(1.0)
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1).long(),
                    ignore_index=-100,
                )

        return {"loss": loss, "logits": logits, "hidden_states": h_text}

    # ------------------------------------------------------------------
    # Save / load only trainable parts
    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            "proj_a.weight": self.proj_a.weight.detach().cpu(),
            "proj_a.bias": self.proj_a.bias.detach().cpu(),
            "type_emb.weight": self.type_emb.weight.detach().cpu(),
            "cond_scale": self.cond_scale.detach().cpu(),
        }
        if self.delta_emb.numel() > 0:
            sd["delta_emb"] = self.delta_emb.detach().cpu()
        if self.proj_a_corr_A is not None:
            sd["proj_a_corr_A"] = self.proj_a_corr_A.detach().cpu()
            sd["proj_a_corr_B"] = self.proj_a_corr_B.detach().cpu()
        if self.lora_cfg is not None:
            for n, t in lora_state_dict(self.encoder.backbone).items():
                sd[f"lora::{n}"] = t
        return sd

    def load_trainable(self, state_dict: Dict[str, torch.Tensor]):
        self.proj_a.weight.data.copy_(state_dict["proj_a.weight"])
        self.proj_a.bias.data.copy_(state_dict["proj_a.bias"])
        self.type_emb.weight.data.copy_(state_dict["type_emb.weight"])
        if "cond_scale" in state_dict:  # absent in pre-calibration checkpoints
            self.cond_scale.data.copy_(state_dict["cond_scale"])
        if "delta_emb" in state_dict and self.delta_emb.numel() > 0:
            self.delta_emb.data.copy_(state_dict["delta_emb"])
        if "proj_a_corr_A" in state_dict:
            if self.proj_a_corr_A is None:
                raise ValueError(
                    "checkpoint has a Proj_A low-rank correction but the "
                    "model was built with proj_a_mode != 'wdec-frozen' — "
                    "use load_editor_from_checkpoint")
            self.proj_a_corr_A.data.copy_(state_dict["proj_a_corr_A"])
            self.proj_a_corr_B.data.copy_(state_dict["proj_a_corr_B"])
        lora_sd = {k[len("lora::"):]: v for k, v in state_dict.items()
                   if k.startswith("lora::")}
        if lora_sd and self.lora_cfg is None:
            raise ValueError(
                "checkpoint contains LoRA adapters but the model was built "
                "with lora_r=0 — use load_editor_from_checkpoint, which "
                "reads the checkpoint's lora config")
        if self.lora_cfg is not None:
            load_lora_state_dict(self.encoder.backbone, lora_sd)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trainable": self.trainable_state_dict(),
            "d_sae": int(self.d_sae),
            "d_model": int(self.d_model),
            "train_token_ids": self.train_token_ids,
            "lora": self.lora_cfg,
            "proj_a_mode": self.proj_a_mode,
            "proj_a_rank": int(self.proj_a_rank),
        }, path)


def load_editor_from_checkpoint(
    llm2vec_dir: str, ckpt_path: str, d_sae: int,
    dtype: torch.dtype = torch.bfloat16,
) -> SAEEditor:
    blob = torch.load(ckpt_path, map_location="cpu")
    lora = blob.get("lora") or {}
    editor = SAEEditor(
        llm2vec_dir, d_sae=d_sae, dtype=dtype,
        train_token_ids=blob.get("train_token_ids", {}),
        lora_r=int(lora.get("r", 0)),
        lora_alpha=float(lora.get("alpha", 32.0)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        proj_a_mode=blob.get("proj_a_mode", "learned"),
        proj_a_rank=int(blob.get("proj_a_rank", 32)),
        # w_dec omitted: proj_a.weight is restored from the checkpoint.
    )
    editor.load_trainable(blob["trainable"])
    return editor
