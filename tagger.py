"""
SAE-conditioned two-tag tagger (v2 — LEWIS-faithful).

Forward (see README §4.2):

    [INT_amp, INT_sup, e(text_1), …, e(text_T)]
            │
            └── LLM2Vec'd Gemma (frozen) ─►  hidden states (B, T+2, d_model)
                                                │
                                         drop conditioning prefix
                                                │
                              ┌─────────────────┴──────────────────┐
                    per-token 3-class op head          per-token binary ins head
                              │                                    │
                    op ∈ {KEEP, REPL, DEL}          "insert phrase before this token?"

Following LEWIS (Reid & Zhong 2021, §2.1), each token carries TWO tags: a
binary insertion indicator for the boundary to its left and a 3-class
non-insertion op for the token itself. Insertion therefore never competes
with KEEP inside a softmax — in the v1 4-class design the INS tag sat on a
gap-adjacent token whose content is unchanged (i.e. a KEEP-looking token),
and held-out INS F1 was exactly 0.

Trainable: Proj_A (shared signature with editor), type_emb[0..2],
cond_scale, the two small heads, and — when lora_r > 0 (the
LEWIS-faithful default in train_tagger.py) — a fresh LoRA adapter on the
backbone's attention/MLP projections (LEWIS fine-tunes its RoBERTa
tagger; the tagger's adapter is independent of the editor's).

The tagger does not need [INS]/[DEL]/[SEP] embedding deltas because its
input is the user-provided (or corrupted) text and never contains those
markers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewis_ops import NUM_OPS3
from lora import apply_lora, load_lora_state_dict, lora_state_dict
from model import BidirectionalLLM


class SAETagger(nn.Module):
    def __init__(
        self,
        llm2vec_dir: str,
        d_sae: int,
        head_hidden: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        lora_r: int = 0,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        proj_a_mode: str = "learned",
        proj_a_rank: int = 32,
        w_dec: torch.Tensor = None,
    ):
        super().__init__()
        self.encoder = BidirectionalLLM(llm2vec_dir, dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # LEWIS fine-tunes its RoBERTa tagger; lora_r > 0 is the faithful
        # setting (each model gets its OWN fresh adapter — LEWIS's tagger
        # and generator are separate networks). 0 = frozen ablation.
        self.lora_cfg = None
        if lora_r > 0:
            n_wrapped = apply_lora(self.encoder.backbone, r=lora_r,
                                   alpha=lora_alpha, dropout=lora_dropout)
            self.lora_cfg = {"r": int(lora_r), "alpha": float(lora_alpha),
                             "dropout": float(lora_dropout)}
            print(f"[tagger] LoRA r={lora_r} on {n_wrapped} backbone modules")

        d_model = self.encoder.config.hidden_size
        self.d_model = int(d_model)
        self.d_sae = int(d_sae)

        # Proj_A (own copy; warm-started from the editor's). See SAEEditor
        # for the proj_a_mode semantics ("wdec-frozen" grounds the map in
        # the SAE decoder geometry with a trainable low-rank correction).
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
            self.proj_a_corr_A = nn.Parameter(
                torch.empty(self.proj_a_rank, d_sae, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.proj_a_corr_A, a=math.sqrt(5))
            self.proj_a_corr_B = nn.Parameter(
                torch.zeros(d_model, self.proj_a_rank, dtype=torch.float32))
        else:
            self.proj_a_corr_A = None
            self.proj_a_corr_B = None

        self.type_emb = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)

        # Conditioning scale calibration — same scheme as SAEEditor: RMS-
        # normalize each cond vector to the median token-embedding row RMS
        # (Gemma multiplies inputs_embeds by sqrt(hidden_size); z values are
        # raw SAE deltas of wildly varying magnitude). cond_scale is a
        # learnable global gain (init 1.0).
        with torch.no_grad():
            emb_w = self.encoder.get_input_embeddings().weight
            row_rms = emb_w.float().pow(2).mean(dim=-1).sqrt()
            target_rms = row_rms.median()
        self.register_buffer("cond_target_rms", target_rms.to(torch.float32))
        self.cond_scale = nn.Parameter(torch.ones(1))

        self.op_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, NUM_OPS3),
        )
        self.ins_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def _calibrate_cond(self, x: torch.Tensor) -> torch.Tensor:
        """RMS-normalize a (B, d_model) cond vector to the calibrated target."""
        # eps inside the sqrt — same NaN guard as SAEEditor._calibrate_cond.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x / rms * (self.cond_target_rms * self.cond_scale)

    def _proj(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj_a(z)
        if self.proj_a_corr_A is not None:
            x = x + (z @ self.proj_a_corr_A.t()) @ self.proj_a_corr_B.t()
        return x

    def proj_a_trainable_parameters(self):
        """Parameters governed by the Proj_A freeze/unfreeze schedule."""
        if self.proj_a_mode == "wdec-frozen":
            return [self.proj_a_corr_A, self.proj_a_corr_B]
        return list(self.proj_a.parameters())

    def cond_embeds(self, z_amp: torch.Tensor, z_sup: torch.Tensor) -> torch.Tensor:
        amp = self._calibrate_cond(self._proj(z_amp.to(self.proj_a.weight.dtype)))
        sup = self._calibrate_cond(self._proj(z_sup.to(self.proj_a.weight.dtype)))
        amp = amp + self.type_emb(torch.full((amp.shape[0],), 1, device=amp.device, dtype=torch.long))
        sup = sup + self.type_emb(torch.full((sup.shape[0],), 2, device=sup.device, dtype=torch.long))
        return torch.stack([amp, sup], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        attention_mask: torch.Tensor,      # (B, T)
        z_amp: torch.Tensor,               # (B, d_sae)
        z_sup: torch.Tensor,               # (B, d_sae)
        op_labels: Optional[torch.Tensor] = None,   # (B, T) 3-class; -100 = ignore
        ins_labels: Optional[torch.Tensor] = None,  # (B, T) {0,1}; -100 = ignore
        class_weights: Optional[torch.Tensor] = None,   # (NUM_OPS3,)
        ins_pos_weight: Optional[torch.Tensor] = None,  # scalar tensor
        ins_loss_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        with torch.no_grad():
            tok_embs = self.encoder.get_input_embeddings()(input_ids)

        cond = self.cond_embeds(z_amp, z_sup).to(tok_embs.dtype)
        full_embs = torch.cat([cond, tok_embs], dim=1)
        full_mask = torch.cat([
            torch.ones(B, 2, dtype=attention_mask.dtype, device=input_ids.device),
            attention_mask,
        ], dim=1)
        h = self.encoder(
            inputs_embeds=full_embs, attention_mask=full_mask,
        ).last_hidden_state                       # (B, T+2, d_model)
        h_text = h[:, 2:, :]                       # (B, T, d_model)
        h_text = h_text.to(self.op_head[0].weight.dtype)
        op_logits = self.op_head(h_text)                    # (B, T, NUM_OPS3)
        ins_logits = self.ins_head(h_text).squeeze(-1)      # (B, T)

        loss = None
        op_loss = None
        ins_loss = None
        if op_labels is not None:
            op_loss = F.cross_entropy(
                op_logits.reshape(-1, NUM_OPS3),
                op_labels.reshape(-1).long(),
                ignore_index=-100,
                weight=class_weights.to(op_logits.dtype) if class_weights is not None else None,
            )
            loss = op_loss
        if ins_labels is not None:
            flat_logits = ins_logits.reshape(-1)
            flat_labels = ins_labels.reshape(-1)
            valid = flat_labels != -100
            if valid.any():
                ins_loss = F.binary_cross_entropy_with_logits(
                    flat_logits[valid].float(),
                    flat_labels[valid].float(),
                    pos_weight=ins_pos_weight.float() if ins_pos_weight is not None else None,
                )
                loss = ins_loss * ins_loss_weight if loss is None else loss + ins_loss * ins_loss_weight
        return {"loss": loss, "op_loss": op_loss, "ins_loss": ins_loss,
                "op_logits": op_logits, "ins_logits": ins_logits}

    def predict_ops(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_amp: torch.Tensor,
        z_sup: torch.Tensor,
        ins_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (op3, ins_before), both (B, T) long."""
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, z_amp, z_sup)
        op3 = out["op_logits"].argmax(dim=-1)
        ins = (torch.sigmoid(out["ins_logits"].float()) >= ins_threshold).long()
        return op3, ins

    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            "proj_a.weight": self.proj_a.weight.detach().cpu(),
            "proj_a.bias": self.proj_a.bias.detach().cpu(),
            "type_emb.weight": self.type_emb.weight.detach().cpu(),
            "cond_scale": self.cond_scale.detach().cpu(),
        }
        for k, v in self.op_head.state_dict().items():
            sd[f"op_head.{k}"] = v.detach().cpu()
        for k, v in self.ins_head.state_dict().items():
            sd[f"ins_head.{k}"] = v.detach().cpu()
        if self.proj_a_corr_A is not None:
            sd["proj_a_corr_A"] = self.proj_a_corr_A.detach().cpu()
            sd["proj_a_corr_B"] = self.proj_a_corr_B.detach().cpu()
        if self.lora_cfg is not None:
            for n, t in lora_state_dict(self.encoder.backbone).items():
                sd[f"lora::{n}"] = t
        return sd

    def load_trainable(self, sd: Dict[str, torch.Tensor]):
        if any(k.startswith("head.") for k in sd):
            raise ValueError(
                "checkpoint uses the v1 4-class head — retrain the tagger "
                "with the v2 two-tag scheme (op_head + ins_head)")
        self.proj_a.weight.data.copy_(sd["proj_a.weight"])
        self.proj_a.bias.data.copy_(sd["proj_a.bias"])
        self.type_emb.weight.data.copy_(sd["type_emb.weight"])
        if "cond_scale" in sd:  # absent in pre-calibration checkpoints
            self.cond_scale.data.copy_(sd["cond_scale"])
        op_sd = {k[len("op_head."):]: v for k, v in sd.items() if k.startswith("op_head.")}
        self.op_head.load_state_dict(op_sd)
        ins_sd = {k[len("ins_head."):]: v for k, v in sd.items() if k.startswith("ins_head.")}
        self.ins_head.load_state_dict(ins_sd)
        if "proj_a_corr_A" in sd:
            if self.proj_a_corr_A is None:
                raise ValueError(
                    "checkpoint has a Proj_A low-rank correction but the "
                    "model was built with proj_a_mode != 'wdec-frozen' — "
                    "use load_tagger_from_checkpoint")
            self.proj_a_corr_A.data.copy_(sd["proj_a_corr_A"])
            self.proj_a_corr_B.data.copy_(sd["proj_a_corr_B"])
        lora_sd = {k[len("lora::"):]: v for k, v in sd.items()
                   if k.startswith("lora::")}
        if lora_sd and self.lora_cfg is None:
            raise ValueError(
                "checkpoint contains LoRA adapters but the model was built "
                "with lora_r=0 — use load_tagger_from_checkpoint, which "
                "reads the checkpoint's lora config")
        if self.lora_cfg is not None:
            load_lora_state_dict(self.encoder.backbone, lora_sd)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trainable": self.trainable_state_dict(),
            "d_sae": int(self.d_sae),
            "d_model": int(self.d_model),
            "lora": self.lora_cfg,
            "proj_a_mode": self.proj_a_mode,
            "proj_a_rank": int(self.proj_a_rank),
        }, path)


def load_tagger_from_checkpoint(
    llm2vec_dir: str, ckpt_path: str, d_sae: int,
    dtype: torch.dtype = torch.bfloat16,
) -> SAETagger:
    blob = torch.load(ckpt_path, map_location="cpu")
    lora = blob.get("lora") or {}
    tagger = SAETagger(
        llm2vec_dir, d_sae=d_sae, dtype=dtype,
        lora_r=int(lora.get("r", 0)),
        lora_alpha=float(lora.get("alpha", 32.0)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        proj_a_mode=blob.get("proj_a_mode", "learned"),
        proj_a_rank=int(blob.get("proj_a_rank", 32)),
    )
    tagger.load_trainable(blob["trainable"])
    return tagger
