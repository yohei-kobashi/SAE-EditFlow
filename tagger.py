"""
SAE-conditioned 4-class tagger.

Forward (see README §4.2):

    [INT_amp, INT_sup, e(text_1), …, e(text_T)]
            │
            └── LLM2Vec'd Gemma (frozen) ─►  hidden states (B, T+2, d_model)
                                                │
                                         drop conditioning prefix
                                                │
                                    per-token 4-class MLP head
                                                │
                                 op ∈ {KEEP, REPL, INS, DEL}

Trainable: Proj_A (shared signature with editor; trained jointly or
independently), type_emb[0..2], a small 4-class head.

The tagger does not need [INS] or [DEL] embedding deltas because its input
is the user-provided (or corrupted) text and never contains those markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewis_ops import NUM_OPS
from model import BidirectionalLLM


class SAETagger(nn.Module):
    def __init__(
        self,
        llm2vec_dir: str,
        d_sae: int,
        head_hidden: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.encoder = BidirectionalLLM(llm2vec_dir, dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        d_model = self.encoder.config.hidden_size
        self.d_model = int(d_model)
        self.d_sae = int(d_sae)

        # Proj_A (own copy; checkpoints can be tied to editor's at deployment).
        self.proj_a = nn.Linear(d_sae, d_model, bias=True)
        nn.init.normal_(self.proj_a.weight, std=0.02)
        nn.init.zeros_(self.proj_a.bias)

        self.type_emb = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, NUM_OPS),
        )

    def cond_embeds(self, z_amp: torch.Tensor, z_sup: torch.Tensor) -> torch.Tensor:
        amp = self.proj_a(z_amp.to(self.proj_a.weight.dtype))
        sup = self.proj_a(z_sup.to(self.proj_a.weight.dtype))
        amp = amp + self.type_emb(torch.full((amp.shape[0],), 1, device=amp.device, dtype=torch.long))
        sup = sup + self.type_emb(torch.full((sup.shape[0],), 2, device=sup.device, dtype=torch.long))
        return torch.stack([amp, sup], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        attention_mask: torch.Tensor,      # (B, T)
        z_amp: torch.Tensor,               # (B, d_sae)
        z_sup: torch.Tensor,               # (B, d_sae)
        labels: Optional[torch.Tensor] = None,  # (B, T) -100 = ignore
        class_weights: Optional[torch.Tensor] = None,  # (NUM_OPS,)
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
        logits = self.head(h_text.to(self.head[0].weight.dtype))   # (B, T, NUM_OPS)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, NUM_OPS),
                labels.reshape(-1).long(),
                ignore_index=-100,
                weight=class_weights.to(logits.dtype) if class_weights is not None else None,
            )
        return {"loss": loss, "logits": logits}

    def predict_ops(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_amp: torch.Tensor,
        z_sup: torch.Tensor,
    ) -> torch.Tensor:
        """Return argmax ops (B, T) long."""
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, z_amp, z_sup)
        return out["logits"].argmax(dim=-1)

    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            "proj_a.weight": self.proj_a.weight.detach().cpu(),
            "proj_a.bias": self.proj_a.bias.detach().cpu(),
            "type_emb.weight": self.type_emb.weight.detach().cpu(),
        }
        for k, v in self.head.state_dict().items():
            sd[f"head.{k}"] = v.detach().cpu()
        return sd

    def load_trainable(self, sd: Dict[str, torch.Tensor]):
        self.proj_a.weight.data.copy_(sd["proj_a.weight"])
        self.proj_a.bias.data.copy_(sd["proj_a.bias"])
        self.type_emb.weight.data.copy_(sd["type_emb.weight"])
        head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
        self.head.load_state_dict(head_sd)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trainable": self.trainable_state_dict(),
            "d_sae": int(self.d_sae),
            "d_model": int(self.d_model),
        }, path)


def load_tagger_from_checkpoint(
    llm2vec_dir: str, ckpt_path: str, d_sae: int,
    dtype: torch.dtype = torch.bfloat16,
) -> SAETagger:
    blob = torch.load(ckpt_path, map_location="cpu")
    tagger = SAETagger(llm2vec_dir, d_sae=d_sae, dtype=dtype)
    tagger.load_trainable(blob["trainable"])
    return tagger
