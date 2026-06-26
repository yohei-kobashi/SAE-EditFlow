"""
Length predictor head (ablation only).

Reads encoder hidden states at OP_INS positions and predicts the slot count
over {1..L_MAX}. Trained on INS samples whose gold span length is recorded
by `corruption.py`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthHead(nn.Module):
    def __init__(self, d_model: int, l_max: int = 5, hidden: int = 256):
        super().__init__()
        self.l_max = int(l_max)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, l_max),
        )

    def forward(
        self,
        hidden_at_gap: torch.Tensor,        # (B, d_model)
        gold_length: Optional[torch.Tensor] = None,  # (B,) long in [0, l_max)
    ) -> Dict[str, torch.Tensor]:
        logits = self.net(hidden_at_gap.to(self.net[0].weight.dtype))
        loss = None
        if gold_length is not None:
            loss = F.cross_entropy(logits, gold_length.clamp(min=0, max=self.l_max - 1).long())
        return {"loss": loss, "logits": logits}

    def predict(self, hidden_at_gap: torch.Tensor) -> torch.Tensor:
        return self.forward(hidden_at_gap)["logits"].argmax(dim=-1) + 1   # back to 1..L_MAX
