"""Intervener — learned intervention generator (INTERVENER_PLAN.md).

The proposed model under the 2026-07-17 decision: the editor's OUTPUT
INTERFACE decides causal status. This model never emits tokens; it maps
(src, SAE spec) -> residual-stream deltas that are injected into the FROZEN
gemma-2-2b-it at layer L, and the edited text is produced by the frozen
LM's own generation. do(residual) with learned content — ReFT-class
intervention, conditioned on the instance-level SAE spec.

Architecture: reuses SAEEditFlow's encoder + feature-token conditioning
verbatim (editflow.py; rate/Q heads unused, t fed as 0). Two zero-init
heads on top of the per-position hidden states:
  * delta_prefill: (B, T, d) — added at the src-token span inside the
    rewrite prompt during prefill (reading side).
  * delta_decode:  (B, d)    — masked-mean pooled, added at every decode
    step (writing side; steer scope=all's learned counterpart).
Zero-init => identity start (no intervention until trained).

v2 (steer_alpha > 0): residual parameterization around the steering
champion. forward() additionally returns base = steer_alpha *
(z_amp - z_sup) @ W_dec, applied at ALL positions (prompt + decode) —
exactly the C1' steer rendering. The heads become corrections on top,
so zero-init now starts AT the champion (exact 0.2385) instead of at
identity, which killed v1 (copy-attractor collapse: NLL is dominated
by unchanged tokens, so identity-start heads learned "copy" and true
became indistinguishable from random).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from editflow import SAEEditFlow


class Intervener(nn.Module):
    def __init__(self, llm2vec_dir: str, d_sae: int,
                 dtype: torch.dtype = torch.bfloat16, lora_r: int = 32,
                 w_dec: Optional[torch.Tensor] = None,
                 steer_alpha: float = 0.0):
        super().__init__()
        self.flow = SAEEditFlow(
            llm2vec_dir, d_sae, dtype=dtype, lora_r=lora_r,
            cond_mode="feature-tokens", rate_param="hazard", w_dec=w_dec)
        d = self.flow.d_model
        self.delta_prefill = nn.Linear(d, d)
        self.delta_decode = nn.Linear(d, d)
        for lin in (self.delta_prefill, self.delta_decode):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.steer_alpha = float(steer_alpha)
        if self.steer_alpha > 0.0:
            if w_dec is None:
                raise ValueError("steer_alpha base requires w_dec")
            # non-persistent: rebuilt from the SAE at load time, so the
            # checkpoint stays head+LoRA sized.
            self.register_buffer(
                "w_dec_base", w_dec.detach().float().clone(),
                persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                z_amp: torch.Tensor, z_sup: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """input_ids: (B, T) src tokens (llm2vec tokenizer = gemma vocab).
        Returns delta_pre (B, T, d) float32 and delta_dec (B, d) float32."""
        B = input_ids.shape[0]
        t0 = torch.zeros(B, device=input_ids.device)
        out = self.flow(input_ids, attention_mask, z_amp, z_sup, t0)
        h = out["hidden"].float()                        # (B, T, d)
        delta_pre = self.delta_prefill(h)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        delta_dec = self.delta_decode(pooled)            # (B, d)
        out_d = {"delta_pre": delta_pre * m, "delta_dec": delta_dec}
        if self.steer_alpha > 0.0:
            out_d["base"] = self.steer_alpha * (
                (z_amp.float() - z_sup.float()) @ self.w_dec_base)  # (B, d)
        return out_d

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {f"flow::{k}": v for k, v in
              self.flow.trainable_state_dict().items()}
        for name in ("delta_prefill", "delta_decode"):
            lin = getattr(self, name)
            sd[f"{name}.weight"] = lin.weight.detach().cpu()
            sd[f"{name}.bias"] = lin.bias.detach().cpu()
        return sd

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]):
        flow_sd = {k[len("flow::"):]: v for k, v in sd.items()
                   if k.startswith("flow::")}
        # editflow's loader is load_trainable (verbatim signature, line 429)
        self.flow.load_trainable(flow_sd)
        for name in ("delta_prefill", "delta_decode"):
            lin = getattr(self, name)
            lin.weight.data.copy_(sd[f"{name}.weight"])
            lin.bias.data.copy_(sd[f"{name}.bias"])


class InjectHook:
    """Forward hook on it_model.model.layers[L]: prefill (T>1) adds the
    per-position field at the src span [lo, hi); decode steps (T==1) add
    delta_dec. Autograd flows through the added tensors (training uses a
    single teacher-forced forward = the prefill branch with the response
    span carrying delta_dec)."""

    def __init__(self):
        self.enabled = False
        self.delta_pre = None     # (T_src, d) for the current example
        self.span = None          # (lo, hi) src-token span in prompt ids
        self.delta_dec = None     # (d,)
        self.resp_from = None     # int | None — teacher-forcing: add
        #                           delta_dec at positions >= resp_from
        self.base = None          # (d,) | None — steer_alpha residual
        #                           base, ALL positions (champion scope=all)

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return None
        h = output[0] if isinstance(output, tuple) else output
        if self.base is not None:
            h = h + self.base.to(h.dtype).view(1, 1, -1)
        if h.shape[1] == 1:                              # decode step
            if self.delta_dec is not None:
                h = h + self.delta_dec.to(h.dtype).view(1, 1, -1)
        else:                                            # prefill / TF pass
            add = torch.zeros_like(h)
            if self.delta_pre is not None and self.span is not None:
                lo, hi = self.span
                n = min(hi - lo, self.delta_pre.shape[0], h.shape[1] - lo)
                if n > 0:
                    add[:, lo:lo + n, :] = self.delta_pre[:n].to(h.dtype)
            if self.delta_dec is not None and self.resp_from is not None:
                add[:, self.resp_from:, :] = (
                    add[:, self.resp_from:, :]
                    + self.delta_dec.to(h.dtype).view(1, 1, -1))
            h = h + add
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


def find_subseq(hay: List[int], needle: List[int]) -> Optional[int]:
    """First occurrence of needle in hay (token-id lists), else None."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if hay[i:i + m] == needle:
            return i
    return None
