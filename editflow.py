"""
SAE-EF model (EDIT_FLOWS_PLAN.md §2) — frozen bidirectional Gemma with
per-position edit-operation rate heads.

    [INT_amp, INT_sup, T(t), e(x_t_1..T)]
          │                     │
          └── LLM2Vec'd Gemma (frozen + LoRA r=16) ──► h (B, T, d)
                    │
      λ head: Linear(d,3) → softplus  → per-position (λ^ins, λ^del, λ^sub)
      Q^sub  : frozen lm_head(h + sub_shift)   — replacement token at i
      Q^ins  : frozen lm_head(h + ins_shift)   — token inserted AFTER i

Reuses the editor's conditioning machinery verbatim (Proj_A wdec-frozen +
type_emb + cond_scale, initializable from a v6 SAEEditor checkpoint,
including its LoRA adapters) so the SAE-feature interface — and the
true/empty/random verification protocol — carries over unchanged. The
input x_t is a PLAIN token sequence: no [MASK]/[INS]/[SEP] markers, so no
delta embedding rows. λ^ins(i) = rate of inserting after position i (gap
representation, a simplification of the paper's blank tokens).

Trainable: LoRA, Proj_A low-rank correction, type_emb, cond_scale,
t-projection, λ head, sub_shift/ins_shift. Frozen: encoder, lm_head,
W_dec base, input embeddings.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from lora import apply_lora, load_lora_state_dict, lora_state_dict
from model import BidirectionalLLM


def timestep_features(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Sinusoidal features of t ∈ [0,1] (diffusion-style), (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                          device=t.device) / half)
    args = t.float().unsqueeze(-1) * 1000.0 * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SAEEditFlow(nn.Module):
    def __init__(
        self,
        llm2vec_dir: str,
        d_sae: int,
        dtype: torch.dtype = torch.bfloat16,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        proj_a_rank: int = 32,
        w_dec: Optional[torch.Tensor] = None,
        t_dim: int = 128,
        lam_bias_init: float = -4.0,
        t_film: bool = False,
        cond_mode: str = "pooled",
        rate_param: str = "free",
        w_max: float = 20.0,
    ):
        """t_film (Z1a): modulate the λ head's input with FiLM(t) — REFUTED
        + premise-breaking (README §13.8 Z1 verdict; the additive β(t) is an
        input-independent rate path). Kept for checkpoint compat only.
        cond_mode (Z1b): 'pooled' = the editor-style 2-vector prefix;
        'feature-tokens' = one prefix token PER commanded feature —
        W_dec[f] base (frozen, stored in proj_a) + type_emb sign + a
        magnitude projection — so attention can bind individual features
        to individual edit sites (essential problem #1).
        rate_param (S1, EDIT_FLOWS_ZERO §5): 'free' = λ = softplus(head) —
        the paper's generic CTMC head; 'hazard' = λ = w(t)·sigmoid(head).
        In the editing regime κ is known and each op fires once, so the
        target rate is exactly w(t)·1[pending]: give the hazard factor
        analytically and learn ONLY P(pending). Magnitude tracking becomes
        exact by construction, the thr{F} decode reads as p ≥ F (a
        calibrated probability), and there is no input-independent rate
        path to leak premise protection."""
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
            print(f"[editflow] LoRA r={lora_r} on {n_wrapped} backbone modules")

        causal = AutoModelForCausalLM.from_pretrained(llm2vec_dir,
                                                      torch_dtype=dtype)
        self.lm_head = causal.get_output_embeddings()
        for p in self.lm_head.parameters():
            p.requires_grad_(False)
        del causal

        d_model = int(self.encoder.config.hidden_size)
        self.d_model = d_model
        self.d_sae = int(d_sae)
        self.proj_a_rank = int(proj_a_rank)
        self.t_dim = int(t_dim)
        if cond_mode not in ("pooled", "feature-tokens"):
            raise ValueError(f"unknown cond_mode {cond_mode!r}")
        self.cond_mode = cond_mode
        self.t_film = bool(t_film)
        if rate_param not in ("free", "hazard"):
            raise ValueError(f"unknown rate_param {rate_param!r}")
        self.rate_param = rate_param
        self.w_max = float(w_max)

        # Conditioning — identical structure to SAEEditor (wdec-frozen mode)
        # so a v6 editor checkpoint initializes it 1:1.
        self.proj_a = nn.Linear(d_sae, d_model, bias=True)
        nn.init.normal_(self.proj_a.weight, std=0.02)
        nn.init.zeros_(self.proj_a.bias)
        if w_dec is not None:
            if tuple(w_dec.shape) != (d_sae, d_model):
                raise ValueError(f"W_dec shape {tuple(w_dec.shape)} != "
                                 f"({d_sae}, {d_model})")
            self.proj_a.weight.data.copy_(
                w_dec.t().to(self.proj_a.weight.dtype))
        for p in self.proj_a.parameters():
            p.requires_grad_(False)
        self.proj_a_corr_A = nn.Parameter(
            torch.empty(self.proj_a_rank, d_sae, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.proj_a_corr_A, a=math.sqrt(5))
        self.proj_a_corr_B = nn.Parameter(
            torch.zeros(d_model, self.proj_a_rank, dtype=torch.float32))
        self.type_emb = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        with torch.no_grad():
            emb_w = self.encoder.get_input_embeddings().weight
            row_rms = emb_w.float().pow(2).mean(dim=-1).sqrt()
            target_rms = row_rms.median()
        self.register_buffer("cond_target_rms", target_rms.to(torch.float32))
        self.cond_scale = nn.Parameter(torch.ones(1))

        # Time token
        self.t_proj = nn.Linear(self.t_dim, d_model)
        nn.init.normal_(self.t_proj.weight, std=0.02)
        nn.init.zeros_(self.t_proj.bias)

        # Heads. λ bias init ≪ 0 → softplus ≈ 0 rates at step 0: the model
        # starts as "edit nothing" (premise-protection prior + stability).
        self.lam_head = nn.Linear(d_model, 3)
        nn.init.normal_(self.lam_head.weight, std=0.02)
        nn.init.constant_(self.lam_head.bias, float(lam_bias_init))
        self.sub_shift = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        self.ins_shift = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))
        # Z1a: FiLM(t) on the λ head input. Zero-init → exact identity at
        # step 0; trained with the boosted rate-head LR group.
        self.lam_film = None
        if self.t_film:
            self.lam_film = nn.Linear(self.t_dim, 2 * d_model)
            nn.init.zeros_(self.lam_film.weight)
            nn.init.zeros_(self.lam_film.bias)
        # Z1b: magnitude projection for feature tokens. Zero-init → the
        # token starts as pure W_dec direction + sign embedding.
        self.mag_proj = None
        if self.cond_mode == "feature-tokens":
            self.mag_proj = nn.Linear(1, d_model)
            nn.init.zeros_(self.mag_proj.weight)
            nn.init.zeros_(self.mag_proj.bias)

    # ------------------------------------------------------------------
    def _calibrate(self, x: torch.Tensor, scale: bool = True) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        out = x / rms * self.cond_target_rms
        return out * self.cond_scale if scale else out

    def _proj(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj_a(z)
        return x + (z @ self.proj_a_corr_A.t()) @ self.proj_a_corr_B.t()

    def cond_embeds(self, z_amp: torch.Tensor, z_sup: torch.Tensor) -> torch.Tensor:
        amp = self._calibrate(self._proj(z_amp.to(self.proj_a.weight.dtype)))
        sup = self._calibrate(self._proj(z_sup.to(self.proj_a.weight.dtype)))
        B = amp.shape[0]
        amp = amp + self.type_emb(torch.full((B,), 1, device=amp.device,
                                             dtype=torch.long))
        sup = sup + self.type_emb(torch.full((B,), 2, device=sup.device,
                                             dtype=torch.long))
        return torch.stack([amp, sup], dim=1)          # (B, 2, d)

    def time_embeds(self, t: torch.Tensor) -> torch.Tensor:
        feat = timestep_features(t, self.t_dim)
        emb = self.t_proj(feat)
        return self._calibrate(emb, scale=False).unsqueeze(1)   # (B, 1, d)

    def feature_token_embeds(self, z_amp: torch.Tensor, z_sup: torch.Tensor):
        """(B, P, d) prefix + (B, P) mask: one token per commanded feature.
        Base = W_dec[f] (frozen, = proj_a.weight column), + type_emb sign
        (amp=1 / sup=2, warm-startable from the editor), + mag_proj of
        log1p(value). Right-padded; an empty spec yields P=1 all-masked."""
        B = z_amp.shape[0]
        W = self.proj_a.weight                          # (d_model, d_sae)
        device = z_amp.device
        rows, counts = [], []
        for b in range(B):
            toks = []
            for z, sign in ((z_amp[b], 1), (z_sup[b], 2)):
                nz = torch.nonzero(z > 0).flatten()
                if nz.numel() == 0:
                    continue
                base = W[:, nz].t().to(torch.float32)   # (n, d)
                mag = self.mag_proj(
                    torch.log1p(z[nz]).unsqueeze(-1).to(torch.float32))
                sgn = self.type_emb(torch.full((nz.numel(),), sign,
                                               device=device,
                                               dtype=torch.long))
                toks.append(self._calibrate(base) + sgn + mag)
            rows.append(torch.cat(toks, dim=0) if toks
                        else torch.zeros(0, self.d_model, device=device))
            counts.append(rows[-1].shape[0])
        P = max(1, max(counts))
        out = torch.zeros(B, P, self.d_model, device=device)
        mask = torch.zeros(B, P, dtype=torch.long, device=device)
        for b, r in enumerate(rows):
            if r.shape[0]:
                out[b, :r.shape[0]] = r
                mask[b, :r.shape[0]] = 1
        return out, mask

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T) plain x_t tokens
        attention_mask: torch.Tensor,   # (B, T)
        z_amp: torch.Tensor,            # (B, d_sae)
        z_sup: torch.Tensor,            # (B, d_sae)
        t: torch.Tensor,                # (B,) in [0, 1]
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        with torch.no_grad():
            tok = self.encoder.get_input_embeddings()(input_ids)
        if self.cond_mode == "feature-tokens":
            cond, cond_mask = self.feature_token_embeds(z_amp, z_sup)
            cond = cond.to(tok.dtype)
        else:
            cond = self.cond_embeds(z_amp, z_sup).to(tok.dtype)
            cond_mask = torch.ones(B, 2, dtype=attention_mask.dtype,
                                   device=input_ids.device)
        temb = self.time_embeds(t).to(tok.dtype)
        full = torch.cat([cond, temb, tok], dim=1)
        full_mask = torch.cat([
            cond_mask.to(attention_mask.dtype),
            torch.ones(B, 1, dtype=attention_mask.dtype,
                       device=input_ids.device),
            attention_mask,
        ], dim=1)
        n_prefix = cond.shape[1] + 1
        h = self.encoder(inputs_embeds=full,
                         attention_mask=full_mask).last_hidden_state
        h_text = h[:, n_prefix:, :]                     # (B, T, d)
        lam_in = h_text.float()
        if self.lam_film is not None:
            gb = self.lam_film(timestep_features(t, self.t_dim))  # (B, 2d)
            gamma, beta = gb.chunk(2, dim=-1)
            lam_in = lam_in * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        logits3 = self.lam_head(lam_in)                 # (B, T, 3) float32
        if self.rate_param == "hazard":
            # λ = w(t)·P(pending): analytic hazard × learned probability.
            tt = t.float().clamp(0.0, 1.0)
            w = (3.0 * tt * tt / (1.0 - tt ** 3).clamp_min(1e-9)
                 ).clamp(max=self.w_max)                # (B,)
            p = torch.sigmoid(logits3)
            lam = w.view(-1, 1, 1) * p
        else:
            p = None
            lam = F.softplus(logits3)
        lam = lam * attention_mask.unsqueeze(-1).float()
        out = {"lambda": lam, "hidden": h_text}
        if p is not None:
            out["p"] = p * attention_mask.unsqueeze(-1).float()
        return out

    def q_logits(self, h_sel: torch.Tensor, kind: str) -> torch.Tensor:
        """Token distribution logits from selected hidden states (N, d).
        kind: 'sub' (replacement at i) or 'ins' (insert after i)."""
        shift = self.sub_shift if kind == "sub" else self.ins_shift
        h = h_sel.float() + shift
        return self.lm_head(h.to(self.lm_head.weight.dtype)).float()

    # ------------------------------------------------------------------
    # Init from a v6 SAEEditor checkpoint (conditioning + LoRA warm start)
    # ------------------------------------------------------------------
    def init_from_editor(self, editor_ckpt: str):
        blob = torch.load(editor_ckpt, map_location="cpu", weights_only=False)
        sd = blob["trainable"]
        if blob.get("proj_a_mode") != "wdec-frozen":
            raise ValueError("editor checkpoint is not wdec-frozen; "
                             "conditioning init would not match")
        self.proj_a.weight.data.copy_(sd["proj_a.weight"])
        self.proj_a.bias.data.copy_(sd["proj_a.bias"])
        self.proj_a_corr_A.data.copy_(sd["proj_a_corr_A"])
        self.proj_a_corr_B.data.copy_(sd["proj_a_corr_B"])
        self.type_emb.weight.data.copy_(sd["type_emb.weight"])
        if "cond_scale" in sd:
            self.cond_scale.data.copy_(sd["cond_scale"])
        lora_sd = {k[len("lora::"):]: v for k, v in sd.items()
                   if k.startswith("lora::")}
        if lora_sd and self.lora_cfg is not None:
            load_lora_state_dict(self.encoder.backbone, lora_sd)
            print(f"[editflow] init: conditioning + {len(lora_sd)} LoRA "
                  f"tensors from {editor_ckpt}")
        else:
            print(f"[editflow] init: conditioning from {editor_ckpt} "
                  f"(no LoRA transfer)")

    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            "proj_a.weight": self.proj_a.weight.detach().cpu(),
            "proj_a.bias": self.proj_a.bias.detach().cpu(),
            "proj_a_corr_A": self.proj_a_corr_A.detach().cpu(),
            "proj_a_corr_B": self.proj_a_corr_B.detach().cpu(),
            "type_emb.weight": self.type_emb.weight.detach().cpu(),
            "cond_scale": self.cond_scale.detach().cpu(),
            "t_proj.weight": self.t_proj.weight.detach().cpu(),
            "t_proj.bias": self.t_proj.bias.detach().cpu(),
            "lam_head.weight": self.lam_head.weight.detach().cpu(),
            "lam_head.bias": self.lam_head.bias.detach().cpu(),
            "sub_shift": self.sub_shift.detach().cpu(),
            "ins_shift": self.ins_shift.detach().cpu(),
        }
        if self.lam_film is not None:
            sd["lam_film.weight"] = self.lam_film.weight.detach().cpu()
            sd["lam_film.bias"] = self.lam_film.bias.detach().cpu()
        if self.mag_proj is not None:
            sd["mag_proj.weight"] = self.mag_proj.weight.detach().cpu()
            sd["mag_proj.bias"] = self.mag_proj.bias.detach().cpu()
        if self.lora_cfg is not None:
            for n, tns in lora_state_dict(self.encoder.backbone).items():
                sd[f"lora::{n}"] = tns
        return sd

    _TRAINABLE_KEYS = (
        "proj_a.weight", "proj_a.bias", "proj_a_corr_A", "proj_a_corr_B",
        "type_emb.weight", "cond_scale", "t_proj.weight", "t_proj.bias",
        "lam_head.weight", "lam_head.bias", "sub_shift", "ins_shift",
        "lam_film.weight", "lam_film.bias", "mag_proj.weight",
        "mag_proj.bias",
    )

    def load_trainable(self, sd: Dict[str, torch.Tensor],
                       strict: bool = True):
        """strict=False tolerates keys missing on either side (warm-start
        across Z variants: a pilot checkpoint has no lam_film/mag_proj; a
        pooled model has no mag_proj to receive)."""
        skipped = []
        for name in self._TRAINABLE_KEYS:
            obj = self
            *path, leaf = name.split(".")
            for part in path:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            tensor = getattr(obj, leaf, None) if obj is not None else None
            if tensor is None or name not in sd:
                if strict and (name in sd) != (tensor is not None):
                    raise KeyError(f"trainable key mismatch: {name} "
                                   f"(in ckpt: {name in sd}, in model: "
                                   f"{tensor is not None})")
                if name in sd or tensor is not None:
                    skipped.append(name)
                continue
            tensor.data.copy_(sd[name])
        if skipped:
            print(f"[editflow] load_trainable: skipped {skipped}")
        lora_sd = {k[len("lora::"):]: v for k, v in sd.items()
                   if k.startswith("lora::")}
        if lora_sd:
            if self.lora_cfg is None:
                raise ValueError("checkpoint has LoRA but model built with "
                                 "lora_r=0 — use load_editflow_from_checkpoint")
            load_lora_state_dict(self.encoder.backbone, lora_sd)

    def init_from_editflow(self, ckpt_path: str):
        """Warm start from another SAE-EF checkpoint, tolerating Z-variant
        differences (missing/extra heads are skipped, LoRA transfers)."""
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.load_trainable(blob["trainable"], strict=False)
        print(f"[editflow] init from editflow ckpt {ckpt_path}")

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trainable": self.trainable_state_dict(),
            "d_sae": int(self.d_sae),
            "d_model": int(self.d_model),
            "lora": self.lora_cfg,
            "proj_a_rank": int(self.proj_a_rank),
            "t_dim": int(self.t_dim),
            "t_film": bool(self.t_film),
            "cond_mode": self.cond_mode,
            "rate_param": self.rate_param,
            "w_max": float(self.w_max),
        }, path)


def load_editflow_from_checkpoint(
    llm2vec_dir: str, ckpt_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> SAEEditFlow:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora = blob.get("lora") or {}
    model = SAEEditFlow(
        llm2vec_dir, d_sae=int(blob["d_sae"]), dtype=dtype,
        lora_r=int(lora.get("r", 0)),
        lora_alpha=float(lora.get("alpha", 32.0)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        proj_a_rank=int(blob.get("proj_a_rank", 32)),
        t_dim=int(blob.get("t_dim", 128)),
        t_film=bool(blob.get("t_film", False)),
        cond_mode=blob.get("cond_mode", "pooled"),
        rate_param=blob.get("rate_param", "free"),
        w_max=float(blob.get("w_max", 20.0)),
    )
    model.load_trainable(blob["trainable"])
    return model
