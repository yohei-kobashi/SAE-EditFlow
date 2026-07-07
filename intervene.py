"""
Intervention spec helpers (inference-time).

User-facing format:
    "+1234"  → amplify  feature 1234
    "-5678"  → suppress feature 5678

`build_intervention_vectors` converts a parsed spec + per-feature mean
activations (`mu`, from precompute_sae.py) into the two sparse vectors
`z_amp` and `z_sup` consumed by Proj_A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class FeatureSpec:
    feature_id: int
    sign: str          # "+" or "-"

    @classmethod
    def parse(cls, s: str) -> "FeatureSpec":
        s = s.strip()
        if not s or s[0] not in ("+", "-"):
            raise ValueError(f"feature spec must start with '+' or '-'; got {s!r}")
        return cls(feature_id=int(s[1:]), sign=s[0])

    @classmethod
    def parse_many(cls, specs: Iterable[str]) -> List["FeatureSpec"]:
        return [cls.parse(s) for s in specs]


def build_intervention_vectors(
    spec: Sequence[FeatureSpec],
    mu: np.ndarray,
    strength: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (z_amp, z_sup) as 1D float32 tensors of shape (d_sae,)."""
    d_sae = int(mu.shape[0])
    z_amp = torch.zeros(d_sae, dtype=torch.float32)
    z_sup = torch.zeros(d_sae, dtype=torch.float32)
    for fs in spec:
        v = float(mu[fs.feature_id]) * float(strength)
        if fs.sign == "+":
            z_amp[fs.feature_id] = v
        else:
            z_sup[fs.feature_id] = v
    return z_amp, z_sup


def parse_k_spec(spec) -> tuple:
    """Conditioning-count spec → ("uniform"|"log", lo, hi).

      'LO-HI'     : uniform inclusive
      'log:LO-HI' : log-uniform inclusive — most mass on sparse specs
                    (the realistic user regime) with a dense tail for
                    robustness (log:1-32 puts ~70% of draws at k ≤ 8)
      'K'         : fixed

    Shared by training (--k-amp/--k-sup) and evaluation
    (eval_tagger_editor.py / scripts/sweep_eval_hparams.py) so both sides
    draw from the same distribution family.
    """
    s = str(spec)
    mode = "uniform"
    if s.startswith("log:"):
        mode = "log"
        s = s[4:]
    if "-" in s:
        lo, hi = s.split("-", 1)
        return mode, int(lo), int(hi)
    v = int(s)
    return mode, v, v


def draw_k(rng, spec: tuple) -> int:
    mode, lo, hi = spec
    if lo >= hi:
        return int(lo)
    if mode == "log":
        u = rng.uniform(math.log(lo), math.log(hi + 1))
        return int(min(hi, math.floor(math.exp(u))))
    return int(rng.integers(lo, hi + 1))


def diff_to_sparse(
    z_X: torch.Tensor,
    z_X_prime: torch.Tensor,
    k_top: int,
    k_amp: int,
    k_sup: int,
    rng: np.random.Generator,
    empty_conditioning_prob: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct training-time (z_amp, z_sup) from clean / corrupted SAE
    forwards via diff-based sub-sampling. See README §6.3.3.

    z_X / z_X_prime are (d_sae,) dense tensors holding sentence-level pool-max
    top-K_train sparse SAE vectors.
    """
    if rng.random() < empty_conditioning_prob:
        zero = torch.zeros_like(z_X)
        return zero, zero.clone()

    delta = z_X - z_X_prime
    pos = torch.clamp(delta, min=0.0)
    neg = torch.clamp(-delta, min=0.0)

    z_amp = torch.zeros_like(z_X)
    z_sup = torch.zeros_like(z_X)

    if k_amp > 0:
        # top-K_top candidate features by positive diff
        k = min(k_top, int((pos > 0).sum().item()))
        if k > 0:
            cand_v, cand_i = pos.topk(k)
            n_choose = min(int(k_amp), k)
            chosen = rng.choice(k, size=n_choose, replace=False)
            for c in chosen:
                idx = int(cand_i[int(c)].item())
                z_amp[idx] = float(cand_v[int(c)].item())

    if k_sup > 0:
        k = min(k_top, int((neg > 0).sum().item()))
        if k > 0:
            cand_v, cand_i = neg.topk(k)
            n_choose = min(int(k_sup), k)
            chosen = rng.choice(k, size=n_choose, replace=False)
            for c in chosen:
                idx = int(cand_i[int(c)].item())
                z_sup[idx] = float(cand_v[int(c)].item())

    return z_amp, z_sup
