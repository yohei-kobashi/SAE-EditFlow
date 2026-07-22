"""EF-version editor training under the through-LM objective
(EF_LM_LOSS_PLAN.md, user-approved 2026-07-18).

Per batch:
  1. Corruption records (x0, x1) with layer-L z sidecar conditioning
     (diff_to_sparse draw family identical to prior training).
  2. Intermediate state x_t: with prob --t0-prob keep x_t = x0; else draw
     t ~ U(0,1) and apply each difflib opcode segment of x0->x1 with
     prob t (EF regime: the editor must render the REMAINING edits from
     any partial state; at x_t = x1 it must be silent -> self-calibrated
     stopping at inference).
  3. BARE frame (LinguaLens-faithful, NO instruction): the frozen
     gemma-2-2b-it reads  [BOS] + x_t + "\\n" + x1  and only the x1 span
     is teacher-forced. The editor's delta (lam_i * v_i) is injected at
     layer L over the [BOS]+x_t positions. The behaviour "emit the
     edited sentence after the source" must itself be induced by the
     injection — no prompt contributes editing stance.
  4. Loss = NLL(x1) through the frozen LM (grads to the editor only)
     + per-position norm budget (--norm-alpha * ||dvec_L||, over-budget
       penalty on true rows)
     + null teachers: empty spec (--empty-prob) and mismatched partner
       spec (--mismatch-null-prob, P5): NO NLL target (the bare frame
       cannot force a copy), squared-norm suppression only.

Runner does fail-fast: train --max-steps 10000, probe100, then resume
to the full step count (resume machinery below).
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup, set_seed)

from data import CorruptionDataset, _dense_topk
from intervene import diff_to_sparse, draw_k, parse_k_spec
from intervener import (EFIntervener, REPEAT_PROMPT, chat_prompt_ids,
                        find_subseq)
from model import load_sae_w_dec
from resume_utils import (
    add_resume_args, find_latest_ckpt, load_train_state, save_train_state,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True,
                   help="layer-L z sidecar cache (scripts/make_z_sidecar.py)")
    p.add_argument("--dev-corruption-dir", default=None)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--inject-layer", type=int, required=True)
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", required=True,
                   help="layer-L SAE (W_dec for conditioning + budget)")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--eval-steps", type=int, default=2000)
    p.add_argument("--dev-batches", type=int, default=48)

    # Conditioning draw family (identical to prior training).
    p.add_argument("--k-top", type=int, default=32)
    p.add_argument("--k-amp", default="log:1-32")
    p.add_argument("--k-sup", default="log:1-32")
    p.add_argument("--empty-prob", type=float, default=0.08)
    p.add_argument("--mismatch-null-prob", type=float, default=0.12)

    p.add_argument("--frame", choices=["bare", "repeat"], default="bare",
                   help="'repeat' = v5 (user decision 2026-07-19): the "
                        "chat-templated explicit repeat instruction "
                        "(REPEAT_PROMPT, 99% plain-model copy rate). The "
                        "prompt carries reproduction capability only; "
                        "ALL rows get an NLL target (true -> x1, "
                        "empty/mismatch/copy -> x_t itself) and "
                        "empty+mismatch stay norm-suppressed (copying "
                        "needs no delta under this frame).")
    # EF regime.
    p.add_argument("--t0-prob", type=float, default=0.5,
                   help="prob. that x_t = x0 (full remaining edit); else "
                        "t ~ U(0,1) partial application per opcode.")
    p.add_argument("--agg-aug-prob", type=float, default=0.0,
                   help="probability of replacing the true spec's dense "
                        "inputs with a dilution mix (own*w + others-mean*"
                        "(1-w), w~U(0.3,0.7)) before sparsification — "
                        "adapter for feature-level averaged specs")
    p.add_argument("--agg-aug-n", type=int, default=3,
                   help="how many other batch members to mix in")
    p.add_argument("--agg-cluster-table", default="",
                   help="v6/T1: {dominant_latent: sparse mean delta} JSON "
                        "(build_group_means.py). When set, agg-aug mixes "
                        "the pair's delta with its PSEUDO-FEATURE group "
                        "mean (same dominant SAE latent) instead of "
                        "unrelated batch members — reproduces the "
                        "eval-time feature-mean statistics WITH signal.")
    p.add_argument("--ins-loss-boost", type=float, default=1.0,
                   help="v6/T3: NLL weight multiplier for true rows whose "
                        "remaining edit net-INSERTS tokens — strengthens "
                        "the insertion drive that limits enhancement.")
    p.add_argument("--edit-only-loss", action="store_true",
                   help="2026-07-18 user decision after the L12 copy "
                        "collapse (ef true==random, copy 0.92): restrict "
                        "the NLL to the CHANGED tokens of x1 vs x_t "
                        "(non-equal difflib opcodes + deletion boundary). "
                        "Echoing then earns nothing and the edit content "
                        "exists only in the spec -> spec reading is "
                        "forced. Unchanged tokens keep --bg-weight.")
    p.add_argument("--bg-weight", type=float, default=0.1,
                   help="loss weight on unchanged response tokens under "
                        "--edit-only-loss (small but nonzero so the "
                        "reproduction skill does not silently degrade).")
    p.add_argument("--init-ckpt", default="",
                   help="warm-start the editor from this checkpoint's "
                        "trainable state (e.g. the copy-collapsed run, "
                        "which already induces reproduction).")
    p.add_argument("--init-flow-ckpt", default="",
                   help="plan-A (user-approved 2026-07-19): warm-start "
                        "ONLY the flow encoder (feature-token conditioning "
                        "+ LoRA) from a token-output EF checkpoint whose "
                        "spec-reading is proven (S3 champion, lam-IoU "
                        "0.73 true / 0.30 random). strict=False filtered "
                        "load; rate/content heads stay fresh.")
    p.add_argument("--mismatch-echo", action="store_true",
                   help="plan-B (user-approved 2026-07-19): mismatch rows "
                        "get an ECHO NLL target (labels = x_t itself), "
                        "hi-weighted at the pair's own edit sites — the "
                        "contrast 'wrong spec -> reproduce unchanged, "
                        "especially where the true spec would edit'. "
                        "Mismatch rows leave the norm-suppression and "
                        "lam-BCE pools (delta must be nonzero to echo).")
    p.add_argument("--lam-sup-w", type=float, default=0.0,
                   help="branch B (plan §5, activated 2026-07-19 after "
                        "v2 kept true==random): weak direct BCE on the "
                        "rate field — true rows fire at the remaining-"
                        "edit source positions and nowhere else; "
                        "mismatch/empty rows fire nowhere. The mismatch "
                        "contrast is what forces SPEC-conditional "
                        "firing (eval's true-vs-random separation).")

    # Norm budget.
    p.add_argument("--norm-alpha", type=float, default=0.5,
                   help="per-position budget = norm_alpha * ||dvec_L||")
    p.add_argument("--norm-reg-w", type=float, default=0.05)
    p.add_argument("--null-norm-w", type=float, default=0.1)

    p.add_argument("--max-src-len", type=int, default=160,
                   help="skip records whose x0 or x1 exceeds this")
    p.add_argument("--max-lm-len", type=int, default=384)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    add_resume_args(p)
    return p.parse_args()


class EFCollator:
    def __init__(self, d_sae: int):
        self.d_sae = int(d_sae)

    def __call__(self, batch: List[Dict]) -> Dict:
        return {
            "x0": [list(map(int, r["x_prime_token_ids"])) for r in batch],
            "x1": [list(map(int, r["x_token_ids"])) for r in batch],
            "z_X": torch.stack([_dense_topk(r["z_X_topk"], self.d_sae)
                                for r in batch]),
            "z_X_prime": torch.stack([_dense_topk(r["z_X_prime_topk"],
                                                  self.d_sae)
                                      for r in batch]),
        }


def sample_intermediate(x0: List[int], x1: List[int], t: float,
                        rng: np.random.Generator) -> List[int]:
    """Partial application of the x0->x1 edit: each non-equal difflib
    opcode segment flips to the x1 side independently with prob t.
    t=0 -> x0, t=1 -> x1. Token lists include BOS (equal prefix)."""
    if t <= 0.0:
        return list(x0)
    if t >= 1.0:
        return list(x1)
    out: List[int] = []
    sm = difflib.SequenceMatcher(None, x0, x1, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or rng.random() < t:
            out.extend(x1[j1:j2])
        else:
            out.extend(x0[i1:i2])
    return out


def build_batch(batch: Dict, rng: np.random.Generator, args, ctx) -> Dict:
    rows = []
    B_ = len(batch["x0"])
    for b in range(B_):
        x0, x1 = batch["x0"][b], batch["x1"][b]
        if max(len(x0), len(x1)) > args.max_src_len:
            continue

        null_kind = None
        u = rng.random()
        if u < args.empty_prob:
            null_kind = "empty"
            a = torch.zeros(ctx["d_sae"])
            s = torch.zeros(ctx["d_sae"])
        elif u < args.empty_prob + args.mismatch_null_prob and B_ > 1:
            o = int(rng.integers(0, B_))
            while o == b:
                o = int(rng.integers(0, B_))
            a, s = diff_to_sparse(
                batch["z_X"][o], batch["z_X_prime"][o], k_top=args.k_top,
                k_amp=draw_k(rng, args._k_amp),
                k_sup=draw_k(rng, args._k_sup),
                rng=rng, empty_conditioning_prob=0.0)
            if bool((a > 0).any() or (s > 0).any()):
                null_kind = "mismatch"
            else:
                a, s = diff_to_sparse(
                    batch["z_X"][b], batch["z_X_prime"][b],
                    k_top=args.k_top, k_amp=draw_k(rng, args._k_amp),
                    k_sup=draw_k(rng, args._k_sup), rng=rng,
                    empty_conditioning_prob=0.0)
        else:
            zx_b, zxp_b = batch["z_X"][b], batch["z_X_prime"][b]
            if (args.agg_cluster_table and "agg_table" in ctx
                    and args.agg_aug_prob > 0
                    and rng.random() < args.agg_aug_prob):
                d_own = (zxp_b - zx_b).float()
                if float(d_own.abs().max()) > 0:
                    dom = int(d_own.abs().argmax())
                    gm = ctx["agg_table"].get(dom)
                    if gm is not None:
                        wmix = 0.3 + 0.4 * rng.random()
                        mixed = wmix * d_own + (1 - wmix) * gm
                        zx_b = torch.clamp(-mixed, min=0.0)
                        zxp_b = torch.clamp(mixed, min=0.0)
            elif (args.agg_aug_prob > 0 and B_ > 1
                    and rng.random() < args.agg_aug_prob):
                # aggregated-spec augmentation (2026-07-22, user-approved
                # ③): dilute the pair's own delta with the batch-mean of
                # other pairs' deltas — mimics the feature-level pool-mean
                # spec's statistics (own signal shrunk, unrelated
                # components averaged in) so the editor is not OOD on the
                # evaluation-time averaged specs.
                others = [o for o in range(B_) if o != b]
                n_mix = min(args.agg_aug_n, len(others))
                pick = rng.choice(len(others), size=n_mix, replace=False)
                sel = [others[int(i)] for i in pick]
                w = 0.3 + 0.4 * rng.random()
                mx = torch.stack([batch["z_X"][o] for o in sel]).mean(0)
                mxp = torch.stack(
                    [batch["z_X_prime"][o] for o in sel]).mean(0)
                zx_b = w * zx_b + (1 - w) * mx
                zxp_b = w * zxp_b + (1 - w) * mxp
            a, s = diff_to_sparse(
                zx_b, zxp_b, k_top=args.k_top,
                k_amp=draw_k(rng, args._k_amp),
                k_sup=draw_k(rng, args._k_sup), rng=rng,
                empty_conditioning_prob=0.0)

        t = 0.0 if rng.random() < args.t0_prob else float(rng.random())
        xt = sample_intermediate(x0, x1, t, rng)
        x1_body = x1[1:] if (x1 and x1[0] == ctx["bos_id"]) else list(x1)
        xt_body0 = xt[1:] if (xt and xt[0] == ctx["bos_id"]) else list(xt)

        if args.frame == "repeat":
            xt_text = ctx["tok"].decode(xt_body0, skip_special_tokens=True)
            prompt_ids = chat_prompt_ids(
                ctx["it_tok"], REPEAT_PROMPT.format(src=xt_text))
            needle = ctx["it_tok"](xt_text,
                                   add_special_tokens=False).input_ids
            enc_extra = 0
            lo = find_subseq(prompt_ids, needle)
            if lo is None and len(needle) > 1:
                lo = find_subseq(prompt_ids, needle[1:])
                if lo is not None:
                    enc_extra = 1
                    needle = needle[1:]
            if lo is None:
                continue
            if null_kind is None:
                target_ids = list(x1_body)
            else:                               # empty/mismatch -> copy
                target_ids = ctx["it_tok"](
                    xt_text, add_special_tokens=False).input_ids
            resp0 = target_ids + [ctx["eot_id"]]
            lm_prefix = prompt_ids
            if len(lm_prefix) + len(resp0) > args.max_lm_len:
                continue
        else:
            # bare frame: [BOS]+x_t  \n  x1-body(no BOS)
            lm_prefix = list(xt) + [ctx["nl_id"]]
            if len(lm_prefix) + len(x1_body) > args.max_lm_len:
                continue

        # positions of the response that differ from the input (edit-only
        # loss) and the input-side remaining-edit positions (lam
        # supervision). non-equal opcode spans; deletions mark the
        # boundary token.
        edit_pos = set()
        lam_pos = set()
        xt_body = xt_body0
        bos_off = len(xt) - len(xt_body)
        if args.frame == "repeat":
            # diff in the -it token space: needle (= x_t as it sits in
            # the prompt) vs the true response; encoder position of
            # needle[j] = enc_extra + bos_off + j.
            if null_kind is None and (args.edit_only_loss
                                      or args.lam_sup_w > 0):
                sm = difflib.SequenceMatcher(None, needle, resp0[:-1],
                                             autojunk=False)
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        continue
                    if j2 > j1:
                        edit_pos.update(range(j1, j2))
                    else:
                        edit_pos.add(min(j1, len(resp0) - 2))
                    if i2 > i1:
                        lam_pos.update(range(bos_off + enc_extra + i1,
                                             bos_off + enc_extra + i2))
                    else:
                        lam_pos.add(bos_off + enc_extra
                                    + min(i1, len(needle) - 1))
            rows.append({
                "xt": xt, "prefix": lm_prefix, "resp": resp0,
                "null": null_kind, "za": a, "zs": s, "t": t,
                "hi_pos": edit_pos, "lam_pos": lam_pos,
                "has_labels": True,
                "span_lo": lo, "span_n": len(needle),
                "enc_off": bos_off + enc_extra,
            })
            continue

        need_diff = (args.edit_only_loss or args.lam_sup_w > 0) and (
            null_kind is None
            or (null_kind == "mismatch" and args.mismatch_echo))
        if need_diff:
            sm = difflib.SequenceMatcher(None, xt_body, x1_body,
                                         autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                if j2 > j1:
                    edit_pos.update(range(j1, j2))
                else:                                  # pure deletion
                    edit_pos.add(min(j1, len(x1_body) - 1))
                if i2 > i1:                            # x_t-side span
                    lam_pos.update(range(bos_off + i1, bos_off + i2))
                else:                                  # pure insertion
                    lam_pos.add(bos_off + min(i1, len(xt_body) - 1))

        if null_kind == "mismatch" and args.mismatch_echo:
            # plan-B echo teacher: label = x_t itself; hi-weight at the
            # pair's own edit sites (body frame of the echo target).
            resp = list(xt_body)
            hi_pos = {p - bos_off for p in lam_pos
                      if 0 <= p - bos_off < len(resp)}
            has_labels = True
        else:
            resp = x1_body
            hi_pos = edit_pos
            has_labels = null_kind is None
        rows.append({
            "xt": xt, "prefix": lm_prefix, "resp": resp,
            "null": null_kind, "za": a, "zs": s, "t": t,
            "hi_pos": hi_pos, "lam_pos": lam_pos,
            "has_labels": has_labels,
            "span_lo": None, "span_n": 0, "enc_off": 0,
        })
    if not rows:
        return None

    B = len(rows)
    Te = max(len(r["xt"]) for r in rows)
    enc_ids = torch.full((B, Te), ctx["pad_id"], dtype=torch.long)
    enc_mask = torch.zeros((B, Te), dtype=torch.long)
    lam_tgt = torch.zeros((B, Te), dtype=torch.float32)
    for i, r in enumerate(rows):
        for j in r["lam_pos"]:                     # null rows: all-zero
            if j < Te:
                lam_tgt[i, j] = 1.0
    Tl = max(len(r["prefix"]) + len(r["resp"]) for r in rows)
    lm_ids = torch.full((B, Tl), ctx["it_pad"], dtype=torch.long)
    lm_mask = torch.zeros((B, Tl), dtype=torch.long)
    labels = torch.full((B, Tl), -100, dtype=torch.long)
    ce_w = torch.ones((B, Tl), dtype=torch.float32)
    for i, r in enumerate(rows):
        e = r["xt"]
        enc_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        enc_mask[i, :len(e)] = 1
        full = r["prefix"] + r["resp"]
        lm_ids[i, :len(full)] = torch.tensor(full, dtype=torch.long)
        lm_mask[i, :len(full)] = 1
        if r["has_labels"]:
            rf = len(r["prefix"])
            labels[i, rf:len(full)] = torch.tensor(r["resp"],
                                                   dtype=torch.long)
            if args.edit_only_loss:
                ce_w[i, rf:len(full)] = args.bg_weight
                for j in r["hi_pos"]:
                    if rf + j < Tl:
                        ce_w[i, rf + j] = 1.0
        r["xt_len"] = len(e)
        r["lm_len"] = len(full)
    W = ctx["w_dec"]                               # (d_sae, d) f32
    za = torch.stack([r["za"] for r in rows])
    zs = torch.stack([r["zs"] for r in rows])
    dvec = (za.float() - zs.float()) @ W
    budget = args.norm_alpha * dvec.norm(dim=-1)   # (B,)
    return {
        "rows": rows, "enc_ids": enc_ids, "enc_mask": enc_mask,
        "lm_ids": lm_ids, "lm_mask": lm_mask, "labels": labels,
        "ce_w": ce_w, "lam_tgt": lam_tgt,
        "z_amp": za, "z_sup": zs, "budget": budget,
        "null_mask": torch.tensor([r["null"] is not None for r in rows]),
        "empty_mask": torch.tensor([r["null"] == "empty" for r in rows]),
        "mm_mask": torch.tensor([r["null"] == "mismatch" for r in rows]),
    }


class AddHook:
    """Adds a full (B, T, d) tensor to the layer output (batched
    teacher-forcing injection; autograd flows through self.add)."""

    def __init__(self):
        self.add = None

    def __call__(self, module, inputs, output):
        if self.add is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        if self.add.shape[:2] != h.shape[:2]:
            return None
        h = h + self.add.to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


def ef_loss(model, it_model, hook, built: Dict, args, device: str,
            return_metrics: bool = False):
    out = model(built["enc_ids"].to(device), built["enc_mask"].to(device),
                built["z_amp"].to(device), built["z_sup"].to(device))
    delta, lam = out["delta"], out["lam"]          # (B,Te,d), (B,Te) f32
    B, Tl = built["lm_ids"].shape
    d = delta.shape[-1]
    add = torch.zeros(B, Tl, d, device=device, dtype=torch.float32)
    for i, r in enumerate(built["rows"]):
        if r.get("span_lo") is not None:           # repeat frame: src
            lo, eo = r["span_lo"], r["enc_off"]    # span inside prompt
            n = min(r["span_n"], delta.shape[1] - eo, Tl - lo)
            if n > 0:
                add[i, lo:lo + n] = delta[i, eo:eo + n]
        else:
            n = r["xt_len"]
            add[i, :n] = delta[i, :n]              # bare: identity map

    hook.add = add
    try:
        lm_out = it_model(input_ids=built["lm_ids"].to(device),
                          attention_mask=built["lm_mask"].to(device))
    finally:
        hook.add = None
    logits = lm_out.logits[:, :-1].float()
    tgt = built["labels"][:, 1:].to(device)
    ce_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tgt.reshape(-1), ignore_index=-100,
                             reduction="none").view(B, -1)
    valid = (tgt != -100).float()
    w = built["ce_w"][:, 1:].to(device) * valid
    nll = (ce_tok * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    budget = built["budget"].to(device)            # (B,)
    null = built["null_mask"].to(device)
    empty = built["empty_mask"].to(device)
    mm = built["mm_mask"].to(device)
    true_m = (~null).float()
    has_nll = (valid.sum(dim=1) > 0).float()
    eps = 1e-6

    # rows whose delta must be SILENT: empty always; mismatch only when
    # the echo teacher is off (with echo, mismatch rows need a delta to
    # render the reproduction). Under the repeat frame copying needs NO
    # delta (the prompt does it), so mismatch is silent again.
    if args.frame == "repeat":
        silent = empty | mm
    else:
        silent = (empty | mm) if not args.mismatch_echo else empty
    active_m = (~silent).float()

    pos_norm = delta.norm(dim=-1)                  # (B, Te)
    enc_m = built["enc_mask"].to(device).float()
    over = F.relu(pos_norm - budget.unsqueeze(1)).pow(2) \
        / (budget.unsqueeze(1).pow(2) + eps)
    over = (over * enc_m).sum(dim=1) / enc_m.sum(dim=1).clamp_min(1.0)
    norm_pen = (over * active_m).sum() / active_m.sum().clamp_min(1.0)

    ref = ((budget.pow(2) * true_m).sum()
           / true_m.sum().clamp_min(1.0)).detach().clamp_min(1.0)
    null_sq = (delta.pow(2).sum(dim=-1) * enc_m).sum(dim=1) \
        / enc_m.sum(dim=1).clamp_min(1.0)
    sil_f = silent.float()
    null_pen = ((null_sq / ref) * sil_f).sum() / sil_f.sum().clamp_min(1.0)

    labeled = has_nll
    if args.ins_loss_boost != 1.0:
        bo = torch.tensor(
            [args.ins_loss_boost
             if (r.get("null") is None and r.get("span_n") is not None
                 and len(r.get("resp", [])) - 1 > r["span_n"])
             else 1.0 for r in built["rows"]],
            device=device, dtype=nll.dtype)
    else:
        bo = torch.ones_like(nll)
    nll_lab = (nll * labeled * bo).sum() \
        / (labeled * bo).sum().clamp_min(1.0)
    nll_true = (nll * true_m * has_nll).sum() \
        / (true_m * has_nll).sum().clamp_min(1.0)
    loss = (nll_lab + args.norm_reg_w * norm_pen
            + args.null_norm_w * null_pen)
    if args.lam_sup_w > 0:
        lam_t = built["lam_tgt"].to(device)
        lam_bce = F.binary_cross_entropy(
            lam.clamp(1e-6, 1 - 1e-6), lam_t, reduction="none")
        # mismatch rows leave the BCE pool under the BARE echo teacher
        # (their lambda may fire to render the echo); under the repeat
        # frame they are silent -> all-zero targets stay in the pool.
        if args.frame == "repeat" or not args.mismatch_echo:
            row_m = torch.ones_like(mm, dtype=torch.float32)
        else:
            row_m = (~mm).float()
        wmask = enc_m * row_m.unsqueeze(1)
        lam_bce = (lam_bce * wmask).sum() / wmask.sum().clamp_min(1.0)
        loss = loss + args.lam_sup_w * lam_bce
    if not return_metrics:
        return loss
    lam_mean = (lam * enc_m).sum(dim=1) / enc_m.sum(dim=1).clamp_min(1.0)
    dnorm = (pos_norm * enc_m).sum(dim=1) / enc_m.sum(dim=1).clamp_min(1.0)
    m = {
        "nll_true": float(nll_true),
        "lam_true": float((lam_mean * true_m).sum()
                          / true_m.sum().clamp_min(1.0)),
        "lam_null": float((lam_mean * null.float()).sum()
                          / null.float().sum().clamp_min(1.0))
        if bool(null.any()) else float("nan"),
        "dnorm": float((dnorm * true_m).sum() / true_m.sum().clamp_min(1.0)),
        "budget": float((budget * true_m).sum()
                        / true_m.sum().clamp_min(1.0)),
        "norm_pen": float(norm_pen), "null_pen": float(null_pen),
    }
    return loss, m


def main():
    args = parse_args()
    args._k_amp = parse_k_spec(args.k_amp)
    args._k_sup = parse_k_spec(args.k_sup)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    tok = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    it_tok = AutoTokenizer.from_pretrained(args.it_model)
    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])

    print(f"[ef-lm] W_dec from {args.sae_repo}/{args.sae_path} "
          f"(inject layer {args.inject_layer})")
    w_dec = load_sae_w_dec(args.sae_repo, args.sae_path)
    model = EFIntervener(args.llm2vec_dir, d_sae, dtype=dtype,
                         lora_r=args.lora_r, w_dec=w_dec).to(args.device)

    if args.init_flow_ckpt:
        blob = torch.load(args.init_flow_ckpt, map_location="cpu",
                          weights_only=False)
        sd = blob.get("trainable", blob)
        model.flow.load_trainable(sd, strict=False)
        print(f"[ef-lm] flow encoder warm start from "
              f"{args.init_flow_ckpt} ({len(sd)} keys, strict=False)")
    if args.init_ckpt:
        blob = torch.load(args.init_ckpt, map_location=args.device,
                          weights_only=False)
        model.load_trainable_state_dict(blob["trainable"])
        print(f"[ef-lm] warm start from {args.init_ckpt}")

    it_model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype,
        attn_implementation="eager").to(args.device).eval()
    it_model.requires_grad_(False)
    hook = AddHook()
    it_model.model.layers[args.inject_layer].register_forward_hook(hook)
    print(f"[ef-lm] frozen {args.it_model}, AddHook on "
          f"layers[{args.inject_layer}] output; BARE frame (no prompt)")

    nl_ids = it_tok("\n", add_special_tokens=False).input_ids
    assert len(nl_ids) == 1, f"newline splits into {nl_ids}"
    ctx = {
        "d_sae": d_sae, "tok": tok, "it_tok": it_tok,
        "pad_id": tok.pad_token_id or 0,
        "it_pad": it_tok.pad_token_id or it_tok.eos_token_id,
        "bos_id": int(tok.bos_token_id),
        "nl_id": int(nl_ids[0]),
        "eot_id": int(it_tok.convert_tokens_to_ids("<end_of_turn>")),
        "w_dec": w_dec.float(),
    }
    if args.agg_cluster_table:
        import json as _json
        _tab = _json.loads(Path(args.agg_cluster_table).read_text())
        agg = {}
        for dom, sp in _tab.items():
            v = torch.zeros(d_sae)
            for i, val in sp.items():
                v[int(i)] = float(val)
            agg[int(dom)] = v
        ctx["agg_table"] = agg
        print(f"[ef-lm] T1 cluster table: {len(agg)} pseudo-feature "
              f"group means")
    if args.frame == "repeat":
        print(f"[ef-lm] REPEAT frame: {REPEAT_PROMPT[:60]!r}...")

    lora_params = [p for n, p in model.named_parameters()
                   if "lora_" in n and p.requires_grad]
    small_params = [p for n, p in model.named_parameters()
                    if "lora_" not in n and p.requires_grad]
    print(f"[ef-lm] trainable: small="
          f"{sum(p.numel() for p in small_params):,} "
          f"lora={sum(p.numel() for p in lora_params):,}")
    optim = torch.optim.AdamW([
        {"params": small_params, "lr": args.learning_rate},
        {"params": lora_params, "lr": args.backbone_lr},
    ])
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps)

    dataset = CorruptionDataset(args.corruption_dir, shuffle=True,
                                seed=args.seed)
    collator = EFCollator(d_sae=d_sae)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collator)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_batches = []
    if args.dev_corruption_dir:
        dev_ds = CorruptionDataset(args.dev_corruption_dir, shuffle=False,
                                   seed=args.seed, infinite=False)
        dev_loader = DataLoader(dev_ds, batch_size=args.batch_size,
                                num_workers=0, collate_fn=collator)
        for i, b in enumerate(dev_loader):
            if i >= args.dev_batches:
                break
            dev_batches.append(b)
        print(f"[ef-lm] dev monitoring: {len(dev_batches)} batches")

    def evaluate_dev() -> Dict[str, float]:
        dev_rng = np.random.default_rng(args.seed + 9999)
        rows = []
        model.eval()
        with torch.no_grad():
            for batch in dev_batches:
                built = build_batch(batch, dev_rng, args, ctx)
                if built is None:
                    continue
                _, m = ef_loss(model, it_model, hook, built, args,
                               args.device, return_metrics=True)
                rows.append(m)
        model.train()
        keys = set().union(*rows) if rows else set()
        return {k: float(np.mean([r[k] for r in rows
                                  if k in r and r[k] == r[k]]))
                for k in keys}

    def save_ckpt(path: Path):
        torch.save({"trainable": model.trainable_state_dict(),
                    "config": {"llm2vec_dir": args.llm2vec_dir,
                               "d_sae": d_sae, "lora_r": args.lora_r,
                               "inject_layer": args.inject_layer,
                               "sae_path": args.sae_path,
                               "model_type": "ef_lm",
                               "it_model": args.it_model}}, str(path))

    best_path = out_dir / "eflm-best.pt"
    best_json = out_dir / "best.json"
    best_dev = float("inf")
    if best_json.exists():
        try:
            best_dev = float(json.loads(best_json.read_text())["dev_nll"])
            print(f"[ef-lm] RESUME: best dev nll {best_dev:.4f}")
        except (ValueError, KeyError):
            pass

    def maybe_update_best(at_step: int):
        nonlocal best_dev
        if not dev_batches:
            return
        m = evaluate_dev()
        marker = ""
        if m["nll_true"] < best_dev:
            best_dev = m["nll_true"]
            save_ckpt(best_path)
            best_json.write_text(json.dumps(
                {"step": int(at_step), "dev_nll": float(m["nll_true"])}))
            marker = "  ** new best **"
        print(f"[ef-lm] step={at_step} DEV nll_true={m['nll_true']:.4f} "
              f"lam={m.get('lam_true', 0):.3f} "
              f"lam_null={m.get('lam_null', float('nan')):.3f} "
              f"|d|={m.get('dnorm', 0):.2f} "
              f"budget={m.get('budget', 0):.2f} "
              f"(best {best_dev:.4f}){marker}")

    step = 0
    if args.resume:
        latest = find_latest_ckpt(out_dir, "eflm")
        if latest is not None:
            ckpt_path, ckpt_step = latest
            print(f"[ef-lm] RESUME: {ckpt_path} (step {ckpt_step})")
            blob = torch.load(str(ckpt_path), map_location=args.device,
                              weights_only=False)
            model.load_trainable_state_dict(blob["trainable"])
            restored = load_train_state(ckpt_path, optim, sched,
                                        device=args.device)
            step = restored if restored is not None else ckpt_step
            if step >= args.max_steps:
                print("[ef-lm] RESUME: already past max_steps")
                save_ckpt(out_dir / "eflm-final.pt")
                return

    loss_window = []
    pbar = tqdm(total=args.max_steps, initial=step, desc="[ef-lm]",
                unit="step", dynamic_ncols=True)
    for batch in loader:
        if step >= args.max_steps:
            break
        built = build_batch(batch, rng, args, ctx)
        if built is None:
            continue
        if step == 0:
            b = built["budget"]
            print(f"[ef-lm] budget |{args.norm_alpha}*dvec| first batch: "
                  f"mean={float(b.mean()):.2f} min={float(b.min()):.2f} "
                  f"max={float(b.max()):.2f}")
        loss = ef_loss(model, it_model, hook, built, args, args.device)
        (loss / args.grad_accum_steps).backward()
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()

        loss_window.append(float(loss.item()))
        if step % args.logging_steps == 0:
            avg = sum(loss_window[-args.logging_steps:]) / max(
                1, min(len(loss_window), args.logging_steps))
            print(f"[ef-lm] step={step} loss={avg:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")
        if step > 0 and step % args.save_steps == 0:
            ckpt = out_dir / f"eflm-step{step}.pt"
            save_ckpt(ckpt)
            save_train_state(ckpt, optim, sched, step)
            print(f"[ef-lm] saved {ckpt}")
        if dev_batches and step > 0 and step % args.eval_steps == 0:
            maybe_update_best(step)
        step += 1
        pbar.update(1)
    pbar.close()

    # loop-exit checkpoint: the boundary step (== max_steps) never reaches
    # the in-loop save, which left stage-1 runs without a step10000 ckpt
    # (runner's probe100 gate silently skipped) and made resume re-train
    # from the previous save point.
    exit_ckpt = out_dir / f"eflm-step{step}.pt"
    if step > 0 and not exit_ckpt.exists():
        save_ckpt(exit_ckpt)
        save_train_state(exit_ckpt, optim, sched, step)
        print(f"[ef-lm] saved loop-exit ckpt {exit_ckpt}")

    maybe_update_best(step)
    if dev_batches and best_path.exists():
        save_ckpt(out_dir / "eflm-last.pt")
        import shutil
        shutil.copyfile(best_path, out_dir / "eflm-final.pt")
        print(f"[ef-lm] done at step {step}; eflm-final.pt = best dev "
              f"state ({json.loads(best_json.read_text())})")
    else:
        save_ckpt(out_dir / "eflm-final.pt")
        print(f"[ef-lm] done at step {step}")


if __name__ == "__main__":
    main()
