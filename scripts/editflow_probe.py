"""
SAE-EF probe on LinguaLens (EDIT_FLOWS_PLAN.md §4.4-§5) — the promotion
gates, measured on the SAME 200-pair sample (seed 42) as the editor probes
so rows are directly comparable with probe_local / probe_cmlm.

The model edits x' directly (no tagger, no templates): T decode steps, at
each step fire the top-rate operations (deterministic variant) or sample
fires from the rates (stochastic). Conditioning is training-parity
(edit-local pool + blocklist) with true/empty/random controls, and the
logit-lens bias is added to the Q distributions at every step.

Gate measurements:
  (a) λ-IoU — WHERE quality of the rates alone: forward x' at t=--iou-t,
      rank positions by total rate, IoU vs gold edit sites (count oracle),
      under true/empty/random. Compare with the tagger's OOD span IoU
      (README §13.7: e2e iou ≈ 0.30).
  (b) deterministic vs stochastic decode quality (exact / sim_target).
  (c) empty no-edit rate — fraction of pairs the empty condition leaves
      byte-identical (premise protection; gate ≥ 0.99).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer                             # noqa: E402

from editflow import load_editflow_from_checkpoint                 # noqa: E402
from editflow_ops import (                                         # noqa: E402
    KIND_DEL, KIND_INS, KIND_MOV, KIND_SUB, adj_counts, align_pair,
    apply_step_ops, build_xt, edited_marks_xt, gold_edit_positions,
    lambda_iou, move_reinterpret, slot_ops, w_weight,
)
from eval_lingualens import (                                      # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk, pair_metrics,
    randomize_intervention, sae_z_with_offsets,
)
from model import SAEFeatureExtractor                              # noqa: E402

EDIT_BUCKETS = ((1, 1, "1"), (2, 3, "2-3"), (4, 8, "4-8"),
                (9, 10 ** 9, "9+"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--editflow-ckpt", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"],
                   default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--cond-scope", choices=["global", "local"],
                   default="local")
    p.add_argument("--blocklist", default="")
    p.add_argument("--conditions", default="true,empty,random")

    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--decode", default="det,stoch",
                   help="Comma list of BASE[@MOD...]: bases are det "
                        "(expected-count top-rate ops), stoch (Bernoulli "
                        "fires + Q samples; true-only), thr{F} (fire when "
                        "λ ≥ F·w(t); for hazard models this reads p ≥ F), "
                        "bo{K} (K stoch samples, pick by directional SAE "
                        "achievement — the ranker's flow-native "
                        "replacement; true-only). Modifiers: @cfg{S} "
                        "(CFG scale on λ and Q, paper §CFG), @temp{T} "
                        "(Q sampling temperature for stoch/bo). E.g. "
                        "'det,det@cfg2,thr0.02@cfg2,bo4@temp0.7@cfg2'.")
    p.add_argument("--w-max", type=float, default=20.0,
                   help="w(t) clip — must match training --w-max.")
    p.add_argument("--tagger-ckpt", default="",
                   help="Optional SAETagger checkpoint: adds the tagger's "
                        "COUNT-ORACLE IoU on the same pairs / gold sets, "
                        "making gate (a) apples-to-apples (the e2e iou "
                        "~0.30 is threshold-based, a lower bar).")
    p.add_argument("--steer-lambda", type=float, default=1.0,
                   help="Logit-lens bias on Q^sub/Q^ins at every step; "
                        "0 = off.")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="CFG on λ and Q independently: "
                        "u = u_empty + s·(u_cond − u_empty). 1 = off.")
    p.add_argument("--iou-t", type=float, default=0.7,
                   help="t for the λ-IoU forward (w(t) must be "
                        "informative; rates vanish at t→0 by design).")
    p.add_argument("--max-ops-per-step", type=int, default=8)
    p.add_argument("--max-grow", type=int, default=24,
                   help="Stop editing when the sequence has grown by this "
                        "many tokens (runaway-insertion guard).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


def _bucket(n: int) -> str:
    for lo, hi, name in EDIT_BUCKETS:
        if lo <= n <= hi:
            return name
    return "9+"


def parse_decode(m: str) -> Dict:
    """'BASE[@MOD...]' → spec dict. See --decode help."""
    parts = m.split("@")
    base = parts[0]
    spec = {"name": m, "cfg": None, "temp": 1.0, "bo": 0, "frac": 0.0}
    try:
        if base in ("det", "stoch"):
            spec["decode"] = base
        elif base.startswith("thr"):
            spec["decode"] = "thr"
            spec["frac"] = float(base[3:])
        elif base.startswith("bo"):
            spec["decode"] = "stoch"
            spec["bo"] = int(base[2:])
        else:
            raise ValueError(f"unknown base {base!r}")
        for p in parts[1:]:
            if p.startswith("cfg"):
                spec["cfg"] = float(p[3:])
            elif p.startswith("temp"):
                spec["temp"] = float(p[4:])
            else:
                raise ValueError(f"unknown modifier {p!r}")
    except ValueError as e:
        raise SystemExit(f"bad decode spec {m!r}: {e}")
    return spec


@torch.no_grad()
def rates(model, ids: List[int], za, zs, t: float, device: str,
          adj: Optional[List[int]] = None):
    """One forward → (lam (L,3) float32, hidden (L,d)). adj (S3): per-token
    count of already-edited neighbors; ignored by non-localized models."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(x)
    adj_t = None
    if adj is not None:
        adj_t = torch.tensor([adj], dtype=torch.long, device=device)
    out = model(input_ids=x, attention_mask=attn, z_amp=za, z_sup=zs,
                t=torch.tensor([t], device=device), adj=adj_t)
    return out["lambda"][0], out["hidden"][0]


@torch.no_grad()
def decode_flow(
    model, src_ids: List[int], za, zs, *,
    steps: int, device: str, mode: str = "det",
    thr_frac: float = 0.0, w_max: float = 20.0,
    thr_abs_floor: float = 0.05, temp: float = 1.0,
    lens_bias: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    max_ops_per_step: int = 8, max_grow: int = 24,
    suppress_ids: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Integrate the rate model from t=0 to 1 over `steps` steps."""
    x = list(src_ids)
    edited = [False] * len(x)          # S3 adjacency tracking (see below)
    use_adj = float(getattr(model, "lam_prop", 0.0)) > 0
    zae = torch.zeros_like(za)
    zse = torch.zeros_like(zs)
    h_step = 1.0 / steps
    carry = 0.0
    stall = 0
    for step in range(steps):
        if len(x) - len(src_ids) >= max_grow:
            break
        t = (step + 0.5) * h_step
        adj_l = None
        if use_adj:
            marks = {i for i, e in enumerate(edited) if e}
            adj_l = adj_counts(marks, len(x))
        lam, hid = rates(model, x, za, zs, t, device, adj=adj_l)
        if cfg_scale != 1.0:
            lam_e, hid_e = rates(model, x, zae, zse, t, device, adj=adj_l)
            lam = torch.clamp(lam_e + cfg_scale * (lam - lam_e), min=0.0)
        L = lam.shape[0]
        K = lam.shape[1]                             # 3, or 4 with MOV
        lam = lam.clone()
        lam[0, KIND_DEL] = 0.0                       # <bos> is structural
        lam[0, KIND_SUB] = 0.0
        if K > KIND_MOV:
            lam[0, KIND_MOV] = 0.0

        flat = lam.reshape(-1)                       # (L*K,)
        # Stall exit — LATE HALF ONLY (t > 0.5). The rates are hazards
        # (they grow with t by construction: target w(t) ≈ 0 near t=0),
        # so early quiet steps are normal, not dead; counting them killed
        # every decode at t≈0.12 in the first recal run. Dead rates after
        # t=0.5 mean nothing is coming (empty / satisfied conditioning).
        if float(flat.max()) < 0.02:
            if t > 0.5:
                stall += 1
                if stall >= 6:
                    break
            continue
        stall = 0
        if mode == "det":
            # Expected fire count this step (with fractional carry so mass
            # < 1 per step still integrates), taken from the TOP rates only:
            # ops below 10% of the step's max rate are noise, not "due" —
            # without the floor, a large n_fire drags near-zero ops in and
            # the decode runs away.
            mass = float(h_step * flat.sum()) + carry
            n_fire = int(mass)
            carry = mass - n_fire
            if n_fire <= 0:
                continue
            n_fire = min(n_fire, max_ops_per_step)
            floor_val = 0.1 * float(flat.max())
            order = [i for i in
                     torch.argsort(flat, descending=True).tolist()
                     if float(flat[i]) >= floor_val]
        elif mode == "thr":
            # Calibrated firing: training's per-op optimum at a pending
            # site is λ* = w(t), so "λ ≥ F·w(t)" reads the model's own
            # magnitude against its target instead of against the step
            # budget — immune to a globally under-scaled rate head. The
            # absolute floor keeps noise rates from firing near t=0 where
            # w(t) → 0 makes any relative threshold vacuous.
            floor_val = max(thr_frac * w_weight(t, w_max), thr_abs_floor)
            cand = [i for i in range(len(flat))
                    if float(flat[i]) >= floor_val]
            cand.sort(key=lambda i: -float(flat[i]))
            order = cand
            n_fire = min(len(cand), max_ops_per_step)
            if n_fire <= 0:
                continue
        else:
            p_fire = 1.0 - torch.exp(-h_step * flat)
            fires = [i for i in range(len(flat))
                     if rng.random() < float(p_fire[i])]
            fires.sort(key=lambda i: -float(flat[i]))
            order = fires
            n_fire = min(len(fires), max_ops_per_step)
            if n_fire <= 0:
                continue

        chosen, used_pos = [], set()
        ptr = None                                   # lazy MOV pointer
        for fi in order:
            if len(chosen) >= n_fire:
                break
            pos, kind = fi // K, fi % K
            if pos in used_pos or float(flat[fi]) <= 0:
                continue
            used_pos.add(pos)
            tok = None
            if kind == KIND_MOV:
                if ptr is None:
                    attn1 = torch.ones(1, len(x), dtype=torch.long,
                                       device=hid.device)
                    ptr = model.mov_pointer_logits(
                        hid.unsqueeze(0), attn1)[0]  # (L, L)
                dst = int(ptr[pos].argmax())
                if dst == pos:
                    continue
                chosen.append({"kind": KIND_MOV, "pos": pos, "dst": dst})
                continue
            if kind in (KIND_SUB, KIND_INS):
                logits = model.q_logits(
                    hid[pos].unsqueeze(0),
                    "sub" if kind == KIND_SUB else "ins")[0]
                if cfg_scale != 1.0:
                    logits_e = model.q_logits(
                        hid_e[pos].unsqueeze(0),
                        "sub" if kind == KIND_SUB else "ins")[0]
                    logits = logits_e + cfg_scale * (logits - logits_e)
                if suppress_ids:
                    logits[suppress_ids] = float("-inf")
                if lens_bias is not None:
                    logits = logits + lens_bias
                if mode == "stoch":
                    probs = torch.softmax(logits / max(temp, 1e-3), dim=-1)
                    tok = int(torch.multinomial(probs, 1))
                else:                                 # det / thr: argmax
                    tok = int(logits.argmax())
                if kind == KIND_SUB and tok == x[pos]:
                    continue                          # no-op substitution
            chosen.append({"kind": kind, "pos": pos, "tok": tok})
        if not chosen:
            continue
        x, edited = apply_step_ops(x, chosen, edited)
    return x


def main():
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    mode_specs = [parse_decode(m.strip())
                  for m in args.decode.split(",") if m.strip()]
    modes = [sp["name"] for sp in mode_specs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    from datasets import load_dataset
    print(f"[ef-probe] loading {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen_idx = order[:min(args.sample_size, len(order))]
    print(f"[ef-probe] {len(ds)} pairs, sampling {len(chosen_idx)}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    suppress = [tokenizer.mask_token_id,
                tokenizer.convert_tokens_to_ids("[INS]"),
                tokenizer.convert_tokens_to_ids("[SEP]"),
                tokenizer.convert_tokens_to_ids("[DEL]"),
                tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id]
    suppress = sorted({int(s) for s in suppress if s is not None})

    print(f"[ef-probe] loading model from {args.editflow_ckpt}")
    model = load_editflow_from_checkpoint(
        args.llm2vec_dir, args.editflow_ckpt, dtype=dtype,
    ).to(args.device).eval()
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()

    tagger = None
    if args.tagger_ckpt:
        from tagger import load_tagger_from_checkpoint
        tagger = load_tagger_from_checkpoint(
            args.llm2vec_dir, args.tagger_ckpt, d_sae=model.d_sae,
            dtype=dtype).to(args.device).eval()
        print(f"[ef-probe] tagger count-oracle comparison: "
              f"{args.tagger_ckpt}")

    blk = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        blk = torch.as_tensor(np.asarray(_bl, dtype=np.int64))
        print(f"[ef-probe] blocklist: {len(_bl)} features masked")

    head_w = w_dec = None
    if args.steer_lambda > 0:
        from model import load_sae_w_dec
        w_dec = load_sae_w_dec(args.sae_repo, args.sae_path).to(args.device)
        head_w = model.lm_head.weight.detach().float().to(args.device)
        print(f"[ef-probe] lens bias on Q: λ={args.steer_lambda:g}")

    def lens_bias(za_v, zs_v):
        if head_w is None:
            return None
        d = (za_v.to(args.device) - zs_v.to(args.device)) @ w_dec
        lb = head_w @ d
        s = lb.std()
        if float(s) < 1e-6:
            return None
        return args.steer_lambda * lb / (s + 1e-8)

    T_DIAG = (0.3, 0.5, 0.7, 0.9)

    # Resume: every finished pair is flushed to records.partial.jsonl with
    # ALL its metrics self-contained; on restart, done pairs are skipped
    # and the report aggregates are rebuilt from the records. Per-pair
    # RNGs (seeded on idx) make `random` / stoch draws resume-invariant.
    partial_path = out_dir / "records.partial.jsonl"
    records: List[Dict] = []
    done_idx = set()
    if partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                 # torn tail line from a kill
                records.append(r)
                done_idx.add(int(r["idx"]))
        print(f"[ef-probe] RESUME: {len(records)} pairs loaded from "
              f"{partial_path.name}")
    pf = open(partial_path, "a")

    for step_i, k in enumerate(chosen_idx):
        if int(k) in done_idx:
            continue
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids
        slots = align_pair(src_ids, tgt_ids)
        n_ops = len(slot_ops(slots))
        if n_ops == 0:
            continue
        gold_pos = gold_edit_positions(slots)
        prng = np.random.default_rng(args.seed * 1000003 + int(k))
        srng = random.Random(args.seed * 1000003 + int(k))

        # Conditioning (training parity — same construction as the probes)
        with torch.no_grad():
            if args.cond_scope == "local" or blk is not None:
                s_off, z_s = sae_z_with_offsets(extractor, src, args.device)
                t_off, z_t = sae_z_with_offsets(extractor, tgt, args.device)
                if args.cond_scope == "local":
                    import difflib
                    om_s = [tuple(o) for o in tokenizer(
                        src, add_special_tokens=True,
                        return_offsets_mapping=True)["offset_mapping"]]
                    om_t = [tuple(o) for o in tokenizer(
                        tgt, add_special_tokens=True,
                        return_offsets_mapping=True)["offset_mapping"]]
                    opcodes = difflib.SequenceMatcher(
                        None, src_ids, tgt_ids,
                        autojunk=False).get_opcodes()
                    sr, tr = edit_char_ranges(opcodes, om_s, om_t)
                else:
                    sr, tr = [], []
                z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
                z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
            else:
                z_src = extractor.pool_max_topk(
                    extractor.encode_text(src),
                    args.pool_topk).float().cpu()
                z_tgt = extractor.pool_max_topk(
                    extractor.encode_text(tgt),
                    args.pool_topk).float().cpu()
        za_t, zs_t = diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)
        zvar = {"true": (za_t, zs_t),
                "empty": (torch.zeros_like(za_t), torch.zeros_like(zs_t)),
                "random": (randomize_intervention(za_t, prng),
                           randomize_intervention(zs_t, prng))}

        rec = {"idx": int(k), "src": src, "tgt": tgt, "n_ops": n_ops,
               "outputs": {}}

        # best-of-K selection: directional SAE achievement of a candidate
        # (the ranker's flow-native replacement — EDIT_FLOWS_ZERO §3).
        z_in_global = None

        def sae_gain(out_ids, za_v, zs_v):
            nonlocal z_in_global
            am, sm = za_v > 0, zs_v > 0
            total = float(za_v[am].sum() + zs_v[sm].sum())
            if total <= 0:
                return 0.0
            if z_in_global is None:
                with torch.no_grad():
                    z_in_global = extractor.pool_max_topk(
                        extractor.encode_text(src),
                        args.pool_topk).float().cpu()
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            with torch.no_grad():
                z_out = extractor.pool_max_topk(
                    extractor.encode_text(text),
                    args.pool_topk).float().cpu()
            delta = z_out - z_in_global
            gain = torch.clamp(delta[am], -za_v[am], za_v[am]).sum()
            gain = gain + torch.clamp(-delta[sm], -zs_v[sm], zs_v[sm]).sum()
            return float(gain) / (total + 1e-8)

        for c in conditions:
            za = zvar[c][0].unsqueeze(0).to(args.device)
            zs = zvar[c][1].unsqueeze(0).to(args.device)

            # gate (a): λ-IoU of the rates on x' at t=iou_t
            lam, _ = rates(model, src_ids, za, zs, args.iou_t, args.device)
            lam_tot = lam.sum(dim=-1).cpu().tolist()
            iou = lambda_iou(lam_tot, gold_pos)

            if c == "true":
                # rate-magnitude diagnostic: mean top-|gold| λ across t —
                # compare against the training target w(t). CAVEAT: x0 at
                # high t is OFF-distribution; the on-dist column samples a
                # true z_t state, where the optimum is w(t)·P(pending) and
                # ratios reflect the model's calibrated P.
                rec["rate_diag"] = {}
                rec["rate_diag_ondist"] = {}
                ops_all = slot_ops(slots)
                if bool(getattr(model, "move_ops", False)):
                    ops_all = move_reinterpret(slots, ops_all)
                for td in T_DIAG:
                    lam_d, _ = rates(model, src_ids, za, zs, td,
                                     args.device)
                    tot = lam_d.sum(dim=-1)
                    topv = tot.topk(min(len(gold_pos), tot.shape[0])).values
                    rec["rate_diag"][str(td)] = float(topv.mean())
                    fired = [prng.random() < td ** 3 for _ in ops_all]
                    adj_o = None
                    if float(getattr(model, "lam_prop", 0.0)) > 0:
                        marks_o, xl_o = edited_marks_xt(slots, ops_all,
                                                        fired)
                        adj_o = adj_counts(marks_o, xl_o)
                    x_t, pend = build_xt(slots, ops_all, fired)
                    if pend:
                        lam_o, _ = rates(model, x_t, za, zs, td,
                                         args.device, adj=adj_o)
                        vals = [float(lam_o[op["pos"], op["kind"]])
                                for op in pend]
                        rec["rate_diag_ondist"][str(td)] = (
                            sum(vals) / len(vals))
                if tagger is not None:
                    x_in = torch.tensor([src_ids], dtype=torch.long,
                                        device=args.device)
                    t_out = tagger(x_in, torch.ones_like(x_in), za, zs)
                    opp = torch.softmax(
                        t_out["op_logits"].float(), dim=-1)[0]     # (T,3)
                    insp = torch.sigmoid(
                        t_out["ins_logits"].float())[0]            # (T,)
                    score = (1.0 - opp[:, 0]).clone()
                    # ins-before token j = insert AFTER j-1 → anchor j-1,
                    # matching gold_edit_positions' ins convention
                    score[:-1] = torch.maximum(score[:-1], insp[1:])
                    tiou = lambda_iou(score.cpu().tolist(), gold_pos)
                    if tiou == tiou:
                        rec["tagger_iou"] = tiou

            lb = lens_bias(zvar[c][0], zvar[c][1]) if c != "empty" else None
            rec["outputs"][c] = {"lambda_iou": iou}
            for spec in mode_specs:
                m = spec["name"]
                if spec["decode"] == "stoch" and c != "true":
                    continue
                cfg = spec["cfg"] if spec["cfg"] is not None \
                    else args.cfg_scale
                K = max(1, spec["bo"])
                out_ids, best_gain = None, None
                for ki in range(K):
                    srng_k = random.Random(
                        args.seed * 1000003 + int(k) * 37 + ki) \
                        if K > 1 else srng
                    cand = decode_flow(
                        model, src_ids, za, zs, steps=args.steps,
                        device=args.device, mode=spec["decode"],
                        thr_frac=spec["frac"], w_max=args.w_max,
                        temp=spec["temp"], lens_bias=lb, cfg_scale=cfg,
                        max_ops_per_step=args.max_ops_per_step,
                        max_grow=args.max_grow, suppress_ids=suppress,
                        rng=srng_k)
                    if K == 1:
                        out_ids = cand
                        break
                    g = sae_gain(cand, zvar[c][0], zvar[c][1])
                    if best_gain is None or g > best_gain:
                        best_gain, out_ids = g, cand
                out_text = tokenizer.decode(out_ids,
                                            skip_special_tokens=True)
                pm = pair_metrics(out_text, src, tgt)
                rec["outputs"][c][m] = {
                    "text": out_text, "exact": pm["exact_match"],
                    "sim_target": pm["sim_target"],
                    "copy": pm["copy_rate"],
                    "no_edit": float(out_ids == src_ids)}
        records.append(rec)
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        if (step_i + 1) % 10 == 0:
            print(f"[ef-probe] {step_i + 1}/{len(chosen_idx)} pairs "
                  f"({len(records)} scored)")
    pf.close()

    # ---- aggregates (rebuilt from records — resume-safe) ----------------
    agg = {c: {m: defaultdict(list) for m in modes} for c in conditions}
    iou_agg = {c: [] for c in conditions}
    tagger_iou_agg = []
    lam_diag = {t: [] for t in T_DIAG}
    lam_diag_od = {t: [] for t in T_DIAG}
    bpair = {m: defaultdict(lambda: defaultdict(list)) for m in modes}
    for r in records:
        b = _bucket(int(r["n_ops"]))
        if r.get("tagger_iou") is not None:
            tagger_iou_agg.append(float(r["tagger_iou"]))
        for td in T_DIAG:
            v = (r.get("rate_diag") or {}).get(str(td))
            if v is not None:
                lam_diag[td].append(float(v))
            vo = (r.get("rate_diag_ondist") or {}).get(str(td))
            if vo is not None:
                lam_diag_od[td].append(float(vo))
        for c in conditions:
            co = r["outputs"].get(c)
            if not co:
                continue
            li = co.get("lambda_iou")
            if li is not None and li == li:
                iou_agg[c].append(float(li))
            for m in modes:
                mo = co.get(m)
                if not isinstance(mo, dict):
                    continue
                a = agg[c][m]
                a["exact"].append(mo["exact"])
                a["sim_target"].append(mo["sim_target"])
                a["copy"].append(mo.get("copy", float("nan")))
                a["no_edit"].append(mo.get("no_edit", float("nan")))
                if c == "true":
                    bp = bpair[m][b]
                    bp["exact"].append(mo["exact"])
                    bp["sim_target"].append(mo["sim_target"])

    # ---- report -------------------------------------------------------
    n_scored = len(records)
    lines = ["# SAE-EF probe (LinguaLens)", "",
             f"pairs scored: {n_scored}; steps={args.steps} "
             f"cfg={args.cfg_scale:g} lens λ={args.steer_lambda:g} "
             f"iou_t={args.iou_t:g}",
             f"conditioning: scope={args.cond_scope} "
             f"blocklist={'yes' if blk is not None else 'no'} "
             f"k_amp={args.k_amp} k_sup={args.k_sup}; "
             f"ckpt: {args.editflow_ckpt}", ""]

    lines += ["## Gate (a) — λ-IoU (WHERE from rates alone, count-oracle "
              f"top-k at t={args.iou_t:g})", "",
              "| condition | λ-IoU | n |", "|---|---|---|"]
    payload = {"n_scored": n_scored, "lambda_iou": {}, "conditions": {},
               "buckets": {}}
    for c in conditions:
        v = float(np.mean(iou_agg[c])) if iou_agg[c] else float("nan")
        payload["lambda_iou"][c] = v
        lines.append(f"| {c} | {v:.4f} | {len(iou_agg[c])} |")
    if tagger_iou_agg:
        tv = float(np.mean(tagger_iou_agg))
        payload["lambda_iou"]["tagger_count_oracle"] = tv
        lines.append(f"| tagger (count-oracle, same pairs) | {tv:.4f} "
                     f"| {len(tagger_iou_agg)} |")
    lines += ["", "Compare with tagger OOD span IoU ≈ 0.30 "
              "(README §13.7 e2e; threshold-based — the count-oracle row "
              "above is the apples-to-apples bar).", ""]

    lines += ["## Rate magnitude vs training target (condition = true)", "",
              "The training optimum at a pending site is λ* = "
              "w(t)·P(pending | x_t). The x0 column feeds x0 at every t "
              "(OFF-distribution at high t — ratios < 1 there are partly "
              "CORRECT uncertainty); the on-dist column measures pending "
              "sites on a sampled true z_t state, so its ratio reads as "
              "the model's calibrated P(pending). For hazard-parameterized "
              "models the on-dist ratio IS mean p.", "",
              "| t | w(t) | top-|gold| λ @x0 | ratio | pending λ on-dist "
              "| ratio |",
              "|---|---|---|---|---|---|"]
    payload["rate_calibration"] = {}
    for td in T_DIAG:
        wt = w_weight(td, args.w_max)
        mv = float(np.mean(lam_diag[td])) if lam_diag[td] else float("nan")
        mo = float(np.mean(lam_diag_od[td])) if lam_diag_od[td] \
            else float("nan")
        payload["rate_calibration"][str(td)] = {
            "w": wt, "mean_top_lam_x0": mv, "mean_pending_lam_ondist": mo}
        lines.append(f"| {td:g} | {wt:.3f} | {mv:.4f} | {mv / wt:.3f} "
                     f"| {mo:.4f} | {mo / wt:.3f} |")
    lines.append("")

    lines += ["## Decode quality (gates (b), (c))", "",
              "| condition | mode | exact | sim_target | copy | no_edit |",
              "|---|---|---|---|---|---|"]
    for c in conditions:
        payload["conditions"][c] = {}
        for m in modes:
            a = agg[c][m]
            if not a["exact"]:
                continue
            row = {k: float(np.mean(v)) for k, v in a.items()}
            payload["conditions"][c][m] = row
            lines.append(f"| {c} | {m} | {row['exact']:.4f} "
                         f"| {row['sim_target']:.4f} | {row['copy']:.4f} "
                         f"| {row['no_edit']:.4f} |")
    lines.append("")

    lines += ["## Multi-site breakdown (condition = true)", "",
              "| n_ops | pairs | "
              + " | ".join(f"{m} exact" for m in modes) + " | "
              + " | ".join(f"{m} sim" for m in modes) + " |",
              "|---|---|" + "---|" * (2 * len(modes))]
    for _lo, _hi, name in EDIT_BUCKETS:
        n_pairs = len(bpair[modes[0]][name]["exact"]) \
            if modes and name in bpair[modes[0]] else 0
        if not n_pairs:
            continue
        row = {"pairs": n_pairs}
        ce, cs = [], []
        for m in modes:
            bp = bpair[m][name]
            ex = float(np.mean(bp["exact"])) if bp["exact"] else float("nan")
            sm = float(np.mean(bp["sim_target"])) if bp["sim_target"] \
                else float("nan")
            row[m] = {"exact": ex, "sim_target": sm}
            ce.append(f"{ex:.4f}")
            cs.append(f"{sm:.4f}")
        payload["buckets"][name] = row
        lines.append(f"| {name} | {n_pairs} | " + " | ".join(ce)
                     + " | " + " | ".join(cs) + " |")
    lines += [
        "",
        "Reading guide:",
        "- Gate (a): true λ-IoU ≥ tagger (~0.30) AND ≫ empty/random → the "
        "rates localize OOD from SAE conditioning alone.",
        "- Gate (b): det ≥ stoch on exact/sim → deterministic decode is "
        "safe as the production variant.",
        "- Gate (c): empty no_edit ≥ 0.99 → premise protection holds "
        "structurally (null-record teacher worked).",
        "- Overall: compare bucket exact against the editor-pipeline "
        "probes (probe_cmlm: gold-site ceiling; e2e: 0.114 exact). "
        "n_ops here counts alignment ops (dels included), so buckets are "
        "close to but not identical to the template probes' n_edit.",
        "",
    ]
    (out_dir / "probe_report.md").write_text("\n".join(lines))
    (out_dir / "probe_metrics.json").write_text(json.dumps(payload, indent=2))
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)     # complete → partial retired
    print("\n".join(lines))
    print(f"[ef-probe] wrote {out_dir}/probe_report.md, probe_metrics.json, "
          f"records.jsonl")


if __name__ == "__main__":
    main()
