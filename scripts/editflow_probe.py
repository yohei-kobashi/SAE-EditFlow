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
    KIND_DEL, KIND_INS, KIND_SUB, align_pair, apply_step_ops,
    gold_edit_positions, lambda_iou, slot_ops, w_weight,
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
                   help="Comma list: det = expected-count top-rate ops per "
                        "step; stoch = Bernoulli fires + temperature-1 Q "
                        "samples (true condition only); thr{F} = fire ops "
                        "whose rate ≥ F·w(t) — CALIBRATED thresholding "
                        "against the training target (a pending op's "
                        "optimal rate IS w(t)), robust to a globally "
                        "under-scaled λ head. E.g. 'det,thr0.1,thr0.25'.")
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


@torch.no_grad()
def rates(model, ids: List[int], za, zs, t: float, device: str):
    """One forward → (lam (L,3) float32, hidden (L,d))."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(x)
    out = model(input_ids=x, attention_mask=attn, z_amp=za, z_sup=zs,
                t=torch.tensor([t], device=device))
    return out["lambda"][0], out["hidden"][0]


@torch.no_grad()
def decode_flow(
    model, src_ids: List[int], za, zs, *,
    steps: int, device: str, mode: str = "det",
    thr_frac: float = 0.0, w_max: float = 20.0,
    thr_abs_floor: float = 0.05,
    lens_bias: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    max_ops_per_step: int = 8, max_grow: int = 24,
    suppress_ids: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Integrate the rate model from t=0 to 1 over `steps` steps."""
    x = list(src_ids)
    zae = torch.zeros_like(za)
    zse = torch.zeros_like(zs)
    h_step = 1.0 / steps
    carry = 0.0
    stall = 0
    for step in range(steps):
        if len(x) - len(src_ids) >= max_grow:
            break
        t = (step + 0.5) * h_step
        lam, hid = rates(model, x, za, zs, t, device)
        if cfg_scale != 1.0:
            lam_e, hid_e = rates(model, x, zae, zse, t, device)
            lam = torch.clamp(lam_e + cfg_scale * (lam - lam_e), min=0.0)
        L = lam.shape[0]
        lam = lam.clone()
        lam[0, KIND_DEL] = 0.0                       # <bos> is structural
        lam[0, KIND_SUB] = 0.0

        flat = lam.reshape(-1)                       # (L*3,)
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
        for fi in order:
            if len(chosen) >= n_fire:
                break
            pos, kind = fi // 3, fi % 3
            if pos in used_pos or float(flat[fi]) <= 0:
                continue
            used_pos.add(pos)
            tok = None
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
                if mode == "det":
                    tok = int(logits.argmax())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    tok = int(torch.multinomial(probs, 1))
                if kind == KIND_SUB and tok == x[pos]:
                    continue                          # no-op substitution
            chosen.append({"kind": kind, "pos": pos, "tok": tok})
        if not chosen:
            continue
        x = apply_step_ops(x, chosen)
    return x


def main():
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    mode_specs = []                       # (name, decode_mode, thr_frac)
    for m in args.decode.split(","):
        m = m.strip()
        if not m:
            continue
        if m in ("det", "stoch"):
            mode_specs.append((m, m, 0.0))
        elif m.startswith("thr"):
            mode_specs.append((m, "thr", float(m[3:])))
        else:
            raise SystemExit(f"unknown decode mode {m!r}")
    modes = [name for name, _, _ in mode_specs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    rng = np.random.default_rng(args.seed)
    srng = random.Random(args.seed)

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

    agg = {c: {m: defaultdict(list) for m in modes} for c in conditions}
    iou_agg = {c: [] for c in conditions}
    tagger_iou_agg = []
    T_DIAG = (0.3, 0.5, 0.7, 0.9)
    lam_diag = {t: [] for t in T_DIAG}    # mean λ at top-|gold| sites, true
    bpair = {m: defaultdict(lambda: defaultdict(list)) for m in modes}
    records = []

    for step_i, k in enumerate(chosen_idx):
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids
        slots = align_pair(src_ids, tgt_ids)
        n_ops = len(slot_ops(slots))
        if n_ops == 0:
            continue
        gold_pos = gold_edit_positions(slots)
        bucket = _bucket(n_ops)

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
                "random": (randomize_intervention(za_t, rng),
                           randomize_intervention(zs_t, rng))}

        rec = {"idx": int(k), "src": src, "tgt": tgt, "n_ops": n_ops,
               "outputs": {}}
        for c in conditions:
            za = zvar[c][0].unsqueeze(0).to(args.device)
            zs = zvar[c][1].unsqueeze(0).to(args.device)

            # gate (a): λ-IoU of the rates on x' at t=iou_t
            lam, _ = rates(model, src_ids, za, zs, args.iou_t, args.device)
            lam_tot = lam.sum(dim=-1).cpu().tolist()
            iou = lambda_iou(lam_tot, gold_pos)
            if iou == iou:
                iou_agg[c].append(iou)

            if c == "true":
                # rate-magnitude diagnostic: mean top-|gold| λ across t —
                # compare against the training target w(t)
                for td in T_DIAG:
                    lam_d, _ = rates(model, src_ids, za, zs, td,
                                     args.device)
                    tot = lam_d.sum(dim=-1)
                    topv = tot.topk(min(len(gold_pos), tot.shape[0])).values
                    lam_diag[td].append(float(topv.mean()))
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
                        tagger_iou_agg.append(tiou)

            lb = lens_bias(zvar[c][0], zvar[c][1]) if c != "empty" else None
            rec["outputs"][c] = {"lambda_iou": iou}
            for m, dmode, frac in mode_specs:
                if m == "stoch" and c != "true":
                    continue
                out_ids = decode_flow(
                    model, src_ids, za, zs, steps=args.steps,
                    device=args.device, mode=dmode, thr_frac=frac,
                    w_max=args.w_max, lens_bias=lb,
                    cfg_scale=args.cfg_scale,
                    max_ops_per_step=args.max_ops_per_step,
                    max_grow=args.max_grow, suppress_ids=suppress,
                    rng=srng)
                out_text = tokenizer.decode(out_ids,
                                            skip_special_tokens=True)
                pm = pair_metrics(out_text, src, tgt)
                a = agg[c][m]
                a["exact"].append(pm["exact_match"])
                a["sim_target"].append(pm["sim_target"])
                a["copy"].append(pm["copy_rate"])
                a["no_edit"].append(float(out_ids == src_ids))
                if c == "true":
                    bp = bpair[m][bucket]
                    bp["exact"].append(pm["exact_match"])
                    bp["sim_target"].append(pm["sim_target"])
                rec["outputs"][c][m] = {
                    "text": out_text, "exact": pm["exact_match"],
                    "sim_target": pm["sim_target"]}
        records.append(rec)
        if (step_i + 1) % 10 == 0:
            print(f"[ef-probe] {step_i + 1}/{len(chosen_idx)} pairs")

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
              "λ head calibration: at a pending site the training optimum "
              "is λ* = w(t). Ratios ≪ 1 mean the decode's expected-count "
              "rule under-fires (use thr{F} decode or retrain with a "
              "larger rate-head LR).", "",
              "| t | w(t) target | mean top-|gold| λ | ratio |",
              "|---|---|---|---|"]
    payload["rate_calibration"] = {}
    for td in T_DIAG:
        wt = w_weight(td, args.w_max)
        mv = float(np.mean(lam_diag[td])) if lam_diag[td] else float("nan")
        payload["rate_calibration"][str(td)] = {"w": wt, "mean_top_lam": mv}
        lines.append(f"| {td:g} | {wt:.3f} | {mv:.4f} | {mv / wt:.3f} |")
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
    print("\n".join(lines))
    print(f"[ef-probe] wrote {out_dir}/probe_report.md, probe_metrics.json, "
          f"records.jsonl")


if __name__ == "__main__":
    main()
