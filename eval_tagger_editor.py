"""
Held-out evaluation of the trained SAE-LEWIS tagger and editor (v2).

Runs Level 1 (held-out accuracy) and Level 2 (conditioning causality) on a
DEV corruption cache — a cache generated from Dolma sentences disjoint from
the training cache (see `corruption.py --skip-sentences`), or a set of
shards held out from the training cache.

Level 1 — per-stage metrics (README §13.1):
  tagger : 3-class op head — per-class P/R/F1 over KEEP/REPL/DEL, confusion
           matrix, macro-F1 vs the all-KEEP baseline;
           binary insert head — P/R/F1 of "insert before this token";
           edit-span IoU over the union of both heads' edit positions
  editor : per-position-type argmax accuracy over the template segment of
           the `x' [SEP] x'_c` input (KEEP identity, REPL/INS restoration
           top-1/top-5), marker-emission rate, mean CE loss

Level 2 — conditioning causality (the central SAE-LEWIS validity check):
  every sample is evaluated under three paired conditions —
    true   : diff-based z_amp/z_sup (as in training, empty-prob = 0)
    empty  : z_amp = z_sup = 0
    random : same k / same magnitudes as `true`, random feature indices
  Interpretation:
    Δ(true − empty) ≈ 0  → the model IGNORES the conditioning
    Δ(true − random) ≪ Δ(true − empty)
                         → conditioning is used as an opaque "something
                           changed" flag, not as SAE-grounded features

Outputs `eval_report.md` + `eval_metrics.json` under --output-dir.

Usage:
    python eval_tagger_editor.py \
        --corruption-dir runs/prod/corruption_dev \
        --llm2vec-dir    runs/prod/llm2vec_merged \
        --tagger-ckpt    runs/prod/tagger/tagger-final.pt \
        --editor-ckpt    runs/prod/editor/editor-final.pt \
        --output-dir     runs/prod/eval_tagger_editor
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from data import CorruptionCollator, CorruptionDataset
from editor import load_editor_from_checkpoint
from intervene import diff_to_sparse, draw_k, parse_k_spec
from lewis_ops import NUM_OPS3, OP3_NAMES
from tagger import load_tagger_from_checkpoint

CONDITIONS = ["true", "empty", "random"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True,
                   help="DEV corruption cache (disjoint from training; "
                        "generate with corruption.py --skip-sentences).")
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--tagger-ckpt", default=None,
                   help="tagger-final.pt (skip tagger eval if omitted).")
    p.add_argument("--editor-ckpt", default=None,
                   help="editor-final.pt (skip editor eval if omitted).")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--k-top", type=int, default=8,
                   help="Diff candidate pool size (training used 8).")
    p.add_argument("--k-amp", default="1-4",
                   help="Per-sample k_amp draw: 'LO-HI' (uniform incl.) or "
                        "a fixed int. Training used '1-4'.")
    p.add_argument("--k-sup", default="1-4",
                   help="Per-sample k_sup draw; same syntax as --k-amp.")
    p.add_argument("--ins-threshold", type=float, default=0.5,
                   help="Sigmoid threshold for the tagger's insert head.")
    p.add_argument("--only-families", default="",
                   help="Comma list of transforms.FAMILIES keys: evaluate "
                        "ONLY transform records touching these families "
                        "(the held-out side of a LOFO experiment).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Paired conditioning construction
# ---------------------------------------------------------------------------
_parse_k_spec = parse_k_spec        # shared with training (intervene.py)


def build_conditions(
    z_X: torch.Tensor,          # (B, d_sae)
    z_X_prime: torch.Tensor,    # (B, d_sae)
    k_top: int,
    rng: np.random.Generator,
    k_amp_spec: str = "1-4",
    k_sup_spec: str = "1-4",
    conditions: Optional[List[str]] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Build the paired (z_amp, z_sup) variants for a batch.

    The default per-sample k_amp / k_sup draw matches the training
    distribution (uniform over {1..4}; override via k specs) and
    empty-conditioning is disabled so every sample carries signal under
    `true`. `random` reuses the exact nonzero counts and (permuted)
    magnitudes of `true` at uniformly random feature indices, so any
    accuracy gap vs `true` isolates feature IDENTITY. Only the requested
    `conditions` are built (hyperparameter sweeps use ["true"] alone).
    """
    conditions = list(CONDITIONS) if conditions is None else list(conditions)
    amp_spec = parse_k_spec(k_amp_spec)
    sup_spec = parse_k_spec(k_sup_spec)
    B, d_sae = z_X.shape
    amp_t = torch.zeros_like(z_X)
    sup_t = torch.zeros_like(z_X)
    amp_r = torch.zeros_like(z_X)
    sup_r = torch.zeros_like(z_X)

    for b in range(B):
        k_amp = draw_k(rng, amp_spec)
        k_sup = draw_k(rng, sup_spec)
        a, s = diff_to_sparse(
            z_X[b], z_X_prime[b],
            k_top=k_top, k_amp=k_amp, k_sup=k_sup,
            rng=rng, empty_conditioning_prob=0.0,
        )
        amp_t[b] = a
        sup_t[b] = s
        if "random" not in conditions:
            continue
        for true_vec, rand_vec in ((a, amp_r), (s, sup_r)):
            nz = (true_vec > 0).nonzero(as_tuple=True)[0]
            if len(nz) == 0:
                continue
            vals = true_vec[nz].cpu().numpy()
            rng.shuffle(vals)
            rand_idx = rng.choice(d_sae, size=len(nz), replace=False)
            for i, v in zip(rand_idx, vals):
                rand_vec[b, int(i)] = float(v)

    zeros = torch.zeros_like(z_X)
    all_variants = {
        "true": (amp_t, sup_t),
        "empty": (zeros, zeros.clone()),
        "random": (amp_r, sup_r),
    }
    return {c: all_variants[c] for c in conditions}


# ---------------------------------------------------------------------------
# Tagger evaluation
# ---------------------------------------------------------------------------
def eval_tagger(args, d_sae: int, loader, tagger=None,
                conditions: Optional[List[str]] = None,
                ins_thresholds: Optional[List[float]] = None) -> Dict:
    """Level 1+2 tagger metrics.

    tagger        : pass a preloaded model to skip load/free (sweeps call
                    this repeatedly under different args).
    conditions    : subset of CONDITIONS (default: all three).
    ins_thresholds: extra INS thresholds evaluated from the SAME forwards
                    (reported under conditions[c]["ins_by_threshold"];
                    the primary args.ins_threshold drives "ins" and the
                    span IoU as before).
    """
    conds_list = list(CONDITIONS) if conditions is None else list(conditions)
    own_model = tagger is None
    if own_model:
        dtype = _dtype(args.llm_dtype)
        print(f"[eval] loading tagger from {args.tagger_ckpt}")
        tagger = load_tagger_from_checkpoint(
            args.llm2vec_dir, args.tagger_ckpt, d_sae=d_sae, dtype=dtype,
        ).to(args.device)
        tagger.eval()
    thresholds = [float(args.ins_threshold)] + [
        float(t) for t in (ins_thresholds or [])
        if float(t) != float(args.ins_threshold)]

    rng = np.random.default_rng(args.seed)
    conf = {c: np.zeros((NUM_OPS3, NUM_OPS3), dtype=np.int64) for c in conds_list}
    ins_cnt = {c: {t: defaultdict(int) for t in thresholds}
               for c in conds_list}                        # tp/fp/fn/tn
    iou_sum = {c: 0.0 for c in conds_list}
    iou_n = {c: 0 for c in conds_list}
    seen = 0

    for batch in tqdm(loader, desc="[eval:tagger]", unit="batch"):
        conds = build_conditions(batch["z_X"], batch["z_X_prime"],
                                 args.k_top, rng,
                                 k_amp_spec=getattr(args, "k_amp", "1-4"),
                                 k_sup_spec=getattr(args, "k_sup", "1-4"),
                                 conditions=conds_list)
        ids = batch["tagger_input_ids"].to(args.device)
        attn = batch["tagger_attention_mask"].to(args.device)
        op_gold = batch["tagger_op3_gold"]
        ins_gold = batch["tagger_ins_gold"]
        valid = op_gold != -100

        for c in conds_list:
            z_amp, z_sup = conds[c]
            with torch.no_grad():
                out = tagger(ids, attn,
                             z_amp.to(args.device), z_sup.to(args.device))
            op_pred = out["op_logits"].argmax(dim=-1).cpu()
            ins_prob = torch.sigmoid(out["ins_logits"].float().cpu())
            ins_pred = (ins_prob >= args.ins_threshold).long()

            g = op_gold[valid].numpy()
            p = op_pred[valid].numpy()
            np.add.at(conf[c], (g, p), 1)

            ig = ins_gold[valid]
            for t in thresholds:
                ip_t = (ins_prob >= t).long()[valid]
                ic = ins_cnt[c][t]
                ic["tp"] += int(((ig == 1) & (ip_t == 1)).sum())
                ic["fp"] += int(((ig == 0) & (ip_t == 1)).sum())
                ic["fn"] += int(((ig == 1) & (ip_t == 0)).sum())
                ic["tn"] += int(((ig == 0) & (ip_t == 0)).sum())

            # Edit-span IoU per sample: union of both heads' edit positions
            # (op3 != KEEP or ins fired) vs the gold union.
            for b in range(ids.shape[0]):
                v = valid[b]
                ge = set(((op_gold[b][v] > 0) | (ins_gold[b][v] == 1))
                         .nonzero(as_tuple=True)[0].tolist())
                pe = set(((op_pred[b][v] > 0) | (ins_pred[b][v] == 1))
                         .nonzero(as_tuple=True)[0].tolist())
                union = ge | pe
                if union:
                    iou_sum[c] += len(ge & pe) / len(union)
                    iou_n[c] += 1

        seen += ids.shape[0]
        if seen >= args.max_samples:
            break

    metrics = {"n_samples": seen, "conditions": {}}
    for c in conds_list:
        m = _prf_from_confusion(conf[c])
        m["span_iou"] = iou_sum[c] / max(1, iou_n[c])
        m["confusion"] = conf[c].tolist()
        m["ins"] = _prf_binary(ins_cnt[c][thresholds[0]])
        if len(thresholds) > 1:
            m["ins_by_threshold"] = {
                f"{t:g}": _prf_binary(ins_cnt[c][t]) for t in thresholds}
        metrics["conditions"][c] = m

    # All-KEEP baseline from the gold marginal (same for every condition)
    gold_counts = conf["true"].sum(axis=1)
    baseline_conf = np.zeros_like(conf["true"])
    baseline_conf[:, 0] = gold_counts
    metrics["all_keep_baseline"] = _prf_from_confusion(baseline_conf)

    metrics["diagnostics"] = {
        "cond_scale": float(tagger.cond_scale.item()),
        "proj_a_weight_norm": float(tagger.proj_a.weight.norm().item()),
    }
    if own_model:
        del tagger
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics


def _prf_binary(cnt: Dict[str, int]) -> Dict:
    tp, fp, fn = float(cnt["tp"]), float(cnt["fp"]), float(cnt["fn"])
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "support": int(cnt["tp"] + cnt["fn"])}


def _prf_from_confusion(conf: np.ndarray) -> Dict:
    per_class = {}
    f1s = []
    for i, name in enumerate(OP3_NAMES):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        per_class[name] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": int(conf[i, :].sum()),
        }
        f1s.append(f1)
    total = float(conf.sum()) or 1.0
    return {
        "accuracy": float(np.trace(conf)) / total,
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Editor evaluation
# ---------------------------------------------------------------------------
def eval_editor(args, d_sae: int, loader, tok, editor=None,
                conditions: Optional[List[str]] = None) -> Dict:
    conds_list = list(CONDITIONS) if conditions is None else list(conditions)
    own_model = editor is None
    if own_model:
        dtype = _dtype(args.llm_dtype)
        print(f"[eval] loading editor from {args.editor_ckpt}")
        editor = load_editor_from_checkpoint(
            args.llm2vec_dir, args.editor_ckpt, d_sae=d_sae, dtype=dtype,
        ).to(args.device)
        editor.eval()

    mask_id = int(tok.mask_token_id)
    ins_id = int(tok.convert_tokens_to_ids("[INS]"))
    sep_id = int(tok.convert_tokens_to_ids("[SEP]"))
    del_id = int(tok.convert_tokens_to_ids("[DEL]"))
    marker_ids = torch.tensor([mask_id, ins_id, sep_id, del_id])

    rng = np.random.default_rng(args.seed)
    agg = {c: defaultdict(float) for c in conds_list}
    seen = 0

    for batch in tqdm(loader, desc="[eval:editor]", unit="batch"):
        conds = build_conditions(batch["z_X"], batch["z_X_prime"],
                                 args.k_top, rng,
                                 k_amp_spec=getattr(args, "k_amp", "1-4"),
                                 k_sup_spec=getattr(args, "k_sup", "1-4"),
                                 conditions=conds_list)
        ids = batch["editor_input_ids"].to(args.device)
        attn = batch["editor_attention_mask"].to(args.device)
        tgt = batch["editor_target_ids"]

        # Labels are -100 over the x' [SEP] prefix, so `valid` selects the
        # template segment only.
        valid = tgt != -100
        is_repl = (batch["editor_input_ids"] == mask_id) & valid
        is_ins = (batch["editor_input_ids"] == ins_id) & valid
        is_keep = valid & ~is_repl & ~is_ins

        for c in conds_list:
            z_amp, z_sup = conds[c]
            with torch.no_grad():
                out = editor(ids, attn,
                             z_amp.to(args.device), z_sup.to(args.device))
                logits = out["logits"].float().cpu()

            a = agg[c]
            a["loss_sum"] += float(F.cross_entropy(
                logits[valid], tgt[valid], reduction="sum").item())
            a["loss_n"] += int(valid.sum())

            pred1 = logits.argmax(dim=-1)
            for key, m in (("keep", is_keep), ("repl", is_repl),
                           ("ins", is_ins)):
                a[f"{key}_n"] += int(m.sum())
                a[f"{key}_top1"] += int((pred1[m] == tgt[m]).sum())
            for key, m in (("repl", is_repl), ("ins", is_ins)):
                if m.any():
                    top5 = logits[m].topk(5, dim=-1).indices
                    a[f"{key}_top5"] += int(
                        (top5 == tgt[m].unsqueeze(-1)).any(dim=-1).sum())
            # Marker emission: a special token ([MASK]/[INS]/[SEP]/[DEL])
            # winning the argmax anywhere in the template is always wrong —
            # v2 targets contain no special tokens.
            a["marker_emit"] += int(
                torch.isin(pred1[valid], marker_ids).sum())

        seen += ids.shape[0]
        if seen >= args.max_samples:
            break

    metrics = {"n_samples": seen, "conditions": {}}
    for c in conds_list:
        a = agg[c]
        m = {"mean_loss": a["loss_sum"] / max(1, a["loss_n"])}
        for key in ("keep", "repl", "ins"):
            n = max(1, int(a[f"{key}_n"]))
            m[f"{key}_positions"] = int(a[f"{key}_n"])
            m[f"{key}_top1_acc"] = a[f"{key}_top1"] / n
        for key in ("repl", "ins"):
            m[f"{key}_top5_acc"] = a[f"{key}_top5"] / max(1, int(a[f"{key}_n"]))
        m["marker_emission_rate"] = a["marker_emit"] / max(1, int(a["loss_n"]))
        metrics["conditions"][c] = m

    metrics["diagnostics"] = {
        "cond_scale": float(editor.cond_scale.item()),
        "proj_a_weight_norm": float(editor.proj_a.weight.norm().item()),
        "delta_emb_norms": {
            name: float(editor.delta_emb[slot].norm().item())
            for slot, (name, _tid) in enumerate(editor.train_token_ids.items())
        } if editor.delta_emb.numel() > 0 else {},
    }
    if own_model:
        del editor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics


# ---------------------------------------------------------------------------
# Verdicts + report
# ---------------------------------------------------------------------------
def _verdict(delta_true_empty: float, delta_true_random: float,
             used_thresh: float, weak_thresh: float) -> Dict[str, str]:
    if delta_true_empty >= used_thresh:
        used = "USED"
    elif delta_true_empty >= weak_thresh:
        used = "WEAK"
    else:
        used = "IGNORED"
    if used == "IGNORED":
        grounded = "n/a"
    elif delta_true_random >= 0.5 * delta_true_empty:
        grounded = "SAE-GROUNDED"
    else:
        grounded = "OPAQUE-FLAG-WARNING"
    return {"conditioning": used, "grounding": grounded}


def build_verdicts(tagger_m: Optional[Dict], editor_m: Optional[Dict]) -> Dict:
    v = {}
    if tagger_m:
        f1 = {c: tagger_m["conditions"][c]["macro_f1"] for c in CONDITIONS}
        v["tagger_op3"] = {
            "metric": "macro_f1",
            "true": f1["true"], "empty": f1["empty"], "random": f1["random"],
            "delta_true_empty": f1["true"] - f1["empty"],
            "delta_true_random": f1["true"] - f1["random"],
            **_verdict(f1["true"] - f1["empty"], f1["true"] - f1["random"],
                       used_thresh=0.02, weak_thresh=0.005),
        }
        insf = {c: tagger_m["conditions"][c]["ins"]["f1"] for c in CONDITIONS}
        v["tagger_ins"] = {
            "metric": "ins_f1",
            "true": insf["true"], "empty": insf["empty"], "random": insf["random"],
            "delta_true_empty": insf["true"] - insf["empty"],
            "delta_true_random": insf["true"] - insf["random"],
            **_verdict(insf["true"] - insf["empty"], insf["true"] - insf["random"],
                       used_thresh=0.02, weak_thresh=0.005),
        }
    if editor_m:
        for key, thresh in (("repl_top1_acc", 0.02), ("ins_top1_acc", 0.02)):
            acc = {c: editor_m["conditions"][c][key] for c in CONDITIONS}
            v[f"editor_{key}"] = {
                "metric": key,
                "true": acc["true"], "empty": acc["empty"], "random": acc["random"],
                "delta_true_empty": acc["true"] - acc["empty"],
                "delta_true_random": acc["true"] - acc["random"],
                **_verdict(acc["true"] - acc["empty"], acc["true"] - acc["random"],
                           used_thresh=thresh, weak_thresh=thresh / 4),
            }
    return v


def write_report(out_dir: Path, tagger_m, editor_m, verdicts) -> None:
    lines = ["# Tagger / Editor held-out evaluation (v2)", ""]

    if tagger_m:
        lines += ["## Tagger (Level 1)", "",
                  f"samples: {tagger_m['n_samples']}", ""]
        base = tagger_m["all_keep_baseline"]
        lines += ["| condition | acc | macro-F1 | span IoU | "
                  + " | ".join(f"{n} F1" for n in OP3_NAMES)
                  + " | INS-before F1 |",
                  "|---|---|---|---|" + "---|" * (NUM_OPS3 + 1)]
        for c in CONDITIONS:
            m = tagger_m["conditions"][c]
            row = (f"| {c} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} "
                   f"| {m['span_iou']:.4f} |")
            row += "".join(f" {m['per_class'][n]['f1']:.4f} |" for n in OP3_NAMES)
            row += f" {m['ins']['f1']:.4f} |"
            lines.append(row)
        lines.append(f"| all-KEEP baseline | {base['accuracy']:.4f} "
                     f"| {base['macro_f1']:.4f} | — |"
                     + " — |" * (NUM_OPS3 + 1))
        lines += ["", "Per-class detail (condition = true):"]
        for n in OP3_NAMES:
            pc = tagger_m["conditions"]["true"]["per_class"][n]
            lines.append(f"- {n}: P={pc['precision']:.4f} R={pc['recall']:.4f} "
                         f"F1={pc['f1']:.4f} (support {pc['support']})")
        pc = tagger_m["conditions"]["true"]["ins"]
        lines.append(f"- INS-before (binary head): P={pc['precision']:.4f} "
                     f"R={pc['recall']:.4f} F1={pc['f1']:.4f} "
                     f"(support {pc['support']})")
        d = tagger_m["diagnostics"]
        lines += ["", f"cond_scale = {d['cond_scale']:.4f}   "
                      f"‖Proj_A.W‖ = {d['proj_a_weight_norm']:.2f}", ""]

    if editor_m:
        lines += ["## Editor (Level 1)", "",
                  f"samples: {editor_m['n_samples']}", ""]
        cols = ["mean_loss", "keep_top1_acc", "repl_top1_acc", "repl_top5_acc",
                "ins_top1_acc", "ins_top5_acc", "marker_emission_rate"]
        lines += ["| condition | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for c in CONDITIONS:
            m = editor_m["conditions"][c]
            lines.append("| " + c + " |"
                         + "".join(f" {m[k]:.4f} |" for k in cols))
        d = editor_m["diagnostics"]
        lines += ["",
                  f"cond_scale = {d['cond_scale']:.4f}   "
                  f"‖Proj_A.W‖ = {d['proj_a_weight_norm']:.2f}"]
        if d["delta_emb_norms"]:
            lines.append("delta_emb row norms: " + ", ".join(
                f"{k}={v:.3f}" for k, v in d["delta_emb_norms"].items()))
        lines.append("")

    lines += ["## Conditioning causality (Level 2)", ""]
    lines += ["| probe | metric | true | empty | random | Δ(t−e) | Δ(t−r) "
              "| conditioning | grounding |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, v in verdicts.items():
        lines.append(
            f"| {name} | {v['metric']} | {v['true']:.4f} | {v['empty']:.4f} "
            f"| {v['random']:.4f} | {v['delta_true_empty']:+.4f} "
            f"| {v['delta_true_random']:+.4f} | {v['conditioning']} "
            f"| {v['grounding']} |")
    lines += [
        "",
        "Reading guide:",
        "- `conditioning=IGNORED` — the model does not use z_amp/z_sup. "
        "Check cond_scale (→0 means it learned to mute the prefix), "
        "Proj_A gradients, or over-strong sub-sampling (C11).",
        "- `grounding=OPAQUE-FLAG-WARNING` — random features work almost "
        "as well as the true ones: the conditioning acts as a generic "
        "\"something changed\" flag rather than SAE feature identity. "
        "The random-conditioning ablation (§13.2) will likely fail.",
        "",
    ]
    (out_dir / "eval_report.md").write_text("\n".join(lines))


def _dtype(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def make_loader(args, d_sae: int, tok) -> DataLoader:
    only = [x.strip() for x in
            getattr(args, "only_families", "").split(",") if x.strip()]
    ds = CorruptionDataset(args.corruption_dir, shuffle=False,
                           seed=args.seed, infinite=False,
                           only_t_families=only or None)
    coll = CorruptionCollator(
        d_sae=d_sae, pad_token_id=tok.pad_token_id,
        sep_token_id=tok.convert_tokens_to_ids("[SEP]"),
        del_token_id=tok.convert_tokens_to_ids("[DEL]"),
        bos_token_id=tok.bos_token_id,
    )
    return DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                      collate_fn=coll)


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.tagger_ckpt and not args.editor_ckpt:
        raise SystemExit("[eval] provide --tagger-ckpt and/or --editor-ckpt")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])

    # The two phases iterate the dataset independently but with identical
    # order (shuffle=False) and identical rng seeding, so the paired
    # conditioning is the same for both models.
    tagger_m = None
    editor_m = None
    if args.tagger_ckpt:
        tagger_m = eval_tagger(args, d_sae, make_loader(args, d_sae, tok))
    if args.editor_ckpt:
        editor_m = eval_editor(args, d_sae, make_loader(args, d_sae, tok), tok)

    verdicts = build_verdicts(tagger_m, editor_m)
    payload = {
        "corruption_dir": str(args.corruption_dir),
        "llm2vec_dir": str(args.llm2vec_dir),
        "tagger_ckpt": args.tagger_ckpt,
        "editor_ckpt": args.editor_ckpt,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "tagger": tagger_m,
        "editor": editor_m,
        "verdicts": verdicts,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2))
    write_report(out_dir, tagger_m, editor_m, verdicts)

    print(f"[eval] wrote {out_dir}/eval_metrics.json and eval_report.md")
    for name, v in verdicts.items():
        print(f"[eval] {name}: true={v['true']:.4f} empty={v['empty']:.4f} "
              f"random={v['random']:.4f} → conditioning={v['conditioning']} "
              f"grounding={v['grounding']}")


if __name__ == "__main__":
    main()
