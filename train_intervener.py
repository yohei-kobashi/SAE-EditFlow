"""Intervener training (INTERVENER_PLAN.md) — learned intervention
generator over the corruption cache, supervised through the FROZEN
gemma-2-2b-it.

Per batch:
  1. Load corruption records; (x0, x1) = (x', x); conditioning =
     diff_to_sparse on the cached z(X)/z(X') — identical draw family to
     editflow training (k_top / k-draw specs).
  2. Build the EVAL frame verbatim: chat-templated rewrite prompt
     (eval_clamp_baseline.PROMPT) with src = x0's text, assistant
     response = x1's text (+<end_of_turn>), teacher-forced.
  3. Intervener(src tokens, spec) -> delta_pre (per src position) and
     delta_dec (pooled). An additive tensor is injected at the frozen
     -it's layer L output: delta_pre at the src span inside the prompt,
     delta_dec at every response position (the decode-step counterpart —
     position resp_from-1 predicts the first response token WITHOUT
     delta_dec, matching generation where prefill only touches the span).
  4. Loss = NLL(x1 | do(residual)) through the frozen LM (gradients to
     the generator only)
     + norm-budget regularizer: per-position ||delta|| must stay within
       --norm-alpha * ||dvec(spec)|| (the steer-0.5 rendering norm) —
       keeps the channel intervention-sized, not a free soft prompt
     + null teachers: empty spec (prob --empty-prob) and MISMATCHED
       partner spec (prob --mismatch-null-prob, the P5 lesson) train
       target=x0 (copy) AND squared-norm suppression toward delta=0.

Frozen LM params get requires_grad=False; backward still flows through
its activations into the injected tensor. Cache identity records (empty
diff, x1=x0) are natural extra copy teachers.
"""

from __future__ import annotations

import argparse
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
from intervener import Intervener, find_subseq
from model import load_sae_w_dec
from resume_utils import (
    add_resume_args, find_latest_ckpt, load_train_state, save_train_state,
)
from scripts.eval_clamp_baseline import PROMPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--inject-layer", type=int, default=12,
                   help="frozen -it layer whose OUTPUT receives the delta "
                        "(Gemma Scope layer_L = residual after block L; "
                        "same site as the steer/clamp hooks).")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--dev-corruption-dir", default=None)
    p.add_argument("--eval-steps", type=int, default=2000)
    p.add_argument("--dev-batches", type=int, default=48)

    # Conditioning — same spec family as editflow training.
    p.add_argument("--k-top", type=int, default=32)
    p.add_argument("--k-amp", default="log:1-32")
    p.add_argument("--k-sup", default="log:1-32")
    p.add_argument("--empty-prob", type=float, default=0.08,
                   help="prob. of the EMPTY null teacher: spec zeroed, "
                        "target = x0 (copy) + delta-norm suppression. "
                        "Premise protection (empty -> no intervention).")
    p.add_argument("--mismatch-null-prob", type=float, default=0.12,
                   help="P5 null teacher: a PARTNER record's real spec, "
                        "target = x0 (copy) + delta-norm suppression — "
                        "silence under mismatched conditioning.")

    # v2: residual base + copy-attractor countermeasure.
    p.add_argument("--steer-alpha-base", type=float, default=0.0,
                   help="if > 0, the intervention is steer_alpha*dvec at "
                        "ALL positions (the C1' champion rendering) plus "
                        "the learned heads as corrections — zero-init "
                        "then starts AT steer0.5 (0.2385) instead of at "
                        "identity, which collapsed to copy in v1.")
    p.add_argument("--edit-weight", type=float, default=1.0,
                   help="CE upweight on the response tokens that differ "
                        "from src (LCP/LCS trim). Fights the copy "
                        "attractor: x1~x0 so unweighted NLL is dominated "
                        "by unchanged tokens.")

    # Norm budget (the 'intervention-sized' constraint).
    p.add_argument("--norm-alpha", type=float, default=0.5,
                   help="budget = norm_alpha * ||(za-zs)@W_dec|| per "
                        "example — the steer rendering at the C1' "
                        "operating alpha.")
    p.add_argument("--norm-reg-w", type=float, default=0.05,
                   help="weight of the over-budget penalty on true rows.")
    p.add_argument("--null-norm-w", type=float, default=0.1,
                   help="weight of the squared-norm suppression on "
                        "empty/mismatch rows (normalized by the batch's "
                        "mean true budget).")

    p.add_argument("--max-src-len", type=int, default=192,
                   help="skip records whose x0 or x1 token list exceeds "
                        "this (encoder side).")
    p.add_argument("--max-lm-len", type=int, default=384,
                   help="skip records whose prompt+response exceeds this "
                        "(frozen -it side).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    add_resume_args(p)
    return p.parse_args()


class InterventionCollator:
    """Densifies the z vectors and keeps token ids + texts raw (the spec
    draw / null-teacher sampling happens in the train loop where the rng
    lives). Text fields fall back to None (decoded in the loop)."""

    def __init__(self, d_sae: int):
        self.d_sae = int(d_sae)

    def __call__(self, batch: List[Dict]) -> Dict:
        return {
            "x0": [list(map(int, r["x_prime_token_ids"])) for r in batch],
            "x1": [list(map(int, r["x_token_ids"])) for r in batch],
            "x0_text": [r.get("x_prime_text") for r in batch],
            "x1_text": [r.get("x_text") for r in batch],
            "z_X": torch.stack([_dense_topk(r["z_X_topk"], self.d_sae)
                                for r in batch]),
            "z_X_prime": torch.stack([_dense_topk(r["z_X_prime_topk"],
                                                  self.d_sae)
                                      for r in batch]),
        }


def build_batch(batch: Dict, rng: np.random.Generator, args, ctx) -> Dict:
    """Assemble one training batch: encoder tensors, frozen-LM tensors,
    span maps and per-row bookkeeping. ctx carries tokenizers/ids/W_dec.
    Deterministic given (batch, rng state)."""
    tok, it_tok = ctx["tok"], ctx["it_tok"]
    eot_id, pad_id, it_pad = ctx["eot_id"], ctx["pad_id"], ctx["it_pad"]
    bos_id = ctx["bos_id"]
    rows = []
    B_ = len(batch["x0"])
    for b in range(B_):
        x0, x1 = batch["x0"][b], batch["x1"][b]
        if max(len(x0), len(x1)) > args.max_src_len:
            continue
        src_text = batch["x0_text"][b]
        if src_text is None:
            src_text = tok.decode(x0, skip_special_tokens=True)
        tgt_text = batch["x1_text"][b]
        if tgt_text is None:
            tgt_text = tok.decode(x1, skip_special_tokens=True)

        null_kind = None                     # None | "empty" | "mismatch"
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
            else:                            # partner had no diff: skip null
                a, s = diff_to_sparse(
                    batch["z_X"][b], batch["z_X_prime"][b],
                    k_top=args.k_top, k_amp=draw_k(rng, args._k_amp),
                    k_sup=draw_k(rng, args._k_sup), rng=rng,
                    empty_conditioning_prob=0.0)
        else:
            a, s = diff_to_sparse(
                batch["z_X"][b], batch["z_X_prime"][b], k_top=args.k_top,
                k_amp=draw_k(rng, args._k_amp),
                k_sup=draw_k(rng, args._k_sup), rng=rng,
                empty_conditioning_prob=0.0)
        target_text = src_text if null_kind else tgt_text

        text_in = it_tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(src=src_text)}],
            add_generation_prompt=True, tokenize=False)
        prompt_ids = it_tok(text_in, add_special_tokens=False).input_ids
        resp_ids = it_tok(target_text,
                          add_special_tokens=False).input_ids + [eot_id]
        if len(prompt_ids) + len(resp_ids) > args.max_lm_len:
            continue

        # src span inside the prompt <-> encoder positions. Encoder input
        # is the cache's x0 ids (BOS + sentence, same SP model as -it), so
        # needle[j] <-> x0[enc_off + j] <-> prompt[lo + j].
        needle = it_tok(src_text, add_special_tokens=False).input_ids
        enc_off = 1 if (x0 and x0[0] == bos_id) else 0

        # edit region in the response (true rows): LCP/LCS trim of
        # src-vs-target it-token ids; CE upweight lives there. Pure
        # deletions get the boundary token.
        edit_lo = edit_hi = None
        if null_kind is None and args.edit_weight != 1.0:
            aa, bb = needle, resp_ids[:-1]               # drop <end_of_turn>
            pfx = 0
            while (pfx < min(len(aa), len(bb))
                   and aa[pfx] == bb[pfx]):
                pfx += 1
            sfx = 0
            while (sfx < min(len(aa), len(bb)) - pfx
                   and aa[-1 - sfx] == bb[-1 - sfx]):
                sfx += 1
            edit_lo, edit_hi = pfx, len(bb) - sfx
            if edit_lo >= edit_hi:                       # pure deletion
                edit_lo = max(0, pfx - 1)
                edit_hi = min(len(bb), pfx + 1)

        lo = find_subseq(prompt_ids, needle)
        if lo is None and len(needle) > 1:   # leading-piece drift fallback
            lo = find_subseq(prompt_ids, needle[1:])
            if lo is not None:
                enc_off += 1
                needle = needle[1:]
        rows.append({
            "enc_ids": x0, "enc_off": enc_off,
            "prompt_ids": prompt_ids, "resp_ids": resp_ids,
            "span_lo": lo, "span_n": (len(needle) if lo is not None else 0),
            "null": null_kind, "za": a, "zs": s,
            "edit_lo": edit_lo, "edit_hi": edit_hi,
        })
    if not rows:
        return None

    B = len(rows)
    Te = max(len(r["enc_ids"]) for r in rows)
    enc_ids = torch.full((B, Te), pad_id, dtype=torch.long)
    enc_mask = torch.zeros((B, Te), dtype=torch.long)
    Tl = max(len(r["prompt_ids"]) + len(r["resp_ids"]) for r in rows)
    lm_ids = torch.full((B, Tl), it_pad, dtype=torch.long)
    lm_mask = torch.zeros((B, Tl), dtype=torch.long)
    labels = torch.full((B, Tl), -100, dtype=torch.long)
    ce_w = torch.ones((B, Tl), dtype=torch.float32)
    for i, r in enumerate(rows):
        e = r["enc_ids"]
        enc_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        enc_mask[i, :len(e)] = 1
        full = r["prompt_ids"] + r["resp_ids"]
        lm_ids[i, :len(full)] = torch.tensor(full, dtype=torch.long)
        lm_mask[i, :len(full)] = 1
        rf = len(r["prompt_ids"])
        labels[i, rf:len(full)] = torch.tensor(r["resp_ids"],
                                               dtype=torch.long)
        if r["edit_lo"] is not None:
            ce_w[i, rf + r["edit_lo"]:rf + r["edit_hi"]] = args.edit_weight
        r["resp_from"] = rf
        r["lm_len"] = len(full)
    W = ctx["w_dec"]                                     # (d_sae, d) f32
    za = torch.stack([r["za"] for r in rows])
    zs = torch.stack([r["zs"] for r in rows])
    dvec = (za.float() - zs.float()) @ W                 # (B, d)
    budget = args.norm_alpha * dvec.norm(dim=-1)         # (B,)
    return {
        "rows": rows, "enc_ids": enc_ids, "enc_mask": enc_mask,
        "lm_ids": lm_ids, "lm_mask": lm_mask, "labels": labels,
        "ce_w": ce_w,
        "z_amp": za, "z_sup": zs, "budget": budget,
        "null_mask": torch.tensor([r["null"] is not None for r in rows]),
        "span_found": float(np.mean([r["span_lo"] is not None
                                     for r in rows])),
    }


class AddHook:
    """Adds a full (B, T, d) tensor to the layer output — the batched
    teacher-forcing counterpart of intervener.InjectHook. Autograd flows
    through self.add."""

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


def intervener_loss(model, it_model, hook, built: Dict, args, device: str,
                    return_metrics: bool = False):
    out = model(built["enc_ids"].to(device), built["enc_mask"].to(device),
                built["z_amp"].to(device), built["z_sup"].to(device))
    delta_pre, delta_dec = out["delta_pre"], out["delta_dec"]  # f32
    B, Tl = built["lm_ids"].shape
    d = delta_dec.shape[-1]
    add = torch.zeros(B, Tl, d, device=device, dtype=torch.float32)
    if "base" in out:                    # v2 residual base: ALL positions
        add = add + (out["base"].unsqueeze(1)
                     * built["lm_mask"].to(device).unsqueeze(-1).float())
    pre_norms, pre_sq = [], []
    for i, r in enumerate(built["rows"]):
        if r["span_lo"] is not None and r["span_n"] > 0:
            n = min(r["span_n"], delta_pre.shape[1] - r["enc_off"],
                    Tl - r["span_lo"])
            if n > 0:
                seg = delta_pre[i, r["enc_off"]:r["enc_off"] + n]
                add[i, r["span_lo"]:r["span_lo"] + n] = seg
                pre_norms.append(seg.norm(dim=-1))       # (n,)
                pre_sq.append(seg.pow(2).sum(dim=-1).mean())
            else:
                pre_norms.append(None)
                pre_sq.append(delta_pre.sum() * 0.0)
        else:
            pre_norms.append(None)
            pre_sq.append(delta_pre.sum() * 0.0)
        add[i, r["resp_from"]:r["lm_len"]] = (
            add[i, r["resp_from"]:r["lm_len"]] + delta_dec[i])

    hook.add = add
    try:
        lm_out = it_model(input_ids=built["lm_ids"].to(device),
                          attention_mask=built["lm_mask"].to(device))
    finally:
        hook.add = None
    logits = lm_out.logits[:, :-1].float()               # (B, T-1, V)
    tgt = built["labels"][:, 1:].to(device)              # (B, T-1)
    ce_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tgt.reshape(-1), ignore_index=-100,
                             reduction="none").view(B, -1)
    valid = (tgt != -100).float()
    w = built["ce_w"][:, 1:].to(device) * valid          # edit upweight
    nll = (ce_tok * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    budget = built["budget"].to(device)                  # (B,)
    null = built["null_mask"].to(device)
    eps = 1e-6
    dec_norm = delta_dec.norm(dim=-1)                    # (B,)
    over_dec = F.relu(dec_norm - budget).pow(2) / (budget.pow(2) + eps)
    over_pre = torch.zeros_like(over_dec)
    for i, pn in enumerate(pre_norms):
        if pn is not None and pn.numel():
            over_pre[i] = (F.relu(pn - budget[i]).pow(2)
                           / (budget[i].pow(2) + eps)).mean()
    true_m = (~null).float()
    norm_pen = ((over_dec + over_pre) * true_m).sum() / true_m.sum().clamp_min(1.0)

    ref = ((budget.pow(2) * true_m).sum()
           / true_m.sum().clamp_min(1.0)).detach().clamp_min(1.0)
    null_sq = delta_dec.pow(2).sum(dim=-1) + torch.stack(pre_sq)
    null_pen = ((null_sq / ref) * null.float()).sum() / null.float().sum().clamp_min(1.0)

    loss = (nll.mean() + args.norm_reg_w * norm_pen
            + args.null_norm_w * null_pen)
    if not return_metrics:
        return loss
    m = {
        "nll": float(nll.mean()),
        "nll_true": float((nll * true_m).sum()
                          / true_m.sum().clamp_min(1.0)),
        "dec_norm": float((dec_norm * true_m).sum()
                          / true_m.sum().clamp_min(1.0)),
        "dec_norm_null": float((dec_norm * null.float()).sum()
                               / null.float().sum().clamp_min(1.0))
        if bool(null.any()) else float("nan"),
        "budget": float((budget * true_m).sum()
                        / true_m.sum().clamp_min(1.0)),
        "norm_pen": float(norm_pen), "null_pen": float(null_pen),
        "span_found": built["span_found"],
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

    print(f"[intervener] W_dec from {args.sae_repo}/{args.sae_path}")
    w_dec = load_sae_w_dec(args.sae_repo, args.sae_path)
    model = Intervener(args.llm2vec_dir, d_sae, dtype=dtype,
                       lora_r=args.lora_r, w_dec=w_dec,
                       steer_alpha=args.steer_alpha_base).to(args.device)

    it_model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype,
        attn_implementation="eager").to(args.device).eval()
    it_model.requires_grad_(False)
    hook = AddHook()
    it_model.model.layers[args.inject_layer].register_forward_hook(hook)
    print(f"[intervener] frozen {args.it_model}, AddHook on "
          f"layers[{args.inject_layer}] output")

    ctx = {
        "tok": tok, "it_tok": it_tok, "d_sae": d_sae,
        "pad_id": tok.pad_token_id or 0,
        "it_pad": it_tok.pad_token_id or it_tok.eos_token_id,
        "bos_id": int(tok.bos_token_id),
        "eot_id": int(it_tok.convert_tokens_to_ids("<end_of_turn>")),
        "w_dec": w_dec.float(),
    }

    lora_params = [p for n, p in model.named_parameters()
                   if "lora_" in n and p.requires_grad]
    small_params = [p for n, p in model.named_parameters()
                    if "lora_" not in n and p.requires_grad]
    print(f"[intervener] trainable: small="
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
    collator = InterventionCollator(d_sae=d_sae)
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
        print(f"[intervener] dev monitoring: {len(dev_batches)} batches")

    def evaluate_dev() -> Dict[str, float]:
        dev_rng = np.random.default_rng(args.seed + 9999)
        rows = []
        model.eval()
        with torch.no_grad():
            for batch in dev_batches:
                built = build_batch(batch, dev_rng, args, ctx)
                if built is None:
                    continue
                _, m = intervener_loss(model, it_model, hook, built, args,
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
                               "steer_alpha": args.steer_alpha_base,
                               "it_model": args.it_model}}, str(path))

    best_path = out_dir / "intervener-best.pt"
    best_json = out_dir / "best.json"
    best_dev = float("inf")
    if best_json.exists():
        try:
            best_dev = float(json.loads(best_json.read_text())["dev_nll"])
            print(f"[intervener] RESUME: best dev nll {best_dev:.4f}")
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
        print(f"[intervener] step={at_step} DEV "
              f"nll={m.get('nll', 0):.4f} nll_true={m['nll_true']:.4f} "
              f"|dd|={m.get('dec_norm', 0):.2f} "
              f"|dd|null={m.get('dec_norm_null', float('nan')):.2f} "
              f"budget={m.get('budget', 0):.2f} "
              f"(best {best_dev:.4f}){marker}")

    step = 0
    if args.resume:
        latest = find_latest_ckpt(out_dir, "intervener")
        if latest is not None:
            ckpt_path, ckpt_step = latest
            print(f"[intervener] RESUME: {ckpt_path} (step {ckpt_step})")
            blob = torch.load(str(ckpt_path), map_location=args.device,
                              weights_only=False)
            model.load_trainable_state_dict(blob["trainable"])
            restored = load_train_state(ckpt_path, optim, sched,
                                        device=args.device)
            step = restored if restored is not None else ckpt_step
            if step >= args.max_steps:
                print("[intervener] RESUME: already past max_steps")
                save_ckpt(out_dir / "intervener-final.pt")
                return

    loss_window = []
    pbar = tqdm(total=args.max_steps, initial=step, desc="[intervener]",
                unit="step", dynamic_ncols=True)
    for batch in loader:
        if step >= args.max_steps:
            break
        built = build_batch(batch, rng, args, ctx)
        if built is None:
            continue
        if step == 0:
            b = built["budget"]
            print(f"[intervener] budget |{args.norm_alpha}*dvec| first "
                  f"batch: mean={float(b.mean()):.2f} "
                  f"min={float(b.min()):.2f} max={float(b.max()):.2f} "
                  f"span_found={built['span_found']:.2f}")
        loss = intervener_loss(model, it_model, hook, built, args,
                               args.device)
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
            print(f"[intervener] step={step} loss={avg:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}")
        if step > 0 and step % args.save_steps == 0:
            ckpt = out_dir / f"intervener-step{step}.pt"
            save_ckpt(ckpt)
            save_train_state(ckpt, optim, sched, step)
            print(f"[intervener] saved {ckpt}")
        if dev_batches and step > 0 and step % args.eval_steps == 0:
            maybe_update_best(step)
        step += 1
        pbar.update(1)
    pbar.close()

    maybe_update_best(step)
    if dev_batches and best_path.exists():
        save_ckpt(out_dir / "intervener-last.pt")
        import shutil
        shutil.copyfile(best_path, out_dir / "intervener-final.pt")
        print(f"[intervener] done at step {step}; intervener-final.pt = "
              f"best dev state ({json.loads(best_json.read_text())})")
    else:
        save_ckpt(out_dir / "intervener-final.pt")
        print(f"[intervener] done at step {step}")


if __name__ == "__main__":
    main()
