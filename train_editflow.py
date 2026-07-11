"""
SAE-EF training (EDIT_FLOWS_PLAN.md §1.3, §4.3) — flow matching with edit
operations over the corruption cache.

Per batch:
  1. Load corruption records; coupling (x0, x1) = (x', x) — the cache pair.
  2. Deterministic min-edit alignment (editflow_ops.align_pair).
  3. t ~ U(0,1) per sample; each op fires independently with κ(t) = t³
     → state x_t and its pending ops (build_xt).
  4. Conditioning: diff_to_sparse on the cached z(X)/z(X'), identical to
     editor training (k_top / k-draw specs).
  5. Loss (Bregman / flow matching):
        L = Σ_pos Σ_kind λ̂  −  w(t) · Σ_pending [ log λ̂(kind, pos)
                                                  + log Q̂(tgt | pos) ]
     with w(t) = 3t²/(1−t³) clipped at --w-max. Null records (zero ops,
     zero diff) contribute the pure suppression term — the premise-
     protection teacher: empty conditioning must drive all rates to 0.

Dev monitor (fixed batches + fixed t / conditioning draws):
  dev loss, pending-kind accuracy (argmax λ over 3 kinds at pending
  positions), pending-site IoU (top-|pending| positions by total λ),
  Q top-1 at pending sub/ins, and the empty-vs-true mean rate ratio
  (gate (c) proxy: empty per-token rate must stay ≪ true).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from data import CorruptionDataset, _dense_topk
from editflow import SAEEditFlow
from editflow_ops import (
    KIND_INS, KIND_SUB, align_pair, build_xt, cache_slots, kappa, slot_ops,
    w_weight,
)
from intervene import diff_to_sparse, draw_k, parse_k_spec
from model import load_sae_w_dec
from resume_utils import (
    add_resume_args, find_latest_ckpt, load_train_state, save_train_state,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-from-editor", default=None,
                   help="v6 SAEEditor checkpoint: warm-start the "
                        "conditioning stack (Proj_A correction, type_emb, "
                        "cond_scale) and the LoRA adapters.")
    p.add_argument("--init-from-editflow", default=None,
                   help="SAE-EF checkpoint: warm-start across Z variants "
                        "(missing/extra heads skipped). Applied AFTER "
                        "--init-from-editor if both are given.")

    # Z1 (EDIT_FLOWS_ZERO.md)
    p.add_argument("--t-film", action="store_true",
                   help="Z1a: FiLM(t) on the λ head input — fixes the "
                        "pilot's rate saturation (λ≈0.25 while w(t)→9; "
                        "README §13.8).")
    p.add_argument("--rate-head-lr", type=float, default=3e-3,
                   help="Z1a: separate LR for the rate head (lam_head + "
                        "lam_film) — 10x the small-params LR by default.")
    p.add_argument("--cond-mode", default="pooled",
                   choices=["pooled", "feature-tokens"],
                   help="Z1b: 'feature-tokens' = one prefix token per "
                        "commanded feature (W_dec base + sign + magnitude) "
                        "instead of the pooled 2-vector prefix.")
    p.add_argument("--true-align", action="store_true",
                   help="Z2: build slots from the cache's editor artifacts "
                        "(the ops that GENERATED the pair) instead of "
                        "difflib; falls back per record when they don't "
                        "reconstruct (x0, x1).")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--proj-a-rank", type=int, default=32)
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--dev-corruption-dir", default=None)
    p.add_argument("--eval-steps", type=int, default=2000)
    p.add_argument("--dev-batches", type=int, default=64)

    # Conditioning — same spec family as the v6 editor run.
    p.add_argument("--k-top", type=int, default=32)
    p.add_argument("--k-amp", default="log:1-32")
    p.add_argument("--k-sup", default="log:1-32")
    p.add_argument("--empty-cond-prob", type=float, default=0.0,
                   help="Empty-conditioning dropout on NON-null records. "
                        "Default 0: the cache's null records are the only "
                        "empty-cond teacher, so empty → all-λ-0 (premise "
                        "protection, gate (c)). Raise only to trade the "
                        "no-edit guarantee for a stronger CFG baseline.")
    p.add_argument("--exclude-families", default="")

    # Flow process
    p.add_argument("--w-max", type=float, default=20.0,
                   help="Clip on w(t)=3t²/(1−t³) (t→1 divergence).")
    p.add_argument("--max-len", type=int, default=256,
                   help="Skip records whose x0 or x1 exceeds this.")

    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    add_resume_args(p)
    return p.parse_args()


class EditFlowCollator:
    """Keeps token lists raw (alignment / z_t are built per step in the
    train loop, where the rng lives); densifies the z vectors."""

    def __init__(self, d_sae: int):
        self.d_sae = int(d_sae)

    def __call__(self, batch: List[Dict]) -> Dict:
        return {
            "x0": [list(map(int, r["x_prime_token_ids"])) for r in batch],
            "x1": [list(map(int, r["x_token_ids"])) for r in batch],
            "ei": [r.get("editor_input_token_ids") for r in batch],
            "et": [r.get("editor_target_token_ids") for r in batch],
            "z_X": torch.stack([_dense_topk(r["z_X_topk"], self.d_sae)
                                for r in batch]),
            "z_X_prime": torch.stack([_dense_topk(r["z_X_prime_topk"],
                                                  self.d_sae)
                                      for r in batch]),
        }


def build_batch(batch: Dict, rng: np.random.Generator, args,
                pad_id: int, fixed_t: List[float] = None) -> Dict:
    """Sample (t, fired) per record and assemble padded x_t tensors plus
    flat pending-op index lists. Records longer than --max-len are dropped
    (skipped indices simply don't contribute)."""
    xs, ts, pend_all = [], [], []
    za_l, zs_l = [], []
    for b in range(len(batch["x0"])):
        x0, x1 = batch["x0"][b], batch["x1"][b]
        if max(len(x0), len(x1)) > args.max_len:
            continue
        slots = None
        if getattr(args, "true_align", False) and batch["ei"][b] is not None:
            slots = cache_slots(x0, x1, batch["ei"][b], batch["et"][b],
                                args._mask_id, args._ins_id, args._del_id)
        if slots is None:
            slots = align_pair(x0, x1)
        ops = slot_ops(slots)
        t = float(fixed_t[len(xs) % len(fixed_t)]) if fixed_t \
            else float(rng.random())
        fired = [bool(rng.random() < kappa(t)) for _ in ops]
        x_t, pending = build_xt(slots, ops, fired)
        k_amp = draw_k(rng, args._k_amp)
        k_sup = draw_k(rng, args._k_sup)
        a, s = diff_to_sparse(
            batch["z_X"][b], batch["z_X_prime"][b], k_top=args.k_top,
            k_amp=k_amp, k_sup=k_sup, rng=rng,
            empty_conditioning_prob=args.empty_cond_prob,
        )
        xs.append(x_t)
        ts.append(t)
        pend_all.append(pending)
        za_l.append(a)
        zs_l.append(s)
    if not xs:
        return None
    B = len(xs)
    T = max(len(x) for x in xs)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for b, x in enumerate(xs):
        ids[b, :len(x)] = torch.tensor(x, dtype=torch.long)
        attn[b, :len(x)] = 1
    return {
        "input_ids": ids, "attention_mask": attn,
        "t": torch.tensor(ts, dtype=torch.float32),
        "z_amp": torch.stack(za_l), "z_sup": torch.stack(zs_l),
        "pending": pend_all,
    }


def flow_loss(model, built: Dict, args, device: str,
              return_metrics: bool = False):
    out = model(
        input_ids=built["input_ids"].to(device),
        attention_mask=built["attention_mask"].to(device),
        z_amp=built["z_amp"].to(device),
        z_sup=built["z_sup"].to(device),
        t=built["t"].to(device),
    )
    lam, h = out["lambda"], out["hidden"]                # (B,T,3), (B,T,d)
    B = lam.shape[0]
    eps = 1e-8

    rate_sum = lam.sum(dim=(1, 2))                       # (B,)
    w = torch.tensor([w_weight(float(t), args.w_max) for t in built["t"]],
                     device=device)                       # (B,)

    # Flat pending indices (t_idx = Q target token, -1 for DEL)
    b_idx, p_idx, k_idx, t_idx = [], [], [], []
    for b, pend in enumerate(built["pending"]):
        for op in pend:
            b_idx.append(b)
            p_idx.append(op["pos"])
            k_idx.append(op["kind"])
            t_idx.append(-1 if op["tgt"] is None else int(op["tgt"]))

    ce = torch.zeros(B, device=device)
    metrics = {}
    n_pend = len(b_idx)
    if n_pend:
        bi = torch.tensor(b_idx, device=device)
        pi = torch.tensor(p_idx, device=device)
        ki = torch.tensor(k_idx, device=device)
        lam_sel = lam[bi, pi, ki]                        # (N,)
        log_lam = torch.log(lam_sel + eps)
        ce.index_add_(0, bi, log_lam)

        # Q terms, grouped by kind so lm_head runs once per kind
        for kind_val, kind_name in ((KIND_SUB, "sub"), (KIND_INS, "ins")):
            rows = [i for i in range(n_pend) if k_idx[i] == kind_val]
            if not rows:
                continue
            rb = torch.tensor([b_idx[i] for i in rows], device=device)
            rp = torch.tensor([p_idx[i] for i in rows], device=device)
            tgt = torch.tensor([t_idx[i] for i in rows], device=device)
            logits = model.q_logits(h[rb, rp], kind_name)     # (N, V)
            logq = torch.log_softmax(logits, dim=-1)
            logq_sel = logq.gather(1, tgt.unsqueeze(1)).squeeze(1)
            ce.index_add_(0, rb, logq_sel)
            if return_metrics:
                metrics[f"q_{kind_name}_top1"] = float(
                    (logits.argmax(dim=-1) == tgt).float().mean())

        if return_metrics:
            # pending-kind accuracy: argmax over the 3 rates at the site
            metrics["kind_acc"] = float(
                (lam[bi, pi].argmax(dim=-1) == ki).float().mean())

    loss_b = rate_sum - w * ce                           # (B,)
    loss = loss_b.mean()
    if return_metrics:
        mask = built["attention_mask"].to(device).float()
        metrics["lam_per_tok"] = float(
            (lam.sum(dim=-1) * mask).sum() / mask.sum().clamp_min(1))
        metrics["n_pending"] = float(n_pend) / B
        # pending-site IoU (count oracle): top-|pending_b| by total rate
        ious = []
        lam_tot = lam.sum(dim=-1)
        for b, pend in enumerate(built["pending"]):
            gold = {op["pos"] for op in pend}
            if not gold:
                continue
            L = int(built["attention_mask"][b].sum())
            order = torch.argsort(lam_tot[b, :L], descending=True)
            pred = set(order[:len(gold)].tolist())
            ious.append(len(pred & gold) / len(pred | gold))
        metrics["site_iou"] = float(np.mean(ious)) if ious else float("nan")
        return loss, metrics
    return loss


def main():
    args = parse_args()
    args._k_amp = parse_k_spec(args.k_amp)
    args._k_sup = parse_k_spec(args.k_sup)
    excluded = [x.strip() for x in args.exclude_families.split(",")
                if x.strip()]
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    pad_id = tokenizer.pad_token_id or 0
    args._mask_id = int(tokenizer.mask_token_id)
    args._ins_id = int(tokenizer.convert_tokens_to_ids("[INS]"))
    args._del_id = int(tokenizer.convert_tokens_to_ids("[DEL]"))

    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    print(f"[editflow] loading W_dec from {args.sae_repo}/{args.sae_path}")
    w_dec = load_sae_w_dec(args.sae_repo, args.sae_path)
    model = SAEEditFlow(
        args.llm2vec_dir, d_sae=d_sae, dtype=dtype,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, proj_a_rank=args.proj_a_rank,
        w_dec=w_dec, t_film=args.t_film, cond_mode=args.cond_mode,
    )
    if args.init_from_editor:
        model.init_from_editor(args.init_from_editor)
    if args.init_from_editflow:
        model.init_from_editflow(args.init_from_editflow)
    model = model.to(args.device)
    print(f"[editflow] t_film={args.t_film} cond_mode={args.cond_mode} "
          f"true_align={args.true_align}")

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    rate_names = ("lam_head.", "lam_film.")
    rate_params = [p for n, p in model.named_parameters()
                   if n.startswith(rate_names) and p.requires_grad]
    small_params = [p for n, p in model.named_parameters()
                    if "lora_" not in n and not n.startswith(rate_names)
                    and p.requires_grad]
    print(f"[editflow] trainable: small="
          f"{sum(p.numel() for p in small_params):,} "
          f"rate={sum(p.numel() for p in rate_params):,} "
          f"@ {args.rate_head_lr:g} "
          f"lora={sum(p.numel() for p in lora_params):,}")
    optim = torch.optim.AdamW([
        {"params": small_params, "lr": args.learning_rate},
        {"params": rate_params, "lr": args.rate_head_lr},
        {"params": lora_params, "lr": args.backbone_lr},
    ])
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps)

    dataset = CorruptionDataset(args.corruption_dir, shuffle=True,
                                seed=args.seed,
                                exclude_t_families=excluded)
    collator = EditFlowCollator(d_sae=d_sae)
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
        print(f"[editflow] dev monitoring: {len(dev_batches)} batches")

    # Fixed t grid for dev: reproducible, spans the schedule (skip t≈1).
    dev_t = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]

    def evaluate_dev() -> Dict[str, float]:
        dev_rng = np.random.default_rng(args.seed + 9999)
        rows = []
        with torch.no_grad():
            for batch in dev_batches:
                built = build_batch(batch, dev_rng, args, pad_id,
                                    fixed_t=dev_t)
                if built is None:
                    continue
                loss, m = flow_loss(model, built, args, args.device,
                                    return_metrics=True)
                m["loss"] = float(loss.item())
                # empty-conditioning rate (premise protection proxy)
                out_e = model(
                    input_ids=built["input_ids"].to(args.device),
                    attention_mask=built["attention_mask"].to(args.device),
                    z_amp=torch.zeros_like(built["z_amp"]).to(args.device),
                    z_sup=torch.zeros_like(built["z_sup"]).to(args.device),
                    t=built["t"].to(args.device),
                )
                mask = built["attention_mask"].to(args.device).float()
                m["lam_per_tok_empty"] = float(
                    (out_e["lambda"].sum(dim=-1) * mask).sum()
                    / mask.sum().clamp_min(1))
                rows.append(m)
        keys = set().union(*rows) if rows else set()
        return {k: float(np.mean([r[k] for r in rows
                                  if k in r and r[k] == r[k]]))
                for k in keys}

    best_path = out_dir / "editflow-best.pt"
    best_json = out_dir / "best.json"
    best_dev = float("inf")
    if best_json.exists():
        try:
            best_dev = float(json.loads(best_json.read_text())["dev_loss"])
            print(f"[editflow] RESUME: best dev loss {best_dev:.4f}")
        except (ValueError, KeyError):
            pass

    def maybe_update_best(at_step: int):
        nonlocal best_dev
        if not dev_batches:
            return
        m = evaluate_dev()
        marker = ""
        if m["loss"] < best_dev:
            best_dev = m["loss"]
            model.save(str(best_path))
            best_json.write_text(json.dumps(
                {"step": int(at_step), "dev_loss": float(m["loss"])}))
            marker = "  ** new best **"
        print(f"[editflow] step={at_step} DEV "
              f"loss={m['loss']:.4f} kind_acc={m.get('kind_acc', 0):.4f} "
              f"site_iou={m.get('site_iou', 0):.4f} "
              f"q_sub={m.get('q_sub_top1', 0):.4f} "
              f"q_ins={m.get('q_ins_top1', 0):.4f} "
              f"lam/tok={m.get('lam_per_tok', 0):.4f} "
              f"empty={m.get('lam_per_tok_empty', 0):.4f}"
              f" (best {best_dev:.4f}){marker}")

    step = 0
    if args.resume:
        latest = find_latest_ckpt(out_dir, "editflow")
        if latest is not None:
            ckpt_path, ckpt_step = latest
            print(f"[editflow] RESUME: {ckpt_path} (step {ckpt_step})")
            blob = torch.load(str(ckpt_path), map_location=args.device,
                              weights_only=False)
            model.load_trainable(blob["trainable"])
            restored = load_train_state(ckpt_path, optim, sched,
                                        device=args.device)
            step = restored if restored is not None else ckpt_step
            if step >= args.max_steps:
                print("[editflow] RESUME: already past max_steps")
                model.save(str(out_dir / "editflow-final.pt"))
                return

    loss_window = []
    pbar = tqdm(total=args.max_steps, initial=step, desc="[editflow]",
                unit="step", dynamic_ncols=True)
    for batch in loader:
        if step >= args.max_steps:
            break
        built = build_batch(batch, rng, args, pad_id)
        if built is None:
            continue
        loss = flow_loss(model, built, args, args.device)
        (loss / args.grad_accum_steps).backward()
        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()

        loss_window.append(float(loss.item()))
        if step % args.logging_steps == 0:
            avg = sum(loss_window[-args.logging_steps:]) / max(
                1, min(len(loss_window), args.logging_steps))
            cs = float(model.cond_scale.detach().float())
            print(f"[editflow] step={step} loss={avg:.4f} "
                  f"cond_scale={cs:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if step > 0 and step % args.save_steps == 0:
            ckpt = out_dir / f"editflow-step{step}.pt"
            model.save(str(ckpt))
            save_train_state(ckpt, optim, sched, step)
            print(f"[editflow] saved {ckpt}")
        if dev_batches and step > 0 and step % args.eval_steps == 0:
            maybe_update_best(step)
        step += 1
        pbar.update(1)
    pbar.close()

    maybe_update_best(step)
    if dev_batches and best_path.exists():
        model.save(str(out_dir / "editflow-last.pt"))
        import shutil
        shutil.copyfile(best_path, out_dir / "editflow-final.pt")
        print(f"[editflow] done at step {step}; editflow-final.pt = best dev "
              f"state ({json.loads(best_json.read_text())})")
    else:
        model.save(str(out_dir / "editflow-final.pt"))
        print(f"[editflow] done at step {step}")


if __name__ == "__main__":
    main()
