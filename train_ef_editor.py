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
from intervener import EFIntervener
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

    # EF regime.
    p.add_argument("--t0-prob", type=float, default=0.5,
                   help="prob. that x_t = x0 (full remaining edit); else "
                        "t ~ U(0,1) partial application per opcode.")

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
            a, s = diff_to_sparse(
                batch["z_X"][b], batch["z_X_prime"][b], k_top=args.k_top,
                k_amp=draw_k(rng, args._k_amp),
                k_sup=draw_k(rng, args._k_sup), rng=rng,
                empty_conditioning_prob=0.0)

        t = 0.0 if rng.random() < args.t0_prob else float(rng.random())
        xt = sample_intermediate(x0, x1, t, rng)

        # bare frame: [BOS]+x_t  \n  x1-body(no BOS)
        x1_body = x1[1:] if (x1 and x1[0] == ctx["bos_id"]) else list(x1)
        lm_prefix = list(xt) + [ctx["nl_id"]]
        if len(lm_prefix) + len(x1_body) > args.max_lm_len:
            continue
        rows.append({
            "xt": xt, "prefix": lm_prefix, "resp": x1_body,
            "null": null_kind, "za": a, "zs": s, "t": t,
        })
    if not rows:
        return None

    B = len(rows)
    Te = max(len(r["xt"]) for r in rows)
    enc_ids = torch.full((B, Te), ctx["pad_id"], dtype=torch.long)
    enc_mask = torch.zeros((B, Te), dtype=torch.long)
    Tl = max(len(r["prefix"]) + len(r["resp"]) for r in rows)
    lm_ids = torch.full((B, Tl), ctx["it_pad"], dtype=torch.long)
    lm_mask = torch.zeros((B, Tl), dtype=torch.long)
    labels = torch.full((B, Tl), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        e = r["xt"]
        enc_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        enc_mask[i, :len(e)] = 1
        full = r["prefix"] + r["resp"]
        lm_ids[i, :len(full)] = torch.tensor(full, dtype=torch.long)
        lm_mask[i, :len(full)] = 1
        if r["null"] is None:                      # null rows: no NLL
            rf = len(r["prefix"])
            labels[i, rf:len(full)] = torch.tensor(r["resp"],
                                                   dtype=torch.long)
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
        "z_amp": za, "z_sup": zs, "budget": budget,
        "null_mask": torch.tensor([r["null"] is not None for r in rows]),
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
        n = r["xt_len"]
        add[i, :n] = delta[i, :n]                  # identity position map

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
    nll = (ce_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    budget = built["budget"].to(device)            # (B,)
    null = built["null_mask"].to(device)
    true_m = (~null).float()
    has_nll = (valid.sum(dim=1) > 0).float()
    eps = 1e-6

    pos_norm = delta.norm(dim=-1)                  # (B, Te)
    enc_m = built["enc_mask"].to(device).float()
    over = F.relu(pos_norm - budget.unsqueeze(1)).pow(2) \
        / (budget.unsqueeze(1).pow(2) + eps)
    over = (over * enc_m).sum(dim=1) / enc_m.sum(dim=1).clamp_min(1.0)
    norm_pen = (over * true_m).sum() / true_m.sum().clamp_min(1.0)

    ref = ((budget.pow(2) * true_m).sum()
           / true_m.sum().clamp_min(1.0)).detach().clamp_min(1.0)
    null_sq = (delta.pow(2).sum(dim=-1) * enc_m).sum(dim=1) \
        / enc_m.sum(dim=1).clamp_min(1.0)
    null_pen = ((null_sq / ref) * null.float()).sum() \
        / null.float().sum().clamp_min(1.0)

    nll_true = (nll * true_m * has_nll).sum() \
        / (true_m * has_nll).sum().clamp_min(1.0)
    loss = (nll_true + args.norm_reg_w * norm_pen
            + args.null_norm_w * null_pen)
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
        "d_sae": d_sae,
        "pad_id": tok.pad_token_id or 0,
        "it_pad": it_tok.pad_token_id or it_tok.eos_token_id,
        "bos_id": int(tok.bos_token_id),
        "nl_id": int(nl_ids[0]),
        "w_dec": w_dec.float(),
    }

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
