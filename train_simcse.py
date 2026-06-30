"""
Stage 1b: SimCSE (unsupervised contrastive) on top of the MNTP checkpoint.

LLM2Vec paper §4 trains in three steps: Bi (attention patch) → MNTP →
SimCSE. The MNTP step happens in `train_llm2vec.py`. This script does the
SimCSE step.

Recipe (Gao et al. 2021, applied per LLM2Vec §4.3):
  * Each batch sentence is forwarded through the bidirectional encoder
    TWICE. The two forwards use independent dropout masks, producing a
    POSITIVE pair (z_i, z_i+). All other sentences in the batch are
    negatives.
  * Loss = NT-Xent on the (B, B) similarity matrix with temperature τ.
  * Pooling is mean over non-pad positions (matches `eval_llm2vec.py`).

Notes vs. canonical SimCSE:
  - Canonical SimCSE on BERT relies on its native 10% attention/hidden
    dropout. Gemma-2's default `attention_dropout=0.0`, so the two forwards
    would be IDENTICAL and SimCSE collapses. We inject dropout > 0 (default
    0.1) into every attention module before training.
  - Canonical SimCSE uses τ = 0.05 and batch ≥ 64. Batch ≥ memory cap on
    Gemma-2B in bf16 is what bounds it — pick the largest the GPU fits.
  - LR for FULL FT is much lower than the LoRA recipe (1e-6 vs 3e-5). This
    mirrors the MNTP "blow-up at LR peak" lesson from `train_llm2vec.py`.

The output directory contains a NEW HF-format checkpoint plus
`llm2vec_meta.json` (so downstream consumers — `corruption.py`, `tagger`,
`editor`, eval — can keep using --llm2vec-dir unchanged).

Resume:
  * If `--output-dir/checkpoint-*` exists → load from latest checkpoint.
  * Else if `--llm2vec-dir` exists → load from MNTP and start fresh.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from data import download_dolma_shards, iter_dolma_texts, iter_sentences
from model import _patch_attention_bidirectional
from resume_utils import add_resume_args


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True,
                   help="MNTP-trained checkpoint (output of train_llm2vec.py). "
                        "Used as initial weights when no checkpoint-* exists "
                        "under --output-dir.")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the SimCSE-trained checkpoint. Becomes "
                        "the new --llm2vec-dir for all downstream stages.")
    p.add_argument("--data-cache-dir", default="./dolma_cache")
    p.add_argument("--max-files", type=int, default=None)

    # SimCSE in the original paper uses short sequences (seq_len ≤ 64) so
    # quadratic attention isn't the bottleneck. We allow longer because
    # Dolma sentences are real, not Wikipedia-cropped 32-token snippets.
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--sentence-splitter", choices=["pysbd", "nltk"], default="pysbd")
    p.add_argument("--sent-min-chars", type=int, default=16)
    p.add_argument("--sent-max-chars", type=int, default=512)
    p.add_argument("--no-quality-filter", action="store_true")
    p.add_argument("--pooling", choices=["mean", "weighted_mean", "last"],
                   default="mean",
                   help="How to pool last_hidden_state → sentence vector. "
                        "Should match the pooling used by eval_llm2vec / "
                        "downstream consumers. Default mean is the LLM2Vec "
                        "paper's choice.")

    # SimCSE NEEDS dropout > 0; Gemma-2 default is 0.0. 0.1 mirrors the
    # canonical BERT-base SimCSE recipe.
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Attention dropout to inject before training. >0 is "
                        "required for SimCSE's dropout-as-augmentation; "
                        "Gemma-2's default 0.0 would make the two forwards "
                        "identical and the loss would collapse to log(B).")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="NT-Xent temperature τ. 0.05 is Gao et al. (2021).")

    # Batch / steps. Larger per-device batch = more negatives = better SimCSE.
    # grad-accum does NOT help here (negatives come from the same forward pass,
    # not from accumulated micro-batches), so leave accum=1 unless you need
    # gradient stability.
    p.add_argument("--per-device-batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Note: grad-accum does NOT increase SimCSE in-batch "
                        "negatives — those come only from per_device_batch. "
                        "Set >1 only for gradient noise smoothing.")
    p.add_argument("--learning-rate", type=float, default=1e-6,
                   help="Very low because we do FULL FT (vs canonical LoRA + "
                        "3e-5). Mirrors MNTP's 5e-5 → 1e-5 lesson.")
    p.add_argument("--max-steps", type=int, default=2000,
                   help="SimCSE converges quickly; 1-2k optimizer steps is the "
                        "typical paper recipe.")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3,
                   help="Rolling cap on saved checkpoint-* dirs (Gemma-2B at "
                        "~5GB each).")

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade ~2x throughput for ~2x lower activation memory. "
                        "SimCSE does TWO forwards per step; enable this if you "
                        "OOM at the batch size you want.")
    p.add_argument("--seed", type=int, default=42)
    add_resume_args(p)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class DolmaSentenceStream(IterableDataset):
    """Stream Dolma → sentences → (input_ids, attention_mask) dicts.

    Mirrors `train_llm2vec.DolmaSentenceTokenStream` but yields what
    SimCSE wants (no special_tokens_mask, no MLM collator).
    """

    def __init__(
        self,
        shard_paths,
        tokenizer,
        max_seq_length: int,
        sentence_splitter: str,
        sent_min_chars: int = 16,
        sent_max_chars: int = 512,
        seed: int = 42,
        quality_filter: bool = True,
    ):
        self.shard_paths = list(shard_paths)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sentence_splitter = sentence_splitter
        self.sent_min_chars = sent_min_chars
        self.sent_max_chars = sent_max_chars
        self.seed = seed
        self.quality_filter = quality_filter

    def __iter__(self) -> Iterator[Dict]:
        worker = get_worker_info()
        shards = self.shard_paths
        worker_seed = self.seed
        if worker is not None:
            shards = shards[worker.id::worker.num_workers]
            worker_seed = self.seed + 1000 * worker.id
        text_iter = iter_dolma_texts(shards, min_chars=64)
        sent_iter = iter_sentences(
            text_iter,
            splitter=self.sentence_splitter,
            min_chars=self.sent_min_chars,
            max_chars=self.sent_max_chars,
            max_sentences_per_text=None,
            sample_strategy="random",
            seed=worker_seed,
            quality_filter=self.quality_filter,
        )
        for sent in sent_iter:
            enc = self.tokenizer(
                sent,
                truncation=True,
                max_length=self.max_seq_length,
                add_special_tokens=True,
            )
            if len(enc["input_ids"]) < 4:
                continue
            yield {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }


class SimCSECollator:
    """Right-pad to longest in batch (capped by max_seq_length).

    No two-copy duplication here: the two stochastic forwards are done in
    the training loop, not at collate time. That way each forward
    independently samples dropout masks — they would be correlated if we
    concatenated the inputs into one (2B, T) tensor.
    """

    def __init__(self, pad_token_id: int, max_seq_length: int):
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(b["input_ids"]) for b in batch), self.max_seq_length)
        B = len(batch)
        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            L = min(len(b["input_ids"]), max_len)
            input_ids[i, :L] = torch.tensor(b["input_ids"][:L], dtype=torch.long)
            attention_mask[i, :L] = torch.tensor(b["attention_mask"][:L], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# --------------------------------------------------------------------------- #
# Pooling + losses
# --------------------------------------------------------------------------- #
def _pool(h: torch.Tensor, mask: torch.Tensor, strategy: str) -> torch.Tensor:
    """Pool (B, T, d) → (B, d) using attention mask. Matches eval_llm2vec."""
    m = mask.unsqueeze(-1).to(h.dtype)                       # (B, T, 1)
    if strategy == "mean":
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-9)
    if strategy == "last":
        lengths = mask.sum(dim=1).long()
        idx = (lengths - 1).clamp_min(0)
        return h[torch.arange(h.shape[0], device=h.device), idx]
    if strategy == "weighted_mean":
        T = h.shape[1]
        w = torch.arange(1, T + 1, device=h.device, dtype=h.dtype)
        w = w.unsqueeze(0) * mask.to(h.dtype)                # (B, T)
        return (h * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1).clamp_min(1e-9).unsqueeze(-1)
    raise ValueError(f"unknown pooling: {strategy!r}")


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SimCSE / InfoNCE. Symmetric (both directions averaged).

    z1, z2 : (B, d) L2-normalized.
    Returns (loss, diagnostics).
    """
    sim = z1 @ z2.t() / temperature                          # (B, B)
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

    with torch.no_grad():
        # Top-1 accuracy on the diagonal — a sanity check on the loss.
        acc = (sim.argmax(dim=-1) == labels).float().mean().item()
        # Alignment (lower better) and uniformity (lower better — more
        # spread out on the unit hypersphere). Wang & Isola 2020 metrics.
        alignment = ((z1 - z2).norm(dim=-1).pow(2)).mean().item()
        # Pairwise distances of z1 vs z1 (excluding diagonal)
        d = torch.cdist(z1, z1, p=2).pow(2)                  # (B, B)
        B = d.shape[0]
        off = d.flatten()[1:].view(B - 1, B + 1)[:, :-1].flatten()
        uniformity = (-2.0 * off).exp().mean().log().item()

    return loss, {
        "acc": acc, "alignment": alignment, "uniformity": uniformity,
    }


# --------------------------------------------------------------------------- #
# Resume helpers (HF-format checkpoint dirs)
# --------------------------------------------------------------------------- #
_CKPT_DIR_RE = re.compile(r"^checkpoint-(?P<step>\d+)$")


def _find_latest_ckpt_dir(out_dir: Path) -> Optional[Tuple[Path, int]]:
    """Find the highest-step `checkpoint-{N}/` sub-directory."""
    best: Optional[Tuple[Path, int]] = None
    if not out_dir.exists():
        return None
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        m = _CKPT_DIR_RE.match(p.name)
        if not m:
            continue
        n = int(m.group("step"))
        if best is None or n > best[1]:
            best = (p, n)
    return best


def _prune_old_ckpts(out_dir: Path, keep: int) -> None:
    """Keep only the `keep` highest-step `checkpoint-*` dirs."""
    if keep <= 0:
        return
    pairs: List[Tuple[Path, int]] = []
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        m = _CKPT_DIR_RE.match(p.name)
        if not m:
            continue
        pairs.append((p, int(m.group("step"))))
    pairs.sort(key=lambda x: -x[1])
    for path, _ in pairs[keep:]:
        # Remove sidecar state.pt first (it's INSIDE the dir for us).
        try:
            import shutil
            shutil.rmtree(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer (same as MNTP — INS/DEL/MASK already added) ------------
    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Decide load source: latest checkpoint-* or MNTP dir --------------
    init_step = 0
    init_source = args.llm2vec_dir
    latest = _find_latest_ckpt_dir(out_dir) if args.resume else None
    if latest is not None:
        init_source, init_step = str(latest[0]), latest[1]
        print(f"[simcse] RESUME: continuing from {init_source} (step {init_step})")
    else:
        print(f"[simcse] starting from MNTP ckpt: {init_source}")

    # ---- Load model (AutoModelForCausalLM so the LM head — needed by ------
    # corruption.py for PPL — survives) -------------------------------------
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.llm_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        init_source,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    # Re-apply the bidirectional patch — it's not persisted as part of the
    # HF save (it's a monkey-patch on instance methods + flags).
    _patch_attention_bidirectional(model.model)
    model.config.use_cache = False

    # ---- Inject dropout into every attention module -----------------------
    # Gemma-2 stores attention_dropout as a float on each attention module
    # (used by the SDPA path: dropout_p=self.attention_dropout if training).
    # The default is 0.0 → identical forwards → SimCSE collapses to log(B).
    n_dropout_patched = 0
    for module in model.modules():
        if hasattr(module, "attention_dropout") and isinstance(
            getattr(module, "attention_dropout"), float,
        ):
            module.attention_dropout = float(args.dropout)
            n_dropout_patched += 1
    if hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = float(args.dropout)
    print(f"[simcse] dropout = {args.dropout}  ({n_dropout_patched} attention modules patched)")
    if n_dropout_patched == 0:
        print(
            "[simcse] WARNING: 0 attention modules had `attention_dropout`; "
            "the two SimCSE forwards may be identical and loss will collapse "
            "to log(B). Check the backbone class."
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(args.device)
    model.train()

    # ---- Optim / sched ----------------------------------------------------
    no_decay = ["bias", "LayerNorm.weight", "RMSNorm.weight", "norm.weight"]
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if any(nd in n for nd in no_decay) else decay_params).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    sched = get_linear_schedule_with_warmup(
        optim, args.warmup_steps, args.max_steps,
    )

    # ---- Restore optim/sched/RNG if resuming ------------------------------
    # We DON'T use resume_utils.load_train_state here: that helper expects a
    # `<ckpt>.state.pt` sidecar next to a `<ckpt>.pt` file, but SimCSE
    # checkpoints are HF-format DIRECTORIES (`checkpoint-N/`), and we put
    # the optim/sched/RNG blob INSIDE the dir as `state.pt`. Read it
    # directly here so the dir is self-contained.
    if init_step > 0:
        state_path = Path(init_source) / "state.pt"
        if state_path.exists():
            blob = torch.load(state_path, map_location=args.device, weights_only=False)
            try:
                optim.load_state_dict(blob["optim"])
            except (KeyError, ValueError):
                pass
            if blob.get("sched") is not None:
                try:
                    sched.load_state_dict(blob["sched"])
                except (KeyError, ValueError):
                    pass
            for k_torch in ("torch_rng_state",):
                if k_torch in blob:
                    try:
                        torch.set_rng_state(blob[k_torch])
                    except (RuntimeError, TypeError):
                        pass
            if "numpy_rng_state" in blob:
                try:
                    np.random.set_state(blob["numpy_rng_state"])
                except (TypeError, ValueError):
                    pass
            if torch.cuda.is_available() and "cuda_rng_state_all" in blob:
                try:
                    torch.cuda.set_rng_state_all(blob["cuda_rng_state_all"])
                except (RuntimeError, TypeError):
                    pass
            print(f"[simcse] restored optim/sched/RNG from {state_path}")
        else:
            print(f"[simcse] NOTE: {state_path} missing — using fresh optim/sched")
        if init_step >= args.max_steps:
            print(f"[simcse] already at max_steps ({init_step} >= {args.max_steps}); "
                  f"writing final checkpoint and exiting.")
            _write_final(out_dir, model, tokenizer, args, final_step=init_step)
            return

    # ---- Data -------------------------------------------------------------
    shard_paths = download_dolma_shards(args.data_cache_dir, max_files=args.max_files)
    dataset = DolmaSentenceStream(
        shard_paths, tokenizer,
        max_seq_length=args.max_seq_length,
        sentence_splitter=args.sentence_splitter,
        sent_min_chars=args.sent_min_chars,
        sent_max_chars=args.sent_max_chars,
        seed=args.seed + init_step,        # different stream after resume
        quality_filter=not args.no_quality_filter,
    )
    collator = SimCSECollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_length=args.max_seq_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,                    # NT-Xent over a partial batch is awkward
    )

    # ---- Training loop ----------------------------------------------------
    step = init_step
    accum_count = 0
    loss_window: List[float] = []
    acc_window: List[float] = []
    align_window: List[float] = []
    unif_window: List[float] = []
    pbar = tqdm(
        total=args.max_steps, initial=step,
        desc="[simcse]", unit="step", dynamic_ncols=True, smoothing=0.05,
    )
    t0 = time.time()
    optim.zero_grad()
    for batch in loader:
        if step >= args.max_steps:
            break

        input_ids = batch["input_ids"].to(args.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)

        # Two STOCHASTIC forwards (independent dropout masks). Each goes
        # through model.model (the inner backbone) — we don't need the LM
        # head here, but it's still attached so save_pretrained writes a
        # GemmaForCausalLM compatible checkpoint.
        out1 = model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        )
        z1 = _pool(out1.last_hidden_state, attention_mask, args.pooling)
        out2 = model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        )
        z2 = _pool(out2.last_hidden_state, attention_mask, args.pooling)

        z1 = F.normalize(z1.float(), dim=-1)
        z2 = F.normalize(z2.float(), dim=-1)
        loss, diag = nt_xent_loss(z1, z2, args.temperature)
        (loss / args.grad_accum_steps).backward()
        accum_count += 1

        loss_window.append(float(loss.item()))
        acc_window.append(diag["acc"])
        align_window.append(diag["alignment"])
        unif_window.append(diag["uniformity"])

        if accum_count >= args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad()
            accum_count = 0
            step += 1
            pbar.update(1)

            if step % args.logging_steps == 0:
                w = args.logging_steps
                avg_loss = sum(loss_window[-w:]) / max(1, min(len(loss_window), w))
                avg_acc = sum(acc_window[-w:]) / max(1, min(len(acc_window), w))
                avg_align = sum(align_window[-w:]) / max(1, min(len(align_window), w))
                avg_unif = sum(unif_window[-w:]) / max(1, min(len(unif_window), w))
                rate = step / max(1e-6, time.time() - t0)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.3f}",
                    "align": f"{avg_align:.3f}",
                    "unif": f"{avg_unif:.2f}",
                })
                print(
                    f"[simcse] step={step} loss={avg_loss:.4f} "
                    f"acc={avg_acc:.3f} alignment={avg_align:.4f} "
                    f"uniformity={avg_unif:.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e} rate={rate:.2f} step/s"
                )

            if step > 0 and step % args.save_steps == 0:
                _save_checkpoint(out_dir, step, model, tokenizer, optim, sched, args)
                _prune_old_ckpts(out_dir, keep=args.save_total_limit)

    pbar.close()
    _write_final(out_dir, model, tokenizer, args, final_step=step)
    print(f"[simcse] done at step {step}  out={out_dir}")


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def _save_checkpoint(out_dir, step, model, tokenizer, optim, sched, args):
    ckpt_dir = out_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # safe_serialization=False because Gemma ties lm_head.weight to
    # embed_tokens.weight (same convention as train_llm2vec).
    model.save_pretrained(ckpt_dir, safe_serialization=False)
    tokenizer.save_pretrained(ckpt_dir)
    state = {
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "step":  int(step),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, ckpt_dir / "state.pt")
    print(f"[simcse] saved checkpoint {ckpt_dir}")


def _write_final(out_dir, model, tokenizer, args, final_step: int):
    """Write the terminal HF-format checkpoint + llm2vec_meta.json.

    Downstream consumers (corruption, tagger, editor, eval) point at
    --llm2vec-dir and look for `llm2vec_meta.json` to detect a finished
    run. We mirror the metadata that train_llm2vec writes, plus a SimCSE
    block, so the same dir is a drop-in replacement.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=False)
    tokenizer.save_pretrained(out_dir)

    # Inherit the MNTP meta and amend it with the SimCSE specifics.
    src_meta = Path(args.llm2vec_dir) / "llm2vec_meta.json"
    meta = {}
    if src_meta.exists():
        try:
            meta = json.loads(src_meta.read_text())
        except json.JSONDecodeError:
            meta = {}
    meta["mntp_source_dir"] = str(args.llm2vec_dir)
    meta["simcse"] = {
        "temperature": float(args.temperature),
        "dropout": float(args.dropout),
        "pooling": args.pooling,
        "max_seq_length": int(args.max_seq_length),
        "learning_rate": float(args.learning_rate),
        "max_steps": int(args.max_steps),
        "final_step": int(final_step),
        "per_device_batch_size": int(args.per_device_batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "weight_decay": float(args.weight_decay),
        "loss": "nt_xent (symmetric, in-batch negatives)",
        "augmentation": "dropout-as-augmentation (two stochastic forwards)",
    }
    (out_dir / "llm2vec_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
