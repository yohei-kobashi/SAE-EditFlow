"""
Train the length predictor head (ablation).

For each INS corruption sample we know the gold span length L. We use the
editor's frozen encoder + Proj_A to obtain the hidden state at the first
[INS] position of the gap, then train a small head to classify L ∈ {1..L_MAX}.

Backbone, LM head, Proj_A, and type_emb are all frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from data import CorruptionCollator, CorruptionDataset
from editor import load_editor_from_checkpoint
from intervene import diff_to_sparse
from length_head import LengthHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--editor-ckpt", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--l-max", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=1000)

    p.add_argument("--k-top", type=int, default=8)
    p.add_argument("--empty-cond-prob", type=float, default=0.15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def first_ins_position(input_ids_row: np.ndarray, ins_id: int) -> int:
    """Index of the first [INS] in the row, or -1 if none."""
    arr = np.where(input_ids_row == ins_id)[0]
    return int(arr[0]) if len(arr) else -1


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    ins_id = tokenizer.convert_tokens_to_ids("[INS]")
    meta = json.loads((Path(args.corruption_dir) / "meta.json").read_text())
    d_sae = int(meta["d_sae"])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    editor = load_editor_from_checkpoint(
        args.llm2vec_dir, args.editor_ckpt, d_sae, dtype=dtype,
    ).to(args.device).eval()
    for p in editor.parameters():
        p.requires_grad_(False)

    head = LengthHead(d_model=editor.d_model, l_max=args.l_max).to(args.device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    sched = get_linear_schedule_with_warmup(
        optim, args.warmup_steps, args.max_steps,
    )

    ds = CorruptionDataset(args.corruption_dir, shuffle=True, seed=args.seed)
    coll = CorruptionCollator(d_sae=d_sae, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=coll,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    loss_window = []
    empty_batches = 0
    # If the corruption cache has zero (or near-zero) INS yield, every batch
    # we pull will fail the `gold_length > 0` filter and the loop never
    # increments `step`. Cap consecutive empties so this case fails loudly
    # within seconds instead of stalling for the whole job walltime.
    max_empty_batches = max(200, args.batch_size * 50)
    for batch in loader:
        if step >= args.max_steps:
            break

        # Filter to INS samples only
        ed_in = batch["editor_input_ids"]
        gold_length = batch["ins_span_length"]
        keep = gold_length > 0
        if not keep.any():
            empty_batches += 1
            if empty_batches >= max_empty_batches:
                print(
                    f"[length] no INS samples in {empty_batches} consecutive "
                    f"batches — the corruption cache appears to have zero INS "
                    f"yield. Check meta.json['bucket_yields']['ins']. "
                    f"Stopping early."
                )
                break
            continue
        empty_batches = 0
        idx = keep.nonzero(as_tuple=True)[0]

        z_X = batch["z_X"][idx]
        z_X_prime = batch["z_X_prime"][idx]
        B = z_X.shape[0]
        z_amp = torch.zeros_like(z_X)
        z_sup = torch.zeros_like(z_X)
        for b in range(B):
            k_amp = int(rng.integers(0, 4))
            k_sup = int(rng.integers(0, 4))
            a, s = diff_to_sparse(
                z_X[b], z_X_prime[b],
                k_top=args.k_top, k_amp=k_amp, k_sup=k_sup,
                rng=rng, empty_conditioning_prob=args.empty_cond_prob,
            )
            z_amp[b] = a
            z_sup[b] = s

        with torch.no_grad():
            out = editor(
                input_ids=ed_in[idx].to(args.device),
                attention_mask=batch["editor_attention_mask"][idx].to(args.device),
                z_amp=z_amp.to(args.device),
                z_sup=z_sup.to(args.device),
            )
            hidden = out["hidden_states"]                    # (B, T, d_model)

        # Pick the first [INS] position per row
        ins_positions = []
        for b in range(B):
            row = ed_in[idx][b].numpy()
            pos = first_ins_position(row, ins_id)
            if pos < 0:
                pos = 0
            ins_positions.append(pos)
        ins_positions = torch.tensor(ins_positions, device=args.device, dtype=torch.long)
        gathered = hidden[torch.arange(B, device=args.device), ins_positions]

        gold = (gold_length[idx] - 1).clamp(min=0, max=args.l_max - 1)
        gold = gold.to(args.device)

        out = head(gathered, gold_length=gold)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()

        loss_window.append(float(loss.item()))
        if step % args.logging_steps == 0:
            avg = sum(loss_window[-args.logging_steps:]) / max(1, min(len(loss_window), args.logging_steps))
            print(f"[length] step={step} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")

        if step > 0 and step % args.save_steps == 0:
            torch.save(head.state_dict(), out_dir / f"length-step{step}.pt")

        step += 1

    torch.save(head.state_dict(), out_dir / "length-final.pt")
    print(f"[length] done at step {step}")


if __name__ == "__main__":
    main()
