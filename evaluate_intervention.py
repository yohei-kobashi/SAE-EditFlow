"""
End-to-end SAE-LEWIS inference and a thin metrics layer.

  edit(text, spec, strength):
    1. tokenize
    2. tagger.predict_ops
    3. apply_ops_for_editor → build base editor input
    4. enumerate INS slot counts in {1..L_MAX}^G
    5. editor forward per template → argmax decode
    6. ranker → top-1

Usage:
    python evaluate_intervention.py \
        --llm2vec-dir <path> \
        --tagger-ckpt <path> \
        --editor-ckpt <path> \
        --mu <stage0 mu.npy> \
        --text "..." --spec "+1234" "-5678"
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from editor import load_editor_from_checkpoint
from intervene import FeatureSpec, build_intervention_vectors
from lewis_ops import (
    OP_INS,
    EditorInputs,
    apply_ops_for_editor,
    decode_with_op_mask,
    expand_ins_gap,
)
from model import BidirectionalLLM, SAEFeatureExtractor, load_causal_gemma
from ranker import Ranker, RankerWeights
from tagger import load_tagger_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--tagger-ckpt", required=True)
    p.add_argument("--editor-ckpt", required=True)
    p.add_argument("--mu", required=True, help="Path to mu.npy from precompute_sae.py.")

    p.add_argument("--llm", default="google/gemma-2-2b",
                   help="Base Gemma used by SAE extractor / ranker fluency.")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", default="layer_12/width_16k/average_l0_71/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"], default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--text", required=True)
    p.add_argument("--spec", nargs="+", required=True,
                   help="Intervention spec entries: '+1234' (amp) or '-5678' (sup).")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--l-max", type=int, default=5)

    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


def find_ins_gaps(op_per_pos: np.ndarray) -> List[Tuple[int, int]]:
    """Identify (start, length) of consecutive OP_INS runs."""
    gaps: List[Tuple[int, int]] = []
    n = len(op_per_pos)
    i = 0
    while i < n:
        if int(op_per_pos[i]) == OP_INS:
            j = i
            while j < n and int(op_per_pos[j]) == OP_INS:
                j += 1
            gaps.append((i, j - i))
            i = j
        else:
            i += 1
    return gaps


@torch.no_grad()
def edit_once(
    *,
    text: str,
    spec: List[FeatureSpec],
    strength: float,
    mu: np.ndarray,
    tagger,
    editor,
    ranker: Ranker,
    tokenizer,
    l_max: int,
    device: str,
) -> str:
    d_sae = int(mu.shape[0])
    z_amp_full, z_sup_full = build_intervention_vectors(spec, mu, strength)

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = enc.input_ids                         # (1, T)
    attn = enc.attention_mask
    z_amp_dev = z_amp_full.unsqueeze(0).to(device)
    z_sup_dev = z_sup_full.unsqueeze(0).to(device)

    # 1. Tagger
    ops = tagger.predict_ops(input_ids, attn, z_amp_dev, z_sup_dev)[0].cpu().tolist()
    token_ids = input_ids[0].cpu().tolist()

    # 2. apply_ops_for_editor
    ed_in = apply_ops_for_editor(
        token_ids, ops,
        mask_token_id=tokenizer.mask_token_id,
        ins_token_id=tokenizer.convert_tokens_to_ids("[INS]"),
    )
    del_id = tokenizer.convert_tokens_to_ids("[DEL]")

    # 3. enumerate INS slot counts
    G = len(ed_in.ins_gaps)
    if G == 0:
        slot_choices = [tuple()]
    else:
        slot_choices = list(itertools.product(range(1, l_max + 1), repeat=G))

    candidates: List[List[int]] = []
    ins_counts: List[int] = []
    for choice in slot_choices:
        cur = ed_in
        for gi, k in enumerate(choice):
            cur = expand_ins_gap(cur, gi, k, tokenizer.convert_tokens_to_ids("[INS]"))
        # Editor forward
        ids = torch.tensor([cur.input_ids.tolist()], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        out = editor(
            input_ids=ids, attention_mask=mask,
            z_amp=z_amp_dev, z_sup=z_sup_dev,
        )
        argmax = out["logits"][0].argmax(dim=-1).cpu().numpy()
        decoded = decode_with_op_mask(
            cur.input_ids, argmax, cur.op_per_pos, del_token_id=del_id,
        )
        candidates.append(decoded)
        ins_counts.append(sum(choice) if choice else 0)

    # 4. rank
    scores = ranker.rank(
        candidates=candidates, input_ids=token_ids,
        z_amp=z_amp_full.to(device), z_sup=z_sup_full.to(device),
        num_ins_slots_per_cand=ins_counts,
    )
    best = int(np.argmax(scores))
    return tokenizer.decode(candidates[best], skip_special_tokens=True)


def main():
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    mu = np.load(args.mu)
    d_sae = int(mu.shape[0])

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)

    print("[eval] loading tagger / editor")
    tagger = load_tagger_from_checkpoint(args.llm2vec_dir, args.tagger_ckpt, d_sae, dtype=dtype)
    tagger = tagger.to(args.device).eval()
    editor = load_editor_from_checkpoint(args.llm2vec_dir, args.editor_ckpt, d_sae, dtype=dtype)
    editor = editor.to(args.device).eval()

    print("[eval] loading ranker components (SAE / causal / bidirectional)")
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    )
    causal, _ = load_causal_gemma(args.llm2vec_dir)
    bid = BidirectionalLLM(args.llm2vec_dir, dtype=dtype)
    ranker = Ranker(extractor, causal, bid, RankerWeights(), device=args.device)

    specs = [FeatureSpec.parse(s) for s in args.spec]
    out = edit_once(
        text=args.text, spec=specs, strength=args.strength, mu=mu,
        tagger=tagger, editor=editor, ranker=ranker, tokenizer=tokenizer,
        l_max=args.l_max, device=args.device,
    )
    print("== edited ==")
    print(out)


if __name__ == "__main__":
    main()
