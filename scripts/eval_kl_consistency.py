"""Metric (1) — counterfactual-consistency (EF_LM_LOSS_PLAN §0b, user
decision 2026-07-19): does the intervened state condition generation AS IF
the edited sentence had been the input?

Teacher-forced, generation-free, judge-free. Per pair, three contexts in
the repeat frame:

  B (reference):    prompt(src = x1), response = x1   <- the frozen LM's
                    own definition of "x1 was the input"
  C (null):         prompt(src = x0), response = x1, no hook
  A (intervened):   prompt(src = x0) + delta,  response = x1

At every response position: KL( p_A || p_B ) vs KL( p_C || p_B ), and
NLL_A(x1) vs NLL_C(x1) vs NLL_B(x1). Success = A's distributions move
from C toward B:  KL_reduction = 1 - KL(A||B)/KL(C||B)  (1 = perfect
counterfactual substitution, 0 = no effect).

Arms: ef (editor ckpt), steer (alpha*dvec at every position). Conditions
true/empty/random. ~6 forwards/pair -> minutes for 499 pairs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

from editflow_ops import align_pair, slot_ops                  # noqa: E402
from eval_lingualens import (                                  # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk,
    randomize_intervention, sae_z_with_offsets,
)
from intervener import (EFIntervener, REPEAT_PROMPT,           # noqa: E402
                        chat_prompt_ids, find_subseq)
from model import SAEFeatureExtractor, load_sae                # noqa: E402
from scripts.eval_clamp_baseline import SaeClampHook           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--ef-ckpt", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", required=True)
    p.add_argument("--sae-layer", type=int, required=True)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--blocklist", default="")
    p.add_argument("--conditions", default="true,empty,random")
    p.add_argument("--arms", default="steer")
    p.add_argument("--steer-alpha", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


class SpanHook:
    """Teacher-forcing injection: mode 'ef' adds delta rows at
    [off, off+n) of the full sequence; 'steer' adds alpha*dvec at every
    position."""

    def __init__(self):
        self.mode = None
        self.delta = None
        self.off = 0
        self.dvec = None
        self.alpha = 0.5

    def __call__(self, module, inputs, output):
        if self.mode is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        if self.mode == "ef" and self.delta is not None:
            n = min(self.delta.shape[0], h.shape[1] - self.off)
            if n > 0:
                add = torch.zeros_like(h)
                add[:, self.off:self.off + n] = self.delta[:n].to(h.dtype)
                h = h + add
        elif self.mode == "steer" and self.dvec is not None:
            h = h + (self.alpha * self.dvec).to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


def main():
    args = parse_args()
    conditions = [c for c in args.conditions.split(",") if c]
    arms = [a for a in args.arms.split(",") if a]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    import random
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen = order[:min(args.sample_size, len(order))]

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer,
    ).to(args.device).eval()
    blk = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        blk = torch.as_tensor(np.asarray(_bl, dtype=np.int64))

    it_tok = AutoTokenizer.from_pretrained(args.it_model)
    it_model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype).to(args.device).eval()
    sae = load_sae("jumprelu", args.sae_repo, args.sae_path
                   ).to(args.device).eval()
    W = sae.W_dec.float()

    interv = None
    if "ef" in arms:
        blob = torch.load(args.ef_ckpt, map_location="cpu",
                          weights_only=False)
        cfg = blob["config"]
        assert int(cfg["inject_layer"]) == args.sae_layer
        interv = EFIntervener(args.llm2vec_dir, int(cfg["d_sae"]),
                              dtype=dtype, lora_r=int(cfg.get("lora_r", 32)),
                              w_dec=sae.W_dec.detach().float().cpu(),
                              ).to(args.device).eval()
        interv.load_trainable_state_dict(blob["trainable"])

    hook = SpanHook()
    it_model.model.layers[args.sae_layer].register_forward_hook(hook)
    clamp_hook = SaeClampHook(sae)
    it_model.model.layers[args.sae_layer].register_forward_hook(clamp_hook)
    print(f"[klc] layer {args.sae_layer}, arms {arms}, "
          f"{len(chosen)} pairs (teacher-forced, no generation)")

    def frame_ids(sentence_in: str, resp_ids: list):
        pids = chat_prompt_ids(it_tok,
                               REPEAT_PROMPT.format(src=sentence_in))
        return pids, pids + resp_ids

    @torch.no_grad()
    def resp_logprobs(full_ids, resp_from):
        ids = torch.tensor([full_ids], device=args.device)
        logits = it_model(input_ids=ids).logits[0].float()
        # position t predicts token t+1; response tokens occupy
        # [resp_from, len) -> predictive rows [resp_from-1, len-1)
        rows = logits[resp_from - 1:len(full_ids) - 1]
        return F.log_softmax(rows, dim=-1)          # (Tr, V)

    records = []
    agg = defaultdict(list)
    for step_i, k in enumerate(chosen):
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids
        if len(slot_ops(align_pair(src_ids, tgt_ids))) == 0:
            continue
        prng = np.random.default_rng(args.seed * 1000003 + int(k))
        resp = it_tok(tgt, add_special_tokens=False).input_ids

        with torch.no_grad():
            s_off, z_s = sae_z_with_offsets(extractor, src, args.device)
            t_off, z_t = sae_z_with_offsets(extractor, tgt, args.device)
            om_s = [tuple(o) for o in tokenizer(
                src, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            om_t = [tuple(o) for o in tokenizer(
                tgt, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            opcodes = difflib.SequenceMatcher(
                None, src_ids, tgt_ids, autojunk=False).get_opcodes()
            sr, tr = edit_char_ranges(opcodes, om_s, om_t)
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
        za_t, zs_t = diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)
        zvar = {"true": (za_t, zs_t),
                "empty": (torch.zeros_like(za_t), torch.zeros_like(zs_t)),
                "random": (randomize_intervention(za_t, prng),
                           randomize_intervention(zs_t, prng))}

        # reference B and null C
        hook.mode = None
        pB, fB = frame_ids(tgt, resp)
        lpB = resp_logprobs(fB, len(pB))
        pC, fC = frame_ids(src, resp)
        lpC = resp_logprobs(fC, len(pC))
        pB_p = lpB.exp()
        kl_CB = float((pB_p * (lpB - lpC)).sum(-1).mean())
        nllB = float(-lpB.gather(-1, torch.tensor(
            resp, device=args.device).view(-1, 1)).mean())
        nllC = float(-lpC.gather(-1, torch.tensor(
            resp, device=args.device).view(-1, 1)).mean())

        rec = {"idx": int(k), "kl_CB": kl_CB, "nllB": nllB, "nllC": nllC,
               "arms": {}}
        for c in conditions:
            za, zs = zvar[c]
            for arm in arms:
                if arm == "ef":
                    ei = torch.tensor([src_ids], device=args.device)
                    with torch.no_grad():
                        io = interv(ei, torch.ones_like(ei),
                                    za.unsqueeze(0).to(args.device),
                                    zs.unsqueeze(0).to(args.device))
                    needle = it_tok(src,
                                    add_special_tokens=False).input_ids
                    off = 0
                    lo = find_subseq(pC, needle)
                    if lo is None and len(needle) > 1:
                        lo = find_subseq(pC, needle[1:])
                        if lo is not None:
                            off = 1
                            needle = needle[1:]
                    if lo is None:
                        continue
                    eo = (1 if src_ids[0] == tokenizer.bos_token_id
                          else 0) + off
                    hook.mode = "ef"
                    hook.delta = io["delta"][0,
                                             eo:eo + len(needle)].detach()
                    hook.off = lo
                elif arm == "clamp":
                    clamp_hook.enabled = True
                    clamp_hook.amp_idx = torch.nonzero(
                        za > 0).flatten().to(args.device)
                    clamp_hook.amp_val = 10.0
                    clamp_hook.sup_idx = torch.nonzero(
                        zs > 0).flatten().to(args.device)
                else:
                    dvec = (za.to(args.device).float() @ W
                            - zs.to(args.device).float() @ W)
                    hook.mode, hook.dvec = "steer", dvec
                    hook.alpha = args.steer_alpha
                lpA = resp_logprobs(fC, len(pC))
                hook.mode = None
                clamp_hook.enabled = False
                kl_AB = float((pB_p * (lpB - lpA)).sum(-1).mean())
                nllA = float(-lpA.gather(-1, torch.tensor(
                    resp, device=args.device).view(-1, 1)).mean())
                red = 1.0 - kl_AB / max(kl_CB, 1e-8)
                rec["arms"][f"{arm}|{c}"] = {
                    "kl_AB": kl_AB, "nllA": nllA, "kl_red": red}
                agg[(arm, c, "kl_red")].append(red)
                agg[(arm, c, "nllA")].append(nllA)
        agg[("_", "_", "kl_CB")].append(kl_CB)
        agg[("_", "_", "nllB")].append(nllB)
        agg[("_", "_", "nllC")].append(nllC)
        records.append(rec)
        if (step_i + 1) % 50 == 0:
            print(f"[klc] {step_i + 1}/{len(chosen)}")

    lines = [f"# Counterfactual consistency (KL/NLL, layer "
             f"{args.sae_layer}, n={len(records)})", "",
             f"reference B = frame(x1); null C = frame(x0); "
             f"KL_red = 1 - KL(A||B)/KL(C||B)", "",
             f"floor: KL(C||B) mean {np.mean(agg[('_','_','kl_CB')]):.4f}; "
             f"NLL(x1): B {np.mean(agg[('_','_','nllB')]):.4f} / "
             f"C {np.mean(agg[('_','_','nllC')]):.4f}", "",
             "| arm | condition | KL_reduction | NLL_A(x1) |",
             "|---|---|---|---|"]
    for arm in arms:
        for c in conditions:
            kr = agg.get((arm, c, "kl_red"), [])
            na = agg.get((arm, c, "nllA"), [])
            if kr:
                lines.append(f"| {arm} | {c} | {np.mean(kr):.4f} | "
                             f"{np.mean(na):.4f} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(report)


if __name__ == "__main__":
    main()
