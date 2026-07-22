"""BARE-frame probe for the EF through-LM editor (EF_LM_LOSS_PLAN.md §4).

LinguaLens-faithful frame: NO instruction prompt. The frozen
gemma-2-2b-it reads  [BOS] + src + "\\n"  and generates greedily; the
first generated line is the candidate edit. Information channels:
  * ef    — the learned editor's delta (lam_i * v_i) injected at layer L
            over the [BOS]+src positions during PREFILL only (the edited
            perception persists through the KV cache);
  * steer — fixed rendering h + alpha*(za-zs)@W_dec at EVERY position
            (prefill + decode), the champion mechanics in this frame;
  * steer_local — DIAGNOSTIC 3 (2026-07-19): the same fixed rendering
            injected ONLY at the ORACLE edit positions (src-side difflib
            spans vs tgt), prefill-only — the ef arm's exact injection
            interface with WHERE solved and WHAT linear. Measures the
            capacity ceiling of a linear spec rendering at the right
            positions in the bare frame.
  * oracle_resid — DIAGNOSTIC 5 (2026-07-19, prepared for the v5/v6
            residual-matching objective): inject the IDEAL residual
            delta  Δh* = h_L(tgt frame) − h_L(src frame)  at the
            difflib-aligned src positions (true condition only; no spec
            involved). Tests the objective's premise: does making the
            L12 states match the edited text cause the frozen LM to
            OUTPUT the edited text? --oracle-scale sweeps the push.
  * oracle_resid_b — DIAGNOSTIC 6a: oracle_resid + the BOUNDARY \\n
            position (same layer L). Isolates the boundary-position
            contribution identified by diag5.
  * oracle_resid_all — DIAGNOSTIC 6b (ANALYSIS ONLY, not a method arm:
            multi-layer injection bypasses the L12-feature mediation
            claim): the ideal per-layer deltas injected at EVERY layer's
            output (aligned positions + boundary). Measures the ceiling
            when ALL decoder inflow paths are closed — the lower-KV
            pathway's contribution = (6b − 6a).
  * raw   — no hook (the frame's floor: whatever the LM does after a
            bare sentence).
Conditions true / empty / random as in the standard probe; spec built
on the fly with the layer-L SAE (edit-local pool + per-layer blocklist).

Diagnostics: mean lambda per condition and lambda-IoU (top-|gold| lam
positions vs gold edit token positions) — the through-LM heir of the EF
lambda-IoU. --rounds > 1 re-encodes the output and re-injects
(self-correction ablation; stops early when mean lam < --stop-lam).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

from editflow_ops import align_pair, slot_ops                  # noqa: E402
from eval_lingualens import (                                  # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk, pair_metrics,
    randomize_intervention, sae_z_with_offsets,
)
from intervener import (EFIntervener, REPEAT_PROMPT,           # noqa: E402
                        chat_prompt_ids, find_subseq)
from model import SAEFeatureExtractor, load_sae                # noqa: E402
from scripts.eval_clamp_baseline import (extract_sentence, bname,  # noqa
                                         SaeClampHook)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--ef-ckpt", default="",
                   help="EFIntervener checkpoint; empty = skip the ef arm")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", required=True)
    p.add_argument("--sae-layer", type=int, required=True)
    p.add_argument("--sae-type", default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--reverse-pairs", action="store_true",
                   help="amp direction (user 2026-07-21: 全評価軸でamp/sup両方): "
                        "src=sentence2 (feature absent) -> tgt=sentence1 "
                        "(feature present); the edit ADDS the phenomenon")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--blocklist", default="",
                   help="layer-L blocklist .npy (leave empty to disable)")
    p.add_argument("--conditions", default="true,empty,random")
    p.add_argument("--arms", default="ef,steer,raw")
    p.add_argument("--steer-alpha", type=float, default=0.5)
    p.add_argument("--oracle-scale", type=float, default=1.0,
                   help="multiplier on the oracle residual delta "
                        "(arm oracle_resid)")
    p.add_argument("--a3-prompts", default="",
                   help="steering_prompts.json (gen_a3_prompts.py) — "
                        "enables the 'prompting' arm (A3 = AxBench "
                        "prompt-steering port; no SAE involved). The "
                        "standard probe direction s1->s2 REMOVES the "
                        "feature (their metrics.py: sentence1 = example, "
                        "sentence2 = counterfactual), so true uses the "
                        "'abl' prompt; random = another feature's abl; "
                        "empty = plain Question frame.")
    p.add_argument("--frame", choices=["bare", "repeat"], default="bare",
                   help="'repeat' = v5 frame: chat-templated explicit "
                        "repeat instruction (REPEAT_PROMPT); ef injects "
                        "at the src span inside the prompt, steer at all "
                        "positions, raw = prompt only. oracle_/steer_"
                        "local diag arms are bare-frame only.")
    p.add_argument("--ef-scale", type=float, default=1.0,
                   help="ablation: inference-time multiplier on the "
                        "editor's delta (tests whether the learned "
                        "magnitude is self-calibrated, vs steer's "
                        "alpha cliff)")
    p.add_argument("--spec-scope", choices=["local", "global"],
                   default="local",
                   help="ablation (7): 'global' pools the spec over ALL "
                        "sentence positions instead of the edit span")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--stop-lam", type=float, default=0.05)
    p.add_argument("--max-new-pad", type=int, default=24)
    p.add_argument("--feature-spec", default="",
                   help="per-feature spec JSON (build_feature_specs.py "
                        "l{L}_spec.json). When set, the intervention spec "
                        "is the feature's pool-mean delta (rescaled to the "
                        "pool's per-pair norm median) instead of the "
                        "evaluated pair's own z_tgt−z_src — the pair "
                        "never contributes to its spec. sup direction is "
                        "the stored sign; --reverse-pairs flips it.")
    p.add_argument("--fspec-retrieve", default="",
                   help="improvement A (2026-07-22): retrieval table JSON "
                        "(build_retrieval_table.py). Replaces the pool-MEAN "
                        "spec with the mean of the --retrieve-m pool pairs "
                        "whose SOURCE side is most similar (SAE max-act "
                        "cosine) to the eval src — input-only adaptation.")
    p.add_argument("--retrieve-m", type=int, default=5)
    p.add_argument("--cluster-expand", default="",
                   help="improvement ① : W_dec neighbor table "
                        "(build_cluster_table.py). Each spec component "
                        "shares weight with its decoder-cosine split-"
                        "siblings: v[j] += share * cos_ij * v[i]. Offline "
                        "and input-independent.")
    p.add_argument("--cluster-share", type=float, default=0.5)
    p.add_argument("--amp-only", action="store_true",
                   help="improvement C: after direction resolution keep "
                        "only the additive (v>0) components and renorm — "
                        "concentrates the budget on the insertion drive.")
    p.add_argument("--src-gate", action="store_true",
                   help="improvement ① (2026-07-22): per-instance gating "
                        "of the feature spec by the SOURCE's own SAE "
                        "activations (input-only — no target peeking). "
                        "Suppress-side components (v<0 after direction "
                        "resolution) are kept only where the feature "
                        "actually fires in src; the rescale then uses the "
                        "gated vector's norm.")
    p.add_argument("--fspec-scale", type=float, default=1.0,
                   help="extra multiplier on the feature spec AFTER the "
                        "norm-median rescale (input-side strength sweep; "
                        "only meaningful with --feature-spec)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (default). >0 turns on sampling for "
                        "the frame generation — combine with --gen-seed "
                        "and repeated runs for a sampling-robustness probe")
    p.add_argument("--gen-seed", type=int, default=0,
                   help="torch manual seed for sampled generation runs")
    p.add_argument("--pool-dev", default="",
                   help="eval_split.json path; sample pairs from the "
                        "identification POOL instead of the eval 500 — "
                        "for hyperparameter selection (e.g. fspec-scale) "
                        "without touching the eval sample")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


def oracle_delta(hs, ht, s_ids, t_ids, boundary: bool):
    """Ideal residual delta at difflib-aligned src positions (+ the
    boundary \\n position when boundary=True). hs/ht: (T, d) hidden
    states of the src/tgt frames (ids + [nl])."""
    n_rows = len(s_ids) + (1 if boundary else 0)
    d = torch.zeros(n_rows, hs.shape[-1], device=hs.device)
    sm2 = difflib.SequenceMatcher(None, s_ids, t_ids, autojunk=False)
    for tag2, i1, i2, j1, j2 in sm2.get_opcodes():
        if tag2 in ("equal", "replace"):
            n2 = min(i2 - i1, j2 - j1)
            for k2 in range(n2):
                d[i1 + k2] = ht[j1 + k2] - hs[i1 + k2]
            for k2 in range(n2, i2 - i1):
                d[i1 + k2] = ht[min(j2 - 1, ht.shape[0] - 1)] \
                    - hs[i1 + k2]
        elif tag2 == "delete":
            jb = min(j1, ht.shape[0] - 1)
            for p2 in range(i1, i2):
                d[p2] = ht[jb] - hs[p2]
    if boundary:
        d[len(s_ids)] = ht[len(t_ids)] - hs[len(s_ids)]
    return d


class BareHook:
    """mode 'ef': add the per-position field over the first n PREFILL
    positions only. mode 'steer': h + alpha*dvec at every position,
    prefill and decode alike (champion mechanics)."""

    def __init__(self):
        self.mode = None            # None | "ef" | "steer"
        self.delta = None           # (n, d) float
        self.off = 0                # start position of the delta rows
        self.dvec = None            # (d,) float
        self.alpha = 0.5

    def __call__(self, module, inputs, output):
        if self.mode is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        if self.mode == "ef":
            if h.shape[1] > 1 and self.delta is not None:
                n = min(self.delta.shape[0], h.shape[1] - self.off)
                if n > 0:
                    add = torch.zeros_like(h)
                    add[:, self.off:self.off + n] = \
                        self.delta[:n].to(h.dtype)
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
    if args.pool_dev:
        _ev = set(json.loads(Path(args.pool_dev).read_text())["eval_idx"])
        order = [k for k in order if k not in _ev]
        print(f"[efbare] POOL-DEV mode: sampling from the "
              f"{len(order)}-pair identification pool (eval excluded)")
    chosen = order[:min(args.sample_size, len(order))]
    print(f"[efbare] {len(ds)} pairs, sampling {len(chosen)}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    blk = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        blk = torch.as_tensor(np.asarray(_bl, dtype=np.int64))
        print(f"[efbare] blocklist: {len(_bl)} features masked")

    it_tok = AutoTokenizer.from_pretrained(args.it_model)
    it_model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype).to(args.device).eval()
    sae = load_sae(args.sae_type, args.sae_repo, args.sae_path,
                   sae_k=args.sae_k).to(args.device).eval()
    W = sae.W_dec.float()

    interv = None
    if "ef" in arms:
        if not args.ef_ckpt:
            raise SystemExit("arm 'ef' needs --ef-ckpt")
        blob = torch.load(args.ef_ckpt, map_location="cpu",
                          weights_only=False)
        cfg = blob["config"]
        if int(cfg["inject_layer"]) != args.sae_layer:
            raise SystemExit(f"ckpt layer {cfg['inject_layer']} != "
                             f"--sae-layer {args.sae_layer}")
        interv = EFIntervener(args.llm2vec_dir, int(cfg["d_sae"]),
                              dtype=dtype, lora_r=int(cfg.get("lora_r", 32)),
                              w_dec=sae.W_dec.detach().float().cpu(),
                              ).to(args.device).eval()
        interv.load_trainable_state_dict(blob["trainable"])
        print(f"[efbare] EF editor loaded from {args.ef_ckpt}")

    fspec = None
    if args.feature_spec:
        fspec = json.loads(Path(args.feature_spec).read_text())
        print(f"[efbare] FEATURE-SPEC mode: {len(fspec)} features from "
              f"{args.feature_spec} (pair-independent interventions)")
    ctab = None
    if args.cluster_expand:
        ctab = {int(k): v for k, v in json.loads(
            Path(args.cluster_expand).read_text()).items()}
        print(f"[efbare] CLUSTER-EXPAND: {len(ctab)} latents with "
              f"neighbors, share={args.cluster_share}")
    rtab = None
    if args.fspec_retrieve:
        rtab = json.loads(Path(args.fspec_retrieve).read_text())
        print(f"[efbare] RETRIEVAL spec: {len(rtab)} features, "
              f"m={args.retrieve_m}")

    a3 = None
    if "prompting" in arms or "prompting_edit" in arms:
        if not args.a3_prompts:
            raise SystemExit("arm 'prompting' needs --a3-prompts")
        a3 = json.loads(Path(args.a3_prompts).read_text())
        print(f"[efbare] A3 prompts: {len(a3)} features")

    if args.temperature > 0:
        torch.manual_seed(args.gen_seed or args.seed)
        print(f"[efbare] SAMPLING mode: temperature={args.temperature}, "
              f"gen_seed={args.gen_seed or args.seed}")
    hook = BareHook()
    it_model.model.layers[args.sae_layer].register_forward_hook(hook)
    # A1 arm: LinguaLens-faithful set(10/0) + full recon replacement,
    # every position and step (OpenSAE semantics; empty spec = recon
    # passthrough = their control)
    clamp_hook = SaeClampHook(sae)
    it_model.model.layers[args.sae_layer].register_forward_hook(clamp_hook)
    # diag 6b: one hook per layer (enabled only by oracle_resid_all)
    n_layers = len(it_model.model.layers)
    layer_hooks = [BareHook() for _ in range(n_layers)]
    for _l, _hk in enumerate(layer_hooks):
        it_model.model.layers[_l].register_forward_hook(_hk)
    nl_ids = it_tok("\n", add_special_tokens=False).input_ids
    assert len(nl_ids) == 1
    nl_id = int(nl_ids[0])
    print(f"[efbare] BARE frame on layers[{args.sae_layer}]; "
          f"arms={arms} rounds={args.rounds}")

    @torch.no_grad()
    def gen_continuation(prefix_ids, src_len=None):
        full = (prefix_ids if args.frame == "repeat"
                else prefix_ids + [nl_id])
        ids = torch.tensor([full], device=args.device)
        gen_kw = dict(do_sample=False)
        if args.temperature > 0:
            gen_kw = dict(do_sample=True, temperature=args.temperature)
        out = it_model.generate(
            input_ids=ids,
            max_new_tokens=(src_len or len(prefix_ids)) + args.max_new_pad,
            pad_token_id=it_tok.pad_token_id or it_tok.eos_token_id,
            **gen_kw)
        text = it_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text.split("\n")[0].strip()

    def frame_prompt(src_text, src_ids_):
        """Returns (lm_prefix_ids, span_lo, needle) for the frame."""
        if args.frame == "repeat":
            pids = chat_prompt_ids(it_tok,
                                   REPEAT_PROMPT.format(src=src_text))
            needle = it_tok(src_text, add_special_tokens=False).input_ids
            off = 0
            lo = find_subseq(pids, needle)
            if lo is None and len(needle) > 1:
                lo = find_subseq(pids, needle[1:])
                if lo is not None:
                    off = 1
                    needle = needle[1:]
            return pids, lo, needle, off
        return list(src_ids_), 0, list(src_ids_), 0

    @torch.no_grad()
    def ef_delta(src_ids, za, zs):
        ei = torch.tensor([src_ids], device=args.device)
        io = interv(ei, torch.ones_like(ei),
                    za.unsqueeze(0).to(args.device),
                    zs.unsqueeze(0).to(args.device))
        return io["delta"][0].detach(), io["lam"][0].detach()

    def lam_iou(lam, src_ids, tgt_ids):
        sm = difflib.SequenceMatcher(None, src_ids, tgt_ids,
                                     autojunk=False)
        gold = set()
        for tag, i1, i2, _, _ in sm.get_opcodes():
            if tag != "equal":
                gold.update(range(i1, max(i1 + 1, i2)))
        if not gold:
            return None
        k = min(len(gold), lam.shape[0])
        top = set(torch.topk(lam, k).indices.tolist())
        return len(top & gold) / max(1, len(top | gold))

    partial_path = out_dir / "records.partial.jsonl"
    records, done_idx = [], set()
    for src_path in (out_dir / "records.jsonl", partial_path):
        if not src_path.exists():
            continue
        with open(src_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(r["idx"]) in done_idx:
                    continue
                records.append(r)
                done_idx.add(int(r["idx"]))
    if records:
        print(f"[efbare] RESUME: {len(records)} pairs")
    pf = open(partial_path, "a")

    for step_i, k in enumerate(chosen):
        if int(k) in done_idx:
            continue
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        if args.reverse_pairs:
            src, tgt = tgt, src
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_ids = tokenizer(tgt, add_special_tokens=True).input_ids
        slots = align_pair(src_ids, tgt_ids)
        n_ops = len(slot_ops(slots))
        if n_ops == 0:
            continue
        prng = np.random.default_rng(args.seed * 1000003 + int(k))

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
            if args.spec_scope == "global":
                sr, tr = [], []          # local_pool_topk falls back to
                #                          global pooling on empty ranges
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
        if fspec is not None:
            fs = fspec.get(ex.get("feature") or "?")
            if fs is None:
                continue                     # feature absent from pool
            if rtab is not None:
                ent = rtab.get(ex.get("feature") or "?")
                if ent is None:
                    continue
                g_r = z_s.max(dim=0).values.float().cpu()
                if blk is not None:
                    g_r[blk] = 0.0
                side = "m2" if args.reverse_pairs else "m1"
                sims = []
                for pp in ent:
                    num = sum(float(g_r[int(i)]) * val
                              for i, val in pp[side].items())
                    den = (sum(val * val for val in pp[side].values())
                           ** 0.5) + 1e-9
                    sims.append(num / den)
                order2 = sorted(range(len(sims)), key=lambda i: -sims[i])
                top = order2[:max(1, min(args.retrieve_m, len(order2)))]
                v = torch.zeros_like(z_src)
                for i in top:
                    for fi, val in ent[i]["d"].items():
                        v[int(fi)] += val
                v = v / len(top)
            else:
                v = torch.zeros_like(z_src)
                for fi, val in fs["spec"].items():
                    v[int(fi)] = val
            if args.reverse_pairs:           # spec stored sup (s1->s2)
                v = -v
            if ctab is not None:
                add = torch.zeros_like(v)
                nz2 = torch.nonzero(v).flatten().tolist()
                for i in nz2:
                    for j, c in ctab.get(int(i), ()):
                        add[j] += args.cluster_share * c * float(v[i])
                if blk is not None:
                    add[blk] = 0.0
                v = v + add
            if args.amp_only:
                v = torch.clamp(v, min=0.0)
            if args.src_gate:
                g_src = z_s.max(dim=0).values.float().cpu()
                if blk is not None:
                    g_src[blk] = 0.0
                dead = (v < 0) & (g_src <= 0)
                v[dead] = 0.0
            if (args.src_gate or args.amp_only or rtab is not None
                    or ctab is not None):
                nrm = float(v.norm())        # dynamic renorm paths
                if nrm > 0:
                    v = v * (fs["norm_median"] / nrm)
            elif fs["mean_norm"] > 0:        # rescale to pool's per-pair
                v = v * (fs["norm_median"] / fs["mean_norm"])
            v = v * args.fspec_scale
            za_t, zs_t = diff_intervention(
                torch.zeros_like(v), v, args.k_amp, args.k_sup)
        else:
            za_t, zs_t = diff_intervention(
                z_src, z_tgt, args.k_amp, args.k_sup)
        zvar = {"true": (za_t, zs_t),
                "empty": (torch.zeros_like(za_t), torch.zeros_like(zs_t)),
                "random": (randomize_intervention(za_t, prng),
                           randomize_intervention(zs_t, prng))}

        rec = {"idx": int(k), "src": src, "tgt": tgt, "n_ops": n_ops,
               "feature": ex.get("feature"), "outputs": {}}
        for c in conditions:
            za, zs = zvar[c]
            rec["outputs"][c] = {}
            for arm in arms:
                if arm == "raw" and c != "empty":
                    continue                      # raw is condition-free
                if arm.startswith("oracle_resid") and c != "true":
                    continue                      # oracle uses tgt, no spec
                if args.frame == "repeat" and (
                        arm.startswith("oracle_") or arm == "steer_local"):
                    continue                      # diag arms are bare-only
                hook.mode = None
                if arm == "ef":
                    cur_ids = list(src_ids)
                    cur_text = src
                    texts, lams, ious = [], [], []
                    for rnd in range(max(1, args.rounds)):
                        delta, lam = ef_delta(cur_ids, za, zs)
                        lam_mean = float(lam.mean())
                        lams.append(lam_mean)
                        if rnd == 0:
                            iou = lam_iou(lam, src_ids, tgt_ids)
                            if iou is not None:
                                ious.append(iou)
                        if rnd > 0 and lam_mean < args.stop_lam:
                            break
                        pids, lo, needle, extra = frame_prompt(cur_text,
                                                               cur_ids)
                        if args.frame == "repeat":
                            if lo is None:
                                break
                            eo = (1 if cur_ids and cur_ids[0]
                                  == tokenizer.bos_token_id else 0) + extra
                            n = min(len(needle), delta.shape[0] - eo)
                            hook.delta = delta[eo:eo + n] * args.ef_scale
                            hook.off = lo
                        else:
                            hook.delta = delta * args.ef_scale
                            hook.off = 0
                        hook.mode = "ef"
                        out_text = gen_continuation(pids,
                                                    src_len=len(cur_ids))
                        hook.mode = None
                        if not out_text:
                            break
                        texts.append(out_text)
                        cur_text = out_text
                        cur_ids = tokenizer(
                            out_text, add_special_tokens=True).input_ids
                    out_text = texts[-1] if texts else ""
                    extra = {"lam_mean": lams,
                             "lam_iou": (ious[0] if ious else None),
                             "n_rounds": len(texts)}
                elif arm == "steer":
                    dvec = (za.to(args.device).float() @ W
                            - zs.to(args.device).float() @ W)
                    hook.mode, hook.dvec = "steer", dvec
                    hook.alpha = args.steer_alpha
                    pids, _, _, _ = frame_prompt(src, src_ids)
                    out_text = gen_continuation(pids,
                                                src_len=len(src_ids))
                    hook.mode = None
                    extra = {}
                elif arm == "prompting_edit":
                    # A3' (main prompting row, user-approved 2026-07-19):
                    # natural-language EDIT instruction — the no-SAE
                    # prompting ceiling on our task. s1->s2 removes the
                    # feature (dataset convention), hence "remove".
                    feat = (ex.get("feature") or "").replace("_", " ")
                    verb = "add" if args.reverse_pairs else "remove"
                    if c == "true":
                        instr = (f"Rewrite the input sentence to {verb} "
                                 f"any {feat}. Output only the rewritten "
                                 f"sentence.")
                    elif c == "random":
                        others = [f2 for f2 in sorted(a3) if
                                  f2.replace('_', ' ') != feat] \
                            if a3 else ["metaphor"]
                        rf2 = others[int(prng.integers(0, len(others)))]
                        instr = (f"Rewrite the input sentence to {verb} "
                                 f"any {rf2.replace('_', ' ')}. Output "
                                 f"only the rewritten sentence.")
                    else:
                        instr = ("Rewrite the input sentence. Output "
                                 "only the rewritten sentence.")
                    pids2 = chat_prompt_ids(
                        it_tok, instr + "\n\nInput: " + src)
                    hook.mode = None
                    with torch.no_grad():
                        g = it_model.generate(
                            input_ids=torch.tensor([pids2],
                                                   device=args.device),
                            max_new_tokens=len(src_ids) + args.max_new_pad,
                            do_sample=False,
                            pad_token_id=it_tok.pad_token_id
                            or it_tok.eos_token_id)
                    out_text = it_tok.decode(
                        g[0, len(pids2):],
                        skip_special_tokens=True).split("\n")[0].strip()
                    extra = {}
                elif arm == "prompting":
                    feat = ex.get("feature") or ""
                    dkey = "enh" if args.reverse_pairs else "abl"
                    if c == "true":
                        sp = a3.get(feat, {}).get(dkey, "")
                    elif c == "random":
                        others = [f2 for f2 in sorted(a3)
                                  if f2 != feat]
                        sp = a3[others[int(prng.integers(
                            0, len(others)))]][dkey]
                    else:                          # empty
                        sp = ""
                    content = ((sp + "\n\nQuestion: " + src)
                               if sp else ("Question: " + src))
                    pids2 = chat_prompt_ids(it_tok, content)
                    hook.mode = None
                    with torch.no_grad():
                        g = it_model.generate(
                            input_ids=torch.tensor([pids2],
                                                   device=args.device),
                            max_new_tokens=len(src_ids) + args.max_new_pad,
                            do_sample=False,
                            pad_token_id=it_tok.pad_token_id
                            or it_tok.eos_token_id)
                    out_text = it_tok.decode(
                        g[0, len(pids2):],
                        skip_special_tokens=True).split("\n")[0].strip()
                    extra = {}
                elif arm == "clamp":
                    clamp_hook.enabled = True
                    clamp_hook.amp_idx = torch.nonzero(
                        za > 0).flatten().to(args.device)
                    clamp_hook.amp_val = 10.0      # their enhancement value
                    clamp_hook.sup_idx = torch.nonzero(
                        zs > 0).flatten().to(args.device)
                    pids, _, _, _ = frame_prompt(src, src_ids)
                    out_text = gen_continuation(pids,
                                                src_len=len(src_ids))
                    clamp_hook.enabled = False
                    extra = {}
                elif arm.startswith("oracle_resid"):
                    with torch.no_grad():
                        os_ = it_model(
                            input_ids=torch.tensor(
                                [src_ids + [nl_id]], device=args.device),
                            output_hidden_states=True, use_cache=False)
                        ot_ = it_model(
                            input_ids=torch.tensor(
                                [tgt_ids + [nl_id]], device=args.device),
                            output_hidden_states=True, use_cache=False)
                    if arm == "oracle_resid_all":      # diag 6b
                        for _l in range(n_layers):
                            dl = oracle_delta(
                                os_.hidden_states[_l + 1][0].float(),
                                ot_.hidden_states[_l + 1][0].float(),
                                src_ids, tgt_ids, boundary=True)
                            layer_hooks[_l].mode = "ef"
                            layer_hooks[_l].delta = dl * args.oracle_scale
                        out_text = gen_continuation(list(src_ids))
                        for _hk in layer_hooks:
                            _hk.mode = None
                        extra = {}
                    else:                              # diag 5 / 6a
                        hs = os_.hidden_states[args.sae_layer + 1][0].float()
                        ht = ot_.hidden_states[args.sae_layer + 1][0].float()
                        dloc = oracle_delta(
                            hs, ht, src_ids, tgt_ids,
                            boundary=(arm == "oracle_resid_b"))
                        hook.mode = "ef"
                        hook.delta = dloc * args.oracle_scale
                        out_text = gen_continuation(list(src_ids))
                        hook.mode = None
                        extra = {"d_norm": float(dloc.norm(dim=-1).mean())}
                    del os_, ot_
                elif arm == "steer_local":
                    # diagnostic 3: oracle WHERE (gold src-side edit
                    # positions) x linear WHAT (alpha*dvec), prefill-only
                    # via the ef injection interface.
                    dvec = (za.to(args.device).float() @ W
                            - zs.to(args.device).float() @ W)
                    sm2 = difflib.SequenceMatcher(None, src_ids, tgt_ids,
                                                  autojunk=False)
                    gold = set()
                    for tag2, i1, i2, _, _ in sm2.get_opcodes():
                        if tag2 != "equal":
                            gold.update(range(i1, max(i1 + 1, i2)))
                    gold = sorted(p for p in gold if p < len(src_ids))
                    dloc = torch.zeros(len(src_ids), W.shape[1],
                                       device=args.device)
                    for p in gold:
                        dloc[p] = args.steer_alpha * dvec
                    hook.mode, hook.delta = "ef", dloc
                    out_text = gen_continuation(list(src_ids))
                    hook.mode = None
                    extra = {"n_gold": len(gold)}
                else:                              # raw
                    hook.mode = None
                    pids, _, _, _ = frame_prompt(src, src_ids)
                    out_text = gen_continuation(pids,
                                                src_len=len(src_ids))
                    extra = {}
                out_text = extract_sentence(out_text, src)
                pm = pair_metrics(out_text, src, tgt)
                rec["outputs"][c][arm] = {
                    "text": out_text, "exact": pm["exact_match"],
                    "sim_target": pm["sim_target"],
                    "copy": pm["copy_rate"],
                    "no_edit": pm["copy_rate"], **extra}
        hook.mode = None
        records.append(rec)
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        if (step_i + 1) % 10 == 0:
            print(f"[efbare] {step_i + 1}/{len(chosen)} pairs "
                  f"({len(records)} scored)")
    pf.close()

    # ---- report -----------------------------------------------------------
    lines = [f"# EF through-LM bare-frame probe (layer {args.sae_layer})",
             ""]
    fdesc = ("REPEAT (chat-templated repeat instruction, greedy)"
             if args.frame == "repeat"
             else "BARE (no prompt, [BOS]+src+\\n, greedy)")
    lines.append(f"pairs scored: {len(records)}; frame = {fdesc}; "
                 f"arms {arms}; "
                 f"steer alpha {args.steer_alpha}; rounds {args.rounds}; "
                 f"sae {args.sae_path}")
    lines += ["", "| condition | arm | exact | sim_target | copy |",
              "|---|---|---|---|---|"]
    agg = defaultdict(list)
    for r in records:
        for c, arms_d in r["outputs"].items():
            for a, m in arms_d.items():
                agg[(c, a)].append(m)
    for (c, a), ms in sorted(agg.items()):
        lines.append(
            f"| {c} | {a} | "
            f"{np.mean([m['exact'] for m in ms]):.4f} | "
            f"{np.mean([m['sim_target'] for m in ms]):.4f} | "
            f"{np.mean([m['copy'] for m in ms]):.4f} |")
    if "ef" in arms:
        lines += ["", "## lambda diagnostics", "",
                  "| condition | mean lam (round 1) | lam-IoU |",
                  "|---|---|---|"]
        for c in conditions:
            ms = agg.get((c, "ef"), [])
            lam1 = [m["lam_mean"][0] for m in ms if m.get("lam_mean")]
            ious = [m["lam_iou"] for m in ms if m.get("lam_iou") is not None]
            lines.append(
                f"| {c} | {np.mean(lam1) if lam1 else float('nan'):.4f} | "
                f"{np.mean(ious) if ious else float('nan'):.4f} |")
        lines += ["", "## Multi-site breakdown (condition = true, arm = ef)",
                  "", "| n_ops | pairs | exact | sim |", "|---|---|---|---|"]
        buckets = defaultdict(list)
        for r in records:
            m = r["outputs"].get("true", {}).get("ef")
            if m:
                buckets[bname(r["n_ops"])].append(m)
        for bn in ("1", "2-3", "4-8", "9+"):
            ms = buckets.get(bn, [])
            if ms:
                lines.append(
                    f"| {bn} | {len(ms)} | "
                    f"{np.mean([m['exact'] for m in ms]):.4f} | "
                    f"{np.mean([m['sim_target'] for m in ms]):.4f} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report)
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)
    print(report)
    print(f"[efbare] wrote {out_dir}/report.md, records.jsonl")


if __name__ == "__main__":
    main()
