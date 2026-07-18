"""BARE-frame probe for the EF through-LM editor (EF_LM_LOSS_PLAN.md §4).

LinguaLens-faithful frame: NO instruction prompt. The frozen
gemma-2-2b-it reads  [BOS] + src + "\\n"  and generates greedily; the
first generated line is the candidate edit. Information channels:
  * ef    — the learned editor's delta (lam_i * v_i) injected at layer L
            over the [BOS]+src positions during PREFILL only (the edited
            perception persists through the KV cache);
  * steer — fixed rendering h + alpha*(za-zs)@W_dec at EVERY position
            (prefill + decode), the champion mechanics in this frame;
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
from intervener import EFIntervener                            # noqa: E402
from model import SAEFeatureExtractor, load_sae                # noqa: E402
from scripts.eval_clamp_baseline import extract_sentence, bname  # noqa: E402


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
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--stop-lam", type=float, default=0.05)
    p.add_argument("--max-new-pad", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


class BareHook:
    """mode 'ef': add the per-position field over the first n PREFILL
    positions only. mode 'steer': h + alpha*dvec at every position,
    prefill and decode alike (champion mechanics)."""

    def __init__(self):
        self.mode = None            # None | "ef" | "steer"
        self.delta = None           # (n, d) float
        self.dvec = None            # (d,) float
        self.alpha = 0.5

    def __call__(self, module, inputs, output):
        if self.mode is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        if self.mode == "ef":
            if h.shape[1] > 1 and self.delta is not None:
                n = min(self.delta.shape[0], h.shape[1])
                add = torch.zeros_like(h)
                add[:, :n] = self.delta[:n].to(h.dtype)
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

    hook = BareHook()
    it_model.model.layers[args.sae_layer].register_forward_hook(hook)
    nl_ids = it_tok("\n", add_special_tokens=False).input_ids
    assert len(nl_ids) == 1
    nl_id = int(nl_ids[0])
    print(f"[efbare] BARE frame on layers[{args.sae_layer}]; "
          f"arms={arms} rounds={args.rounds}")

    @torch.no_grad()
    def gen_continuation(prefix_ids):
        ids = torch.tensor([prefix_ids + [nl_id]], device=args.device)
        out = it_model.generate(
            input_ids=ids, max_new_tokens=len(prefix_ids) + args.max_new_pad,
            do_sample=False,
            pad_token_id=it_tok.pad_token_id or it_tok.eos_token_id)
        text = it_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text.split("\n")[0].strip()

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
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
        za_t, zs_t = diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)
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
                hook.mode = None
                if arm == "ef":
                    cur_ids = list(src_ids)
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
                        hook.mode, hook.delta = "ef", delta
                        out_text = gen_continuation(cur_ids)
                        hook.mode = None
                        if not out_text:
                            break
                        texts.append(out_text)
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
                    out_text = gen_continuation(list(src_ids))
                    hook.mode = None
                    extra = {}
                else:                              # raw
                    hook.mode = None
                    out_text = gen_continuation(list(src_ids))
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
    lines.append(f"pairs scored: {len(records)}; frame = BARE "
                 f"(no prompt, [BOS]+src+\\n, greedy); arms {arms}; "
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
