"""FIC generation (metric (2), plan §1) — LinguaLens-verbatim frame:
bare sentence -> free continuation, temperature 1.0, 100 new tokens, NO
prompt (except the 'prompting' arm, whose prompt IS the method). The
JUDGED text is the full decode (sentence + continuation), official-style.

Per feature x n_pairs pairs x two directions:
  enhancement: dataset convention sentence1 = example (feature PRESENT),
    sentence2 = counterfactual -> start from s2, spec = diff(s2 -> s1)
  ablation:    start from s1, spec = diff(s1 -> s2)
Conditions: targeted / random (same shape) / control (raw continuation;
the clamp arm additionally gets its faithful 'recon' control =
reconstruction passthrough, LinguaLens's multiply x1).

Arms: steer (alpha*dvec every position), clamp (set 10/0 + recon
replacement), prompting (A3-literal steering prompts — their native
prominence task), and ef (--ef-ckpt: the editor's delta injected over the
sentence prefill — use the BARE-frame champion v4 here; user decision
2026-07-21: FIC covers all four arms in both frames, the repeat side by
reusing the exact-probe records via eval_fic_judge.py).

Resume-safe (records keyed). Judge lives in a separate script.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

from eval_lingualens import (                                  # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk,
    randomize_intervention, sae_z_with_offsets,
)
from intervener import EFIntervener, chat_prompt_ids           # noqa: E402
from model import SAEFeatureExtractor, load_sae                # noqa: E402
from scripts.eval_clamp_baseline import SaeClampHook           # noqa: E402


class EFPrefillHook:
    """Add the editor's per-position delta over the first n prefill
    positions (bare frame: sentence starts at position 0). Decode steps
    (seq len 1) are untouched — same mechanics as eval_ef_bare's BareHook
    'ef' mode."""

    def __init__(self):
        self.enabled = False
        self.delta = None            # (n, d)

    def __call__(self, module, inputs, output):
        if not self.enabled or self.delta is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            n = min(self.delta.shape[0], h.shape[1])
            add = torch.zeros_like(h)
            add[:, :n] = self.delta[:n].to(h.dtype)
            h = h + add
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


def stable_hash(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path", required=True)
    p.add_argument("--sae-layer", type=int, required=True)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--n-pairs", type=int, default=5)   # user decision
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--blocklist", default="")
    p.add_argument("--arms", default="steer,clamp,prompting")
    p.add_argument("--ef-ckpt", default="",
                   help="EFIntervener checkpoint for the ef arm — use the "
                        "BARE-frame champion (v4) here; this script's frame "
                        "is the bare free continuation (user 2026-07-21: "
                        "both-frames FIC; the repeat side reuses the exact "
                        "probe records)")
    p.add_argument("--a3-prompts", default="runs/a3_prompts/steering_prompts.json")
    p.add_argument("--steer-alpha", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=100)  # repo default
    p.add_argument("--temperature", type=float, default=1.0)   # repo default
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


class SteerAllHook:
    def __init__(self):
        self.enabled = False
        self.vec = None

    def __call__(self, module, inputs, output):
        if not self.enabled or self.vec is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = h + self.vec.to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


def main():
    args = parse_args()
    arms = [a for a in args.arms.split(",") if a]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    ds = ds.filter(lambda r: r["language"] == args.language)
    by_feat = defaultdict(list)
    for i in range(len(ds)):
        r = ds[i]
        by_feat[r["feature"]].append(i)
    rng = np.random.default_rng(args.seed)
    chosen = {}
    for f in sorted(by_feat):
        idxs = list(by_feat[f])
        rng.shuffle(idxs)
        chosen[f] = idxs[:args.n_pairs]
    print(f"[fic] {len(chosen)} features x <= {args.n_pairs} pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer,
    ).to(args.device).eval()
    blk = None
    if args.blocklist:
        blk = torch.as_tensor(np.load(args.blocklist).astype(np.int64))
    it_tok = AutoTokenizer.from_pretrained(args.it_model)
    it_model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype).to(args.device).eval()
    sae = load_sae("jumprelu", args.sae_repo, args.sae_path
                   ).to(args.device).eval()
    W = sae.W_dec.float()
    a3 = (json.loads(Path(args.a3_prompts).read_text())
          if "prompting" in arms else None)

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
        print(f"[fic] EF editor loaded from {args.ef_ckpt}")

    steer_hook = SteerAllHook()
    clamp_hook = SaeClampHook(sae)
    ef_hook = EFPrefillHook()
    layer = it_model.model.layers[args.sae_layer]
    layer.register_forward_hook(steer_hook)
    layer.register_forward_hook(clamp_hook)
    layer.register_forward_hook(ef_hook)

    rec_path = out_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in open(rec_path):
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"[fic] RESUME: {len(done)} records")
    fh = open(rec_path, "a")

    @torch.no_grad()
    def gen(prefix_ids, seed_key):
        torch.manual_seed(stable_hash(*seed_key) % (2 ** 31))
        ids = torch.tensor([prefix_ids], device=args.device)
        g = it_model.generate(
            input_ids=ids, max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0, temperature=args.temperature,
            pad_token_id=it_tok.pad_token_id or it_tok.eos_token_id)
        return it_tok.decode(g[0], skip_special_tokens=True)  # FULL text

    def build_spec(src, tgt):
        with torch.no_grad():
            s_off, z_s = sae_z_with_offsets(extractor, src, args.device)
            t_off, z_t = sae_z_with_offsets(extractor, tgt, args.device)
            om_s = [tuple(o) for o in tokenizer(
                src, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            om_t = [tuple(o) for o in tokenizer(
                tgt, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            s_ids = tokenizer(src, add_special_tokens=True).input_ids
            t_ids = tokenizer(tgt, add_special_tokens=True).input_ids
            ops = difflib.SequenceMatcher(
                None, s_ids, t_ids, autojunk=False).get_opcodes()
            sr, tr = edit_char_ranges(ops, om_s, om_t)
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
        return diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)

    def emit(key, feature, pi, d, arm, cond, text):
        fh.write(json.dumps({
            "key": key, "feature": feature, "pair": pi, "dir": d,
            "arm": arm, "cond": cond, "output": text},
            ensure_ascii=False) + "\n")
        fh.flush()

    for f in sorted(chosen):
        for pi, di in enumerate(chosen[f]):
            ex = ds[int(di)]
            s1, s2 = ex["sentence1"], ex["sentence2"]
            for d, (start, target) in (("enh", (s2, s1)),
                                       ("abl", (s1, s2))):
                bare_ids = it_tok(start, add_special_tokens=True).input_ids
                prng = np.random.default_rng(
                    stable_hash(args.seed, f, pi, d) % (2 ** 31))
                za = zs = None
                # control (shared raw)
                key = f"{f}|{pi}|{d}|_|control"
                if key not in done:
                    steer_hook.enabled = clamp_hook.enabled = False
                    ef_hook.enabled = False
                    emit(key, f, pi, d, "_", "control",
                         gen(bare_ids, (key,)))
                    done.add(key)
                for arm in arms:
                    for cond in (["targeted", "random"]
                                 + (["recon"] if arm == "clamp" else [])):
                        key = f"{f}|{pi}|{d}|{arm}|{cond}"
                        if key in done:
                            continue
                        if za is None:
                            za, zs = build_spec(start, target)
                        if cond == "random":
                            za_c = randomize_intervention(za, prng)
                            zs_c = randomize_intervention(zs, prng)
                        else:
                            za_c, zs_c = za, zs
                        steer_hook.enabled = clamp_hook.enabled = False
                        ef_hook.enabled = False
                        prefix = bare_ids
                        if arm == "ef":
                            with torch.no_grad():
                                ei = torch.tensor([bare_ids],
                                                  device=args.device)
                                io = interv(
                                    ei, torch.ones_like(ei),
                                    za_c.unsqueeze(0).to(args.device),
                                    zs_c.unsqueeze(0).to(args.device))
                                ef_hook.delta = io["delta"][0].detach()
                            ef_hook.enabled = True
                        elif arm == "steer" and cond != "recon":
                            steer_hook.vec = args.steer_alpha * (
                                za_c.to(args.device).float() @ W
                                - zs_c.to(args.device).float() @ W)
                            steer_hook.enabled = True
                        elif arm == "clamp":
                            clamp_hook.enabled = True
                            if cond == "recon":
                                clamp_hook.amp_idx = None
                                clamp_hook.sup_idx = None
                            else:
                                clamp_hook.amp_idx = torch.nonzero(
                                    za_c > 0).flatten().to(args.device)
                                clamp_hook.amp_val = 10.0
                                clamp_hook.sup_idx = torch.nonzero(
                                    zs_c > 0).flatten().to(args.device)
                        elif arm == "prompting":
                            sp_dir = a3.get(f, {}).get(d, "")
                            if cond == "random":
                                others = [f2 for f2 in sorted(a3)
                                          if f2 != f]
                                sp_dir = a3[others[int(prng.integers(
                                    0, len(others)))]][d]
                            prefix = chat_prompt_ids(
                                it_tok, sp_dir + "\n\nQuestion: " + start)
                        text = gen(prefix, (key,))
                        steer_hook.enabled = clamp_hook.enabled = False
                        ef_hook.enabled = False
                        emit(key, f, pi, d, arm, cond, text)
                        done.add(key)
        print(f"[fic] {f} done ({len(done)} records)")
    fh.close()
    print(f"[fic] GENERATION DONE -> {rec_path} ({len(done)} records)")


if __name__ == "__main__":
    main()
