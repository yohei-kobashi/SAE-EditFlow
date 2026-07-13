"""
B1 baseline (PAPER_OUTLINE §5, claim C1): LinguaLens direct SAE
intervention, reproduced as faithfully as possible from the official
code (THU-KEG/LinguaLens + THU-KEG/OpenSAE, verified 2026-07-14) but on
OUR backbone: Gemma-2-2B(-it) + Gemma Scope layer-12/16k.

Faithful mechanics (OpenSAE transformer_with_sae.py):
  * hook on the SAE layer's output, prompt_only=False → the intervention
    applies at EVERY position of EVERY forward, prompt and generated
    tokens alike (no position selection — the documented C1 premise);
  * "set" intervention: active target features are overwritten with the
    value; INACTIVE ones are force-inserted (OpenSAE replaces the
    token's min-activation slot) — in dense terms exactly z[idx]=value.
    Enhancement value 10.0 and ablation value 0.0 in their code; we
    additionally sweep {5, 20} (Gemma Scope's activation scale differs
    from Llama's) and a task-native `clampZ` (set each amp feature to
    its commanded magnitude);
  * the residual is REPLACED by the SAE reconstruction of the modified
    features (not a delta) — so even the control passes through the
    reconstruction. Their control condition (multiply x1 = pure
    reconstruction passthrough) is our `empty` condition (mode `recon`);
    an extra `raw` mode (hook disabled) isolates reconstruction damage.

Adaptations for the minimal-pair editing regime (recorded for the
paper): their protocol free-generates from a prompt (5x temp 0.7) and
judges feature prominence; ours must EDIT, so the model is the
instruction-tuned sibling (gemma-2-2b-it — mirroring their use of an
instruct Llama with a base-trained SAE) given a neutral rewrite prompt
with NO feature text (the clamp IS the conditioning channel), greedy
decode; enhancement (z_amp indices) and ablation (z_sup indices) apply
SIMULTANEOUSLY, as the commanded delta demands. Records land in the
probe format, so compare_ef_pipeline.py and the FRR judge apply
unchanged — FRR is the LinguaLens-basis metric this method was built
for.

Usage (miyabi):
    python scripts/eval_clamp_baseline.py \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --blocklist runs/blocklist/blocklist.npy \
        --output-dir runs/prod_gemma_v6/clamp_baseline500 \
        --sample-size 500 --device cuda
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
from model import SAEFeatureExtractor, load_sae                # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")

    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"],
                   default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)

    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k-amp", type=int, default=64)
    p.add_argument("--k-sup", type=int, default=64)
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--blocklist", default="")
    p.add_argument("--conditions", default="true,empty,random")
    p.add_argument("--clamp-values", default="5,10,20",
                   help="enhancement 'set' values swept on `true` "
                        "(their code uses 10); ablation is always set-0. "
                        "empty/random run at the SECOND value (10).")
    p.add_argument("--intervention", choices=["clamp", "steer"],
                   default="clamp",
                   help="B1 'clamp' = OpenSAE-faithful set+reconstruction "
                        "replacement. B3 'steer' = steering vector: "
                        "h + alpha*(za@W_dec - zs@W_dec), a pure delta "
                        "add (no SAE in the loop) rendering the commanded "
                        "feature delta in residual space (SAE-TS spirit); "
                        "--clamp-values are read as the alpha sweep.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


PROMPT = """You are a precise text editor. Rewrite the sentence below, \
making only the minimal change that feels required; if nothing needs \
changing, output it unchanged. Output ONLY the rewritten sentence, \
nothing else.

Sentence: {src}"""


class SaeClampHook:
    """OpenSAE-faithful intervention: encode the layer output with the
    SAE, 'set' the target features (force-insert included), replace the
    residual with the reconstruction. enabled=False → raw passthrough."""

    def __init__(self, sae):
        self.sae = sae
        self.enabled = False
        self.amp_idx = None          # LongTensor or None
        self.amp_val = None          # float, or per-feature FloatTensor
        self.sup_idx = None

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return None
        h = output[0] if isinstance(output, tuple) else output
        dt = h.dtype
        z = self.sae.encode(h.to(self.sae.W_enc.dtype))
        if self.amp_idx is not None and self.amp_idx.numel():
            if isinstance(self.amp_val, torch.Tensor):
                z[..., self.amp_idx] = self.amp_val.to(z.dtype)
            else:
                z[..., self.amp_idx] = float(self.amp_val)
        if self.sup_idx is not None and self.sup_idx.numel():
            z[..., self.sup_idx] = 0.0
        h_new = self.sae.decode(z).to(dt)
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new


class SteerHook:
    """B3: steering-vector intervention — the commanded feature delta
    rendered in residual space, h + alpha*dvec, added at every position
    (pure delta: NO SAE reconstruction in the forward pass)."""

    def __init__(self):
        self.enabled = False
        self.dvec = None             # (d_llm,) FloatTensor
        self.alpha = 1.0

    def __call__(self, module, inputs, output):
        if not self.enabled or self.dvec is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h_new = h + (self.alpha * self.dvec).to(h.dtype)
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        return h_new


def extract_sentence(text: str, src: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip().strip('"').strip()
        if not line:
            continue
        for pre in ("Rewritten sentence:", "Sentence:", "Output:"):
            if line.lower().startswith(pre.lower()):
                line = line[len(pre):].strip().strip('"').strip()
        if line:
            return line
    return src


def bname(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4-8" if n <= 8 else "9+"


def main():
    args = parse_args()
    conditions = [c for c in args.conditions.split(",") if c]
    clamp_vals = [float(x) for x in args.clamp_values.split(",") if x]
    ctrl_val = clamp_vals[min(1, len(clamp_vals) - 1)]
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
    print(f"[b1] {len(ds)} pairs, sampling {len(chosen)}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    blk = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        blk = torch.as_tensor(np.asarray(_bl, dtype=np.int64))
        print(f"[b1] blocklist: {len(_bl)} features masked")

    it_tok = AutoTokenizer.from_pretrained(args.it_model)
    it_model = AutoModelForCausalLM.from_pretrained(
        args.it_model, torch_dtype=dtype).to(args.device).eval()
    # Gemma Scope layer_L = residual AFTER block L → hook block L's output
    sae = load_sae(args.sae_type, args.sae_repo, args.sae_path,
                   sae_k=args.sae_k).to(args.device).eval()
    if args.intervention == "clamp":
        hook = SaeClampHook(sae)
        mode_prefix = "clamp"
    else:
        hook = SteerHook()
        mode_prefix = "steer"
    it_model.model.layers[args.sae_layer].register_forward_hook(hook)
    print(f"[b1] rewriter {args.it_model}, {args.intervention} hook on "
          f"layers[{args.sae_layer}] output (all positions, "
          f"prompt+generation)")

    @torch.no_grad()
    def rewrite(src: str) -> str:
        text_in = it_tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(src=src)}],
            add_generation_prompt=True, tokenize=False)
        enc = it_tok(text_in, return_tensors="pt",
                     add_special_tokens=False).to(args.device)
        gen = it_model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=it_tok.pad_token_id or it_tok.eos_token_id)
        return extract_sentence(
            it_tok.decode(gen[0, enc["input_ids"].shape[1]:],
                          skip_special_tokens=True), src)

    partial_path = out_dir / "records.partial.jsonl"
    records, done_idx = [], set()
    if partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(r)
                done_idx.add(int(r["idx"]))
        print(f"[b1] RESUME: {len(records)} pairs")
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
        za_t, zs_t = diff_intervention(z_src, z_tgt, args.k_amp,
                                       args.k_sup)
        zvar = {"true": (za_t, zs_t),
                "empty": (torch.zeros_like(za_t),
                          torch.zeros_like(zs_t)),
                "random": (randomize_intervention(za_t, prng),
                           randomize_intervention(zs_t, prng))}

        rec = {"idx": int(k), "src": src, "tgt": tgt, "n_ops": n_ops,
               "outputs": {}}
        for c in conditions:
            za, zs = zvar[c]
            amp = torch.nonzero(za > 0).flatten().to(args.device)
            sup = torch.nonzero(zs > 0).flatten().to(args.device)
            if c == "true":
                modes = [(f"{mode_prefix}{v:g}", v) for v in clamp_vals]
                if args.intervention == "clamp":
                    modes.append(("clampZ", za[amp.cpu()].to(args.device)))
            elif c == "empty":
                modes = ([("recon", ctrl_val), ("raw", None)]
                         if args.intervention == "clamp"
                         else [("raw", None)])
            else:
                modes = [(f"{mode_prefix}{ctrl_val:g}", ctrl_val)]
            if args.intervention == "steer":
                W = sae.W_dec.float()                    # (d_sae, d_llm)
                dvec = (za.to(args.device).float() @ W
                        - zs.to(args.device).float() @ W)
            rec["outputs"][c] = {}
            for mname, val in modes:
                if mname == "raw":
                    hook.enabled = False       # plain model, no hook
                elif args.intervention == "clamp":
                    hook.enabled = True        # recon replacement always
                    hook.amp_idx, hook.sup_idx = amp, sup
                    hook.amp_val = val
                else:
                    hook.enabled = True
                    hook.dvec = dvec
                    hook.alpha = float(val)
                out_text = rewrite(src)
                pm = pair_metrics(out_text, src, tgt)
                rec["outputs"][c][mname] = {
                    "text": out_text, "exact": pm["exact_match"],
                    "sim_target": pm["sim_target"],
                    "copy": pm["copy_rate"],
                    "no_edit": pm["copy_rate"]}
        hook.enabled = False
        records.append(rec)
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        if (step_i + 1) % 10 == 0:
            print(f"[b1] {step_i + 1}/{len(chosen)} pairs "
                  f"({len(records)} scored)")
    pf.close()

    title = ("B1 LinguaLens-clamp" if args.intervention == "clamp"
             else "B3 steering-vector")
    lines = [f"# {title} baseline (LinguaLens)", ""]
    lines.append(f"pairs scored: {len(records)}; rewriter {args.it_model}; "
                 f"intervention={args.intervention} on "
                 f"layers[{args.sae_layer}]; value/alpha sweep "
                 f"{clamp_vals}; conditioning identical to the EF probe")
    lines += ["", "| condition | mode | exact | sim_target | copy |",
              "|---|---|---|---|---|"]
    for c in conditions:
        modes = sorted({m for r in records
                        for m in r["outputs"].get(c, {})})
        for m in modes:
            rows = [r["outputs"][c][m] for r in records
                    if m in r["outputs"].get(c, {})]
            if rows:
                lines.append(
                    f"| {c} | {m} | "
                    f"{np.mean([r['exact'] for r in rows]):.4f} | "
                    f"{np.mean([r['sim_target'] for r in rows]):.4f} | "
                    f"{np.mean([r['copy'] for r in rows]):.4f} |")
    lines += ["", "## Multi-site breakdown (condition = true)", ""]
    modes = sorted({m for r in records for m in r["outputs"]["true"]})
    lines.append("| n_ops | pairs | " +
                 " | ".join(f"{m} exact" for m in modes) + " | " +
                 " | ".join(f"{m} sim" for m in modes) + " |")
    lines.append("|---" * (2 + 2 * len(modes)) + "|")
    byb = defaultdict(list)
    for r in records:
        byb[bname(r["n_ops"])].append(r)
    for b in ("1", "2-3", "4-8", "9+"):
        rs = byb.get(b, [])
        if not rs:
            continue
        cells = [f"{np.mean([r['outputs']['true'][m]['exact'] for r in rs]):.4f}"
                 for m in modes]
        cells += [f"{np.mean([r['outputs']['true'][m]['sim_target'] for r in rs]):.4f}"
                  for m in modes]
        lines.append(f"| {b} | {len(rs)} | " + " | ".join(cells) + " |")

    report = "\n".join(lines)
    print(report)
    (out_dir / "report.md").write_text(report + "\n")
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)
    print(f"[b1] wrote {out_dir}/report.md, records.jsonl")


if __name__ == "__main__":
    main()
