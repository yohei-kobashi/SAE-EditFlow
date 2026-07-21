"""Per-feature intervention identification on the canonical pool — ALL arms.

User decision 2026-07-22: exact/FIC interventions must not be derived from
the evaluated pair (LinguaLens critique, audit §5). Instead every arm
identifies its intervention on the ~4,451-pair pool of eval_split.json:

  ef      spec(f) = mean over pool pairs of the pooled signed SAE delta
          (z_tgt − z_src, bare-sentence encode, edit-span local max-pool +
          per-layer blocklist + pool-topk — byte-identical conventions to
          eval_ef_bare). Mean, not max: no winner's curse (audit §5).
  FRC     LinguaLens PS/PN/FRC top-r (identify_features_frc scoring,
          no blocklist — their protocol), emitted in the identified-JSON
          format eval_clamp_baseline --feature-sets consumes.
  AUROC   AxBench SAE-A selection (max-pooled activation vs s1/s2 labels,
          select_features_auroc scoring), same JSON format.

ONE LLM forward per sentence serves all layers (output_hidden_states) and
all three selectors. Per-layer sidecar jsonl makes every stage resumable.
Split-half cosine of spec(f) is emitted as the stability figure (contrast
with FRC top-3 selection instability, audit §5).

Usage (GPU):
    python scripts/build_feature_specs.py --out-dir runs/feature_specs \
        --split runs/tables/eval_split.json --device cuda
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

from eval_lingualens import (                                  # noqa: E402
    edit_char_ranges, local_pool_topk)
from model import load_sae                                     # noqa: E402

# same table as run_ef_editor.sh
LAYER_CFG = {
    4:  ("layer_4/width_16k/average_l0_60/params.npz",
         "runs/blocklist_l4/blocklist.npy"),
    12: ("layer_12/width_16k/average_l0_82/params.npz",
         "runs/blocklist/blocklist.npy"),
    20: ("layer_20/width_16k/average_l0_71/params.npz",
         "runs/blocklist_l20/blocklist.npy"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="runs/feature_specs")
    p.add_argument("--split", default="runs/tables/eval_split.json")
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--layers", default="4,12,20")
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--top-store", type=int, default=128,
                   help="spec entries kept per feature (|value| top)")
    p.add_argument("--top-r", type=int, default=16,
                   help="FRC/AUROC ranking depth emitted")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    return p.parse_args()


def auroc(pos_vals, neg_vals):
    """Rank-statistic AUROC with tie averaging (AxBench selector metric)."""
    vals = [(v, 1) for v in pos_vals] + [(v, 0) for v in neg_vals]
    vals.sort(key=lambda x: x[0])
    n_pos, n_neg = len(pos_vals), len(neg_vals)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum, i = 0.0, 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j][0] == vals[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0            # 1-based average rank
        rank_sum += avg_rank * sum(1 for k in range(i, j) if vals[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    eval_idx = set(split["eval_idx"])

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    assert len(ds) == split["n_total"], \
        f"dataset size {len(ds)} != split n_total {split['n_total']}"
    pool = [k for k in range(len(ds)) if k not in eval_idx]
    print(f"[specs] pool {len(pool)} pairs (excl. {len(eval_idx)} eval), "
          f"layers {layers}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.llm_dtype]
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=dtype).to(args.device).eval()
    saes, blks = {}, {}
    for L in layers:
        sae_path, blk_path = LAYER_CFG[L]
        saes[L] = load_sae("jumprelu", args.sae_repo, sae_path
                           ).to(args.device).eval()
        blks[L] = torch.as_tensor(
            np.asarray(np.load(blk_path), dtype=np.int64))
        print(f"[specs] L{L}: SAE {sae_path}, blocklist {len(blks[L])}")

    # ---- resume-safe sidecars (one per layer) ---------------------------
    side, done = {}, {}
    for L in layers:
        sp = out_dir / f"l{L}_pairs.jsonl"
        done[L] = set()
        if sp.exists():
            with open(sp) as f:
                for line in f:
                    if line.strip():
                        done[L].add(int(json.loads(line)["idx"]))
            print(f"[specs] L{L} RESUME: {len(done[L])} pairs cached")
        side[L] = open(sp, "a")

    @torch.no_grad()
    def all_layer_z(text):
        """{layer: (offsets, z (T,d_sae))} from ONE forward."""
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=256, return_offsets_mapping=True,
                  add_special_tokens=True)
        offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
        inp = {k: v.to(args.device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        out = llm(**inp, output_hidden_states=True, use_cache=False)
        res = {}
        for L in layers:
            h = out.hidden_states[L + 1][0]      # model.py convention
            res[L] = (offsets, saes[L].encode(
                h.to(saes[L].W_enc.dtype)))
        return res

    todo = [k for k in pool
            if any(k not in done[L] for L in layers)]
    print(f"[specs] encoding {len(todo)} pairs")
    for n_done, k in enumerate(todo):
        ex = ds[int(k)]
        s, t = ex["sentence1"], ex["sentence2"]
        feat = ex.get("feature") or "?"
        zs_all = all_layer_z(s)
        zt_all = all_layer_z(t)
        s_ids = tok(s, add_special_tokens=True).input_ids
        t_ids = tok(t, add_special_tokens=True).input_ids
        opcodes = difflib.SequenceMatcher(
            None, s_ids, t_ids, autojunk=False).get_opcodes()
        for L in layers:
            if k in done[L]:
                continue
            (s_off, z_s), (t_off, z_t) = zs_all[L], zt_all[L]
            sr, tr = edit_char_ranges(opcodes, s_off, t_off)
            zp_s = local_pool_topk(z_s, s_off, sr, args.pool_topk, blks[L])
            zp_t = local_pool_topk(z_t, t_off, tr, args.pool_topk, blks[L])
            delta = (zp_t - zp_s)
            nz = torch.nonzero(delta).flatten()
            rec = {
                "idx": int(k), "feature": feat,
                "delta": {int(i): round(float(delta[i]), 5) for i in nz},
                "dnorm": round(float(delta.norm()), 5),
                # FRC acts: active anywhere, NO blocklist (their protocol)
                "act_s": torch.nonzero((z_s > 0).any(0)).flatten().tolist(),
                "act_t": torch.nonzero((z_t > 0).any(0)).flatten().tolist(),
                # AUROC: max-pooled per sentence, sparse (missing = 0)
                "max_s": {int(i): round(float(v), 5) for i, v in zip(
                    torch.nonzero(z_s.max(0).values > 0).flatten().tolist(),
                    z_s.max(0).values[z_s.max(0).values > 0].tolist())},
                "max_t": {int(i): round(float(v), 5) for i, v in zip(
                    torch.nonzero(z_t.max(0).values > 0).flatten().tolist(),
                    z_t.max(0).values[z_t.max(0).values > 0].tolist())},
            }
            side[L].write(json.dumps(rec) + "\n")
            side[L].flush()
            done[L].add(k)
        if (n_done + 1) % 200 == 0:
            print(f"[specs] {n_done + 1}/{len(todo)} pairs encoded")
    for L in layers:
        side[L].close()

    # ---- aggregation per layer ------------------------------------------
    d_sae = int(saes[layers[0]].W_enc.shape[1])
    for L in layers:
        by_ph = defaultdict(list)
        with open(out_dir / f"l{L}_pairs.jsonl") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if int(r["idx"]) not in eval_idx:
                        by_ph[r["feature"]].append(r)
        spec_out, frc_out, auroc_out = {}, {}, {}
        rep = [f"# feature specs L{L} (pool {len(pool)} pairs, "
               f"pool-topk {args.pool_topk})", "",
               "| feature | n | spec_nz | ‖mean‖ | norm_med | sh_cos |"
               " FRC#1 | AUROC#1 |", "|---|---|---|---|---|---|---|---|"]
        for ph in sorted(by_ph):
            rows = sorted(by_ph[ph], key=lambda r: int(r["idx"]))
            n = len(rows)
            acc = torch.zeros(d_sae, dtype=torch.float64)
            accA = torch.zeros(d_sae, dtype=torch.float64)
            accB = torch.zeros(d_sae, dtype=torch.float64)
            dnorms = []
            pos_count, neg_count = defaultdict(int), defaultdict(int)
            for j, r in enumerate(rows):
                for i, v in r["delta"].items():
                    acc[int(i)] += v
                    (accA if j % 2 == 0 else accB)[int(i)] += v
                dnorms.append(r["dnorm"])
                for fset, cnt in ((r["act_s"], pos_count),
                                  (r["act_t"], neg_count)):
                    for fi in set(fset):
                        cnt[fi] += 1
            mean = acc / n
            cos = float(torch.nn.functional.cosine_similarity(
                accA, accB, dim=0)) if accA.norm() > 0 and \
                accB.norm() > 0 else 0.0
            order = torch.argsort(mean.abs(), descending=True)
            keep = [int(i) for i in order[:args.top_store]
                    if abs(float(mean[i])) > 0]
            spec_out[ph] = {
                "n": n,
                "spec": {int(i): round(float(mean[i]), 5) for i in keep},
                "mean_norm": round(float(mean.norm()), 5),
                "norm_median": round(float(np.median(dnorms)), 5),
                "splithalf_cos": round(cos, 4),
            }
            # FRC (identify_features_frc scoring; s1 positive, s2 counter)
            scored = []
            for fi, pc in pos_count.items():
                ps = pc / n
                pn = 1.0 - neg_count.get(fi, 0) / n
                if ps + pn > 0:
                    scored.append((fi, 2 * ps * pn / (ps + pn)))
            scored.sort(key=lambda x: -x[1])
            frc_out[ph] = [[int(f), round(v, 4)]
                           for f, v in scored[:args.top_r]]
            # AUROC (select_features_auroc scoring) over candidate feats
            cands = set(pos_count) | set(neg_count)
            asc = []
            for fi in cands:
                pv = [r["max_s"].get(str(fi), r["max_s"].get(fi, 0.0))
                      for r in rows]
                nv = [r["max_t"].get(str(fi), r["max_t"].get(fi, 0.0))
                      for r in rows]
                asc.append((fi, auroc(pv, nv)))
            asc.sort(key=lambda x: -x[1])
            auroc_out[ph] = [[int(f), round(v, 4)]
                             for f, v in asc[:args.top_r]]
            rep.append(
                f"| {ph} | {n} | {len(keep)} "
                f"| {spec_out[ph]['mean_norm']:.3f} "
                f"| {spec_out[ph]['norm_median']:.3f} | {cos:.3f} "
                f"| {frc_out[ph][0][0] if frc_out[ph] else '-'} "
                f"| {auroc_out[ph][0][0] if auroc_out[ph] else '-'} |")
        cs = [v["splithalf_cos"] for v in spec_out.values()]
        rep += ["", f"mean split-half cosine of spec: "
                f"{sum(cs) / len(cs):.4f} (n={len(cs)} features)"]
        (out_dir / f"l{L}_spec.json").write_text(json.dumps(spec_out))
        (out_dir / f"l{L}_frc_r{args.top_r}.json").write_text(
            json.dumps(frc_out))
        (out_dir / f"l{L}_auroc_r{args.top_r}.json").write_text(
            json.dumps(auroc_out))
        # native-protocol slices (consumers use the whole list):
        # LinguaLens intervenes on FRC top-3, AxBench SAE-A on AUROC top-1
        (out_dir / f"l{L}_frc_r3.json").write_text(json.dumps(
            {ph: lst[:3] for ph, lst in frc_out.items()}))
        (out_dir / f"l{L}_auroc_r1.json").write_text(json.dumps(
            {ph: lst[:1] for ph, lst in auroc_out.items()}))
        (out_dir / f"l{L}_report.md").write_text("\n".join(rep) + "\n")
        print(f"[specs] L{L}: {len(spec_out)} features, "
              f"mean sh-cos {sum(cs) / len(cs):.4f}")
    print("==================== BUILD-SPECS-DONE ====================")


if __name__ == "__main__":
    main()
