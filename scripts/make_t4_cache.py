"""T4 (user-approved 2026-07-23): LinguaLens-train adaptation cache.

Builds a corruption-cache-format shard set from the TRAIN section of
eval_split v2 (all pairs minus eval500 minus dev500): each pair yields
two rows teaching "feature-level ctx spec -> this pair's edit":

  ablation row     x0 = sentence1 (feature present) -> x1 = sentence2
                   conditioning delta = +spec  (spec = mean(z_s2 - z_s1))
  enhancement row  x0 = sentence2 -> x1 = sentence1, delta = -spec

The conditioning is stored so that the trainer's convention
(delta = z_X - z_X_prime, with x0 = x_prime, x1 = x) reproduces the
EVAL-time operating point: v is rescaled to the pool per-pair norm-median
and multiplied by the eval scale (3.5), exactly like eval_ef_bare
--feature-spec --fspec-scale.  Zero-shot is given up by design — the
paper reports zero-shot and pool-adapted rows separately.

Usage:
    python scripts/make_t4_cache.py \
        --spec runs/feature_specs/l12_specctx.json \
        --split runs/tables/eval_split.json \
        --out runs/prod_gemma_v4/t4_ctx_l12 --scale 3.5
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=3.5)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--tokenizer", default="runs/mcgill_gemma_repro_3k/final")
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--meta-from", required=True,
                   help="existing cache meta.json to inherit trainer "
                        "fields from (d_sae etc.)")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    spec = json.loads(Path(args.spec).read_text())
    sp = json.loads(Path(args.split).read_text())
    banned = set(sp["eval_idx"]) | set(sp.get("dev_idx", []))

    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(args.meta_from).read_text())
    meta.update({
        "kind": "t4_lingualens_train_adaptation",
        "t4_spec": args.spec, "t4_scale": args.scale,
        "t4_split": args.split, "t4_note": "train section only "
        "(eval500 + dev500 excluded); 2 rows per pair (abl/enh)"})
    (out / "meta.json").write_text(json.dumps(meta))

    def sparse(v_items):
        return [{"f": int(f), "v": round(float(x), 5)}
                for f, x in v_items if x > 0]

    rows, n_skip = [], 0
    for k in range(len(ds)):
        if k in banned:
            continue
        ex = ds[int(k)]
        fs = spec.get(ex.get("feature") or "?")
        if fs is None:
            n_skip += 1
            continue
        mult = args.scale * (
            fs["norm_median"] / fs["mean_norm"] if fs["mean_norm"] > 0
            else 1.0)
        v = {int(i): float(x) * mult for i, x in fs["spec"].items()}
        s1 = tok(ex["sentence1"], add_special_tokens=True).input_ids
        s2 = tok(ex["sentence2"], add_special_tokens=True).input_ids
        if max(len(s1), len(s2)) > 176:
            n_skip += 1
            continue
        pos = [(i, x) for i, x in v.items() if x > 0]
        neg = [(i, -x) for i, x in v.items() if x < 0]
        # ablation: x0=s1 -> x1=s2, delta(zX - zX') = +v
        rows.append({"x_prime_token_ids": s1, "x_token_ids": s2,
                     "z_X_topk": sparse(pos), "z_X_prime_topk": sparse(neg),
                     "bucket": "t4_abl", "feature": ex.get("feature")})
        # enhancement: x0=s2 -> x1=s1, delta = -v
        rows.append({"x_prime_token_ids": s2, "x_token_ids": s1,
                     "z_X_topk": sparse(neg), "z_X_prime_topk": sparse(pos),
                     "bucket": "t4_enh", "feature": ex.get("feature")})

    n = 0
    for i0 in range(0, len(rows), args.shard_size):
        pth = out / f"shard-t4-{i0 // args.shard_size:05d}.jsonl.gz"
        with gzip.open(str(pth) + ".tmp", "wt", encoding="utf-8") as w:
            for r in rows[i0:i0 + args.shard_size]:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        Path(str(pth) + ".tmp").rename(pth)
        n += len(rows[i0:i0 + args.shard_size])
    print(f"[t4cache] {n} rows ({n // 2} pairs x 2 dirs), "
          f"{n_skip} skipped -> {out}")
    print("T4-CACHE-BUILT")


if __name__ == "__main__":
    main()
