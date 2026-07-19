"""A3 arm prep (AxBench prompt-steering port, audit §3): generate the
per-feature steering prompts with gpt-4o-mini, exactly as AxBench does
(utils/prompt_utils.get_steering_prompts + T_GENERATE_PREPEND_STEERING_
PROMPT, verbatim). Ablation direction uses the minimal 'avoid'
modification of the Objective line (their repo has no suppression
template — documented adaptation, user-approved defaults 2026-07-19).

Pure stdlib + API -> runs on prepost:
    cd ~/SAE-LEWIS && python3 scripts/gen_a3_prompts.py \
        --features runs/frc/identified_l12_16k_r3.json \
        --output-dir runs/a3_prompts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_paper_metrics import Judge  # noqa: E402

# AxBench templates/prompt_templates.py T_GENERATE_PREPEND_STEERING_PROMPT
# — verbatim.
T_ENH = """Generate a prompt to guide a language \
model in producing responses.

Objective:
Direct the model to include content related to {CONCEPT} (the concept) in its responses.
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.

Return only the final prompt without any additional text."""

# minimal 'avoid' modification of the Objective line (ablation direction)
T_ABL = """Generate a prompt to guide a language \
model in producing responses.

Objective:
Direct the model to avoid any content related to {CONCEPT} (the concept) in its responses.
Ensure the responses never reference this concept, even if the question seems to call for it.
Optionally, provide in-context examples to reinforce this behavior.

Return only the final prompt without any additional text."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True,
                   help="json whose keys are the feature names "
                        "(e.g. runs/frc/identified_l12_16k_r3.json)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lm-model", default="gpt-4o-mini")   # their default
    args = p.parse_args()

    feats = sorted(json.load(open(args.features)).keys())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            try:
                c = json.loads(line)
                cache[c["key"]] = c["prompt"]
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"[a3] RESUME: {len(cache)} cached")
    cfh = open(cache_path, "a")
    judge = Judge(args.lm_model, max_tokens=400)

    n_new = 0
    for f in feats:
        concept = f.replace("_", " ")
        for d, tpl in (("enh", T_ENH), ("abl", T_ABL)):
            key = f"{f}|{d}"
            if key in cache:
                continue
            prompt = judge(tpl.format(CONCEPT=concept)).strip()
            cache[key] = prompt
            cfh.write(json.dumps({"key": key, "prompt": prompt},
                                 ensure_ascii=False) + "\n")
            cfh.flush()
            n_new += 1
            if n_new % 20 == 0:
                print(f"[a3] +{n_new}")
    cfh.close()

    result = {}
    for f in feats:
        result[f] = {"enh": cache.get(f"{f}|enh", ""),
                     "abl": cache.get(f"{f}|abl", "")}
    (out / "steering_prompts.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1))
    print(f"[a3] wrote {out}/steering_prompts.json "
          f"({len(result)} features x 2 directions)")


if __name__ == "__main__":
    main()
