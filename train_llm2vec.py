"""
Stage 1: LLM2Vec-style MNTP training.

Turn causal Gemma into a bidirectional encoder via:
  (1) bidirectional attention patch from `model._patch_attention_bidirectional`,
  (2) MNTP (Masked Next Token Prediction) on sentence-segmented Dolma.

This file mirrors the canonical LLM2Vec recipe
(https://github.com/McGill-NLP/llm2vec, `experiments/run_mntp.py`):

  * Tokens are masked with BERT-style 15% probability + 80/10/10 split
    (MASK / random / original), via HF's `DataCollatorForLanguageModeling`.
  * The forward pass goes through `GemmaForCausalLM.forward(labels=...)`
    which internally applies the standard causal-LM +1 shift:
        loss = CE(logits[..., :-1, :], labels[..., 1:])
    Combined with mask-at-i / label-at-i this means the masked token at
    position i is predicted from the hidden state at position i-1 — i.e.,
    the LLM2Vec MNTP objective (the LM head's "predict next token" mapping
    is preserved, attention is bidirectional via `_update_causal_mask`).

We also add the SAE-LEWIS-specific special tokens [INS] and [DEL] to the
tokenizer before training. Their embedding rows are jointly learned during
MNTP. [MASK] is added if missing (Gemma has no native mask token).

Output: a HF-format checkpoint at --output-dir, loaded later by
`BidirectionalLLM(<output-dir>)` from `model.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import download_dolma_shards, iter_dolma_texts, iter_sentences
from model import _patch_attention_bidirectional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--data-cache-dir", default="./dolma_cache")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--max-seq-length", type=int, default=256)
    # Canonical LLM2Vec MNTP uses BERT-style 15% / 80-10-10 masking via
    # `DataCollatorForLanguageModeling`. Probability is configurable; the
    # 80/10/10 split is baked into the HF collator and not exposed.
    p.add_argument("--mlm-probability", type=float, default=0.15,
                   help="Fraction of non-special tokens to mask per sequence "
                        "(canonical LLM2Vec default: 0.15).")
    p.add_argument("--sentence-splitter", choices=["pysbd", "nltk"], default="pysbd")
    p.add_argument("--sent-min-chars", type=int, default=16)
    p.add_argument("--sent-max-chars", type=int, default=2000)
    p.add_argument("--max-sentences-per-text", type=int, default=None,
                   help="Cap on qualifying sentences kept per source document. "
                        "None = use every sentence.")
    p.add_argument("--sentence-sample-strategy",
                   choices=["head", "random", "stride"], default="random",
                   help="Per-document sampling when --max-sentences-per-text "
                        "is set. Default 'random' to avoid the lead-bias of "
                        "'head'. No effect when --max-sentences-per-text is "
                        "None.")
    p.add_argument("--no-quality-filter", action="store_true",
                   help="Disable the looks-like-sentence heuristic filter.")

    p.add_argument("--per-device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)

    # ---- LoRA controls ----------------------------------------------------
    # Canonical LLM2Vec uses LoRA. Empirical evidence on Gemma-2B: full
    # fine-tune ended at MNTP loss ≈ 4.6 and STS-B Spearman ≈ 0.43 (no
    # better than the base LM). Switching to LoRA hits the paper's range.
    p.add_argument("--use-lora", dest="use_lora", action="store_true",
                   default=True,
                   help="Default. Canonical LLM2Vec recipe — train a LoRA "
                        "adapter on top of the frozen base, then merge it "
                        "into the saved checkpoint.")
    p.add_argument("--no-use-lora", dest="use_lora", action="store_false",
                   help="Full fine-tune (every parameter trainable). NOT "
                        "canonical; reproduce only for ablation. Adjust LR "
                        "down to ~1e-5 — the LoRA default 3e-4 will blow up.")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank (LLM2Vec paper default).")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA scaling factor (paper default).")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                   help="Modules to LoRA-fy. Gemma-2's attention + MLP linear "
                        "projections; matches the canonical LLM2Vec list.")

    # LR / max_steps defaults are tuned to the LoRA path (3e-4 / 1000 steps =
    # paper recipe). For --no-use-lora override both at the CLI: 1e-5 / 10000
    # steps is the empirically-stable full-FT range.
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="Default tuned for --use-lora (canonical LLM2Vec = "
                        "3e-4 with LoRA r=16, warmup ≈ 10%% of max_steps). "
                        "When passing --no-use-lora, override to ~1e-5 — "
                        "full-FT Gemma-2B at LoRA's LR blows up at the LR "
                        "peak (we saw 5.0 → 4.3 → 5.4 → stuck).")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Canonical LLM2Vec MNTP recipe trains for ~1k LoRA "
                        "steps on Wikipedia. Override to 10000 for full FT.")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--llm-dtype", default="bfloat16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # Resume: default ON. HF Trainer auto-detects the latest checkpoint-*
    # directory in --output-dir (model + optim + sched + RNG state, plus
    # LoRA adapter state when training a PeftModel).
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Default. Resume from the latest checkpoint-* under "
                        "--output-dir if one exists.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore any existing checkpoint-* and start fresh.")
    return p.parse_args()


class DolmaSentenceTokenStream(IterableDataset):
    """Stream Dolma → sentences → tokenized examples.

    Each yielded example is a dict with `input_ids` and `special_tokens_mask`
    — the keys expected by `DataCollatorForLanguageModeling`.
    """

    def __init__(
        self,
        shard_paths,
        tokenizer,
        max_seq_length,
        sentence_splitter: str,
        sent_min_chars: int = 16,
        sent_max_chars: int = 2000,
        max_sentences_per_text: Optional[int] = None,
        sample_strategy: str = "random",
        seed: int = 42,
        quality_filter: bool = True,
    ):
        self.shard_paths = list(shard_paths)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sentence_splitter = sentence_splitter
        self.sent_min_chars = sent_min_chars
        self.sent_max_chars = sent_max_chars
        self.max_sentences_per_text = max_sentences_per_text
        self.sample_strategy = sample_strategy
        self.seed = seed
        self.quality_filter = quality_filter

    def __iter__(self) -> Iterator[Dict]:
        worker = get_worker_info()
        shards = self.shard_paths
        worker_seed = self.seed
        if worker is not None:
            shards = shards[worker.id::worker.num_workers]
            worker_seed = self.seed + 1000 * worker.id
        text_iter = iter_dolma_texts(shards, min_chars=64)
        sent_iter = iter_sentences(
            text_iter,
            splitter=self.sentence_splitter,
            min_chars=self.sent_min_chars,
            max_chars=self.sent_max_chars,
            max_sentences_per_text=self.max_sentences_per_text,
            sample_strategy=self.sample_strategy,
            seed=worker_seed,
            quality_filter=self.quality_filter,
        )
        for sent in sent_iter:
            enc = self.tokenizer(
                sent,
                truncation=True,
                max_length=self.max_seq_length,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
            ids = enc["input_ids"]
            if len(ids) < 4:
                continue
            yield {
                "input_ids": ids,
                "special_tokens_mask": enc["special_tokens_mask"],
            }


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    # Special tokens for the SAE-LEWIS pipeline. [MASK] is added if missing
    # (Gemma has no native mask). [INS] / [DEL] are new and learned during MNTP.
    added = 0
    if tokenizer.mask_token is None:
        added += tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    added += tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[INS]", "[DEL]"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"[llm2vec] tokens: [MASK]={tokenizer.mask_token_id} "
        f"[INS]={tokenizer.convert_tokens_to_ids('[INS]')} "
        f"[DEL]={tokenizer.convert_tokens_to_ids('[DEL]')}  added={added}"
    )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.llm_dtype]
    # We use SDPA (3-5x faster than eager). For Gemma-2 the SDPA path
    # respects our bidir patch: it computes
    #   is_causal = q_len > 1 and attention_mask is None and module.is_causal
    # `_patch_attention_bidirectional` sets module.is_causal=False AND the
    # patched _update_causal_mask returns a non-None padding-only 4D mask,
    # so SDPA uses the explicit mask = bidirectional. (Canonical LLM2Vec
    # forces eager only because it uses a subclass-level override that
    # only registers an "eager" attention class. We monkey-patch the
    # existing module, so all kernels work.)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    if added > 0 or model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    # Make the inner Gemma backbone bidirectional. The outer
    # GemmaForCausalLM.forward still applies the standard causal-LM +1
    # shift (loss = CE(logits[:-1], labels[1:])), which is exactly the
    # LLM2Vec MNTP objective: predict the masked token at position i from
    # the bidirectional hidden state at position i-1.
    _patch_attention_bidirectional(model.model)
    model.config.use_cache = False

    # ---- LoRA wrapping (canonical LLM2Vec recipe) -------------------------
    # Done AFTER the bidir patch — peft only rewires the linear projections
    # (q/k/v/o/gate/up/down) and doesn't touch GemmaModel._update_causal_mask
    # or per-attention-module is_causal flags, so the patch survives the
    # PEFT wrap. The same GemmaModel instance lives at
    # peft_model.base_model.model.model after wrapping; its patched methods
    # are preserved (they're bound to the instance, not the class).
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # PEFT silently switches off gradient on the base — keep that
        # behaviour but explicitly disable cache (mirrors the line above
        # for the non-LoRA branch).
        model.base_model.model.config.use_cache = False

    shard_paths = download_dolma_shards(args.data_cache_dir, max_files=args.max_files)
    dataset = DolmaSentenceTokenStream(
        shard_paths, tokenizer,
        max_seq_length=args.max_seq_length,
        sentence_splitter=args.sentence_splitter,
        sent_min_chars=args.sent_min_chars,
        sent_max_chars=args.sent_max_chars,
        max_sentences_per_text=args.max_sentences_per_text,
        sample_strategy=args.sentence_sample_strategy,
        seed=args.seed,
        quality_filter=not args.no_quality_filter,
    )

    # Canonical LLM2Vec collator: 15% / 80-10-10 / token-level masking.
    # Labels are at the SAME positions as masked tokens (-100 elsewhere);
    # the +1 shift that turns this into MNTP is applied inside the model's
    # CausalLM forward pass.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=None,
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        report_to=[],
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        seed=args.seed,
        # Gemma ties lm_head.weight to embed_tokens.weight. safetensors
        # refuses to serialize tied tensors via the Trainer's default save
        # path, so switch to the legacy pytorch_model.bin format.
        save_safetensors=False,
    )

    # Pass the raw GemmaForCausalLM model directly: its built-in
    # `forward(input_ids, labels=labels)` returns the +1-shifted CE loss,
    # which is the MNTP objective. No wrapper that would re-compute the
    # loss in a non-shifted way.
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=dataset, data_collator=collator,
    )
    # HF Trainer auto-detects the latest checkpoint-* sub-directory under
    # output_dir when resume_from_checkpoint=True; if there is no
    # checkpoint it starts fresh, so the flag is safe even on a clean run.
    resume_arg = True if args.resume and any(
        Path(args.output_dir).glob("checkpoint-*")
    ) else False
    if resume_arg:
        print(f"[llm2vec] RESUME: continuing from latest checkpoint-* in {args.output_dir}")
    trainer.train(resume_from_checkpoint=resume_arg)

    out_dir = Path(args.output_dir)
    # Merge LoRA into the base before saving so downstream stages (corruption,
    # tagger, editor, eval) load a plain HF GemmaForCausalLM via
    # AutoModelForCausalLM.from_pretrained — no PEFT-awareness needed.
    if args.use_lora:
        print("[llm2vec] merging LoRA adapter into base model...")
        model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=False)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "llm2vec_meta.json").write_text(json.dumps({
        "base_llm": args.llm,
        "max_seq_length": args.max_seq_length,
        "mlm_probability": args.mlm_probability,
        "mntp_objective": "predict-i-from-h(i-1) (canonical LLM2Vec)",
        "vocab_size": len(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "ins_token_id": tokenizer.convert_tokens_to_ids("[INS]"),
        "del_token_id": tokenizer.convert_tokens_to_ids("[DEL]"),
        "seed": args.seed,
        "lora": ({
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": list(args.lora_target_modules),
            "merged": True,
        } if args.use_lora else None),
        # # of Dolma shards (from URL-list head) consumed by training. Used
        # by `eval_llm2vec.py` to choose held-out shards via
        # `start_index=dolma_max_files`. `None` = streamed every available
        # shard, in which case eval cannot be strictly held-out at the
        # shard level.
        "dolma_max_files": (int(args.max_files) if args.max_files is not None else None),
    }, indent=2))
    print(f"[llm2vec] saved to {out_dir}")


if __name__ == "__main__":
    main()
