# LinguaLens minimal-pair token-diff distribution
Dataset: `THU-KEG/LinguaLens-Data` (split=train, total 7251 pairs).  Tokenizer: `google/gemma-2-2b`.  Sample size: 4951 (seed=42).
Three token-level edit metrics (all `tok_edit ≥ editor_ops ≥ n_hunks`):
* **`tok_edit`** — classical Levenshtein, insert/delete/substitute cost 1. The minimum number of single-token edits. Over-counts INS (a contiguous block insertion of K tokens scores K, not 1).
* **`editor_ops`** — editor inference cost: REPL/DEL hunks contribute one marker per token, INS hunks contribute one marker per gap (length head emits the rest). What the editor actually has to emit at inference time.
* **`n_hunks`** — number of contiguous non-equal diff blocks. Corresponds 1-to-1 to SAE-LEWIS's `N_total` (one hunk = one op). Use this to compare against the corruption pipeline's bucket weights.
* `set_diff` = `|set(tok1) △ set(tok2)|` — order-insensitive, dedup-counted token set symmetric difference.

## Overall

| metric | mean | p05 | p25 | p50 | p75 | p90 | p95 | p99 |
|--------|------|-----|-----|-----|-----|-----|-----|-----|
| `len(tok1)`     | 9.14 | 6.0 | 7.0 | 9.0 | 10.0 | 12.0 | 14.0 | 21.0 |
| `len(tok2)`     | 8.42 | 5.0 | 6.0 | 8.0 | 10.0 | 12.0 | 13.0 | 21.0 |
| `|Δlen|`        | -    | 0.0 | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 |
| `tok_edit`      | 3.20 | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 | 7.0 | 9.0 |
| `editor_ops`    | 2.68 | 1.0 | 1.0 | 2.0 | 4.0 | 5.0 | 6.0 | 8.0 |
| `n_hunks`       | 1.36 | 1.0 | 1.0 | 1.0 | 2.0 | 2.0 | 2.0 | 3.0 |
| `set_diff`      | 4.05 | 1.0 | 2.0 | 4.0 | 5.0 | 7.0 | 8.0 | 11.0 |

**Op-type histogram (overall)** (hunk counts across all pairs in the slice; compare ratio against the corruption pipeline's default `(REPL=0.55, INS=0.25, DEL=0.20)`):

| op | total hunks | ratio |
|----|-------------|-------|
| REPL | 4705 | 0.700 |
| INS | 1220 | 0.182 |
| DEL | 794 | 0.118 |

**tok_edit histogram (overall):**

| value | count | % | cum % |
|-------|-------|---|-------|
| 0 | 36 | 0.7% | 0.7% |
| 1 | 1118 | 22.6% | 23.3% |
| 2 | 1074 | 21.7% | 45.0% |
| 3 | 761 | 15.4% | 60.4% |
| 4 | 740 | 14.9% | 75.3% |
| 5 | 578 | 11.7% | 87.0% |
| 6 | 320 | 6.5% | 93.5% |
| 7 | 170 | 3.4% | 96.9% |
| 8 | 73 | 1.5% | 98.4% |
| 9 | 58 | 1.2% | 99.5% |
| 10 | 13 | 0.3% | 99.8% |
| 11 | 5 | 0.1% | 99.9% |
| 12 | 3 | 0.1% | 100.0% |
| 13 | 1 | 0.0% | 100.0% |
| 14 | 1 | 0.0% | 100.0% |
| 15 | 0 | 0.0% | 100.0% |
| 16 | 0 | 0.0% | 100.0% |
| 17 | 0 | 0.0% | 100.0% |
| 18 | 0 | 0.0% | 100.0% |
| 19 | 0 | 0.0% | 100.0% |
| 20 | 0 | 0.0% | 100.0% |

**editor_ops histogram (overall):**

| value | count | % | cum % |
|-------|-------|---|-------|
| 0 | 36 | 0.7% | 0.7% |
| 1 | 1532 | 30.9% | 31.7% |
| 2 | 1237 | 25.0% | 56.7% |
| 3 | 729 | 14.7% | 71.4% |
| 4 | 638 | 12.9% | 84.3% |
| 5 | 428 | 8.6% | 92.9% |
| 6 | 199 | 4.0% | 96.9% |
| 7 | 92 | 1.9% | 98.8% |
| 8 | 20 | 0.4% | 99.2% |
| 9 | 28 | 0.6% | 99.8% |
| 10 | 5 | 0.1% | 99.9% |
| 11 | 3 | 0.1% | 99.9% |
| 12 | 1 | 0.0% | 99.9% |
| 13 | 1 | 0.0% | 100.0% |
| 14 | 2 | 0.0% | 100.0% |
| 15 | 0 | 0.0% | 100.0% |
| 16 | 0 | 0.0% | 100.0% |
| 17 | 0 | 0.0% | 100.0% |
| 18 | 0 | 0.0% | 100.0% |
| 19 | 0 | 0.0% | 100.0% |
| 20 | 0 | 0.0% | 100.0% |

**n_hunks histogram (overall) -- compare to SAE-LEWIS bucket weights:**

| value | count | % | cum % |
|-------|-------|---|-------|
| 0 | 36 | 0.7% | 0.7% |
| 1 | 3299 | 66.6% | 67.4% |
| 2 | 1433 | 28.9% | 96.3% |
| 3 | 178 | 3.6% | 99.9% |
| 4 | 5 | 0.1% | 100.0% |
| 5 | 0 | 0.0% | 100.0% |
| 6 | 0 | 0.0% | 100.0% |
| 7 | 0 | 0.0% | 100.0% |
| 8 | 0 | 0.0% | 100.0% |
| 9 | 0 | 0.0% | 100.0% |
| 10 | 0 | 0.0% | 100.0% |
| 11-50 | 0 | 0.0% | 100.0% |

**set_diff histogram (overall):**

| value | count | % | cum % |
|-------|-------|---|-------|
| 0 | 119 | 2.4% | 2.4% |
| 1 | 333 | 6.7% | 9.1% |
| 2 | 1079 | 21.8% | 30.9% |
| 3 | 777 | 15.7% | 46.6% |
| 4 | 912 | 18.4% | 65.0% |
| 5 | 497 | 10.0% | 75.1% |
| 6 | 442 | 8.9% | 84.0% |
| 7 | 344 | 6.9% | 91.0% |
| 8 | 235 | 4.7% | 95.7% |
| 9 | 101 | 2.0% | 97.7% |
| 10 | 52 | 1.1% | 98.8% |
| 11 | 34 | 0.7% | 99.5% |
| 12 | 15 | 0.3% | 99.8% |
| 13 | 4 | 0.1% | 99.9% |
| 14 | 4 | 0.1% | 99.9% |
| 15 | 3 | 0.1% | 100.0% |
| 16 | 0 | 0.0% | 100.0% |
| 17 | 0 | 0.0% | 100.0% |
| 18 | 0 | 0.0% | 100.0% |
| 19 | 0 | 0.0% | 100.0% |
| 20 | 0 | 0.0% | 100.0% |

## Per language

### English  (n = 4951)

| metric | mean | p05 | p25 | p50 | p75 | p90 | p95 | p99 |
|--------|------|-----|-----|-----|-----|-----|-----|-----|
| `len(tok1)`     | 9.14 | 6.0 | 7.0 | 9.0 | 10.0 | 12.0 | 14.0 | 21.0 |
| `len(tok2)`     | 8.42 | 5.0 | 6.0 | 8.0 | 10.0 | 12.0 | 13.0 | 21.0 |
| `|Δlen|`        | -    | 0.0 | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 |
| `tok_edit`      | 3.20 | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 | 7.0 | 9.0 |
| `editor_ops`    | 2.68 | 1.0 | 1.0 | 2.0 | 4.0 | 5.0 | 6.0 | 8.0 |
| `n_hunks`       | 1.36 | 1.0 | 1.0 | 1.0 | 2.0 | 2.0 | 2.0 | 3.0 |
| `set_diff`      | 4.05 | 1.0 | 2.0 | 4.0 | 5.0 | 7.0 | 8.0 | 11.0 |

**English: Op-type histogram** (hunk counts across all pairs in the slice; compare ratio against the corruption pipeline's default `(REPL=0.55, INS=0.25, DEL=0.20)`):

| op | total hunks | ratio |
|----|-------------|-------|
| REPL | 4705 | 0.700 |
| INS | 1220 | 0.182 |
| DEL | 794 | 0.118 |

**English: tok_edit histogram:**

| value | count | % | cum % |
|-------|-------|---|-------|
| 0 | 36 | 0.7% | 0.7% |
| 1 | 1118 | 22.6% | 23.3% |
| 2 | 1074 | 21.7% | 45.0% |
| 3 | 761 | 15.4% | 60.4% |
| 4 | 740 | 14.9% | 75.3% |
| 5 | 578 | 11.7% | 87.0% |
| 6 | 320 | 6.5% | 93.5% |
| 7 | 170 | 3.4% | 96.9% |
| 8 | 73 | 1.5% | 98.4% |
| 9 | 58 | 1.2% | 99.5% |
| 10 | 13 | 0.3% | 99.8% |
| 11 | 5 | 0.1% | 99.9% |
| 12 | 3 | 0.1% | 100.0% |
| 13 | 1 | 0.0% | 100.0% |
| 14 | 1 | 0.0% | 100.0% |
| 15 | 0 | 0.0% | 100.0% |
| 16 | 0 | 0.0% | 100.0% |
| 17 | 0 | 0.0% | 100.0% |
| 18 | 0 | 0.0% | 100.0% |
| 19 | 0 | 0.0% | 100.0% |
| 20 | 0 | 0.0% | 100.0% |

**English: editor_ops histogram:**

| value | count | % | cum % |
|-------|-------|---|-------|
| 0 | 36 | 0.7% | 0.7% |
| 1 | 1532 | 30.9% | 31.7% |
| 2 | 1237 | 25.0% | 56.7% |
| 3 | 729 | 14.7% | 71.4% |
| 4 | 638 | 12.9% | 84.3% |
| 5 | 428 | 8.6% | 92.9% |
| 6 | 199 | 4.0% | 96.9% |
| 7 | 92 | 1.9% | 98.8% |
| 8 | 20 | 0.4% | 99.2% |
| 9 | 28 | 0.6% | 99.8% |
| 10 | 5 | 0.1% | 99.9% |
| 11 | 3 | 0.1% | 99.9% |
| 12 | 1 | 0.0% | 99.9% |
| 13 | 1 | 0.0% | 100.0% |
| 14 | 2 | 0.0% | 100.0% |
| 15 | 0 | 0.0% | 100.0% |
| 16 | 0 | 0.0% | 100.0% |
| 17 | 0 | 0.0% | 100.0% |
| 18 | 0 | 0.0% | 100.0% |
| 19 | 0 | 0.0% | 100.0% |
| 20 | 0 | 0.0% | 100.0% |

**English: n_hunks histogram:**

| value | count | % | cum % |
|-------|-------|---|-------|
| 0 | 36 | 0.7% | 0.7% |
| 1 | 3299 | 66.6% | 67.4% |
| 2 | 1433 | 28.9% | 96.3% |
| 3 | 178 | 3.6% | 99.9% |
| 4 | 5 | 0.1% | 100.0% |
| 5 | 0 | 0.0% | 100.0% |
| 6 | 0 | 0.0% | 100.0% |
| 7 | 0 | 0.0% | 100.0% |
| 8 | 0 | 0.0% | 100.0% |
| 9 | 0 | 0.0% | 100.0% |
| 10 | 0 | 0.0% | 100.0% |
| 11-50 | 0 | 0.0% | 100.0% |

## Top features by n_hunks p50 (largest first)
| feature | n | n_hunks p50 | editor_ops p50 | tok_edit p50 | REPL% | INS% | DEL% | mean len1 | mean len2 |
|---------|---|-------------|----------------|--------------|-------|------|------|-----------|-----------|
| interrogative | 50 | 3.0 | 4.0 | 4.0 | 94% | 3% | 3% | 8.1 | 8.2 |
| first_conditional | 50 | 3.0 | 4.0 | 4.0 | 46% | 22% | 32% | 13.3 | 12.3 |
| non_defining_relative_clauses | 50 | 2.0 | 5.0 | 8.0 | 21% | 49% | 30% | 13.1 | 12.1 |
| active_verbs | 51 | 2.0 | 7.0 | 7.0 | 50% | 5% | 45% | 10.0 | 12.0 |
| tag_questions | 50 | 2.0 | 4.5 | 7.0 | 60% | 38% | 1% | 10.7 | 5.6 |
| cleft_sentences | 50 | 2.0 | 3.0 | 6.5 | 48% | 44% | 8% | 10.0 | 7.6 |
| extraposition | 50 | 2.0 | 6.0 | 6.0 | 50% | 8% | 42% | 9.5 | 8.5 |
| relative_clauses | 50 | 2.0 | 2.5 | 6.0 | 40% | 36% | 24% | 10.1 | 9.3 |
| object_expletives | 50 | 2.0 | 4.0 | 6.0 | 64% | 26% | 11% | 11.6 | 9.2 |
| indirect_speech | 50 | 2.0 | 3.0 | 5.0 | 98% | 1% | 1% | 9.2 | 10.7 |
| direct_object | 50 | 2.0 | 4.0 | 5.0 | 80% | 4% | 16% | 6.3 | 8.3 |
| optative | 50 | 2.0 | 5.0 | 5.0 | 100% | 0% | 0% | 7.9 | 6.0 |
| subject_verb_inversion | 50 | 2.0 | 4.0 | 5.0 | 65% | 15% | 21% | 10.4 | 10.0 |
| politeness | 50 | 2.0 | 5.0 | 5.0 | 100% | 0% | 0% | 8.6 | 6.0 |
| passive_voice | 50 | 2.0 | 4.0 | 5.0 | 12% | 49% | 39% | 10.9 | 8.9 |
| echo_questions | 50 | 2.0 | 4.0 | 5.0 | 66% | 30% | 4% | 15.9 | 13.6 |
| commisive | 50 | 2.0 | 5.0 | 5.0 | 86% | 9% | 4% | 10.1 | 7.4 |
| clausal_subjects | 50 | 2.0 | 3.0 | 5.0 | 58% | 25% | 17% | 8.8 | 8.1 |
| expletive | 50 | 2.0 | 2.5 | 4.0 | 61% | 25% | 14% | 8.1 | 6.8 |
| subject_auxiliary_inversion | 50 | 2.0 | 2.0 | 4.0 | 67% | 19% | 14% | 10.4 | 10.1 |
| spatial_or_directional_prefix | 50 | 2.0 | 4.0 | 4.0 | 30% | 30% | 40% | 8.7 | 9.2 |
| temporal_prefix | 50 | 2.0 | 4.0 | 4.0 | 31% | 32% | 38% | 8.9 | 9.9 |
| s_genitive | 50 | 2.0 | 4.0 | 4.0 | 25% | 38% | 38% | 8.6 | 7.8 |
| existential | 50 | 2.0 | 4.0 | 4.0 | 65% | 19% | 16% | 9.5 | 8.1 |
| static_dynamic | 50 | 2.0 | 3.0 | 3.0 | 79% | 20% | 1% | 8.1 | 6.9 |

## Bottom features by n_hunks p50 (smallest first)
| feature | n | n_hunks p50 | editor_ops p50 | tok_edit p50 | REPL% | INS% | DEL% | mean len1 | mean len2 |
|---------|---|-------------|----------------|--------------|-------|------|------|-----------|-----------|
| intensifiers | 50 | 1.0 | 1.0 | 1.0 | 0% | 100% | 0% | 10.5 | 9.4 |
| past | 50 | 1.0 | 1.0 | 1.0 | 100% | 0% | 0% | 6.5 | 6.5 |
| comparative | 50 | 1.0 | 1.0 | 1.0 | 66% | 34% | 0% | 9.1 | 8.5 |
| past_tense | 50 | 1.0 | 1.0 | 1.0 | 98% | 2% | 0% | 7.5 | 7.5 |
| adjectival_suffix | 50 | 1.0 | 1.0 | 1.0 | 96% | 4% | 0% | 7.3 | 7.2 |
| past_participle | 50 | 1.0 | 1.0 | 1.0 | 100% | 0% | 0% | 7.8 | 7.8 |
| adverbial_suffix | 50 | 1.0 | 1.0 | 1.0 | 92% | 8% | 0% | 7.3 | 7.0 |
| nominal_suffix | 50 | 1.0 | 1.0 | 1.0 | 92% | 8% | 0% | 7.7 | 7.6 |
| third_person_singular | 50 | 1.0 | 1.0 | 1.0 | 96% | 4% | 0% | 7.4 | 7.3 |
| past_tense_irregular | 50 | 1.0 | 1.0 | 1.0 | 100% | 0% | 0% | 9.6 | 9.6 |
| anaphor | 50 | 1.0 | 1.0 | 1.0 | 73% | 27% | 0% | 8.7 | 8.3 |
| given_known | 50 | 1.0 | 1.0 | 1.0 | 30% | 70% | 0% | 8.4 | 7.6 |
| quantifier | 50 | 1.0 | 1.0 | 1.0 | 23% | 77% | 0% | 9.1 | 7.8 |
| verbal_suffix | 50 | 1.0 | 1.0 | 1.0 | 92% | 8% | 0% | 10.1 | 9.9 |
| non_synecdoche_metonymy | 50 | 1.0 | 1.0 | 1.0 | 94% | 0% | 6% | 7.4 | 7.6 |
| agentive_suffix | 50 | 1.0 | 1.0 | 1.0 | 100% | 0% | 0% | 8.2 | 8.2 |
| existential_quantifiers | 50 | 1.0 | 1.0 | 1.0 | 35% | 65% | 0% | 7.2 | 5.8 |
| negation_prefix | 50 | 1.0 | 1.0 | 1.0 | 100% | 0% | 0% | 7.0 | 6.6 |
| noun_plural | 50 | 1.0 | 1.0 | 1.0 | 100% | 0% | 0% | 7.8 | 7.8 |
| past_participle_irregular | 50 | 1.0 | 1.0 | 1.0 | 100% | 0% | 0% | 8.2 | 8.2 |
| punctual_durative | 50 | 1.0 | 1.0 | 1.0 | 39% | 51% | 10% | 6.3 | 5.6 |
| superlative | 50 | 1.0 | 1.0 | 1.0 | 64% | 36% | 0% | 9.1 | 8.6 |
| person | 50 | 1.0 | 1.0 | 1.5 | 100% | 0% | 0% | 19.6 | 20.1 |
| temporal | 50 | 1.0 | 2.0 | 2.0 | 96% | 0% | 4% | 8.3 | 10.1 |
| copular_be | 50 | 1.0 | 2.0 | 2.0 | 75% | 21% | 4% | 6.2 | 6.1 |

## Notes

* Gemma is a subword (SentencePiece) tokenizer.  For Chinese the Gemma tokenizer falls through to roughly character-level segmentation, which inflates token counts and edit distance relative to a Chinese-native tokenizer.  Treat the English and Chinese rows separately when comparing to the SAE-LEWIS compound-corruption N distribution (which is currently calibrated on English Dolma only).
* The edit distance is computed over the FULL tokenization including BOS but excluding nothing else — i.e. exactly the number of token-level operations the SAE-LEWIS editor would have to emit if `sentence1` were corrupted into `sentence2`.
* Re-run with `--sample-size 7251` to use the full dataset, or `--language English` / `--language Chinese` to slice.
