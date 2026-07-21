# 公式実装忠実性の監査(2026-07-18、EF_LM_LOSS_PLAN §2)

対象: THU-KEG/LinguaLens(+OpenSAE)、stanfordnlp/axbench の公式コードと
我々の再現・移植コードの逐語照合。

## 1. LinguaLens(lingualens/intervener.py + OpenSAE)

| 項目 | 公式 | 我々 | 判定 |
|---|---|---|---|
| 介入mode | set(enhancement=10.0 / ablation=0.0)、prompt_only=False | 同 | ✅ |
| set意味論 | activeな対象latentは値で上書き、**非activeはminスロット強制置換**(OpenSAE `_apply_intervention_add_or_set`) | SaeClampHook 同実装 | ✅ |
| control | multiply×1.0(intervention_indices付き)= SAE再構成パススルー | `recon`モード 同 | ✅ |
| 生成 | 素プロンプト(chat templateなし)、temperature=1.0、do_sample、max_new_tokens=100 | 同 | ✅ |
| 試行数 | repo既定 num_generations=10(batch既定5)。論文は50 | n=5採用(ユーザー決定)は**repo batch既定と一致** | ✅ |
| **judge対象テキスト** | `tokenizer.decode(generated_ids[0])` = **プロンプト込み全文** | v1は継続のみをdecode | 🔴 **不一致 → 修正済み**(eval_ll_repro_gen.py、2026-07-18) |
| judge | 論文プロトコル(repoに判定コードなし)、**GPT-4o** | gpt-4o | ✅ |
| FIC式 | E_abl=(Pt−Pb)/Pt、E_enh=(Pt−Pb)/(1−Pb)、負値はw=0.5で符号反転算入、調和平均(App. E.2) | judge_ll_repro.py(Table 2検算済み) | ✅ |
| FIC原実験の範囲 | **5 feature × 手書きプロンプト1本(App. D.1)× リサンプル** — LinguaLens-Data不使用 | 全feature×実ペア版は「指標のデータセット拡張」と明記する | 📌 論文での書き分け必須 |

## 2. AxBench steering(axbench/models/interventions.py)

```python
# AdditionIntervention.forward (verbatim):
steering_vec = subspaces["max_act"] * subspaces["mag"] * self.proj.weight[subspaces["idx"]]
output = base + steering_vec   # 全位置にブロードキャスト
```

- **h + factor(mag) × max_act × W_dec[latent]、全位置** — 我々の再現
  (eval_axbench_repro_gen.py AdditionHook)と一致 ✅
- 生成: temperature=1.0、do_sample、eval_output_length=128 ✅
- A2腕(LinguaLens-Data移植)の設計: この加算機構を事例レベル仕様の
  dvec に適用(h + α·dvec 相当)。強度規約の対応は α ↔ factor×max_act。

## 3. AxBench prompting(models/prompt.py + utils/dataset.py)

公式の組み立て(逐語確認):
1. **steering promptをgpt-4o-miniに生成させる**(meta-prompt
   T_GENERATE_PREPEND_STEERING_PROMPT: "Direct the model to include
   content related to {CONCEPT} ... even if it doesn't directly answer
   the question")
2. `steered_input = f"{steering_prompt}\n\nQuestion: {instruction}"`
3. chat template適用 → temperature=1.0、128トークン生成

**A3腕(LinguaLens-Data移植)の設計案(要ユーザー確認)**:
- enhancement: 彼らのmeta-promptそのまま、CONCEPT=featureの用語+gloss。
  `steered_input = f"{steering_prompt}\n\nQuestion: {src文}"` + chat template
- ablation: 彼らのrepoに抑制用テンプレートが無いため、meta-promptの
  Objective行を "avoid content related to {CONCEPT}" に置換(最小改変、
  論文に明記)
- 生成は彼らの既定(temp 1.0、128tok)

## 4. 修正済み事項(このコミットまで)

- eval_ll_repro_gen.py: judge対象をプロンプト込み全文に(§1の🔴)
- judge_axbench_repro.py: parse_ratingを公式パーサ等価に(gpt-4o-miniは
  "Rating: 0" と裸で回答 — [[x]]必須の旧regexが全ゼロの根本原因だった)
- train_ef_editor.py: max_steps境界のループ脱出ckpt保存(probe100ゲートが
  無言でスキップされていた)

## 5. LinguaLens同定の同定/評価データ分離 — 深掘り(2026-07-21、ユーザー指摘)

問い: 「特定されたactivationsが本当にfeatureに対応したものなのか、
識別データに依存したものなのか」をLinguaLensの手続きは区別できるか。

**一次資料の確定事項(arXiv 2502.20344v2 + 公式repo + OpenSAE README)**:

| 項目 | 事実 | 含意 |
|---|---|---|
| 同定データ | 現象あたり全50ペアでPS/PN/FRC。split記述なし | in-sample選択 |
| 候補数 | OpenSAE 262,144 latent("64x larger than the hidden size") | n=50の統計で26万候補からmax選択 = 勝者の呪いの典型構図 |
| 検証 | "the activation distributions of the top-10 ranked vectors are passed to an LLM agent" — **同一データの活性分布** | 独立検証ではない |
| held-out / CV / 多重比較補正 | 論文中に記述なし | データ依存性は測定されていない |
| 介入評価 | 手書きプロンプト自由生成(App. D.1、§1の通りLinguaLens-Data不使用)、Table 2は5 featureのみ | データ再利用はないが、同定の因果的検証は5/145 featureに限定。しかも成功は「その特徴が現象と相関する」ことしか示さず、**選ばれた特徴集合が安定か**は問えない |

**答え: 区別できない。** 同定はin-sample、検証も同一データ、介入は
5 featureの存在証明のみ。ユーザーの指摘(同定用と評価用のデータは
分けるべき)はLinguaLens原法に対する正当な方法論的批判として成立する。

**我々の実測(Gemma Scope 16k再現、identify_features_frcのactsキャッシュで
split-half、99現象×20反復、scripts/analyze_frc_splithalf.py、
runs/tables/frc_splithalf_l{4,12,20}.md)**:

| 層 | in@1 | out@1 | ovl@1 | in@3 | out@3 | ovl@3 | ovl@10 |
|---|---|---|---|---|---|---|---|
| L4 | 0.902 | 0.861 | 0.42 | 0.869 | 0.828 | 0.55 | 0.60 |
| L12 | 0.898 | 0.846 | 0.36 | 0.864 | 0.810 | 0.49 | 0.49 |
| L20 | 0.883 | 0.840 | 0.43 | 0.850 | 0.802 | 0.52 | 0.51 |

- **FRC値は汎化する**(縮小0.04-0.06): 選ばれた特徴は偶然の産物ではなく
  現象と真に相関している — 存在主張は支持される。
- **選択の同一性はデータ依存**: half入れ替えでtop-1特徴の一致は36-43%、
  top-3でも半分入れ替わる。近傍に同程度のFRCを持つ特徴が多数あり、
  「どれを選ぶか」は標本ノイズが決める。LinguaLensのtop-3介入・top-10
  LLM検証は、この不安定な集合の上に立っている(候補26万の彼らの設定では
  16kの我々よりさらに不安定なはず)。
- **両立の機構(2026-07-22追記)**: 全識別データでのtop-64 FRC値
  (identified_l12_16k_r64.json)より、#1と#2のFRC差は中央値0.022
  (45/99現象で<0.02)、top-1から0.05以内の特徴は中央値2本・平均4.1本。
  一方half(n≈23)でのPS/PN推定の二項SEは≈0.074 — **順位を決める差(~0.02)
  が推定ノイズ(~0.07)の1/3**。よって「どれが#1か」は標本が決めるが、
  同点クラスタのどれが選ばれても真のFRCは~0.9級なのでout-of-sample値は
  保持される。クラスタの実体はSAEのfeature splitting(1現象の相関する
  表層手がかりが複数latentに分裂)で、B-2安定核分析(核∩FRC3=8/22)とも
  整合。
- 我々の主評価(ペア由来spec)はこの選択段階を持たないため本問題の圏外。
  FRC同定を使う副実験(P-B/P-J/e2e)には識別/評価ペア分離ガードを実装
  済みだが、除外レシピの不整合(overlap 48/500)が別途要修正(04参照)。
