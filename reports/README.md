# reports/ — 論文執筆用資料(日本語執筆の下敷き)

作成 2026-07-17。ユーザが introduction / previous works / method / experiment を
日本語で執筆するための、確定数値・主張・地雷(書いてはいけないこと)の集約。

> **🔵 2026-07-17 目的の再固定(PAPER_OUTLINE.md 🔵節が正 — 本ディレクトリの
> 各ファイルより優先)**: 主claim = C1'(同定活性への**介入**による編集 =
> 同定の因果証明)。routed/EF(C0)は情報・実用の副claimに降格。
> 「EF」は我々の編集器の腕の略記であり、Havasi et al. のモデルそのものでは
> ない(枠組みは継承、regime/条件付け/hazard/デコードは我々)。
> 新中心図 = 介入本数の局在性スペクトル(k掃引 + P-O介入版)。
> 編集不能featureはSAE側/手法側の判別木でLimitationへ。
> **貢献の位置づけ: 新たな活性抽出手法の提案ではない** — 「activationsの
> 因果妥当性は編集で評価されるべきで、評価の幅が広がる」という
> **編集ベース因果評価枠組み**の提案(既存の同定手法 FRC/AUROC/事例delta
> を同じ枠組みで評価する)。
> 01〜05は追って本節の序列に合わせて改訂する。

> **⚫ 2026-07-17 追記(最優先・PAPER_OUTLINE ⚫節が正): EF系は論文から
> 完全除外(ユーザー決定)。** S1/S2/S3はどれもSAE活性の因果証明になって
> いない(条件付け=P(edit|Z=z)のまま)ため、**S系列・ef32・routed(C0
> 0.2839/0.2892)・C2・C3・ours_ef腕・P-O ef版のすべてを論文に載せない**。
> 本ディレクトリ内のEF/routedの数値・記述(01/03/04の該当箇所)は履歴で
> あり執筆には使わない。効果器は steer/clamp のみ、ヘッドラインは C1'
> 一本。仕様構築(delta top-k)はSAEのみ依存なので C1'/P-I/P-J/P-B/
> FRR(steer行)/judge自己一致/局在性(k掃引・S_min steer版)/再現アンカー
> /B2(SAE不使用参照)は全て生存。判別木の論文版はEF列を落とす。

## ファイル構成

| ファイル | 対応する章 | 内容 |
|---|---|---|
| `01_introduction.md` | Introduction | 動機の流れ、目標文、貢献リスト、冒頭で引く数値、タイトル案 |
| `02_previous_works.md` | Related Work | 2軸分類、4軸サーベイの確定分、検証済み逐語引用、棄却リスト |
| `03_method.md` | Method | タスク定義、アーキテクチャ、Edit Flow機構、学習データ、routed規則 |
| `04_experiment.md` | Experiments | 設定、ベースライン、全確定数値の表、統計手法、統制実験 |
| `05_discussion_limitations.md` | Analysis / Limitations | 3層分解の物語、仕様出所の限界、判明済みの弱点 |
| `06_pipeline_and_theory.md` | **(最新の正)** | 🔵/⚫以降の理論的背景(条件付けvs介入、外生的検証器、免許規則、形1/形2)と現行パイプラインの状態表 |

## ソース・オブ・トゥルース(数値の一次出典)

- `PAPER_OUTLINE.md` — 主張C0〜C4と全結果台帳(§6b〜6l)。**数値はここが正**。
- `RELATED_WORK.md` — 関連研究の英語草稿v2・検証済み引用・棄却リスト。
- `runs/tables/` — main_metrics_{499,997}、frr_per_feature_*(judge別)、per_feature CSV。
- `runs/paper_metrics/report.md` — P-N(両論文の指標での検証)。
- `README.md` §13.8 / `EDIT_FLOWS_ZERO.md` — S0〜S4の学習履歴とゲート判定。

## 🔴 全章共通の用語規則(違反すると主張が壊れる)

1. **本手法に "intervention" / "介入" / "steering" を使わない。**
   本手法は凍結Gemmaの活性を一切変更しない(層0への prefix トークン追加=
   証拠の提示)。正しい語は **"conditioning"(条件付け)/ "specification"(仕様)**。
   介入と呼べるのは B1(クランプ)・B3(steer)・LinguaLens・AxBench 側のみ。
2. routed(0.2839/0.2892)を「介入の結果」と呼ばない — headは条件付け編集器、
   介入はfallback。介入列のヘッドラインはC1'(P-K)が別に持つ。
3. 「言語学的な最小単位」と書かない(形態素と誤読)。単位は「言語現象」、
   編集は「最小対変換(minimal-pair transformation)」。
4. 編集語彙(INS/DEL/SUB)に新規性を主張しない(Gu 2019 / Malmi 2019 が先行)。
   新規性は**条件付け信号**(SAE特徴辞書)にのみ置く。
5. judge評価は「**reliability** を測る」と書く。「品質/validity を測る」は禁止
   (Norman et al. "Reliability without Validity" がそのまま反論)。
6. 使用禁止タイトル: ~~Lifting SAE Interventions into Discrete Edit Operations~~ /
   ~~Structural Interventions on SAE Features~~ / ~~Editing as Intervention~~。

## 数値クイックシート(確定値のみ。詳細は 04)

| 主張 | 数値 | 状態 |
|---|---|---|
| C0 routed exact | **0.2839**@997 / **0.2892**@未接触498(vs steer 0.2269、p≈0.007) | 確定・事前登録・凍結 |
| C1' 仕様2×2(499) | LinguaLens完全版 0.0160 / AxBench完全版 0.0701 / 我々の仕様×clamp 0.1743 / ×steer 0.2337 | 確定 |
| 無介入床 / recon統制 | raw 0.0601 / recon 0.0100 | 確定 |
| EF vs v6パイプライン | 0.1904 vs 0.1102(+73%、p<1e-4)、holdout 0.1833 vs 0.1033 | 確定 |
| B2(プロンプト) | 0.1242、empty copy 0.4770(統制崩壊) | 確定 |
| FRR(GPT-4o) | ef32 0.8735 > routed 0.8306 > steer 0.7327(3 judge順位一致) | 確定 |
| steer net-FRR | 0.4062(GPT-4o)vs EF側 0.69 | 確定 |
| P-I WHERE | true発火 3.57 vs random 1.08、符号検定 393/31、p=5.6e-81 | 確定 |
| P-J AxBench逐語 r=1 | 因果信号ゼロ(0.06/0.07、平均比0.86×)— 検出は0.939で完璧 | 確定 |
| P-B FRC条件付け | exact 0.0140(vs random 0.0000、事例仕様の~7%) | 確定 |
| P-N E_abl | 我々の仕様 +0.631/+0.370 vs 両プロトコル +0.091/+0.115 | 確定 |
| λ-IoU(WHERE) | 0.7449 vs empty 0.1539 / random 0.3252 | 確定 |
| 前提保護 | empty no_edit 1.0000(全F・全ckpt) | 確定 |

## 未確定・進行中(執筆時に「暫定」と明記するもの)

- **S6 学習が進行中**(v7キャッシュ + mismatched-z null教師 + MOVE op)。
  GOなら手法節の学習データ記述に v7/P5 を昇格、NO-GOなら S3 のまま。
  ゲート: MOVE≥4/12(薄い)、thr0.1≥0.2104、random no_edit≥0.87、λ-IoU≥0.74。
- **P-O(事例レベル最小条件集合)**: 実装済み・S6決着後に実行。
- BLEU/chrF は sacrebleu 未導入で空欄(表の空欄埋めのみ)。
- gain-router 0.3012 は未昇格(未接触標本での再確認が要る — 使わない)。
- 定性例収集(`run_examples.sh`)は未実行。
