# 01 — Introduction 執筆資料(2026-07-23 feature-specプロトコルへ全面改訂)

> 🔶 大前提: SAE介入(spec)を信号としてeditorに入力し、editorの出力
> embedding(Δh)を凍結LMのresidual streamに戻す。テキストは凍結LMの
> 生成として出る。
> 🚫 数値はすべてfeature-specプロトコル(03§3'、同定/評価分離)の確定値
> のみ(README規則0'、oracle値は付録限定)。手法名は記述形
> 「SAE-conditioned edit-flow intervention」(固有名不使用、規則📛)。
> 方向用語は enhancement(足す)/ ablation(消す)(規則5')。
> ⚠️ 改善構成(⑦文脈内spec / T系 / ③invstd)の採否確定後、
> 【最終構成で更新】印の数値を差し替えること。

## 1. 論文の一文

> **SAEの活性同定を因果的に検証するため、featureごとのSAE活性spec
> (同定プールの符号付き平均)と入力文を読み、編集ベクトルΔhを凍結LMの
> residual streamに注入する学習editorを提案する。テキストは凍結LM自身が
> 生成するため、最小対編集の成功(元の文でなくペアの文を復唱する)が
> 「同定された活性がその言語学的featureを担う」ことの因果的証拠になる。**

タイトル(確定): *Rendering SAE Interventions as Residual-Stream Edits
with a Learned Edit-Flow Editor*

## 2. 動機の流れ(推奨の段落構成)

### 段落1 — SAEと言語featureの同定
SAEは残差ストリームを解釈可能な活性の組に分解する
(`cunningham2024sparse`=ICLR 2024[arXiv 2023]、
`bricken2023monosemanticity`; SAEは`lieberum2024gemmascope`)。
※引用年はaaai2027.bibの出版年に統一(Cunninghamは2024と書く)。LinguaLens(EMNLP 2025)は言語現象のminimal
pairとFRCで「featureに対応する活性」を同定し、AxBenchは概念検出のAUROC
で単一latentを選択する。

### 段落2 — しかし同定の因果検証が機能していない
標準の検証手段=活性への直接介入(定数クランプ・単一方向加算)は効果が
弱く不安定(LinguaLens/AxBench自身の報告)。さらに介入は全トークン一律で、
複雑なfeatureに厳密に対応した介入ができない。加えて同定自体にも問題:
(i) 同定と評価に同じデータを使う(held-out・CVの記述なし)、(ii) top-r
選択はデータ標本に強く依存する(我々のsplit-half実測: top-1一致36-43%)。

### 段落3 — 提案: 介入を固定ルールでなく学習editorに生成させる
featureごとのspec(同定プール=評価外4,451ペアの符号付き平均delta、
オフライン構築・入力独立)と入力文を双方向エンコーダ(LLM2Vec)で読み、
Edit Flows基盤の学習editorがΔhを凍結LMのresidual streamに注入する。
復唱プロンプト下で編集が成功すれば、ペア文がそのまま出る=活性同定の
因果的証拠。同定/評価のデータ分離を全アーム(本手法/LinguaLens準拠/
AxBench準拠)に統一適用する。

### 段落4 — 発見の予告

> **相関的に同定できる ≠ 因果的に編集できる — その橋を学習介入が架ける**

- 同一の同定プールから作った介入で、exact(L12 ablation)は
  本手法 **0.128** vs 較正steer 0.086 / AxBench準拠 0.070 /
  LinguaLens準拠 0.016【最終構成で更新: ⑦採用なら0.148】
- 特異性: 本手法はrandom spec指定でほぼ無編集(0.000-0.054)。
  promptingはexact 0.180と強いがrandom指定でも0.088編集してしまう
  (介入としての特異性を欠く)
- 因果効果の統合指標FIC: 本手法 **0.463**(L12)> prompting 0.410、
  LinguaLens準拠 0.05-0.07、AxBench準拠 0.11-0.17。ablation成分は
  全層0.85-0.94で4腕最良
- 同定の安定性: mean集約spec のsplit-half cos **0.833-0.838** vs
  LinguaLens top-1選択の half間一致 **36-43%** — 集約が選択不安定を解く

## 3. 貢献リスト

(i) **SAE-conditioned edit-flow intervention**: featureレベルSAE spec+
入力文を読みΔhを凍結LMのresidual streamに注入する学習editor。トークンを
出力しない(出力インターフェース=Δh)ので検証器は凍結LM自身=外生的。
学習はDolma由来の合成破損ペアのみ(評価データにゼロショット)
【T4採用時: zero-shot版/pool適応版の2行構成】。

(ii) **同定/評価を分離した因果検証プロトコル**: 評価500/同定プール4,451
を全アームに統一。復唱枠+greedy+true/random/empty統制で、最小対編集の
成功を因果的証拠として測る(exact net・統合FIC)。

(iii) **適用による発見**: (a) 同一同定情報でも介入の書き方で編集力が
桁で変わる(0.016-0.086 vs 0.128)、(b) promptingは強いが特異性を欠き、
統合FICでは学習介入が上回る、(c) mean集約specはtop-r選択の不安定性
(36-43%)を解消する(0.83)。

(iv) **再現アンカーと批判の実測**: LinguaLens/AxBenchの介入を同一
スタック(gemma-2-2b-it+Gemma Scope 16k)に忠実移植し、LinguaLensの
同定/評価非分離・選択不安定をsplit-halfで定量(audit §5)。

## 4. Introで引く数値(最小セット、L12・feature-spec)

- exact(ablation): **0.128** / steer 0.086 / AxB 0.070 / LL 0.016
  【最終構成で更新】
- 特異性: random 0.000-0.054 vs prompting random 0.088
- 統合FIC: **0.463** vs prompting 0.410(ablation成分 0.937)
- 安定性: spec split-half 0.833-0.838 vs top-1一致 36-43%

## 5. この章の地雷

- **oracle-spec値(0.3166、0.2485、0.1804等)を絶対に書かない**(規則0')。
- 手法の固有名(SAE-EditFlow等)を本文に書かない — 記述形のみ(規則📛)。
- **promptingとの関係は正確に**: raw exactでは及ばない(0.128 vs 0.180)
  ことを正直に書き、特異性(random 0.088)と統合FIC(0.463 vs 0.410)で
  上回ると主張する — 「promptingに全面的に勝つ」とは書かない。
  abstractの表現(「一部を除き及ばなかったものの…FICでは上回る」)と整合。
- 「介入ベースSAE評価は初」等のR5禁止主張をしない(README規則4:
  CausalGym/SAEBench/RAVEL/ReFT系の先行を必ず踏む)。
- 方向はenhancement/ablation(amp/supは本文禁止、規則5')。
- specは**featureレベル・入力独立・オフライン**と明示(「事例レベル
  delta」は旧プロトコル=付録の上界診断のみ)。
- editorを使わない枠組みで書かない(steer/clampはベースライン、規則0)。
