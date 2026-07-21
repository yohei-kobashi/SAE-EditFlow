# 01 — Introduction 執筆資料(🔶2026-07-18 editor前提へ改訂)

> 🔶 大前提(2026-07-18): SAE介入(仕様)を信号としてeditorに入力し、
> editorの出力embedding(Δh)を凍結LMのresidual streamに戻す。
> editorを使わない枠組みの記述は削除済み。
> **🚫(07-22)本ファイルの冒頭数値はfeature-specプロトコルの再測定値で
> 差し替えるまで仮置き。旧oracle-spec値(0.3166等)を本文に使わない
> (README規則0')。§4の0.17〜0.23(旧rewrite枠)も同様に付録限定。**

## 1. 論文の一文

> **SAE活性への介入を信号として受け取り、編集内容をembedding(Δh)として
> 凍結LMのresidual streamに返す学習editor(Intervener)を提案する。
> テキストは凍結LM自身の生成として出るため、編集の成立はSAE活性同定の
> 因果的証明になる — minimal-pair編集の実行を成功基準に、既存の同定手法
> (FRC / AUROC / 事例レベルdelta)を同じ物差しで測る。**

編集という評価器により評価の幅が広がる: (i) exactという厳密なテキスト
接地基準、(ii) 介入本数の局在性スペクトル、(iii) net-FRRの特異性、
(iv) 現象別の到達層(判別木)。

## 2. 動機の流れ(推奨の段落構成)

### 段落1 — SAEと言語現象の同定
SAEは残差ストリームを解釈可能な特徴辞書に分解する(Cunningham 2023,
Bricken 2023; Gemma Scope)。LinguaLens(EMNLP 2025)は言語現象のminimal
pairデータとFRC(PS/PNの調和平均)で「現象に対応する活性」を同定し、
AxBenchは概念検出のAUROCで同定を評価した。

### 段落2 — しかし同定の検証が検出的に閉じている
既存の検証は「発火統計が現象を追跡するか」(検出)か「顕著さがjudgeで
動くか」(自由生成の定性評価)に留まる。**検出できることと、その活性が
現象を因果的に担うことは別**であり(probing批判の系譜)、その乖離は
測られていない。CausalGymは言語minimal pairで介入手法を因果評価したが、
成功基準は**次トークン挙動のフリップ**であり、SAE latentは対象外。
SAEBench/RAVELは介入ベースのSAE評価を持つが、評価器は属性予測であり
**テキスト編集の実行**ではない。

### 段落3 — 提案: SAE仕様に条件付けられたeditor+編集可能性を因果基準に
本研究は、同定された活性集合(仕様)を信号として受け取り、編集を
**residual streamへのembedding(Δh)として描画する学習editor
(Intervener)**を提案する。テキストは凍結LM自身の生成として出る
(do(residual)、ReFT類縁)ため、編集の成立は「LMの計算がその活性方向に
因果的に応答する」ことの証明になる。固定描画(steering)・LinguaLens式
clampはeditorの自明な特殊ケース/既存機構としてベースラインに置き、
random仕様・無介入・再構成の統制との分離込みで、活性集合の因果妥当性を
測る。編集は文法性を保った最小対変換として評価可能であり(exact)、
介入本数を振ることで表現の局在性まで測れる。

### 段落4 — 発見の予告(3層分解 = フック)

> **検出できる ≠ 因果的に担う ≠ 介入で書ける**

- 検出は完璧: AUROC top-1 mean **0.939**(AxBench自身のSAE-A 0.917超え)
- その top-1 は因果的に何も動かさない(true/random比 **0.86×**)。
  FRC top-3 は微弱だが純粋(67/0、p=1e-20)
- 仕様を現象レベル(彼らのプロトコル)から事例レベルdeltaに替えると、
  同一の介入機構で exact が 0.016/0.070 → **0.17〜0.23** — **介入の
  編集力は仕様が決める**
- 71/98現象で介入編集が成立(因果証明)、21現象はWHERE陽性でも実行
  不可(効果器側)、残りが同定側/タスク側 — 判別木

## 3. 貢献リスト

(i) **Intervener(提案editor)**: 事例レベルSAE仕様+src文を双方向
エンコーダで読み、位置依存の介入場Δh(prefill)+全域Δh(decode)を
凍結LMのresidual streamに注入する学習editor。トークンは出力しない
(出力インターフェース=Δh)ので因果証明の枠組みが保たれる。ReFT/LoReFTが
機構の先行 — 新規性は「事例レベルSAE仕様に条件付けられた介入生成」と
編集=因果評価枠組み内での使用。ノルム予算で「介入サイズ」を明示。

(ii) **編集ベース因果評価枠組み**: minimal-pair編集の実行を成功基準に、
統制(random/empty/raw/recon)と多軸指標(exact・FRR/net-FRR・彼ら自身の
指標)で活性集合を評価するプロトコル。CausalGymの因果評価をテキスト編集
まで拡張し、SAEBench系の介入評価に編集という評価器を加える。

(iii) **適用による発見**: (a) 仕様が介入の編集力を決める(C1' 2×2
ベースライン — 両論文の完全プロトコルは無介入床以下〜同等、事例レベル
仕様は同じ機構で3〜15×)、(b) 検出の完璧さと因果的実在の乖離(P-J)、
(c) 局在性スペクトル(介入k掃引・最小介入集合S_minと安定核×FRC3)、
(d) 現象別の判別木(A71/C21/B2/D4)。

(iv) **再現アンカーと相互評価**: LinguaLens介入評価(FIC)とAxBench
steering(SAE/SAE-A)の忠実再現、および相互評価(彼らのプロトコルを
我々のベンチで、LinguaLens機構を彼らのベンチで)。

## 4. Introで引く数値(最小セット)

- 仕様2×2: 0.0160 / 0.0701 vs **0.2337**(raw床 0.0601)
- 検出≠因果: 0.939 vs 0.86×(同一パイプライン内)
- 因果床: P-I 393/31、p=5.6e-81
- 判別木: A 71/98

## 5. タイトル案(🔶editor前提、仮 — 要再検討)

editorが提案物の中心になったため旧タイトル案(評価枠組み単独)は要改訂。
方向性: SAE仕様に条件付けられた介入生成(editor)+編集による因果検証を
両方含む題。例:
1. *SAE-LEWIS: Rendering SAE Interventions as Residual-Stream Edits with
   a Learned Editor*
2. *Edit to Verify: A Learned Intervention Generator for Causally
   Validating SAE Activations by Minimal-Pair Editing*

**使用禁止(トークン出力の旧枠組み)**:
~~Commanding Edits with SAE Features~~(条件付け枠組みの題)、
~~Lifting SAE Interventions into Discrete Edit Operations~~。

## 6. この章の地雷

- **editorを使わない枠組みで書かない**(🔶): steering/clampは
  「ベースライン(既存機構の忠実実装/固定描画の特殊ケース)」とだけ
  書く。提案はIntervener(Δh出力のeditor)。
- editorの出力インターフェースを必ず明示する: **トークンではなくΔh** —
  ここが条件付け(probing系)との因果的な分水嶺。
- R5の先行を必ず踏む: CausalGym(挙動フリップ)/SAEBench(介入評価)/
  ReFT/LoReFT(学習型介入の機構先行)/ Beyond Input Activations(因果的
  latent選択の発想)— 「初」を置く場所は**事例レベルSAE仕様に条件付け
  られた介入生成とテキスト編集という評価器**のみ。
- 免許規則: 事例レベルkを表現幅と書かない(05参照)。
- promptingに勝つ主張はしない(B2はSAE不使用参照としてのみ)。
