# 02 — Previous Works(Related Work)執筆資料

出典: `RELATED_WORK.md`(英語草稿v2 = §D + 第5ラウンド §R5、全主張が
一次資料PDFで検証済み)。日本語で書く際も**構成・主張・引用は §D/§R5 を
そのまま翻訳・要約**すればよい。以下はその骨格と、日本語化の要点・地雷。

> **🔶 改訂ガイド(2026-07-18 editor前提 — 最優先)**: 大前提 = SAE介入を
> 信号としてeditorに入力し、editorの出力embedding(Δh)をresidual stream
> に戻す(手法名は記述形「SAE-conditioned edit-flow intervention」)。論文のRelated Workの背骨は **軸5(因果評価と
> SAE評価ベンチマーク、本ファイル末尾)** と D.1/D.2(SAE同定と懐疑)/
> D.4(steering)/D.5(judge)に加え、**学習型介入の系譜(ReFT/LoReFT —
> 提案editorの機構先行)** を必ず置く。
> **退役(論文に使わない)**: トークン出力の離散編集セル — D.0の
> 2軸分類表の「本研究」行、D.0bの「本研究」列、**D.3のうちトークン出力
> 編集器としての位置づけ**(LEWIS/LevT等は「editorの条件付けの先行」と
> して短く触れるのは可 — 我々のeditorは出力チャネルがΔhである点が差分)、
> **D.6全体**(ルーティング — routedの除外により消滅)。
> 貢献文(2026-07-23 feature-specプロトコルへ更新): 「**featureレベル
> SAE spec(同定プールの符号付き平均)に条件付けられた介入生成器を提案し、
> CausalGymの言語学的因果評価をテキスト編集の実行まで拡張、SAEBench系の
> 介入評価に編集という新しい評価器を加える**」。
> ⚠️ 07-22プロトコル移行により「事例レベル仕様(この1ペアのdelta)」を
> 前提とした差別化(D.0b旧版の「集約しないのは我々だけ」)は**無効** —
> 現行の我々もプール平均で集約する。差別化は下の改訂版D.0bのとおり。

## 0. 章の背骨 — 2軸分類(D.0)

| 手法 | WHERE | WHAT | 条件付け信号 |
|---|---|---|---|
| LinguaLens (Jing+ 2025) | 選択なし(全位置一律・定数クランプ 0/10) | 連続 | SAE特徴(現象あたり**3本**) |
| AxBench SAE/SAE-A (Wu+ PMLR v267) | 選択なし | 連続 (h+αw) | SAE特徴(概念あたり**1本**) |
| ActAdd / CAA / RepE | 選択なし | 連続 | 対照ペア由来の方向 |
| Levenshtein Transformer (Gu+ 2019) | 編集目標から推論 | **離散** | source のみ |
| LEWIS (Reid & Zhong 2021) | 編集目標から推論 | **離散** | source + スタイル分類器attention |
| Susanto+ 2020 / EDITOR (Xu & Carpuat 2021) | 編集目標から推論 | **離散** | **表層の用語辞書/語彙選好** |
| **本研究** | editorのλ場が推論(編集スパンに局在) | **連続 Δh**(residual stream注入 — 文は凍結LMが書く) | **SAE特徴spec**(featureあたり k=64、プール平均) |

空白セルは「SAE特徴条件付け × 学習された位置依存の介入」。左下(離散
編集器)は条件付けが表層辞書に留まり文を編集器自身が書く(因果検証に
ならない)。右上(SAE介入)は全位置一律の固定ルール(clamp/単一方向加算)
で作用が粗い。本研究はSAE specを読む学習editorが位置依存のΔhを描画し、
テキスト生成は凍結LMに残す — 条件付けの解釈可能性と因果検証の両立。

## 0b. 3つの選択方式の対比表(D.0b — 実験節でも使う中心表)

| | 本研究(feature-spec、2026-07-23改訂) | LinguaLens | AxBench |
|---|---|---|---|
| 選択の単位 | この現象(同定プール集約) | この現象(コーパス集約) | この概念(コーパス集約) |
| 集約 | **符号付き連続量のmean**(top-r選択なし) | 平均(EALE)→ **top-3選択** | 平均/AUROC → **top-1選択** |
| 活性の使い方 | 連続・符号付き delta の平均 — **大きさ保存** | **二値**(発火したか)— 大きさを捨てる | 連続 max-pooled |
| 測る場所 | **編集スパン内のみ**(local max-pool) | 文中どこでも | passage全体 |
| ラベル | minimal pairの向きのみ | sentence1/sentence2 | SAE-A: 要 / vanilla: 不要 |
| 選択基準 | \|mean delta\| top-k(k=64) | \|EALE\|>75%ile → FRC = H(PS,PN) | vanilla: 概念出所latent / SAE-A: AUROC最大 |
| 介入に使う本数 | **64**(feature) | **3**(現象) | **1**(概念) |
| 同定/評価分離 | **あり**(評価500非接触の4,451で構築) | なし(全ペアin-sample) | あり(36例文で同定) |

**非対称は5つ(すべて同一同定プール上で成立 — データ量の差ではない)**:
(1) **大きさを捨てるのはLinguaLensだけ**(二値クランプ0/10)— 我々は
符号付き連続量を保存し、editorに強度を読ませる。(2) **編集スパンに局在
させるのは我々だけ**(local max-pool)。(3) **極小top-r選択をしないのは
我々だけ** — 彼らはtop-3/top-1に絞り(勝者の呪い: split-half top-1一致
36-43%)、我々はmean集約のままk=64を渡す(spec split-half cos 0.83)。
(4) 介入の書き方: 固定ルール(全位置一律クランプ/加算)vs **学習editor
の位置依存Δh**。(5) AxBench vanillaは「探索」をしていない(概念の出所
latentをそのまま使う)— 0.695 は選択失敗ではなく「出所latentが自分の
概念を検出できない」という数字。
旧版の「集約しないのは我々だけ(事例レベルdelta)」は**旧プロトコル**
(oracle-spec、付録の上界診断)にのみ当てはまる — 本文では使わない。

**🔴 LinguaLensは論文とコードが食い違う**(公式repoで確認): 論文のPS/PNは
条件付き確率、repoの実装は周辺比率。minimal pairでは独立性が成立しないため
両者は一致しない(例: 両文で発火する話題特徴が半数にあると論文定義FRC=0、
コード0.5)。**我々のFRC再実装はコード側に忠実**(人々が実際に走らせる成果物を
再現)。彼らのdo記法をそのまま引かず、「実装は周辺比率であり因果量ではない」
と1文書く。

## D.1 — SAE特徴: 検出は指令ではない

- 系譜: SAE辞書学習(Cunningham/Bricken/Templeton)→ Gemma Scope(本研究の
  重み)→ LinguaLens(言語現象がこの空間で回収可能; FRC=PS/PNの調和平均、
  EALE 75%ile事前フィルタ、GPT-4oでtop-10検証)。
- **決定的な指摘**: LinguaLensはその特徴を**活性空間のノブとしてのみ**使う
  (ablation=0 / enhancement=10 のクランプ、出力は自由生成)。検証済み逐語:
  "In the ablation experiment, we set the target feature's activation to 0,
  and in the enhancement experiment, we set it to 10."
- LinguaLensの同定は**構成上無競争**: 検出ベースライン無し、AUROC/AUC/F1は
  論文中0回、PS/PNはminimal pair上(最も易しい識別)。AxBenchは反証可能に
  設計されている(12手法・対応t検定)からこそ vanilla SAE 0.695(12中11位)
  と「教師あり選択の価格」(0.695→0.917)を出せる。→ 分野には言語SAE特徴の
  **存在主張**(LinguaLens)と単一latent検出の**比較主張**(AxBench)がある
  が、「同定された特徴が**行動の根拠に足るか**」の評価が無い。そこが本研究の編集ベース因果検証(feature-spec exact/FIC)。
- LinguaLensにおける「編集」はデータ構築側のみ("produced through a minimal
  edit that deletes or substitutes the trigger")。
- **同定/評価のデータ分離が無い(2026-07-21深掘り、詳細は
  audit_ll_axbench.md §5)**: PS/PN/FRCは現象あたり全50ペアでin-sample計算、
  262,144候補latent(OpenSAE、64x拡張)からのtop-r選択もGPT-4o検証
  (top-10の活性分布)も同一データ。held-out・CV・多重比較補正の記述なし。
  介入実験(Table 2)は手書きプロンプト自由生成なのでデータ再利用はないが、
  5 featureのみで「同定された特徴集合の同一性」は検証されない。我々の
  Gemma Scope再現でのsplit-half実測(99現象×20反復×3層): FRC値自体は
  汎化する(top-1 in 0.88-0.90 → out 0.84-0.86)が、**選ばれるtop-1特徴の
  half間一致は36-43%、top-3でも約50%** — 「どの特徴が現象に対応するか」
  という同定の出力は識別ペアの標本に強く依存する。16k候補でこれなので
  262k候補の彼らの設定ではさらに悪化が予想される。論文では「特徴は
  現象に十分対応(スコア汎化)、ただし特定の特徴集合の同一性はデータ
  依存(選択不安定)」と書き分ける。
- **反例なし(確定)**: "No prior work, to our knowledge, uses SAE features
  as the conditioning signal for a model that emits edit operations."
- Amnesic probing(Elazar et al., TACL 2021)= 本研究の「検出≠因果」主張の系譜。"probing
  performance is not correlated to task importance"。**ヘッジ3点**:
  (1) "the canonical" でなく "a canonical" と書く(先行にHewitt & Liang 2019、
  Ravichander 2021、Belinkov 2022)。(2) 彼らは「モデル自身が使うか」、我々は
  「我々が外部指令として使えるか」— 隣接する問いで、精神の拡張であって
  文字通りの適用ではない("extends the spirit rather than the letter")。
  (3) 我々の乖離(FRC-r3 0.016 / AUROC-r1 0.070 vs 学習介入 0.128)はexactでの測定であり、amnesic probingのtask lossとは別の量。

## D.1b — 言語学×SAEのランドスケープ(2026-07-23調査・追加)

LinguaLens以外に「言語学的観点からLLMをSAEで分析する」主要研究。
Related Workの1段落(または脚注群)として引き、位置づけを固定する。
bibキーはaaai2027.bib登録済み。

- **Brinkmann et al.(NAACL 2025)** `brinkmann2025grammatical` —
  最近接。**軸5-7の差分5点で引く**(言語学的SAE因果介入の先行)。
- **Minegishi et al.(ICLR 2025)** `minegishi2025polysemous` —
  多義語の意味表現でSAEの単義性を評価。語彙意味論の観点だが
  **介入なし**(表現評価)— 「検出・表現評価に留まる」群の代表。
- **Deng et al.(ACL 2025)** `deng2025languagespecific` /
  **Andrylie et al.(arXiv 2025)** `andrylie2025languageconcepts` —
  言語固有featureの多言語SAE同定。概念単位は「言語」であり
  言語学的現象(文法feature)ではない — 周辺。
- **SASFT(arXiv 2025)** `deng2025sasft` — SAE誘導fine-tuningで
  コードスイッチ抑制。応用側(検証でなく制御)。
- **BLiMP(TACL 2020)** `warstadt2020blimp` / **MultiBLiMP 1.0
  (TACL 2025)** `jumelet2025multiblimp` — 言語学的minimal pairベンチの
  系譜(SAE不使用・データ側)。LinguaLens-Dataの前提となる伝統として
  1文で引く(最小対=言語現象を単離する標準手段、という我々の評価設計の
  正当化にも使える)。
- 位置づけの一文(推奨): 「言語学的観点のSAE研究は、表現の評価
  (Minegishi)、多言語共有の実証(Brinkmann/Deng)、応用制御(SASFT)へ
  広がるが、**同定プロトコルの検証と編集実行を成功基準にした因果検証**は
  行われていない — そこが本研究」。

## D.2 — 懐疑的証拠と、それが我々に届かない理由

- AxBenchの否定的結論は **steering軸に閉じている**: "even at SAE scale,
  representation steering is still far behind simple prompting and
  finetuning baselines"。手法集合に編集手法は無い。
- **AxBench自身が detect-then-intervene の脱連関を実証**(SAE-A): 検出最良
  latentをAUROCで選び("select the highest-scoring feature by this metric")、
  同じlatentのdecoder方向でsteer("adding their decoder features directly to
  the residual stream")→ 検出 0.695→**0.917**(+32%)、steering
  0.165→**0.157**(−5%)、winrate 48.8%。著者自身: "**better classification
  does not directly lead to better steering**"。Probe(0.940/0.098)、SSV
  (0.912/0.026)も同型。→ 我々の「検出≠因果編集」乖離の独立再現(別modality)として引く —
  反論ではない。我々への免罪符にもしない("We do not claim AxBench licenses
  our conditioning")。
- 検出軸の弱さ(0.695)は我々の限界の住処: 本手法は概念→latent引き当てを
  一切行わない(条件付けは観測された特徴差分のtop-k)が、**運用時に仕様を
  概念から作る問題はこの0.695の世界に落ちる**(→ Limitationsで正面から書く)。
- その他: Kantamneni et al. (ICML 2025) SAEプローブはロジスティック回帰に
  一貫して勝てない(ただし「一貫した優位の不在」であり個別勝ちはある)。
  frozen/random SAE (arXiv:2602.14111): 見出しの0.73はcosine球内で学習する
  Soft-Frozen、完全ランダムは0.55-0.62 < 学習済み0.72-0.74。DeepMind自身が
  Gemma Scope公開時に「公平なベースラインに勝つか」を未解決問題として列挙。

## D.3 — 離散編集: 操作は貢献ではない

- 定義枠: Malmi et al. NAACL 2022 tutorial("predicting edit operations
  which are applied to the inputs")。
- 語彙の先行: LevT(挿入・削除を原子化)、LaserTagger、LEWIS(名前の由来;
  insert/keep/replace/delete)。**"We claim no novelty in the edit
  vocabulary."**
- **最近接の先行 = 制約付きLevT系譜**: Susanto et al. ACL 2020(Wiktionary/
  IATE用語辞書を推論時注入)、EDITOR(TACL 2021、"specify preferences in
  output lexical choice")。彼らも**仕様で離散編集を条件付けている**。
  → 差分は仕様の**種類**: 表層の用語辞書 vs **モデル内部から回収した特徴辞書**。
  "The contribution is the *kind* of specification, not the existence of one."
- **Edit Flowsとの関係(過剰主張しない)**: 生成機構はEdit Flows(Havasi et
  al. 2025)のCTMC。**rate×tokenの因子分解 u_t = λ·Q は式(13)の継承**であり
  我々の2ヘッドはその積そのもの。"decoder" は彼らの論文に0回 — opの適用は
  CTMCの状態遷移そのもので学習対象ではない(彼我共通)。我々が変えたのは:
  スクラッチ生成→ソース係留、ランダムアラインメント→最小編集の決定的
  アラインメント、自由レート→hazard解析形、条件付け=prefix/画像/CFG→
  SAE特徴トークン。

## D.4 — Steeringの入力毎の不安定性

- Tan et al. (NeurIPS 2024): "steerability takes on a large range of values
  across different inputs, including negative values"(anti-steerability)。
- 我々の対比: 彼らは**入力毎**(データセット平均は正のまま)、我々は
  feature別のsteer失敗(較正済steerでも壊れ文19%、feature別exactの
  負のnet)を測る — ⚠️引用する数値は現行feature-specプロトコルの
  feature別表(04§9系)から引き直すこと(旧net-FRR値は付録限定 —
  旧測定では集約平均が負のfeatureが実在し「同じ失敗のより鋭い形」だった)。
- 彼らはCAA/Llama-2/多肢選択でSAE無し → 「現象の引用」であってSAE steering
  の測定としては引かない。**OOD側の主張(プロンプト変更に脆い)は引用禁止**
  (検証で棄却済み)。

## D.5 — Judgeを裁く(新規性は限界コストのみ)

- 先行(すべて実在・確認済み): Sage(ラベルフリーjudge評価)、Shi et al.
  Repetition Stability(同一クエリ反復)、Haldar & Hockenmaier "Rating
  Roulette"(intra-rater Krippendorff's α)、Wang et al. Conflict Rate
  (順序入替)、Zheng et al.(順序ランダム化)、Norman et al. MVVP。
- **我々の主張はここだけ**: 先行は全て**専用の再実行予算**(2〜20×)で重複を
  人工的に作る。我々は**評価データ内に自然発生する exact 一致ペア**から
  追加コストゼロで自己一致率を回収する(exact一致ペアでは judge(src,out) と
  judge(src,tgt) が同一比較になるため)。
- degeneracy反論(常にAを選ぶjudgeが満点)への免疫: gold順とsystem順を
  **独立のrngストリーム**から引くため、always-A judge は 0.50 に落ちる
  (結合していれば1.00だった — 実際バグ下のgemmaがその退化を示した)。
- **(c-4) attenuation は新規性が残る**: 対応のある差は Rogan–Gladen 反転で
  `p_A − p_B = (θ_A − θ_B)(q0+q1−1)` — スケールされるがシフトしない。
  前提=非差異的誤分類で、**per-system自己一致率がその検査**(1つの測定が
  judge選定根拠と妥当性検査の2役)。Chen et al. (arXiv:2601.05420) を引く。

## D.6 — 必要編集量によるルーティング(⚫退役 — routing除外により論文不使用、記録のみ)

- 先行: RouteLLM(学習ルータ)、MoECE(エラー型でゲート)、ESC、APR。
  近い順に: **AdaEdit**(Cheng et al., Findings of ACL 2026)— diff/全文再生成
  を事例毎に切替、ただし**教師ありFTで暗黙内在化**。**CAST**(Lee et al.,
  ICLR 2025)— steeringの推論時ゲート自体は先行、ただし基準はプロンプトの
  意味カテゴリ・両枝ともAR生成・ラベル付きグリッドサーチ。
- 我々の差分は**着想ではなく条件付け信号**: count-rule は「離散編集器自身の
  λ場が発火させたハンク数」= **明示的・教師なし・自己言及的**な編集規模
  推定器。ルータの学習なし、ラベルなし、検証集合でのフィット無し。

## 軸5 — 因果評価とSAE評価ベンチマーク(R5、2026-07-17 逐語検証済み)

🔵再固定の新主張((e)編集ベース因果評価枠組み、(f)介入本数=局在性
スペクトル)を守る章。9本すべて自前PDF抽出で検証済み(`RELATED_WORK.md`
§R5)。**反例なし。ただし以下の並べ方を守らないと過剰主張になる。**

### 軸5-1. CausalGym(Arora, Jurafsky & Potts, ACL 2024)— 最近接

- 内容: SyntaxGym由来のテンプレートから**span-alignedな言語学的minimal
  pairs**を大量生成し、介入手法(DAS・probe・PCA・k-means・means差)を
  因果ベンチマーク。枠組みは明示的にdo演算子:
  > "The core idea of intervention is adopted directly from the
  > do-operator used in causal inference; we test the intervention's
  > effect on model output to establish a causal relationship."
- **成功基準が決定的に違う**: 彼らは log odds-ratio(式9)= 介入が
  **モデルの次トークン選好**(is↔are など)をどれだけ動かすか。
  出力テキストの編集は一切ない。
- **SAEが不在**: 全文に "SAE" / "sparse autoencoder" は**0回**
  (2024年2月、Gemma Scope以前)。手法集合は教師あり特徴化が中心。
- **書き方**: 「言語現象×minimal pair×因果介入」という土俵の先行として
  真っ先に引き、差分を2点に絞る — (i) 評価器: 次トークン挙動のフリップ →
  **テキストのminimal-pair編集の実行**(exact・文法性・局在性が測れる)、
  (ii) 対象: 学習された特徴化(DAS等)→ **SAE辞書latent**(既存の同定
  手法 FRC/AUROC が出したもの)。「彼らの評価をSAEに適用した」ではなく
  「彼らの因果基準を**編集の実行**まで強めた」と書く。

### 軸5-2. SAEBench / RAVEL — 「介入ベースのSAE評価」は先行する

- SAEBench(arXiv:2503.09532):
  > "a comprehensive evaluation suite that measures SAE performance
  > across eight diverse metrics, spanning interpretability, feature
  > disentanglement and practical applications like unlearning"
  収録評価にRAVEL(latent介入で属性予測を変え、他属性を保存できるか=
  disentanglement)、feature absorption 等。
- **全文に "minimal pair" 0回・"text edit" 0回** — 編集という評価器は
  存在しない。
- **書き方**: 「SAEを介入で評価する」という上位概念に新規性を
  置いてはならない(SAEBench/RAVELが先行)。我々の追加は**評価器**:
  属性予測のフリップではなく**自由テキストの最小対編集**を成功基準に
  すること。これにより (i) exactという厳密なテキスト接地、(ii) 介入
  本数の局在性、(iii) 検出的に選ばれたlatentの因果不活性(AUROC-r1 steerのexact崩壊)の検出、が同じ枠組みで測れる — この3点が「評価の幅が
  広がる」の中身。

### 軸5-3. Sparse Feature Circuits(Marks et al., ICLR 2025)

- 逐語: "causally implicated subnetworks of human-interpretable features
  for explaining language model behaviors" — SAE特徴の因果性研究の主要
  系譜。SHIFTは分類器の脱バイアス("surgically removing sensitivity to
  unintended signals")であり**テキスト編集ではない**。"minimal pair" 0回。
- 書き方: SAE特徴の因果帰属(挙動への寄与)の代表として1-2文。
  回路帰属(挙動の説明)vs 編集実行(出力の生成)の対比。

### 軸5-4. Hase et al.(NeurIPS 2023)— localization ≠ editing の系譜

- 逐語: "localization conclusions from representation denoising (also
  known as Causal Tracing) **do not provide any insight** into which
  model MLP layer would be best to edit" / "which layer we edit is a
  far better predictor of performance."
- 書き方: 我々の3層分解(検出≠指令≠介入で書ける)の**重み編集ドメインの
  先行**として、amnesic probing(D.1)と並置する。「局在化の知見が編集を
  導かない」という乖離は彼らが重みで、AxBenchがsteeringで、我々が
  離散編集で測った — 3ドメイン一致の乖離、という強い書き方ができる。

### 軸5-5. Engels et al.(ICLR 2025)— (f)局在性スペクトルの動機

- 逐語: "a rigorous definition of irreducible multi-dimensional features
  based on whether they can be decomposed into either independent or
  non-co-occurring lower-dimensional features"(曜日・月の円環特徴)。
- 書き方: 「featureが複数活性に分散しうる」は既知 — (f)の**動機**として
  引く。我々の差分は**編集可能性を基準にした分散度の測定器**(介入k掃引
  のexact曲線 + S_minの事例内サイズ×事例間Jaccard)。幾何学的定義
  (彼ら)に対する**操作的・因果的定義**(我々)という対比。

### 軸5-6. 🆕 対抗仮説と先行発想(引用しないと突かれる2本)

- **arXiv:2510.01246**(top-1 steering): "many dimensions among the
  top-k latents capture nonsemantic features such as punctuation ...
  we propose focusing on a single, most relevant SAE latent (top-1),
  eliminating redundant features" — **「top-1で足りる(むしろ良い)」
  という局在性の対抗主張**。ただし基準は命令追従のsteering品質
  (minimal pair 0回・exactなし)。我々のk掃引が「厳密編集には多数
  要る」を出したら、**矛盾ではなく「評価器が結論を変える」**として
  書く — 生成品質基準なら1本、正確な編集基準なら~32本。これ自体が
  (e)「編集で評価すべき」の論拠になる。
- **Beyond Input Activations**(EMNLP 2025): "activated latents do not
  contribute equally ... only latents with high causal influence are
  effective for model steering" — **「入力側活性でなく出力への因果影響で
  latentを選ぶべき」という発想は先行**。我々の新規性を「発想」に置かず、
  **両論文の公式選択法(FRC/AUROC)同士を同一同定プール上の編集の因果
  基準で対決させた測定**に置く。

### 軸5-7. 🆕 Brinkmann et al.(NAACL 2025、arXiv:2501.06346)—
言語学的SAE因果検証の最近接(2026-07-23ユーザー指摘で追加)

- 内容(abstract/検索ベース、**逐語引用は投稿前に自前PDF抽出で確認**):
  Llama-3-8B / Aya-23-8B のSAEで**形態統語概念(数・性・時制)**の
  featureを分類器ベースで同定し、(i) 多言語共有featureのablationで
  分類性能が全言語でチャンス水準へ低下、(ii) 機械翻訳中の操作で対象
  文法属性を選択的に変更。主主張=「文法概念のfeature方向は言語間で
  共有される」(表現の多言語普遍性)。
- **届く範囲(正直に)**: 「言語学的概念のSAE latentへの因果介入」は
  先行される。CausalGym(言語×因果、SAE無し)とBrinkmann(言語×SAE×
  因果)に挟まれ、**「言語学的featureのSAE因果検証は初」という主張は
  完全に禁止**(棄却リスト追加)。
- **差分=我々の新規性の置き場(5点)**:
  1. **問いの向き**: 彼らは表現の**多言語普遍性**を示すために介入を
     道具として使う。我々は**検証器そのもの**を問題にする(「同定活性の
     因果検証はどの介入で可能になるか」)。
  2. **介入の書き方**: 彼らの介入は固定ルール型(feature ablation/
     steering)— まさに我々がベースラインとして測定し「編集を実行
     できない」(FRC-r3 0.016 / AUROC-r1 0.070 / 較正steer 0.086)と
     示した枠内。**学習された介入生成器は存在しない**。
  3. **評価器**: 彼らは分類器精度の低下+翻訳出力の属性変化。我々は
     **最小対編集の厳密なテキスト実行**(exact、復唱枠、greedy)+
     統合FIC+random/empty特異性統制。「編集の実行」という評価器は
     彼らに無い(SAEBench/RAVELへの差分と同型で、Brinkmannにも通用)。
  4. **同定プロトコル**: 我々は同定/評価のデータ分離を全アームに敷き、
     選択安定性をsplit-halfで定量(0.83 vs 36-43%)。彼らの分類器選択の
     分離・安定性は主題化されていない(要PDF確認)。
  5. **範囲の軸**: 彼ら=形態統語の少数概念×多言語(言語横断の広さ)。
     我々=形態・統語・意味・語用の**99現象**(意味・語用まで)×英語
     (現象横断の広さ)。直交する広さで、比較でなく相補。
- **書き方(推奨)**: 「文法概念がSAE latentとして因果的に操作可能で
  あること自体はBrinkmannらが翻訳属性の操作で示した。我々はその介入を
  固定ルールから**学習された生成器**に置き換え、**最小対編集の実行**と
  いう強いテキスト接地基準で、**同定プロトコル自体**(分離・安定性)を
  検証対象にする」— 相補の先行として真っ先に引き、対決しない。
  D.1のLinguaLens批判(選択不安定)はBrinkmannの分類器選択にも同じ
  問いを立てられる(強めの武器だが、PDF確認前は示唆に留める)。

### 軸5のランドスケープ脚注(1文ずつで可)

- arXiv:2507.08473 — 説明文を介さないSAE評価(活性予測)。評価方法論の
  多様化の例として。
- EACL 2026 SRW — SAE言語steeringのタスク性能影響。steering評価の
  実務側の例として。

## 🔴 棄却リスト — 書いてはいけないこと(X / X-2 / 軸4より)

- 「LinguaLens自身がSAE介入の信頼性の低さを認めている」(有利すぎる論拠)。
- 「Edit Flowsの語彙はちょうどINS/DEL/SUB」「LaserTaggerはちょうど3種」等の
  語彙の断定(要再確認のまま棄却)。
- Tan et al. のOOD側主張。
- 「LEWISはSAE解釈可能性の系譜全体に先行」(Yun+ 2021, SPINE 2018が先行。
  「Gemma Scope(2024)とLLM-SAE系譜(2023)に先行」と狭く書く)。
- 「LEWISには言語的特徴条件付けが一切ない」(スタイル分類器attentionは
  学習された帰属信号 — 最近接の隣人。差別化は**辞書としての解釈可能性**)。
- 「AdaEditはフォーマット切替であってルーティングではない」「AdaEditは
  コスト効率基準だから対象外」(どちらも0-3棄却 — 逃げられない)。
- 「EdiT5は経路分岐を持つ」(事実誤り: 全入力が無条件固定順)。
- 「CASTより先にsteeringゲートを付けた」(CASTが先行)。
- 「SAEの不振はNeuronpediaラベルのartifact」(強い版は棄却; AxBenchの留保は
  "at least partially" 止まり)。
- 「judge品質を測れる」「人手ラベル不要のjudge評価は新規」「反復一致率は
  新規」「順序ランダム化は新規」。
- (R5追加)「**介入ベースのSAE評価は我々が初**」(SAEBench/RAVELが先行 —
  新規性は**編集という評価器**にのみ置く)。
- (R5追加)「**因果基準でlatentを選ぶ発想は我々が初**」(Beyond Input
  Activations が先行 — 我々は「FRC/AUROCの公式選択法同士の因果対決」)。
- (R5追加)「**言語学的minimal pairでの因果介入評価は我々が初**」
  (CausalGymが先行 — 差分は評価器[挙動フリップ→テキスト編集の実行]と
  対象[学習特徴化→SAE辞書latent])。
- (R5追加)「featureが複数活性に分散すること自体の発見」(Engelsが先行 —
  我々は**編集基準の操作的測定器**が新規)。
- 🆕(2026-07-23)「**言語学的feature/文法概念のSAE latentへの因果介入は
  初**」(Brinkmann et al. NAACL 2025が先行 — 新規性は**学習介入生成器**と
  **編集実行という評価器**と**同定プロトコルの検証**にのみ置く。軸5-7)。

## 手続き規則(執筆時)

1. **逐語引用は全て自前のPDF抽出で再確認**(要約器の捏造を計9件検出済み)。
   手順: `curl -sL -o x.pdf <URL>` → pypdf全ページ → NFKD正規化+リガチャ
   展開+ハイフン折返し除去 → 自分で regex 検索。
2. grepの0ヒットを信じない(リガチャ偽陰性)。
3. 「調べていない」と「探したが無い」を区別: 「探したが無い」と書けるのは
   **(a)(b)(d)のみ**。(c)は先行あり。
4. 投稿時に版番号・題名を再確認(LinguaLensは旧題と現題が異なる。Tan v6、
   Edit Flows v3、AxBenchはPMLR v267 camera-readyの数値を引く)。
   AxBenchの掲載venue(ICML/PMLR表記)は投稿前に最終確認(残タスク)。
