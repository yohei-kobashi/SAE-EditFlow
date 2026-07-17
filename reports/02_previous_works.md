# 02 — Previous Works(Related Work)執筆資料

出典: `RELATED_WORK.md`(英語草稿v2 = §D + 第5ラウンド §R5、全主張が
一次資料PDFで検証済み)。日本語で書く際も**構成・主張・引用は §D/§R5 を
そのまま翻訳・要約**すればよい。以下はその骨格と、日本語化の要点・地雷。

> **🔵 2026-07-17 再固定後の章構成**: 本研究の貢献は「編集ベースの因果評価
> 枠組み」なので、Related Work の背骨は2本になる —
> (1) 従来の2軸分類(§0-D.6、条件付け信号の新規性)に加えて、
> (2) **新設の軸5「因果評価とSAE評価ベンチマーク」(本ファイル末尾近くの
> §軸5)**: CausalGym・RAVEL/SAEBench・SFC・Hase・Engels との位置関係が、
> (e)評価枠組み・(f)局在性スペクトルの主張を守る。貢献文はこの合成:
> 「**CausalGymの言語学的因果評価をテキスト編集の実行まで拡張し、
> SAEBench系の介入評価に編集という新しい評価器を加える**」。

## 0. 章の背骨 — 2軸分類(D.0)

| 手法 | WHERE | WHAT | 条件付け信号 |
|---|---|---|---|
| LinguaLens (Jing+ 2025) | 選択なし(全位置一律・定数クランプ 0/10) | 連続 | SAE特徴(現象あたり**3本**) |
| AxBench SAE/SAE-A (Wu+ PMLR v267) | 選択なし | 連続 (h+αw) | SAE特徴(概念あたり**1本**) |
| ActAdd / CAA / RepE | 選択なし | 連続 | 対照ペア由来の方向 |
| Levenshtein Transformer (Gu+ 2019) | 編集目標から推論 | **離散** | source のみ |
| LEWIS (Reid & Zhong 2021) | 編集目標から推論 | **離散** | source + スタイル分類器attention |
| Susanto+ 2020 / EDITOR (Xu & Carpuat 2021) | 編集目標から推論 | **離散** | **表層の用語辞書/語彙選好** |
| **本研究** | 編集目標から推論(λ-IoU 0.74) | **離散** (INS/DEL/SUB) | **SAE特徴**(事例あたり k=32) |

空白は「離散 × SAE特徴条件付け」のセル。左下(離散編集)は条件付けが表層に
留まり、右上(SAE介入)は作用が連続に留まる。

## 0b. 3つの選択方式の対比表(D.0b — 実験節でも使う中心表)

| | 本研究 | LinguaLens | AxBench |
|---|---|---|---|
| 選択の単位 | **この1ペア(事例)** | この現象(コーパス集約) | この概念(コーパス集約) |
| 集約 | **なし** | 平均(EALE = (1/N)Σ τ_k) | 平均/AUROC |
| 活性の使い方 | 連続・符号付き delta `z_tgt−z_src` | **二値**(発火したか)— 大きさを捨てる | 連続 max-pooled |
| 測る場所 | **編集スパン内のみ**(local) | 文中どこでも | passage全体 |
| ラベル | **不要** | sentence1/sentence2 | SAE-A: 要 / vanilla: 不要 |
| 選択基準 | delta の大きさ top-k | \|EALE\|>75%ile → FRC = H(PS,PN) | vanilla: 概念出所latent / SAE-A: AUROC最大 |
| 介入に使う本数 | **32**(事例) | **3**(現象) | **1**(概念) |

**非対称は5つあり、どれもkの値とは無関係**: (1) 集約しないのは我々だけ —
LinguaLensの原始量 τ_k(s)=a(1)−a(0) は我々のdeltaそのものであり、彼らは平均し
我々は平均しない(→ P-Bは「集約するか否か」のablation)。(2) 大きさを捨てる
のはLinguaLensだけ。(3) 編集スパンに局在させるのは我々だけ。(4) ラベル不使用
は我々だけ。(5) AxBench vanillaは「探索」をしていない(概念の出所latentを
そのまま使う)— 0.695 は選択失敗ではなく「出所latentが自分の概念を検出でき
ない」という数字。

**🔴 LinguaLensは論文とコードが食い違う**(公式repoで確認): 論文のPS/PNは
条件付き確率、repoの実装は周辺比率。minimal pairでは独立性が成立しないため
両者は一致しない(例: 両文で発火する話題特徴が半数にあると論文定義FRC=0、
コード0.5)。**我々のP-B/P-Jはコード側に忠実**(人々が実際に走らせる成果物を
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
  が、「同定された特徴が**行動の根拠に足るか**」の評価が無い。そこがP-B。
- LinguaLensにおける「編集」はデータ構築側のみ("produced through a minimal
  edit that deletes or substitutes the trigger")。
- **反例なし(確定)**: "No prior work, to our knowledge, uses SAE features
  as the conditioning signal for a model that emits edit operations."
- Amnesic probing(Elazar et al., TACL 2021)= P-Bの系譜。"probing
  performance is not correlated to task importance"。**ヘッジ3点**:
  (1) "the canonical" でなく "a canonical" と書く(先行にHewitt & Liang 2019、
  Ravichander 2021、Belinkov 2022)。(2) 彼らは「モデル自身が使うか」、我々は
  「我々が外部指令として使えるか」— 隣接する問いで、精神の拡張であって
  文字通りの適用ではない("extends the spirit rather than the letter")。
  (3) P-Bの10×はexactでの測定であり、amnesic probingのtask lossとは別の量。

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
  (0.912/0.026)も同型。→ **P-Bの独立再現(別modality)として引く** —
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
- 我々の対比: 彼らは**入力毎**(データセット平均は正のまま)、我々の負の
  per-phenomenon net-FRR(past_tense −0.13*、expressive −0.50、
  subject_verb_inversion −0.30)は**集約平均が負** — 同じ失敗のより鋭い形。
  (*数値は04の per-feature 表から引くこと)
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

## D.6 — 必要編集量によるルーティング

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
  本数の局在性、(iii) 検出完璧なlatentの因果不活性(P-J: AUROC 0.939
  でも0.86×)の検出、が同じ枠組みで測れる — この3点が「評価の幅が
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
  latentを選ぶべき」という発想は先行**。P-Jの新規性を「発想」に置かず、
  **両論文の公式選択法(FRC/AUROC)同士を編集・readoutの因果基準で
  対決させた初の測定**(+検出sanity gate 0.939で選択失敗説を封じた)
  に置く。

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
  Activations が先行 — P-Jは「FRC/AUROCの公式選択法同士の因果対決」)。
- (R5追加)「**言語学的minimal pairでの因果介入評価は我々が初**」
  (CausalGymが先行 — 差分は評価器[挙動フリップ→テキスト編集の実行]と
  対象[学習特徴化→SAE辞書latent])。
- (R5追加)「featureが複数活性に分散すること自体の発見」(Engelsが先行 —
  我々は**編集基準の操作的測定器**が新規)。

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
