# Related Work — 草稿・サーベイ確定分・書いてはいけないこと

> **§D に英語草稿**(確定分のみで執筆可能。(c)(d)は第3ラウンド待ちのため
> プレースホルダ)。§0–Z は根拠と地雷の記録。

---

## D. 英語草稿(v1、2026-07-15。**全主張が一次資料で検証済み**)

### D.0 骨格 — 2軸分類

介入手法を **WHERE(作用位置の決め方)** × **WHAT(作用の型)** の2軸で
分類すると、本研究の位置が一意に定まる。この表が Related Work の背骨。

| 手法 | WHERE | WHAT | 条件付け信号 |
|---|---|---|---|
| LinguaLens (Jing+ 2025) | 選択なし(全位置一律・定数クランプ 0/10) | 連続 | **SAE特徴**(現象あたり**3本**) |
| AxBench の SAE/SAE-A | 選択なし | 連続 (h+αw) | SAE特徴(概念あたり**1本**) |
| ActAdd / CAA / RepE | 選択なし | 連続 | 対照ペア由来の方向 |
| Levenshtein Transformer | 編集目標から推論 | **離散** | source sequence のみ |
| LEWIS | 編集目標から推論 | **離散** | source + **スタイル分類器attention** |
| Susanto+ / EDITOR | 編集目標から推論 | **離散** | **表層の用語辞書 / 語彙選好** |
| **本研究** | 編集目標から推論(λ-IoU 0.74 vs empty 0.15 / random 0.33) | **離散** (INS/DEL/SUB) | **SAE特徴**(事例あたり k=32) |

**空白は「離散 × SAE特徴条件付け」のセル**。左下(離散編集)は条件付けが
表層に留まり、右上(SAE介入)は作用が連続に留まる。

### D.1 SAE features: detection is not command

Sparse autoencoders recover interpretable feature dictionaries from frozen
LMs [Cunningham et al. 2023; Bricken et al. 2023; Templeton et al. 2024],
and Gemma Scope [Lieberum et al. 2024] released JumpReLU SAEs for every
layer of Gemma 2 — the weights this work builds on. LinguaLens [Jing et al.,
EMNLP 2025] showed that linguistic phenomena are recoverable in this space:
it ranks base vectors by Feature Representation Confidence, the harmonic
mean of causal necessity and sufficiency, `FRC_k = 2·PS_k·PN_k/(PS_k+PN_k)`,
after an EALE sensitivity pre-filter at the 75th percentile, and verifies
the top-10 with an LLM agent.

**Crucially, LinguaLens uses those features only as an activation-space
knob**: it clamps a target feature's activation during the forward pass
(0 for ablation, 10 for enhancement) and reads off free-form generated text.
Editing appears in LinguaLens only on the *data-construction* side — its
counterfactuals are "produced through a minimal edit that deletes or
substitutes the trigger while preserving semantic content". **No prior work,
to our knowledge, uses SAE features as the conditioning signal for a model
that emits edit operations.**

That a property is decodable does not mean it is usable. Amnesic probing
[Elazar et al., TACL 2021] is a canonical statement of the point — "probing
performance is not correlated to task importance" — and it, along with
[Ravichander et al., EACL 2021; Hewitt & Liang, EMNLP 2019; Belinkov, CL
2022], reframed the question from *what is encoded* to *how information is
used*. Those works ask whether **the model itself** relies on a property
internally; we ask the adjacent question of whether **we** can use a feature
as an external command. Our P-B result extends the spirit rather than the
letter: **the very features FRC identifies as representing a phenomenon are
poor conditioning signals for editing it** — LinguaLens intervenes on 3 base
vectors per phenomenon, whereas conditioning our editor on FRC-identified
phenomenon features collapses exact match by ~10× relative to the k=32
instance-level features it needs. Detecting a feature and commanding with it
are different capabilities.

### D.2 The skeptical evidence, and why it does not reach us

The strongest negative results concern **steering specifically**. AxBench
[Wu et al., PMLR v267] evaluates on Gemma-2-2B/9B with Gemma Scope — our own
base model and SAE lineage — and concludes that "even at SAE scale,
representation steering is still far behind simple prompting and finetuning
baselines". Its two axes are concept detection and steering (`h + αw`); its
method set contains no editing method.

**AxBench's own data separates the axes we need separated**: "better
classification does not directly lead to better steering". Supervised-
selected SAE-A detects at 0.917 against vanilla SAE's 0.695 yet steers
*worse* (0.157 vs 0.165); Probe detects at 0.940 and steers at 0.098.
Detection rank and steering rank do not align. **We do not claim this
directly licenses our use** — AxBench never evaluates an editing condition —
but it does establish that a negative result on the steering axis is not a
negative result about SAE features as such, which is the axis we occupy.

Two threats remain live and we state them plainly. First, on **detection**,
vanilla SAE is "significantly outperformed by five supervised methods"
(0.695, 11th of 12). Our selection is unsupervised, so 0.695 is the relevant
number, not 0.917. We note that AxBench's detection protocol uses **exactly
one latent per concept**, so it measures single-feature identification, not
conditioning by a k=32 instance-level set; how far 0.695 transfers to our
regime is therefore not settled by that experiment. Second, [Kantamneni et
al., ICML 2025] find SAE probes fail to consistently beat logistic
regression across 113 datasets — though the finding is an absence of
*consistent* advantage, with SAEs winning on individual datasets — and
frozen/random-decoder baselines [arXiv:2602.14111] approach trained SAEs on
sparse probing. On causal editing that paper's headline 0.73 comes from its
Soft-Frozen variant, which still trains within a cosine-similarity ball; its
fully-random Frozen Decoder falls to 0.55–0.62, below trained SAEs'
0.72–0.74. DeepMind themselves listed whether SAEs beat fair baselines on
real tasks as an open problem when releasing Gemma Scope.

### D.3 Discrete editing: the operations are not the contribution

Text-editing models "produce the output text by predicting edit operations
which are applied to the inputs", in contrast to seq2seq methods that
"produce the output from scratch, token by token" [Malmi et al., NAACL 2022
tutorial]. The operation vocabulary is well-established: Levenshtein
Transformer [Gu et al., NeurIPS 2019] made insertion and deletion atomic,
LaserTagger [Malmi et al., EMNLP 2019] predicts tags over the input, and
LEWIS [Reid & Zhong, Findings of ACL 2021] — whose name we borrow — uses
insert/keep/replace/delete for style transfer. **We claim no novelty in the
edit vocabulary.**

What differs is the conditioning. LevT injects only the source sequence by
cross-attention; LaserTagger encodes only the input; LEWIS adds a supervised
style classifier's attention over a binary attribute. The closest prior work
to ours **does** condition discrete edits on an external specification: the
constrained-LevT lineage [Susanto et al., ACL 2020; Xu & Carpuat, TACL 2021]
injects terminology constraints at inference time from Wiktionary and IATE
dictionaries, and EDITOR lets users "specify preferences in output lexical
choice". **The specification there is a surface terminology dictionary; ours
is a feature dictionary recovered from the model's own internals.** The
contribution is the *kind* of specification, not the existence of one.

Edit Flows [Havasi et al. 2025] supplies our generative machinery — a
discrete flow over sequences via a CTMC, with the hazard factorization and
localization of its Appendix C.1 — but generates from scratch rather than
against a source, and conditions only on prefixes, images, or a CFG scale.

### D.4 Steering and its per-input unreliability

Activation steering [ActAdd; CAA; ITI; RepE] adds a direction to the
residual stream. Tan et al. [NeurIPS 2024] establish that "steerability
takes on a large range of values across different inputs, including negative
values, where SVs produce the opposite of the desired behaviour", coining
*anti-steerability*; on some datasets nearly half of inputs are
anti-steerable. **Their result is per-input within datasets whose mean
steerability stays positive; our negative per-phenomenon net-FRR (past_tense
−0.13, expressive −0.50, subject_verb_inversion −0.30) is an aggregate mean
going negative — a sharper form of the same failure.** Their study covers
CAA on Llama-2/Qwen-1.5 in a multiple-choice setting and involves no SAEs,
so we cite it for the phenomenon, not as a measurement of SAE steering.

### D.5 Judging the judge (反例(c): **確定**、2026-07-15 第3ラウンド)

> **🔴 新規性の置き場所を全面的に移した。** 「人手ラベル不要のjudge評価」も
> 「同一比較の反復による一致率」も**既に取られている**:
> - **Sage** (arXiv:2512.16041): "assesses the quality of LLM judges
>   **without necessitating any human annotation**... local self-consistency
>   (pair-wise preference stability) and global logical consistency" —
>   **ラベルフリーjudge評価という上位概念は先行**。ただし専用650問を新規
>   キュレーションし 19,500 judgments を回す。
> - **Shi et al.** (AACL-IJCNLP 2025) **Repetition Stability**: "evaluates
>   the reliability of LLM judges when presented with **identical queries
>   multiple times**... percentage of the most frequent selections across
>   multiple trials" — **(c)-2の字義通りの先行例**。専用サブセット+専用予算。
> - **Haldar & Hockenmaier** (Findings of EMNLP 2025, "Rating Roulette"):
>   "we ran each judge LLM on the same set of generations **independently for
>   three runs**... to measure **intra-rater variance**" / "we define
>   **self-reliability** as the agreement of a judge with itself over multiple
>   runs with the same settings"、intra-rater Krippendorff's α。**MT-Bench部は
>   pairwise 3値 = 我々のFRRと同型**。3倍コスト。
> - **Wang et al.** (ACL 2024) **Conflict Rate**: "the proportion of
>   conflicting results given by the same evaluator when simply changing the
>   position of two models" — ラベルフリー診断量。順序入替=2倍のAPI呼び出し。
> - **Norman et al.** (arXiv:2606.19544) MVVP: "N∈[3,5] independent
>   evaluations per item... reports **test–retest reliability,
>   self-consistency, and position flip rate**"。
>
> **残る差分は限界コストのみ**: 先行研究はすべて**専用の再実行予算**
> (2〜20倍のjudge推論)を払って重複を**人工的に作る**。我々は**評価データ内に
> 自然発生する exact 一致ペアから追加コストゼロで回収**する。5系統の検索で
> この構成の先行例は発見できず。→ **(c)の主張は「無償の自然発生重複の回収」
> にのみ置く**。「ラベルフリー」「反復一致率」を新規性として書いたら潰される。
>
> **🔴 用語を直す**: 「judge**品質**を測れる」と書いてはならない。Norman et
> al. の題名がそのまま反論 — **"Reliability without Validity"**。我々が測るのは
> **reliability** であって **validity** ではない。
>
> **🟢 ただし consistency–bias paradox は我々には効かない(検算済み)**:
> 「常に位置Aを選ぶ degenerate judge が自己一致指標で満点を取る」という
> 標準的反論は、gold順とsystem順が**相関する**場合にのみ成立する。我々は
> `rng_gold`/`rng_sys` を分離済み(§6e-1のバグ修正)なので **always-A judge は
> 0.50 = チャンスに落ちる**。観測値(GPT-4o 0.9860 / gemma 0.9717 /
> nano 0.8789)は全て0.5から大きく上。**rngバグ下なら 1.00 を取っていた** —
> gemmaの「1.0000 / flips=0」列はまさにその退化だった。**バグ修正が、この
> 反論に対する免疫そのもの**であり、論文でそう書ける。
>
> **🔴 順序ランダム化に新規性を主張しない**: Zheng et al. (NeurIPS 2023 D&B)
> が既に "more aggressive approach" として明示している既知の選択肢。
>
> **🟢 (c)-4 attenuation は新規性が残る**: LLM judge文脈で同じ議論を明示的に
> 行った先行研究は見つからず(§6e-4)。ただし arXiv:2601.05420 が実データで
> **q0≠q1(非対称誤り率)を観測**しているため、非差異性の前提を明示すること。

LLM judges carry position bias [Wang et al., ACL 2024; Zheng et al., NeurIPS
2023 D&B] and are self-inconsistent under repeated identical queries
[Stureborg et al. 2024; Shi et al., AACL 2025; Haldar & Hockenmaier,
Findings of EMNLP 2025]. Measuring that reliability without human labels is
established practice: Conflict Rate [Wang et al.] re-queries with the
positions swapped, Repetition Stability [Shi et al.] re-queries "identical
queries multiple times", Rating Roulette [Haldar & Hockenmaier] runs each
judge three times to obtain intra-rater Krippendorff's α, and Sage [Feng et
al. 2025] assesses judges "without necessitating any human annotation". We
follow this line rather than depart from it; our order randomization is the
"more aggressive approach" already named by Zheng et al.

**Our only contribution here is a marginal-cost one.** Every method above
manufactures its duplicates and pays a dedicated re-run budget of 2–20×
judge inference. We pay nothing: because a pair whose output exactly matches
the target makes judge(src, out) and judge(src, tgt) the *same comparison*,
FRR restricted to exact-match pairs *is* the judge's self-consistency —
recovered from duplicates that the evaluation already contains. We are
careful about what this licenses: following Norman et al. [2026], what we
obtain is **reliability, not validity**. It also escapes the standard
degeneracy objection — a judge that always picks position A would score
perfectly on a consistency metric — only because we draw the gold and system
presentation orders from independent streams, under which such a judge
scores 0.50; had the two orders been coupled it would score 1.00.

**The measurement then does double duty, and this is the part that matters.**
Noisy-judge inference [Chen et al., arXiv:2601.05420] models judge error by
sensitivity q1 and specificity q0 and corrects via Rogan–Gladen,
`θ = (p + q0 − 1)/(q0 + q1 − 1)`. Inverting gives `p = (1−q0) + θ(q0+q1−1)`,
so for two systems **the intercept cancels**:
`p_A − p_B = (θ_A − θ_B)(q0 + q1 − 1)`. A paired gap is therefore *scaled*,
never shifted — attenuated toward zero with its sign preserved — **provided
the judge's error rates do not depend on the system** (non-differential
misclassification). That assumption is not free in our setting: steer emits
full regenerations while ef32 emits minimal edits, so judging difficulty
could plausibly differ. **Our per-system self-consistency is exactly the
diagnostic**: 0.9860 / 0.9926 / 0.9781 for ef32 / routed / steer under
GPT-4o, a spread of 1.45 points, under which the worst-case gap remains
attenuated. Unlike Chen et al.'s estimators we need no human calibration set,
because we need only the sign and a lower bound, which non-differential error
supplies for free.

> **未確定**: (c)「評価データ内の同一比較を利用した人手ラベル不要のjudge
> 信頼性測定」の**明示的な先行研究があるか**は第3ラウンドが2度停止したため
> 未決着。**「先行研究はない」とはまだ書かない**。上の草稿は「新しい指標を
> 提案するのではなく、既存の指標(self-consistency)を評価データから無料で
> 得る」という**控えめな主張**にしてあるので、反例が出ても崩れない構成。
> 検証待ちの候補: arXiv:2512.16041 (Sage/IPI — ただし位置バイアス軸=(c-3)で
> あって同一比較の反復ではない、と検証者が指摘), arXiv:2606.19544
> ("Reliability without Validity"), arXiv:2405.01724 (Stureborg et al.),
> arXiv:2412.12509 (Schroeder & Wood-Doughty), arXiv:2510.27106
> (Rating Roulette), arXiv:2511.21140, arXiv:2601.20913。

### D.6 Routing by required edit size (反例(d): **確定**、第4ラウンド)

Systems that choose between generation strategies per input are usually
routed by a *learned* criterion: RouteLLM [Ong et al. 2024] trains on human
preference data to pick a strong or weak model by query difficulty and cost;
MoECE [Qorib et al., Findings of EMNLP 2024] gates GEC experts by *error
type*; ESC [Qorib et al., NAACL 2022] combines systems at the level of
individual *edits* rather than selecting a system's output; APR [AAAI 2024]
routes prompts by output quality. Closer to us, AdaEdit [Cheng et al.,
Findings of ACL 2026] switches per instance between a diff and a full
regeneration, choosing whichever is shorter — a monotone proxy for edit size
— but **internalizes that choice through supervised finetuning**, relabelling
its ground truth offline as `E_j = arg min_{S ∈ {C'_j, Diff(C_j,C'_j)}} |S|`
so that "the model implicitly learns to predict the more efficient format".
Conditional activation steering [Lee et al., ICLR 2025] does gate steering at
inference per input — **so gating steering is not itself new** — but its
condition is the prompt's semantic category, its branch is steering on/off
with both paths autoregressive, and its threshold comes from a grid search
over labelled examples.

Our router differs on the conditioning signal, not the idea. The count rule
is **explicit, unsupervised, and self-referential**: the number of edit hunks
the discrete editor's own λ field fires *is* the estimate of how much editing
this instance needs, so no router is trained, no labels are used, and no gate
is fit on a validation set. Editing-model practice, by contrast, treats "is
this task edit-shaped?" as a design-time decision over a corpus — EdiT5
[Mallinson et al. 2022] passes every input through tagging, pointing and
insertion unconditionally, and derives its "editing is favorable here" rule
from average decoder steps per task, offline.

---

第1ラウンド(deep-research、2026-07-15): 6角度 → 26出典 → 130主張抽出 →
25主張を3票の敵対的検証 → **12確定 / 13棄却**。棄却分は「使えない」の記録
として §X に残す(同じ罠を二度踏まないため)。

第2ラウンド(2026-07-15): 24出典 → 120主張 → 25検証 → **12確定 / 13棄却**。
軸4を確定。

**現状のカバレッジ**: 軸1(probing)・軸2(離散編集)・軸3(steering限界)
・軸4(SAE介入・SAE懐疑論)は確定。**反例(c)(judge自己一致率)と(d)
(ルーティング)は2ラウンド経ても検証済み知見ゼロ = 未調査**。
→ **「反例は見つからなかった」と書いてよいのは(a)(b)だけ**。(c)(d)は
「調べていない」が正しい状態で、「探したが無い」と混同してはならない。

---

## 0. 新規性主張の現状(最重要)

| 主張 | 状態 | 主張を置く場所 |
|---|---|---|
| (a) SAE特徴を条件とした離散編集 | **反例なし(確定)** | **条件付け信号**。編集語彙(INS/DEL/SUB)ではない |
| (b) SAE特徴を編集の仕様として使用 | **反例なし(確定)** | **仕様の種類**(モデル内部の特徴辞書 vs 表層の用語辞書)。「仕様で条件付ける離散編集」自体は制約付きLevTが先行 |
| (c) exact一致ペアのFRR = judge自己一致率 | **大部分が既出** | **限界コストのみ**(無償の自然発生重複の回収)。「ラベルフリー」「反復一致率」「順序ランダム化」は全て先行 |
| (c-4) 減衰(attenuation)の議論 | **新規性が残る** | LLM judge文脈での明示的な先行研究なし(§6e-4) |
| (d) 編集サイズによるルーティング | **反例なし(確定)** | **条件付け信号**(手法A自身のλ場が教師なしの編集規模自己推定器)。「編集↔再生成の切替」も「steeringのゲーティング」も先行 |

### 🎯 4ラウンドが1点に収束した — **新規性は一貫して「条件付け信号」**

(a)(b)(d) は独立に調べたのに、**生き残った差分がすべて同じ場所**に落ちた:

- (a) 編集操作は既存 → **何がそれを条件付けるか**が差分
- (b) 仕様による条件付けは既存 → **仕様がどこに住んでいるか**が差分
- (d) 手法の切替は既存 → **何が切替を決めるか**が差分

**論文の実際の貢献は「SAE特徴という条件付け信号を、離散編集という作用に接続
したこと」** であり、編集語彙でも、切替の着想でも、judge評価の枠組みでもない。
Introduction/Contributions はこの1行に収斂させる。(c)だけは性質が違うので
**限界コストの改善**として控えめに別立てする。

### 🔴 新規性の置き場所 — 編集語彙に置くと即死する

**INS/DEL/SUB という編集操作語彙それ自体に新規性を主張してはならない。**
Levenshtein Transformer (Gu et al., NeurIPS 2019) と LaserTagger
(Malmi et al., EMNLP-IJCNLP 2019) が先行し、GECToR/Seq2Edits も replace系
操作を持つ。LEWIS 自身の新規性すら編集語彙ではなく (a) 複数スパン同時編集
(b) スタイル分類器attentionテンプレート+スタイル特化MLM穴埋めによる教師
なし並行ペア合成の側にあった。

**差分は条件付け信号の側にのみ厳密に置く**: 「編集操作の語彙は既存
(Gu 2019 / Malmi 2019)。我々の差分は、その操作を**モデル内部から回収した
解釈可能な特徴辞書**で条件付ける点である」。

---

## 軸1. LLM内部の言語学的特徴 / detecting ≠ using

### LinguaLens (arXiv:2502.20344, EMNLP 2025 Main) — 我々の評価基盤かつ対比対象

SAE特徴を**活性空間の介入ノブとしてのみ**使用: forward propagation中に
対象特徴の活性値を固定(ablation=0 / enhancement=10)、出力は free-form
generated text。逐語:

> "When we modify the values of SAE's activation during forward propagation,
> we expect that such targeted interventions will influence the model's
> behavior"
> "In the ablation experiment, we set the target feature's activation to 0,
> and in the enhancement experiment, we set it to 10."

論文が挙げるSAE特徴の全8用途(活性マッピング、FRCによるbase vector
ランキング、EALE事前フィルタ、GPT-4o検証、反事実解析、介入対象、言語間
比較、因果効果測定)は**すべて解析または活性空間介入**で、編集仕様として
の用途は皆無。公式リポジトリの `Intervener` クラスも「intervention indices
指定→該当層で介入→テキスト生成」で編集操作語彙を持たない。

**🔴 先回りすべき留保**: LinguaLensは編集を**データ構築側**で使っている:

> "A counterfactual sentence s− is produced through a minimal edit that
> deletes or substitutes the trigger while preserving semantic content"

反証にはならない((a)は "as model output" と限定されるため)が、差分を
**「編集をデータ構築の道具として使う vs. SAE特徴を条件に編集操作を生成
する」**と明示しないと査読者に突かれる。

### Elazar et al., "Amnesic Probing" (arXiv:2006.00995, TACL 2021 vol.9 pp.160-175)

P-B(FRC同定した現象特徴で編集すると約10倍崩れる)の**系譜**。逐語:

> "Our findings demonstrate that conventional probing performance is not
> correlated to task importance, and we call for increased scrutiny of
> claims that draw behavioral or causal conclusions from probing results."
> "we point out the inability to infer behavioral conclusions from probing
> results and offer an alternative method that focuses on how the
> information is being used, rather than on what information is encoded."

**🔴 必須のヘッジ3点**:
1. **優先権を過剰主張しない**: "the canonical" ではなく **"a canonical /
   most-cited"**。競合定式化を併記すること — Ravichander, Belinkov & Hovy,
   "Probing the Probing Paradigm: Does Probing Accuracy Entail Task
   Relevance?" (EACL 2021, https://aclanthology.org/2021.eacl-main.295/);
   Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks"
   (EMNLP 2019); Belinkov, "Probing Classifiers: Promises, Shortcomings,
   and Advances" (Computational Linguistics 2022)。
2. **技法(INLP)は係争中**: Kumar, Tan & Sharma, "Probing Classifiers are
   Unreliable for Concept Removal and Detection" (NeurIPS 2022,
   arXiv:2207.04153); "Improving Causal Interventions in Amnesic Probing
   with Mean Projection or LEACE" (Findings of ACL 2025, arXiv:2506.11673)。
   ただし両者とも**原理を強化する方向**で反証ではない。
3. **軸のずれを正直に書く**: amnesic probingの軸は「モデルが**内部で**その
   特性を使うか」、P-Bの軸は「**我々が**特徴を外部制御信号として使えるか」。
   detecting≠using → detecting≠commanding は厳密な一般化ではなく**軸を
   またぐアナロジー**。系譜として提示し、直接的な延長として断定しない。

---

## 軸2. 離散編集 / 非自己回帰編集

### 定義枠: Malmi et al., "Text Generation with Text-Editing Models" (NAACL 2022 Tutorial, arXiv:2206.07043)

逐語(PDFのリガチャ・ハイフン折返しで素朴なgrepは0ヒット、pypdf抽出で復元):

> "Text-editing models are sequence-transduction methods that produce the
> output text by predicting edit operations which are applied to the inputs.
> In contrast, the traditional seq2seq methods produce the output from
> scratch, token by token."

著者は LaserTagger/Felix/EdiT5/Seq2Edits の当事者(Google Research)。
Table 2 が軸2の対象集合を列挙(EditNTS, Felix, GECToR, HCT, LaserTagger,
LevT, LEWIS, Masker, PIE, Seq2Edits, SL)。
**留保**: 7ページのtutorial abstractで網羅的サーベイではない → **定義枠・
整理枠として引用**し、カバレッジの典拠にはしない。

### 系譜と操作語彙

| 手法 | 年/会場 | 操作語彙 | **条件付け信号** |
|---|---|---|---|
| Levenshtein Transformer (Gu et al.) | NeurIPS 2019, arXiv:1905.11006 | insertion + deletion のみ(**SUBなし**) | source sequence(cross-attention) |
| LaserTagger (Malmi et al.) | EMNLP-IJCNLP 2019, arXiv:1909.01187 | タグ付け(語彙の正確な内容は**要再確認**) | 入力文のみ(BERT符号化) |
| LEWIS (Reid & Zhong) | Findings of ACL-IJCNLP 2021, arXiv:2105.08206 | insert/keep/replace/delete | source text + **スタイル分類器attention** |
| Edit Flows (Havasi et al.) | arXiv:2506.09018 | (**要再確認** — 0-3で棄却) | prefix / 画像 / CFGスケールのみ |

LevT 逐語: "Unlike previous approaches, the atomic operations of our model
are insertion and deletion." デコーダの分類器は Deletion / Placeholder /
Token Classifier の3つのみ。**訂正**: 置換は「複数イテレーションにまたがる」
のではなく**1イテレーション内**で実現 — "We essentially decompose one
iteration of our sequence generator into three phases: 'delete tokens –
insert placeholders – replace placeholders with new tokens'"。

LEWIS 逐語: "We propose a coarse-to-fine editor for style transfer that
transforms text using Levenshtein edit operations (e.g. insert, replace,
delete)." / §2.1 "The set of coarse Levenshtein transition types are
insert, keep, replace, and delete."

### 反例探索(a)(b) 陰性 — 条件付け信号の全在庫

- **LevT**: Appendix A 逐語 "we always omit the source information x in
  conditional sequence generation tasks such as machine translation which is
  handled by the cross-attention with an encoder on x." 全文検索で
  `style`=0, `attribute`=0。
  **🔴 査読で起きやすい混同**: §4 "Collaborate with Oracle" の "LevT has
  better interpretability and controllability" は**操作語彙**の性質であって
  条件付け信号の性質ではない。oracleは ground-truth へのアラインメント由来
  のトークン位置レベルactionで、言語的特性の仕様ではなく、policyを条件付け
  るのではなくdecode時に置き換える。この2つの "interpretable" を明示的に
  区別すること。
- **LEWIS**: 全文で sparse autoencoder / autoencoder / dictionary learning
  / sparse coding / interpretable features / feature dictionary /
  monosemanticity が**全て0ヒット**。
- **LaserTagger**: BERT-baseが入力系列のみを受ける。control code / style
  label / 潜在ベクトル一切なし。Google公式ブログの "controllable" は条件
  付けではなく**出力語彙の制限**を指す("By controlling the output phrase
  vocabulary...LaserTagger is less susceptible to hallucination")ため反証
  にならない。
- **Edit Flows**: 28ページv3 PDFを全文抽出しカウント: `sparse autoencoder`=0,
  `steer`=0, `style transfer`=0, `activation`=0, `probe`=0,
  `monosemantic`=0, `sae`=0。`interpret`=2は無関係な用法("interpreted as
  an insertion")、`feature`=1はソフトウェア的用法。39箇所の condition* 文脈
  を全列挙した結果、条件付けの全在庫は (i) ランダムprefix (ii) 画像early
  fusion (iii) CFGスケール(1.5/0.5/1.0)のみ。

### 🔴 主張(b)への最大の脅威(**未検証** — 第2ラウンドで確認中)

**制約付きLevT系譜**: Susanto et al. (ACL 2020) / EDITOR (Xu & Carpuat,
TACL 2021)。これらは「**離散編集を仕様(lexical constraints)で条件付ける**」
ことをまさに行っている。SAE-LEWISの差分は「**仕様がモデル内部から回収された
解釈可能な特徴空間に存在する**(表層の語リストではない)」点に置く必要がある。
一次資料未読 — Related Work執筆前に必ず確認。

### 🔴 軸2の残る穴

GECToR / EdiT5 / FELIX / Seq2Edits / discrete flow matching・diffusion text
editing は**通過した検証が存在しない**(Seq2Edits関連2件はいずれも1-2で棄却)。
特に「三操作語彙は新規でない」という重要な留保の根拠(GECToR/Seq2Editsが
replace系を持つ)**自体が未検証**。

---

## 軸3. Steering の限界

### Tan et al., "Analysing the Generalisation and Reliability of Steering Vectors" (NeurIPS 2024 main, arXiv:2407.12404)

我々の per-feature net-FRR 負値(past_tense −0.13, expressive −0.50,
subject_verb_inversion −0.30)の**公表済み先行事例**。逐語:

> "we rigorously investigate these properties, and show that steering vectors
> have substantial limitations both in- and out-of-distribution.
> In-distribution, steerability is highly variable across different inputs."
> "For all behaviours evaluated, steerability takes on a large range of values
> across different inputs, including negative values, where SVs produce the
> opposite of the desired behaviour."
> "Many of these datasets have almost half of the inputs being anti-steerable,
> implying that the effect of steering is highly unreliable."

"anti-steerability" は同論文の造語。著者: Tan, Chanin, Lynch, Kanoulas,
Paige, Garriga-Alonso, Kirk (UCL / FAR AI)。公表された異議は皆無で、後続
研究(arXiv:2505.22637; arXiv:2505.24859 — steeringの影響は "inconsistent,
negligible, or even counterproductive on some samples")は同方向に収束。

**🔴 引用時に必ず述べる2つのミスマッチ**:
1. **手法**: 評価対象は Contrastive Activation Addition (difference-of-means)
   **のみ**、モデルは Llama-2-7b-Chat / Qwen-1.5-14b-Chat、40のMWE系
   データセット+TruthfulQA中心の**多肢選択型**。論文中に**SAEは一切登場
   しない** → 「activation steering一般がper-input不信頼」の典拠として引用し、
   「SAE特徴steeringの直接測定」として引用してはならない。
2. **粒度**: Tan et al.の anti-steerability は**データセット内のper-input
   比率**(データセット平均のsteerabilityは概ね正のまま)。SAE-LEWISの
   net-FRR負値は **per-phenomenon の集計平均**であり、集計平均が負というのは
   **厳密により強い主張**。この非対称性は我々に有利 — Tan et al.は現象の
   存在の先行事例、負のnet-FRRは**より鋭い結果であって言い直しではない**。

**引用書式**: arXivは米綴り "Analyzing the Generalization..."、NeurIPS
proceedingsは英綴り "Analysing the Generalisation..." → 会議を引くときは後者。
二次アグリゲータの一部が "ICML 2024" と誤記しているので **NeurIPS 2024**。

---

## 軸4. SAE介入技術 / SAE懐疑論 — 第2ラウンドで確定(2026-07-15)

### 結論: AxBenchは我々を沈めない。**steering軸に閉じている**

**AxBench** (arXiv:2501.17148, "Steering LLMs? Even Simple Baselines
Outperform Sparse Autoencoders", Wu, Arora, Geiger, Wang, Huang, Jurafsky,
Manning, Potts, PMLR v267)。題名への回答として明言するのは
**representation steering がprompting/finetuningに劣ること のみ**:

> §7 "To answer the question in the title of this work: our evaluation shows
> that even at SAE scale, **representation steering** is still far behind
> simple prompting and finetuning baselines."
> §3 "We evaluate along two axes: **concept detection** C and **model
> steering** S."

steeringの定義は h_i + αw の activation addition。手法集合(DiffMean, PCA,
LAT, Probe, SSV, ReFT-r1, SAE, SAE-A, BoW, I×G, IG, Prompting,
SFT/LoRA/LoReFT)に**離散/編集系は皆無**。偽陰性対策(リガチャ正規化+
ハイフン折返し除去+`grep -a`)を施した全文検索で `levenshtein` 0 /
`edit distance` 0 / `discrete edit` 0 / `text editing` 0 / `rewrite` 0 /
`constrained decoding` 0 / `lexical constraint` 0 / `non-autoregressive` 0。
`insertion` 3件は全て合成データ生成用プロンプト文言、`edit` の全出現は
参考文献の "editors" と Meng et al. ROME の書誌のみ。

### 🟢 AxBench自身が検出軸とsteering軸の脱連関を実証している

> p.7 "Importantly, we note that SAE-A slightly underperforms the
> unsupervised SAE; **better classification does not directly lead to better
> steering**."

**著者自身のデータが存在証明**(camera-ready v3 / PMLR v267 の値):

| 手法 | 検出 (Table 1 Avg) | steering (Table 2 Avg) |
|---|---|---|
| SAE-A(教師ありAUROC選択) | **0.917** | **0.157** |
| SAE(教師なし) | 0.695 | **0.165** ← 検出で0.222劣るのにsteeringは上 |
| Probe | 0.940 | 0.098 |
| SSV | 0.912 | 0.026 |

**検出順位とsteering順位は一致しない**。Figure 1 自体が両軸を直交2軸として
プロットしている。→ 「SAE steeringが弱い」は「SAE特徴が現象を同定・条件
付けできない」を**含意しない**。

**🔴 較正(必ず守る)**: AxBenchは **editing condition を一度も評価して
いない**。よって detection↔steering の脱連関から「編集条件としての有用性
≠ steeringノブとしての有用性」への橋渡しは**類推**。**"direct support" と
書いてはならない**。正しい書き方: 「AxBenchはdetectionとsteeringが別軸で
あることを実証しており、我々の条件付け用途はsteering軸の否定的結果に直撃
されない」と、類推であることを明示する。なお SAE-A vs SAE のsteering差
(0.157 vs 0.165)は小さく原文も "slightly"、**有意検定は報告されていない**。

### 🔴 実質的脅威は検出軸 — vanilla SAE 0.695(12手法中11位)

Abstractの "On both evaluations, SAEs are not competitive" は**検出側に
ついては本文より強い断定**。本文逐語:

> "Overall, we find that DiffMean, Probe, and ReFT-r1 are the best performers
> with no statistically significant difference (p < 0.05) between any of them
> under a paired t-test. Prompt, SAE-A, and SSV are not far behind and
> significantly outperform the remaining methods. LAT also performs better
> than random. **Vanilla SAEs are thus significantly outperformed by five
> supervised methods**, all of which are much cheaper to train using a
> limited amount of synthetic data."

Table 1 Avg 実測: DiffMean **0.942** / Probe 0.940 / ReFT-r1 0.938 /
Prompt 0.929 / SAE-A 0.917 / BoW 0.914 / SSV 0.912 / LAT 0.712 /
**SAE 0.695** / PCA 0.652。
**SAE-LEWISが教師なしで特徴選択するなら該当するのは0.695の方**。

### 🟢 ただし0.695のk=32レジームへの転移には構造的限界がある(我々の防御)

AxBenchのSAE検出プロトコルは**概念1つにつきGemmaScope latentを1本だけ**
使う(encoder/decoder列ペア1組)。そのlatent単体の活性でトークンをスコア
リングし平均プーリングでpassage-level化する。→ 測っているのは**単一特徴に
よる概念同定**であって、**約32個のinstance-level特徴の集合による条件付け
ではない**。論文内に多latent版は存在しない(3-0で確認)。

### 🔴 モデル差による逃げは効かない

AxBenchは Gemma-2-2B (L10/L20) と Gemma-2-9B (L20/L31) で Gemma Scope
事前学習済みSAEを使用。**2BについてはAxBench自身がbase LM学習SAE
(= SAE-LEWISが使う `gemma-scope-2b-pt-res` 系そのもの)を使用**。
→ **同一ベースモデル・同一SAE系列**。正面から用途の切り分けで応答するしかない。

### AxBenchへの唯一の正面反論も steering軸に閉じている

Jørgensen & Hansen (arXiv:2605.31183) はAxBenchの中核所見を**生き残らせて
いる**: 自身が「promptingにはなお及ばない」と明示的に譲歩し、達成した同等性
は**LoRAに対してのみ**。→ 論争全体がsteering軸に閉じており、AxBenchの否定的
結果は「SAE steeringはpromptingに対して弱い」にnarrowされるだけ。

### その他の懐疑論

**Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda, "Are Sparse
Autoencoders Useful? A Case Study in Sparse Probing"** (arXiv:2502.16681,
ICML 2025)。SAE probeは4つの困難レジーム(data scarcity, class imbalance,
label noise, covariate shift)**すべてでデータセット平均としてロジスティック
回帰を下回り**、113データセット全体で従来手法に対する改善なし。
**🟢 限定**: 敗北は ensemble / "quiver of arrows" 基準での「**一貫した優位の
不在**」であり、**個別データセットではSAEが勝つ場合があると著者自身が明記**。

**SAEBench** (arXiv:2503.09532) の証拠は**両義的**(2-1で票が割れた):
k-sparse probingでSAEは「K本の残差ストリーム次元を直接probingする」
ベースライン(Gemma-2-2B L12で0.65)を有意に上回るが、(i) 当メトリクスは
アーキテクチャ・幅・sparsity間の識別力が乏しく Gao et al. (2024) の
"probe based metricはかなりノイジー" と整合し、(ii) Appendix Eで著者自身が
そのベースラインを **"a notably weak baseline"** と呼び、学習済みモデルの
残差ストリームprobingは**94%精度**に達すると記載。→ 結論は比較対象次第で
変わるので、**SAEBenchを一方向の典拠に使わない**。

### 🔴 最強の反論源: frozen/random SAE baselines (arXiv:2602.14111)

**abstractの見出し数値には反論可能な限定がある**(一次資料Table 2を逐語照合):
causal editing (RAVEL) で **0.73 を出したのは Soft-Frozen Decoder**
(ランダム初期化から cos類似 τ=0.8 内に留めつつ**学習を許す**変種)。
**方向を完全ランダム固定する Frozen Decoder は 0.55–0.62 に落ち**、
フル学習の 0.72–0.74 に明確に劣る。→ **「完全ランダム方向でも編集できる」
という強い主張は成立していない**。
一方 sparse probing では完全ランダムのFrozen Decoderが 0.669–0.702 と
フル学習 0.721 に肉薄 → **probing系指標こそ最も弁別力を欠く**。

**🔴 使えない防御**: 「同論文の causal editing はRAVELによる活性空間介入で
あり、SAE-LEWISの表層離散編集とはタスクが異なる」という切り分けは **0-3で
棄却**。単独の防御には使えない。Soft-Frozen/Frozen の区別で反論すること。

### 肯定側の母集団

**Gemma Scope** (arXiv:2408.05147, Lieberum et al., Google DeepMind,
2024-08) — 我々が使うSAE重みそのものの出所であり、AxBenchとの共通の土俵を
成立させている。**🟢 使える**: SAE公開側の当事者であるDeepMind自身が2024年
8月時点で「SAEが公正なベースラインと比較して実タスク性能を改善できるか」
「steering vectorとSAE feature steering/clampingの比較」「SAEが本当にモデル
の『真の』概念を見つけているか」を **open problem として列挙**しており、
**SAEの下流有効性が当初から未検証だったことの一次的根拠**になる。

**未確認(書誌情報だけなら足りるが、逐語引用するなら要追加確認)**:
Towards Monosemanticity (Bricken et al. 2023), Scaling Monosemanticity /
Golden Gate Claude (Templeton et al. 2024), Cunningham et al.
(arXiv:2309.08600), JumpReLU SAE (arXiv:2407.14435), TopK SAE (OpenAI),
OpenSAE。

### 軸4で棄却された主張(**書いてはいけない**)

| 棄却 | 票 | なぜ |
|---|---|---|
| 「AxBenchのSAE不振はNeuronpediaラベルによる特徴選択のconfoundのせい」 | 0-3 | **この言い訳は使えない** |
| 「AxBenchは steering as production intervention と features as causal ... を明示的に分離している」 | 0-3 | そんな分離は書かれていない |
| 「Kantamneni et al.は自らscopeをprobingに限定し他タスクへの一般化を自ら否認」 | 0-3 | 自己否認はしていない |
| 「懐疑論文自身がSAEが最先端である応用の存在を譲歩している」 | 0-3 | 有利すぎる譲歩は取れない |
| 「arXiv:2602.14111のcausal editingはRAVEL活性空間介入で我々とタスクが違う」 | 0-3 | **単独の防御に使えない** |
| AxBench v1 の数値 0.918 / 0.305 / 0.315 | 改訂済 | v3本文に0ヒット。v1のSAE-A 0.918はGemma-2-2Bの2設定平均をSAEの4設定平均と比べる**非対称比較**。**camera-ready (v3/PMLR v267) の値を引くこと** |
| 「最良のDiffMean(0.938)」 | 転記ミス | **0.938はReFT-r1、DiffMeanは0.942** |

---

## W. ブロッカー2件を一次資料PDFで解決(2026-07-15、自前でpypdf抽出)

要約器の誤りが多発したため、**arXiv/ACL AnthologyのPDFを直接落として
pypdfで全文抽出し、正規化(リガチャ+ハイフン折返し)した上で自分でgrep**
した。以下は**すべて実物からの逐語**。

### W-1. 🟢 P-Bの対比対象が確定 — 「介入は3本、編集には32本」

第1ラウンドで0-3棄却された定式化は**実際にはほぼ正しかった**(検証側の
偽陰性)。実物 §3.2 / Appendix E.1 逐語:

> "harmonic mean to penalise vectors that are only sufficient or only
> necessary: **FRC_k = 2· PS_k PN_k / (PS_k + PN_k)**."
> "The Feature Representation Confidence (FRC) is computed as the **harmonic
> mean of PN and PS**: FRC = 2 PN PS / (PN+PS). The harmonic mean is chosen
> because it ensures that FRC remains low if either PN or PS is low..."

**特徴同定パイプライン全体**(逐語):

> "We first perform **sensitivity pre-filtering** by computing EALE_k for
> every base vector and retaining those whose absolute value exceeds the
> **75th percentile**; on this reduced set we estimate PS_k and PN_k from
> every ⟨s+, s−⟩ pair and rank the vectors by their FRC_k; finally, the
> activation distributions of the **top-10 ranked vectors** are passed to an
> LLM agent, which verifies that each vector genuinely encodes the intended
> linguistic feature and flags any inconsistent or spurious patterns."

**🔴 P-Bの核心 — 介入に使う特徴数(逐語)**:

> "Furthermore, for each linguistic feature, we select **three base vectors
> with the highest FRC as representatives for intervention** and compute the
> average results across these three interventions."
> "We select **6 representative features** for the intervention experiments."
> (= 6つの言語現象。SAE特徴数ではない — 混同しないこと)

→ **対比は「LinguaLensは現象あたり3本のbase vectorで介入」 vs 「SAE-LEWISは
編集にk=32のinstance-level特徴を要する」**。**3 vs 32 ≈ 10×** で、P-Bが観測
した**10×の崩壊と数字が一致する**。P-Bはこれで書ける。

その他の逐語(使える):
- EALE = 個別latent効果の集約: τ_k(s) = a(1)_k − a(0)_k、
  "EALE_k = (1/N) Σ τ_k(s_i), which can rank base vectors by their
  sensitivity to the specified phenomenon"
- FIC (Feature Intervention Confidence) = 正規化ablation/enhancement効果の
  調和平均。"if one or both of the E values are negative, we incorporate a
  penalty coefficient w"
- **データ構築側の最小編集(第1ラウンドの警告を実物で確認)**: "produced
  through a **minimal edit** that deletes or substitutes the trigger while
  preserving semantic content"
- **🔴 LinguaLens本体は OpenSAE on Llama-3.1-8B(32層)** — Gemmaではない。
  "For SAEs, we use OpenSAE (THU-KEG, 2025) and its released checkpoints on
  32 layers of Llama-3.1-8B"。我々のB1はGemma+Gemma Scopeへの**忠実移植**
  である旨を明記すること(AxBenchのGemma論点とは土俵が別)。
- **題名**: PDF実物は **"LinguaLens: Towards Interpreting Linguistic
  Mechanisms of Large Language Models via Sparse Auto-Encoder"**。
  ar5ivは旧題 "Sparse Auto-Encoder Interprets Linguistic Features in Large
  Language Models" を返す → **投稿時に版と題名を確認**。

### W-2. 🟢 主張(b)の最近接脅威 = 制約付きLevT系譜 — 差別化点が確定

| | Susanto et al. (ACL 2020) | EDITOR (Xu & Carpuat, TACL 2021) |
|---|---|---|
| 仕様の中身 | **terminology constraints**(Wiktionary / IATE **用語辞書**、En-De bilingual dictionary entries) | **soft lexical constraints** = ユーザの **output lexical choice** の選好(トークン列) |
| 与え方 | **decode時に注入**、学習改変なし | imitation learningで学習 + decode時 parallel edits |
| SAE/解釈可能性の語彙 | **全て0**(sparse autoencoder/autoencoder/dictionary learning/interpretable feature/activation/steering/monosemantic/probe/latent) | **全て0**(同上) |

逐語 — Susanto: "our method **injects terminology constraints at inference
time** without any impact on decoding speed" / "Our method does not require
any modification to the training procedure and can be easily applied at
runtime **with custom dictionaries**" / 制約の出所は "En-De bilingual
dictionary entries extracted from **Wiktionary**" と "the **Interactive
Terminology for Europe (IATE)** terminology database"。

逐語 — EDITOR: "makes sequence generation flexible by seamlessly allowing
users to **specify preferences in output lexical choice**"。

**→ 差別化点は予測通りに確定**: 制約付きLevT系譜は確かに「離散編集を仕様で
条件付ける」が、その**仕様は表層の用語辞書/トークン列**である。SAE-LEWISの
仕様は**モデル内部から回収された解釈可能な特徴辞書**にある。**仕様の「種類」
が違う**と書けばよく、「仕様で条件付ける離散編集は前例がない」とは書かない
(それは即座に潰される)。

### W-3. 🟢 EDITOR は我々のM1 MOVE(NO-GO)の先行研究 — §7 Analysisで引く

EDITOR abstract 逐語:

> "It relies on a novel **reposition operation** designed to **disentangle
> lexical choice from word positioning decisions**, while enabling
> **efficient oracles for imitation learning** and parallel edits at
> decoding time."

我々のM1 MOVEは「syntax-decidableな移動可能性 → λ^mov が不整合な条件付けの
下で発火」で失敗した(empty ranking 0.15→0.39、random no_edit 0.50)。
EDITORは**まさにその oracle 設計を解いている**(語彙選択と位置決定の分離)。
→ M1のNO-GO考察は「op語彙の拡張は必要だが十分でない」で止めず、
**「EDITORはrepositionのoracleを設計して解いた。我々のMOVEは条件付け
コントラスティブな教師なしにop語彙だけ足したので失敗した」**と、
**先行研究に照らした具体的な失敗原因**として書ける(P5 corruption提案の
直接の論拠にもなる)。

---

## Z. 残る未達

| 項目 | 状態 | 影響 |
|---|---|---|
| **(c) judge自己一致率** | 第3ラウンド実行中 | 「探したが無い」とまだ書けない |
| **(d) 編集サイズによるルーティング** | 第3ラウンド実行中 | routedシステムの新規性主張に直結 |
| AxBenchの正確な会場 | 未解決 | PMLR v267 (wu25a) は確認済だが OpenReview forum K2CckZjNy0 の記述と食い違う。**ICML 2025 / ICLR 2025 のどちらか要確定** |
| vanilla SAE 0.695 への応答方針 | 未決 | 我々の特徴選択が教師あり/教師なしかを明確化し、教師なしなら0.695側にどう応答するか決める(**単一latent vs k=32集合**の構造的切り分けが本命) |
| arXiv:2602.14111 への random-feature control | 未決 | Soft-Frozen/Frozen の区別による反論に加え、実証的controlを実装するか |
| 肯定側母集団の逐語確認 | 未 | Bricken 2023 / Templeton 2024 / Cunningham 2309.08600 / JumpReLU 2407.14435 / TopK / OpenSAE。**書誌情報のみの引用なら不要** |

---

## X. 棄却された主張 — **書いてはいけないこと**

3票の敵対的検証で 0-3 / 1-2 となり**使用禁止**。同じ罠を二度踏まないための記録。

| 棄却された主張 | 票 | なぜ危険か |
|---|---|---|
| ~~「LinguaLensはtop-3のFRC-ranked base vectorのみで条件付け、FRC=PSとPNの調和平均」~~ | ~~0-3~~ | **⚠️ この棄却自体が誤り(検証側の偽陰性)**。§W-1でPDF実物から逐語確認: FRC=調和平均の式も「介入は現象あたりtop-3のbase vector」も**本文にそのまま存在する**。複合主張だったため一部の不正確さで全体が落ちたと見られる。**3票検証にも偽陰性がある**ことの実例 — 棄却されても一次資料で確認する価値がある |
| 「LinguaLens自身がSAE介入の制御信頼性の低さを認めている」 | 0-3 | 「SAE陣営自身による自認」という**有利すぎる論拠は使えない** |
| 「Edit Flowsの操作語彙はちょうどINS/DEL/SUBでMOVEなし」 | 0-3 | Edit Flowsの語彙の正確な内容は再確認が必要 |
| 「LaserTaggerの語彙はKEEP/DELETE/ADDちょうど3種でSUBは非プリミティブ」 | 0-3 | 同上 |
| Seq2Edits関連2件 | 1-2 | 軸2のSeq2Edits記述は**典拠なし** |
| 「steering vectorはOODでプロンプト変更に脆い」 | 0-3 | **Tan et al.のOOD側の主張は引用しない**。in-distribution側のみ3-0で通過 |
| 「LEWISはSAE interpretabilityの系譜全体に先行する」 | 事実誤り | Yun, Chen, Olshausen & LeCun, "Transformer visualization via dictionary learning" (arXiv:2103.15949, DeeLIO@NAACL 2021) がLEWISの**2ヶ月前**、SPINE (Subramanian et al., AAAI 2018) はさらに前。正しくは「LEWISは **Gemma Scope (2024) と LLM-SAEの系譜(Cunningham et al. 2023; Bricken et al. 2023)に**先行する」と狭く書く |
| 「LEWISには言語的特徴による条件付けが一切ない」 | 過剰 | スタイル分類器attentionという**学習された帰属信号**で条件付けており、「特徴を編集仕様に使う」の**最近接の隣人**。差別化は「**辞書としての解釈可能性**」に置く |

---

## X-2. 軸(d)で棄却された主張 — **書いてはいけない**(第4ラウンド)

| 棄却された主張 | 票 | なぜ危険か |
|---|---|---|
| 「AdaEditは**単一モデルの出力フォーマット切替**であって手法間ルーティングではない」 | **0-3** | 🔴 AdaEditを反例から外すために私が使いたくなる論拠。**使えない**。差分は教師あり(SFTで暗黙内在化)vs 明示的教師なし、の1点にのみ置く |
| 「AdaEditの基準は**コスト効率**なので『必要編集量による振り分け』の問いの対象外」 | **0-3** | 🔴 同上。トークン数は編集量の単調プロキシであり、逃げられない |
| 「EdiT5は1モデル内の**経路分岐**を持つ」 | 事実誤り | 全入力が tagging→pointing→insertion を**無条件・固定順**で通過。逐語: "During inference, we first greedily set yt..., then π..., and finally yd..."。ゲート・条件付き項・混合重みは皆無。全文grep: router=0, routing=0, dynamic=0, per-example=0 |
| 「Sequence-to-Action は編集量ベースのルート選択を行う」 | 捏造(第3R) | 要約器の合成。原典にその主張なし |
| 「Steer2Edit は steering→離散編集の切替を行う」 | false friend | そこでの "editing" は **component-levelの重み編集**(steering vectorを診断器にrank-1編集)。離散テキスト編集ではない |
| 「TemplateGEC は seq2edit と seq2seq を切り替える」 | 誤り | Seq2Edit(GECToR)は**検出にのみ**使い、**常に** Seq2Seq へ流すパイプライン。ルーティングではない |

**🔴 CAST (ICLR 2025) による表現制約**: 「steeringに推論時ゲートを付けた初の
研究」と**書いてはならない**。CASTが先行(逐語: `h' ← h + f(sim(h, proj_c h))·α·v`、
`f = 1 if sim > θ else 0`)。ただし反例ではない — ゲート基準は**プロンプトの
意味カテゴリ**(hate speech/adult content等)で編集量ではなく、分岐は
steering ON/OFF の二択で**両枝とも自己回帰生成**(離散編集経路が無い)、
閾値・層・比較方向は**ラベル付き正負例上のF1最大化グリッドサーチ**で決まる
教師ありゲート。camera-ready全26頁で `edit distance`=0, `number of edits`=0。

## Y. 執筆時の手続き規則

1. **逐語引用はすべて自前のPDF全文抽出で再確認する。二次要約器を信用しない。**
   検出された捏造/誤りは**計6件**:
   - Edit Flowsについてabstractと正面から矛盾する記述(第1R)
   - LEWISについて存在しない大文字タグ(KEEP/REPLACE)を「逐語」と称した
     記述(第1R。実物は小文字と `<ins>`/`<repl>` マーカーのみ)
   - LinguaLensの介入特徴数を「5」と回答(実物は「**3本のbase vector**」/
     「**6現象**」)(W-1)
   - LinguaLensのEALEを「NOT FOUND」と回答(実物は定義あり・2ヒット)(W-1)
   - LinguaLensの題名を旧題で回答(W-1)
   - LinguaLensに存在しない一文 "We consider a base vector to be strongly
     related ... if its sufficiency probability exceeds 0.9" を §4.2.1 の
     逐語と称して回答 — **PDF全文に該当なし**(W-1)

   **確立した手順**: `curl -sL -o x.pdf <arxiv/aclanthology PDF URL>` →
   pypdfで全ページ抽出 → NFKD正規化 + `ﬁ`/`ﬂ` 展開 + `-\n` 除去 + 空白畳み
   → 自分で `re.finditer` する。
2. **grepの0ヒットを信じない。** PDFのリガチャ("ﬁrst")とハイフン折返しで
   偽陰性が出る(NAACL tutorialで実際に発生)。上記の正規化を必ず通す。
3. **3票の敵対的検証にも偽陰性がある。** LinguaLensのFRC/top-3の主張は0-3で
   棄却されたが、実物には逐語で存在した(§W-1)。**複合主張は一部の不正確さで
   全体が落ちる**。棄却されても、論証の根幹に関わるものは一次資料で確認する。
4. **「調べていない」と「探したが無い」を区別する。** 現状「反例なし」と
   書いてよいのは (a)(b) のみ。(c)(d) は第3ラウンド待ち。
5. **版番号と題名を投稿時に再確認する。** LinguaLensはarXiv旧題と現題が
   異なる(§W-1)。Tan et al. v6 (2025-05)、Edit Flows v3 が最新。
   AxBenchは camera-ready (v3 / PMLR v267) の数値を引く。
