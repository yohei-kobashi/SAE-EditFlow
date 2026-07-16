# 論文構成案 — SAE特徴を**条件**とした離散編集(SAE-LEWIS / Edit Flows)

作成: 2026-07-13。**2026-07-16 に枠組みを訂正 — 下記🔴を先に読むこと。**

## 🔴 枠組みの訂正: 我々は「介入」していない(2026-07-16)

**旧題「SAE介入の離散編集化」および §2 の表は誤りだった。** コードで確認:
championの条件付け経路 `feature_token_embeds` は

```python
base = W[:, nz].t()                        # W_dec[f]
toks = self._calibrate(base) + sgn + mag   # RMS正規化 + 符号 + 大きさ
```

を **encoder入力の prefix トークン**として与える。すなわち:
1. **層0(入力)** に置く — SAEが住む**層12ではない**
2. **トークンを1個増やす** — 既存の活性を**書き換えない**
3. **RMS再正規化**して埋め込み表のスケールに合わせる — `W_dec` のネイティブ
   スケールですらない(さらに学習された `cond_scale` が掛かる)

**凍結Gemmaの活性は、どの層でも一度も変更されない。** クランプもベクトル
加算もしない。因果的に言えば:

| | やっていること | 因果表現 |
|---|---|---|
| LinguaLens | forward中に `Z_k := 10` と固定 | **`P(Y \| do(Z=z))`** = 介入 |
| ActAdd / steer(B3) | `h := h + α·v` | 介入 |
| **本研究** | `W_dec[f]` を**入力トークンとして足す** | **`P(edit \| Z=z)`** = **条件付け(証拠)** |

Pearlの意味で介入とは変数に値を設定し入力辺を切ること。**我々は何も設定
していない。証拠を与えているだけ**。我々のSAEの使い方は「**解釈可能な方向の
辞書(lookup table)**」であって介入のノブではない。

**この訂正は主張を強くする**(サーベイ4ラウンドが全て「新規性=条件付け信号」
に収束したことと整合): (a) AxBenchのsteering否定結果が**きれいに無関係**に
なる、(b)「介入の離散化」だと**彼らの土俵(活性空間)**で戦うことになり
「なぜ活性空間の指標で比較しない?」と聞かれる、(c)「条件付け」なら
**タスク(目標文を作れるか)**で比較するのが自然で、それは既にやっている。

**正しい2×2**(§2の表を差し替えること):

| | **介入**(活性を書き換える) | **入力**(モデルに与える) |
|---|---|---|
| **再生成**(連続・AR) | LinguaLens, AxBench, ActAdd, steer(B3) | **B2**(ラベルを**言葉**で与える) |
| **離散編集** | **(空白 — 誰もいない)** | **本研究** |

B2と本研究は**同じ「条件付け」の列**。違いは**仕様のモダリティ**(自然言語
ラベル vs 特徴ベクトル `W_dec[f]`)と**作用**(再生成 vs 編集操作)のみ。
B2 exact 0.1242 < 本研究 0.2237。

**用語規則**: 本研究について "intervention" / "介入" / "steering" と書かない。
"**conditioning**" / "**specification**" を使う。`RELATED_WORK.md` は既に
正しい("uses SAE features as the **conditioning signal** for a model that
emits edit operations")。

## タイトル案(訂正後)

1. *Commanding Edits with SAE Features: Discrete Minimal-Pair Transformation without Intervening on Activations*
2. *SAE Features as an Editing Specification, not an Intervention Knob*
3. *SAE-LEWIS: SAE-Conditioned Edit Flows for Linguistic Minimal Pairs*

**旧題(使用禁止)**: ~~*Lifting SAE Interventions into Discrete Edit
Operations*~~ / ~~*Structural Interventions on SAE Features*~~ /
~~*Editing as Intervention*~~ — いずれも「我々が介入している」と主張して
しまう。

## 中心主張(全実測後に改訂 2026-07-14 — 確認測定済み)

- **C0(ヘッドライン、2026-07-16用語訂正)**: SAE**条件付き**ハイブリッド
  システム(EF λ場が編集regimeを判定: ≤1編集なら**条件付き離散編集**、
  それ以外は指令デルタのW_dec**介入**+再生成)が、**全体exactで全方式に勝つ**
  — 未接触の確認標本498ペアで 0.2892 vs steer0.5 0.2269(+80/−49、p≈0.007)・
  ef32単独 0.2369。規則は完全教師なし・事前登録・凍結済み。
  **🔴 0.2839/0.2892 を「介入の結果」と呼んではならない** — headは条件付け
  編集器(EF)であり、介入(steer)はfallback。介入列のヘッドラインは別
  (C1'、P-K)。routed は「条件付けと介入の、教師なし規則による相補的結合」
  として書く。
- **C1'(介入列のヘッドライン、P-K実測待ち)**: 介入の編集性能を決めるのは
  機構ではなく**仕様** — 同じ機構(クランプ/steering)で、現象レベル仕様
  (LinguaLens FRC top-3 / AxBench 単一AUROC latent = 各プロトコル完全版)を
  我々の**事例レベル編集局所delta**に替えるだけで exact ~0(P-J予測)→
  0.17–0.23。**編集器を一切含まない介入のみの主張**で、LinguaLens・AxBench
  両プロトコルに構成的に勝つ(promptingに負けるのは許容)。
  実行: `run_protocol_e2e.sh`(両プロトコルのend-to-end + 機構クロス)。
- **C1(改訂)**: 連続バイアス介入は方向づけとテールの偶発的exactを達成できるが
  (steer0.5 0.23、clamp10 0.17 — 元の「原理的に不可能」は棄却)、**編集の規律を
  持たない**: 動作点が崖(clamp 5/10/20=0.06/0.17/0.03、steer 0.5/1/2=
  0.24/0.12/0.01)、前提保護なし(random copy 0.19–0.42)、特異性なし
  (net-FRR 0.26–0.42 vs EF 0.69)、1-op 0.09 vs EFの0.53。→ B1・B3が実証
- **C2(維持)**: テキスト描画への還元不可 — EFがexactで有意勝ち(78/45、
  p≈0.003)+B2の統制崩壊(empty copy 0.48)。→ B2が実証
- **C3(維持)**: 離散編集内でも単一フローが多段カスケードに exact +45–77%
  (有意)・simタイ。→ B4(S4)が実証
- **C4(新規・解釈可能性)**: (a) FRRは生値では全介入で飽和するが、random対照が
  偽陽性フロア(0.34–0.45)を暴き、**特異的な実現はEFのみ**(net-FRR 0.69)。
  (b) 現象を**検出**する特徴(FRC同定)と編集を**指令**する特徴(編集局所デルタ)は
  別物 — 同定集合条件付けは10×崩壊。(c) gold ΔSLOR=−0.50: 正しい最小対編集は
  流暢さを下げる方向であり、steerの−0.10は「指令された非流暢化の拒否」、EFの
  −0.98は過剰損傷0.47 — 文法性はgold ΔSLORへの近さで測る。

---

## 1. Introduction

- SAEは残差ストリームを解釈可能な基底に分解し、直接因果介入(特徴値の上書き)を
  可能にした。LinguaLens(Jing et al., EMNLP 2025)はこれを言語機構の解釈に大規模
  展開: 145言語特徴のminimal pairデータセット + PS/PN/FRCによる特徴抽出 +
  介入による因果検証。
- しかし介入そのものは3つの技術基盤を欠く(§2の分類で裏付け):
  1. **WHERE推論の欠如** — 介入位置を編集目標から導く機構がない
     (全位置一律 [LinguaLens]、全生成トークンへ強制 [CRL]、人手指定 [SAEdit])。
  2. **構造的な作用語彙の欠如** — 作用が x + c·d の連続バイアスに限られ、
     挿入・削除・並べ替え(トークン間関係の操作)を表現できない。
  3. **文法性保証の欠如** — 介入後の出力品質は劣化する(LinguaLens自身が対照群の
     非対称性を "the intervention affects overall output quality" と説明)。
- **目標文(2026-07-16 改訂 — 旧稿は「介入を…編集操作として実現する」と書いて
  おり、我々が介入していると主張してしまっていた)**:
  本研究が問うのは、**SAE特徴が編集の仕様(specification)として機能するか** —
  すなわち、活性に適用する介入のノブとしてではなく、**離散的な編集操作の系列を
  指令する条件付け信号として**使えるか、である。凍結LMの活性は一切変更しない。
  作用が離散操作であることにより、表現力は語彙的な置換に留まらず、トークンの
  挿入・削除と複数箇所の協調を要する統語レベルの変換にまで及び、編集は文法性を
  保った最小対変換 — 対象の言語現象のみを反転させ、それ以外を保存する編集 —
  として評価可能になる。
  - 用語注意1: 「言語学的な最小単位」とは書かない(形態素と誤読される)。
    単位は「言語現象」、編集は「最小対変換(minimal-pair transformation)」。
  - 用語注意2: 本研究に "intervention" / "介入" / "steering" を使わない
    (冒頭の枠組み訂正)。**"conditioning" / "specification"**。
- 貢献リスト(2026-07-16 改訂):
  (i) **特徴の入り方 × 作用**の2×2分類 — 特徴が**活性への介入**として入るのか
  **モデルへの入力**として入るのか × 作用が**再生成**か**離散編集**か — と、
  空白セル(入力 × 離散編集)の同定;
  (ii) SAE特徴デルタ**条件付き**Edit Flow(hazard分解 + 局所化CTMC)— 3欠落に
  一対一対応(WHERE=rate場λ、構造=離散op、文法性=凍結LM頭Q+empty→no-edit);
  (iii) LinguaLens 500ペアのゼロショットOOD評価で、**活性への介入**(B1クランプ・
  B3 steer)・**テキスト仕様**(B2プロンプト)・多段カスケードに対する系統的比較
  (matched-pair統計付き)。**同じ条件付け列にいるB2との対比が本命** —
  仕様のモダリティ(自然言語ラベル vs 特徴ベクトル)だけが違う。

## 2. Related Work — 2軸分類が背骨

> 🔴 **この表は「作用」列で本研究をLinguaLens/ActAddと同列に並べており、
> 「皆が介入している。彼らは連続に、我々は離散に」と読める = 誤り**(冒頭の
> 枠組み訂正を参照)。**我々は介入していない。** 差し替え先は冒頭の2×2。
> 下表は WHERE の比較としてのみ有効(そこは正しい)。

| 手法 | WHEREの決め方 | 作用 | 目標 |
|---|---|---|---|
| LinguaLens (Jing+ 2025) | 選択なし(全位置一律・定数クランプ10/0) | 連続 | 特徴顕著さの増減(自由生成) |
| CRL (arXiv:2602.10437) | 選択なし(全生成トークンに強制、no-opなし、選ぶのは特徴のみ) | 連続 (x+c·d) | タスク行動(精度・拒否率) |
| SAEdit (arXiv:2510.05081) | **人手指定**(1トークンの埋め込み) | 連続(方向×強度ω) | 画像属性の強度 |
| ActAdd / SAE-TS | 選択なし | 連続 | 行動steering |
| **本研究** | **編集目標から推論**(λ-IoU 0.74 vs empty 0.15 / random 0.33) | **離散**(INS/DEL/SUB) | 最小対変換(exact) |

- SAE介入: LinguaLens(介入は位置・スケール非選択の定数クランプ; 効果は
  人間に区別困難でLLM judge必要、enhance 48–72% vs 対照24–52%、ablationは
  並行経路で補償、品質劣化を自ら報告)。CRL・SAEditはtoken-levelを名乗るが
  「連続バイアスの適用位置/内容の選択」であり構造操作ではない。
  普遍否定は必ず "to our knowledge" + 投稿前にdeep-researchで反例チェック。
- 編集モデル(SAEなし): LEWIS、EdiT5、Levenshtein Transformer — 編集自体は
  可能だが条件付けが解釈可能性と接続しない。本研究は「編集モデルの提案」では
  なく「介入の表現力拡張」であることをここで明確化(C2の土俵設定)。
- 生成基盤: Edit Flows(arXiv:2506.09018)— CTMC上の離散フロー。本研究は
  これをSAE条件付き・編集regime(x0=入力文、x1=目標文)に移植し、
  hazard分解と局所化(付録C.1)を採用。
- minimal pair資源: LinguaLens-Data(英99特徴)。評価にのみ使用
  (訓練はcorruptionキャッシュのみ = ゼロショットOOD)。

## 3. Method

- タスク定義: 入力文 x0 と特徴デルタ (z_amp, z_sup)(= SAE(x0) と SAE(x1) の
  差分の符号分解)から x1 を生成。empty仕様→無編集の保存則を要求。
- アーキテクチャ: 凍結Gemma-2-2B + Gemma Scope layer-12/16k(JumpReLU)+
  LLM2Vec双方向化 + LoRA r=32。条件付けはfeature-tokens(特徴ごとのトークン化、
  local scope、blocklist 32特徴)。
- Edit Flow: 位置ごとの rates λ^{ins,del,sub} + 凍結lm_head Q(logit-lens
  バイアス λ=1)。**hazard分解 λ = w(t)·sigmoid(head)**(w(t)=3t²/(1−t³)、
  κ=t³)で較正を構造化 — mean p が t 安定 ~0.5(S2実測 0.50/0.48/0.46)、
  thr{F} デコード = p≥F の自己較正閾値。
- **局所化(S3)**: 付録C.1のLocalized CTMCを編集regimeへ翻訳 — 訓練は
  t*=u^{1/3}自己発火 + Pois(λ_prop·Δt)近傍伝播、損失重み λ_eff = w+λ_prop·(隣接
  ソース数)、hazardベースを b = w+λ_prop·adj に拡張(adj=編集済み隣接数、
  zero-init埋め込みでwarm-start厳密)。デコードでは編集隣接サイトの発火バーが
  p ≥ F·w/(w+λ_prop·adj) に低下 = 統語変形が要求する協調的多点編集の事前分布。
- 訓練データ: corruption cache(25 families、依存構造述語→round-trip→対称SLOR→
  blocklist)、null-record teacher(empty→no-edit を構造的に教える)。
  LinguaLensは一切使わない(ゼロショットOOD)。
- デコード: thr{F}(F唯一のノブ)+ greedy Q。本番はS3単一ckpt:
  thr0.1=exact-max、thr0.5=バランス(S4ペア統計で確定)。

## 4. 実験設定

- LinguaLens-Data 英語 500ペア(seed 42、パイプラインe2eと同一標本)。
  probe 200ペアはそのprefix → 動作点F選択に使い、**残り300ペアがholdout**。
- 指標: exact / sim_target / copy(+empty・random統制、λ-IoU count-oracle)。
- ペア統計: Δexactは不一致ペアのsign test、Δsimはpaired 95% CI
  (`scripts/compare_ef_pipeline.py`、全システムrecords.jsonlをidxで結合)。
- **FRR(LinguaLens基準の判定、`scripts/judge_feature_realization.py`)**:
  LinguaLensの介入評価はexactではなく**LLM判定の特徴顕著さ**(GPT-4o、
  enhancement/ablation成功率+ランダム特徴対照+FIC; 公式repoに判定コードは
  無く論文側プロトコル — repo確認済み 2026-07-14)。その編集regime適応:
  gold方向 = judge(src vs tgt)(判定equalは除外、システム間でキャッシュ共有)、
  システム判定 = judge(src vs 出力)(copyは自動equal)、
  **FRR = P(判定方向 == gold方向)**。A/B提示順はシード付きランダム化。
  exactはFRRの下界(exact一致は必ず実現)なので、**FRR−exactの差 =
  「方向は正しいが不正確な編集」の量**として読む。random条件のFRRを
  偽陽性フロアとして併記(論文のcontrol群の類似物)。判定はrecordsの
  既存出力に対して走る(再生成なし)。パイロットはローカルjudge
  (gemma-2-9b-it)、**論文表はGPT-4o**(judgeごとに別ディレクトリ)。
  注意: B1(クランプ)とB2(プロンプト)はこの指標でこそ得点し得る —
  C1/C2の判定はexactとFRRの両輪で行い、片方だけで主張しない。

## 5. ベースライン(比較対象)

| ID | 手法 | 守る主張 | 状態 |
|---|---|---|---|
| B1 | **LinguaLens直接介入の忠実再現**(公式repo+OpenSAE精読済 2026-07-14): "set"介入(活性なら上書き、**非活性は最小スロット置換で強制挿入**)、**残差はSAE再構成で完全置換**(デルタ加算ではない)、prompt_only=False=全位置・全ステップ、**対照=multiply×1の再構成パススルー**(=empty `recon`モード、`raw`で再構成ダメージを分離)。Gemma-2-2b-it + Gemma Scope(base学習SAE+instructモデル=彼らのLlama構成と同型)。クランプ値{5,10,20}掃引(彼らは10)+ clampZ(指令量そのまま=best-shot)。z_amp強調とz_sup切除を同時適用。中立Rewriteプロンプト(特徴テキストなし — クランプが唯一の情報チャネル)、greedy。判定はexact+**FRR両輪**(FRRこそ彼らの手法が得点し得る指標)。 | C1 | 実装済・未実行(`run_ef_b1.sh`) |
| B2 | **指示プロンプト書き換え(同一バックボーン)**: 同じ情報(amp/sup特徴の解釈ラベル)を自然言語に描画し、Gemma-2-2B-it に「この言語的変化を適用して書き換えよ」。容量を揃えた公平比較。**リスク最大 → 最初に測る**。 | C2 | **済・C2成立**: prompt8/16 とも exact 0.1242(EF 0.1904 の 65%)/ sim 0.6118。**empty copy 0.4770・random copy 0.2745 = 統制崩壊**(EF は empty 1.00 構造保証)。バケット逆転: 1-op 0.0909(EF の 1/4.3)vs 9+ 0.154(EF 超え)— 自由再生成はテールに強く最小編集に弱い。n_desc 8/16 で不感(ラベル情報は飽和) |
| B3 | **Steering vector**: B1と同じRewrite枠で介入だけ差し替え — (a) ActAdd系(minimal pair活性の平均差ベクトル)、(b) SAE-TS系(目標特徴変化を達成するsteering vector)。「SAEすら不要では」への答え + commanded-deltaに最も近い連続手法。 | C1 | 未実装 |
| B4 | **v6パイプライン**(tagger→enumeration→editor→ranker、refine・fluency gate込み) | C3 | **済(S4)** |
| B5 | input-copy(sim 0.6116)/ empty / random 統制 | 前提保護 | **済(probe組込)** |

- FTスカイライン(LinguaLensペアで直接FT)は**表に入れない**: 評価分布での
  訓練でありregime違い。脚注で「LinguaLens自身がFT優位を認める」に触れ、
  代わりに我々の**ゼロショットOOD**という設定の強さを本文で強調。
- B1–B3は全てrecords.jsonl形式で出力 → compare_ef_pipeline.pyがそのまま
  matched-pair統計まで出す。**200ペアprobeで当たり→500ペア本番**の2段階。

## 6b. 主表の確定値(runs/tables/main_metrics_{499,997}、2026-07-14)

- **997ペア(確認ブロック込み)**: routed **exact 0.2839 / SARI 65.88 / sim
  0.6554** — ef32(0.2237/63.20/0.5809)とsteer(0.2337/58.94/0.6016)を
  **3指標すべてで**上回る。ルーターは勝者を拾うだけでなく正しいregimeを
  選ぶため、副次指標も同時に改善する。oracle(exact@2候補)0.3872。
- 499ペア(全システム): routed 0.2786/66.36/0.6517、oracle(6系)0.5210。
  pipelineのSARI 40.99(copy 0.63の代償 — SARIはcopyを強く罰する)。
  BLEU/chrF列はsacrebleu未導入でnan(pip install sacrebleuで埋まる)。
- **feature別の構造(997、_per_feature.csv)— 論文の中心図**:
  - **EF(離散編集)が支配: 形態・屈折の単一トークン現象** — noun_plural
    0.727 vs steer 0.000、past_tense 0.700 vs 0、anaphor 0.875 vs 0.125、
    superlative 0.667、adjectival_suffix 0.500 vs 0、third_person_singular、
    comparative、negation_prefix、existential_quantifiers…
  - **steer(操舵再生成)が支配: 構造・多語現象** — interrogative 0.700 vs
    EF 0、cleft_sentences 0.643 vs 0.071、passive_voice 0.583 vs 0、
    of_genitive 0.667、clausal_subjects 0.625、future_perfect 0.917、
    subject_auxiliary_inversion 0.500…
  - **routedはfeature別でも概ねmax(EF, steer)を回収**(passive 0.583、
    interrogative 0.500 等 — λ場が「形態=1ハンク/構造=多ハンク」を
    教師なしで見分けている direct evidence)。
  - **両者ゼロの残余フロンティア**: 比喩系(metaphor, personification,
    hyperbole)、省略・外置(elliptical, extraposition, appositives)、
    談話(indirect_speech, echo_questions)、一部の項構造
    (direct_object, factives, resultative)— oracle 0.39が1.0でない理由の
    現象名リスト。in/near/out対応表はこのCSVから作る。

## 6c. 最終動作点のFRR(997ペア、**3 judge**、2026-07-15、
runs/tables/frr_per_feature_{hf_google_gemma-2-9b-it,openai_gpt-4o,openai_gpt-5.4-nano}.*)

主判定=**GPT-4o**(LinguaLens整合に加え、§6e-2で**自己一致率が最高**と実証)。
gemma-2-9b-it(ローカル)と gpt-5.4-nano を頑健性判定として併走。

**下表はrng修正後(§6e-1)の再判定値** — 3 judge完了(`run_frr_rerun.sh`、
2026-07-15)。

| system | gemma-2-9b-it | **GPT-4o(主)** | gpt-5.4-nano | exact |
|---|---|---|---|---|
| ef32 | **0.7894** | **0.8735** | **0.7679** | 0.2237 |
| routed | 0.7691 | 0.8306 | 0.7331 | **0.2839** |
| steer0.5 | 0.7406 | 0.7327 | 0.6922 | 0.2337 |
| steer net-FRR | 0.3234 | **0.4062** | 0.2221 | — |
| steer_rnd(偽陽性フロア) | 0.4172 | **0.3265** | 0.4701 | — |

- **judge不変な主張(これを書く)**: **3 judge全てで FRR順位が完全一致
  (ef32 > routed > steer)、かつ全judgeでroutedがexact首位** —
  「FRRとexactはシステムを異なる順位で並べる」という多軸報告の必然性が
  judge非依存に成立。正しい読みは「操舵再生成は方向づけが上手いのでは
  なく、当たった時に正確に着地する」。
- **バグはこの主張を積極的に壊していた**: バグ下のgemmaは
  ef32 0.8169 > steer 0.7609 > **routed 0.7579(最下位)** で順位が他judgeと
  食い違っていた。膨張がroutedだけを持ち上げないという診断そのままの現れ
  で、修正により3 judge一致に収束した。**修正は主張を弱めるどころか
  強くした**。
- **有意性は自己一致率ではなく対応のある検定で判定**(§6e-3、
  `scripts/frr_paired_test.py`、同一ペア・同一judgeの厳密McNemar)。

**対応のある検定(厳密McNemar、2026-07-15確定)** — 不一致ペア数 A only/B only:

| 対比 | **GPT-4o(主)** | gemma-2-9b-it | gpt-5.4-nano |
|---|---|---|---|
| ef32 > steer | +0.1408 (205/67) **1.9e-17** | +0.0488 (167/119) **5.4e-03** | +0.0757 (180/106) **1.4e-05** |
| **ef32 > routed** | +0.0429 (54/12) **1.7e-07** | +0.0203 (49/29) **3.1e-02** | +0.0348 (67/33) **8.7e-04** |
| routed > steer | +0.0980 (155/59) **3.9e-11** | +0.0285 (118/90) 6.1e-02 n.s. | +0.0409 (135/95) **1.0e-02** |

- **ef32 > steer は3 judge全てでBonferroni補正後も有意**(×3補正:5.8e-17 /
  1.6e-02 / 4.3e-05)。最も硬い主張。
- **ef32 > routed(多軸主張の核)は3 judgeとも同方向**、主judgeで
  補正後 5.1e-07、nanoで 2.6e-03。gemmaのみ補正後 9.2e-02 で有意水準割れ
  (無補正 3.1e-02)— gemmaは§6e-2で**識別力が最低**と独立に測れているので
  検出力不足として整合的に説明でき、方向の一致は保たれる。
- routed > steer は主judgeとnanoで有意、gemmaでは n.s.(補正後 0.183)。
- **多軸トレードオフを生カウントで言える(効く)**: routedはef32と
  **count-ruleがsteerに切り替えたペアでのみ**出力が異なるので、
  ef32 vs routed のMcNemarはそのまま「切り替えが何を買い何を失ったか」。
  GPT-4o判定で **実現を54失い12得て正味−42、その対価にexact一致を+60獲得**
  = **手放した実現1件あたりexact 1.43件**。FRRとexactは冗長ではなく
  **緊張関係**にあり、動作点の選択は用途が決める、という主張の具体的裏付け。
- routedのrandom対照は構造的(EF経路→copy; EF側フロアは旧ラウンドの
  ef64_rnd 0.095を引用)。steerのnet-FRRは3 judgeで0.24–0.41、いずれも
  EF側0.69を大きく下回る=特異性はEFが上、という順序も不変。
- **残余フロンティアの分解(exact=0現象 × FRR)**:
  - **方向実現は可能・正確編集が不可能**(judge横断で安定): metaphor
    (GPT-4o 0.90–1.00、nano 0.80–1.00、gemma 1.00全系)、personification、
    hyperbole(1.00全系・全judge)、echo_questions、universal_quantifiers、
    resultative(steer 0.80–1.00)— 比喩・談話系は「介入としては効くが
    minimal pairとして打てない」。
  - **judge依存で結論が割れる = 単独judgeで書けない**: `extraposition` は
    gemma 0.25–0.42 / nano 0.25–0.50 で「真に未到達」だが、GPT-4o のみ
    0.75–1.00 で「方向実現は可能」側。2:1でGPT-4oが外れ値。
    **→ 「唯一の真に到達不能な現象=extraposition」という旧記述は撤回**し、
    judge分裂の事実として報告する(§6eの自己一致率で重み付け)。
  - 接辞形態系(nominal/verbal_suffix、quantitative_prefix、
    adverbial_suffix 0.455全系)は3 judgeとも低く、判定難度の可能性を併記。
- **feature別net-FRRがsteerの見かけの実現を暴く**(GPT-4o判定でも維持):
  past_tense(steer FRR 0.200・net 0.200 vs ef32 1.000)、expressive
  (net −0.400)、subject_verb_inversion(−0.300)、split_infinitives
  (−0.308)、static_dynamic(−0.500)— randomでも同等以上に「実現」する
  =偽陽性支配の現象群。per-feature net は steer_rnd が新規ブロックのみ
  のため低n(付録扱い)。

## 6e. judge信頼性 — exact一致ペア上の自己一致率(人手ラベル不要)

**発見(2026-07-15)**: 出力がtargetと厳密一致するペアでは、システム判定
`judge(src, out)` は gold判定 `judge(src, tgt)` と**同一の比較**(同じ特徴・
同じ2文字列)。よって自己一致なjudgeは構成上必ず realized=True になる:

> **exact一致ペアに限定したFRR = judgeの自己一致率**、その 1− が
> **FRRのノイズ床**(これ以下のFRR差は解釈不可)。

§4に「exactはFRRの下界」と書いていた不変条件の、直接の実測版。実際に
GPT-4o判定で `negation_prefix` は exact 0.500 > FRR 0.400 と**下界を破って
おり**、judge が同一比較で自己矛盾している証拠(goldとsystemでA/B提示順の
ランダム化が異なる+APIの非決定性)。

意義: **judgeのreliabilityが人手ラベルなし・追加コストゼロで測れる** — 同一
比較で自分と矛盾するjudgeは、より難しい非exactペアを裁く信頼が無い。
実装 `scripts/judge_selfconsistency.py`、実行 `bash run_judge_checks.sh`。

**🔴 用語(サーベイ第3ラウンドで確定した制約)**:
1. **「judge品質を測れる」と書かない。** Norman et al. (arXiv:2606.19544)
   の題名がそのまま反論 — **"Reliability without Validity"**。我々が測るのは
   **reliability であって validity ではない**。
2. **「人手ラベル不要のjudge評価」を新規性にしない。** Sage
   (arXiv:2512.16041) が "assesses the quality of LLM judges without
   necessitating any human annotation" で先行。
3. **「同一比較の反復による一致率」を新規性にしない。** Shi et al. の
   Repetition Stability (AACL 2025)、Haldar & Hockenmaier の intra-rater
   Krippendorff's α (Findings of EMNLP 2025)、Wang et al. の Conflict Rate
   (ACL 2024) が先行。
4. **順序ランダム化を新規性にしない。** Zheng et al. (NeurIPS 2023 D&B) が
   "more aggressive approach" として既出。
5. **残る差分は限界コストのみ** = 先行研究は全て専用の再実行予算(2〜20倍の
   judge推論)を払って重複を**人工的に作る**が、我々は**評価データ内に自然
   発生する exact 一致ペアから追加コストゼロで回収**する。5系統の検索でこの
   構成の先行例は発見できず。**主張はここにだけ置く**。

**🟢 consistency–bias paradox は我々には効かない(検算済み)**: 「常に位置A
を選ぶ degenerate judge が自己一致指標で満点を取る」という標準的反論は、
gold順とsystem順が**相関する場合にのみ**成立する。§6e-1のrng修正で
`rng_gold`/`rng_sys` を分離済みなので、**always-A judge は 0.50 = チャンス
に落ちる**(4通りの場合分けで確認)。観測値は全て0.5から大きく上。
**rngバグ下なら 1.00 を取っていた** — gemmaの「1.0000 / flips=0」列はまさに
その退化だった。**バグ修正がこの反論に対する免疫そのもの**であり、
「順序を独立に引いているので degenerate judge は 0.5 に落ちる」と1文書けば
査読で潰されない。

### 6e-1. 共有rngバグ(2026-07-15に発見・修正、**要再判定**)

`judge_feature_realization.py` は gold判定とsystem判定のA/B提示順を**単一の
rngストリーム**から引いていた。`compare()` が `rng.random()` を1回消費する
ため、system判定の提示順が**goldキャッシュの有無に依存**していた:

| 実行順 | gold呼び出し | system判定が使う引き |
|---|---|---|
| routed(最初、goldを作る) | 実行 | **draw#2**(gold順と独立) |
| ef32 / steer / steer_rnd(goldキャッシュ済) | スキップ | **draw#1 = goldと同一順** |

検証(1000 idx): 旧コードでキャッシュヒット側のsystem順は **1000/1000で
gold順と一致**。よって位置バイアスのあるjudgeは ef32/steer で自分と一致
しやすくなり、**FRRが系統的に膨張**する。routedだけが無バイアスだった。
証拠: gemma(greedy=決定的)の自己一致率が ef32 1.0000/steer 1.0000
(flips=0、同一プロンプトの再実行なので構成上必ず一致)に対し
**routedのみ 0.9517**(flips 13/269)。

**方向が最悪**: 膨張はef32を持ち上げroutedを持ち上げない = 「ef32がFRR首位・
routedがexact首位」という主張と同じ向き。→ 修正済(`rng_gold` / `rng_sys`
の2ストリーム、idxのみをキー、キャッシュ状態非依存・gold順と無相関478/1000)。
`rng_gold` は旧draw#1と同一なので **gold.jsonl(997判定)は再利用可能**、
system判定のみ取り直した(`run_frr_rerun.sh`、旧ファイルは `*.orderbug.jsonl`
に退避し新旧を突き合わせ)。

**膨張量の実測(2026-07-15、3 judge。仮定ではなく測定)** — FRR(旧)−FRR(新):

| system | GPT-4o | gemma-2-9b-it | gpt-5.4-nano | 予測 |
|---|---|---|---|---|
| **routed**(対照=キャッシュミス側) | **±0.0000** | **−0.0112** | **+0.0021** | **0**(元から独立順) |
| ef32 | +0.0030 | +0.0275 | +0.0092 | >0 |
| steer | +0.0132 | +0.0203 | +0.0266 | >0 |
| steer_rnd | +0.0144 | +0.0245 | +0.0062 | >0 |

**対照実験として完璧な形**: キャッシュヒット側3システム × 3 judge =
**9条件すべてが正**(+0.0030〜+0.0275、符号検定 9/9、p≈0.004)。対照の
routedは3 judgeでゼロを跨ぐ(0.0000 / −0.0112 / +0.0021)。**予測した9条件に
だけ人工物が出て、対照3条件には出ない**。個票は動いており(GPT-4o sys一致
0.9619)集計で相殺しているのが「無バイアス」の意味。gold方向一致は全judge
**1.0000** = goldキャッシュが意図通り生き残った検証も兼ねた。

### 6e-2. judge品質(rng修正後、3 judge・全列とも正直な読み)

| judge | 自己一致率 | flips | net-FRR(true−random) | randomフロア(低=良) |
|---|---|---|---|---|
| **GPT-4o(主)** | **0.9860 [0.974, 0.992]** | 10/714 | **0.4062** | **0.3265** |
| gemma-2-9b-it | 0.9717 [0.957, 0.982] | 20/707 | 0.3234 | 0.4172 |
| gpt-5.4-nano | 0.8789 [0.853, 0.901] | 86/710 | 0.2221 | 0.4701 |

- **GPT-4oが3指標すべてで最良**、CIはnanoと明確に分離(0.974–0.992 vs
  0.853–0.901; gemmaとは僅かに重なる)。LinguaLens整合という外形的理由と、
  測定器としての品質が独立に一致した。
- nanoは **gold順を再利用していた旧ef32/steer列でも19–21 flips** — 決定的
  なはずの条件で矛盾する = 純粋なAPI非決定性。gpt-5.x系は `temperature≠1`
  を拒否するため `temperature=1` サンプリングでしか判定できず、回避不能。
  **nanoを主判定にできない技術的理由**として書ける。
- **減衰シグネチャは「正解が既知の対比」でのみ成立**(2026-07-15に主張を
  訂正): 自己一致率の降順 GPT-4o > gemma > nano に対し、
  **net-FRR(0.4062 / 0.3234 / 0.2221)と randomフロア(0.3265 / 0.4172 /
  0.4701)は単調** — 真条件がrandom条件を上回るべきという**正解が分かって
  いる**対比なので、ノイズが差をチャンス方向へ減衰させるのが綺麗に見える
  (nanoのフロア0.4701はほぼチャンス0.5 = randomな特徴を「実現した」と
  言ってしまう)。
- **ただし ef32−steer のgapは単調でない**(GPT-4o 0.1408、gemma 0.0488、
  nano 0.0757)。→ 「judge信頼性がシステム間分離を単調に予測する」という
  前版の主張は**撤回**。gemmaは**自己一致的だが識別力が低い**(同じ答えを
  安定して返すがシステムを区別できない)。二つの実システムの差は真値が
  未知なので減衰の議論を適用できない。**自己一致は必要条件であって十分条件
  ではない** — §6e-2冒頭の「一貫して間違うjudgeはあり得る」という留保が
  実際に観測された形。
- **judge間一致率(rng修正後の確定値、realized判定、n≈963–973)**:

  | judge対 | routed | ef32 | steer | gold方向 |
  |---|---|---|---|---|
  | **gemma ↔ GPT-4o** | **0.8705** | **0.8510** | **0.8756** | **0.8867** |
  | GPT-4o ↔ nano | 0.8276 | 0.8141 | 0.7850 | 0.8074 |
  | gemma ↔ nano | 0.7967 | 0.7884 | 0.7719 | 0.8044 |

  **自己一致率上位2つ(GPT-4o 0.9860・gemma 0.9717)が3システム全てで互いに
  最も一致し、nano(0.8789)が外れる** — judge品質の独立な確認。gold方向
  一致も同じ並び(0.8867 / 0.8074 / 0.8044)。§6e-2の「gemmaは自己一致的
  だが識別力が低い」と矛盾しない: gemmaはGPT-4oと**同じ判定**を返しつつ、
  システム間の**差**を小さくしか出せない(一致と識別力は別の性質)。
- 注意: 自己一致 ≠ 正確さ。主張は「GPT-4oが最良の測定器である傍証」までに
  留め、正確さそのものは人手ラベル無しには主張しない。

### 6e-3. 「ノイズ床」解釈の撤回 — 有意性は対応のある検定で

6e-2の自己一致率から「1−自己一致 = ノイズ床、これ以下のFRR差は解釈不可」
と書いていたが**これは誤りなので撤回**する。judgeのノイズは1判定ごとに
かかりシステムとほぼ独立なので、**980ペアの平均では打ち消し合って集計FRR
から消える**。ノイズがやるのは差を**チャンス方向へ減衰**させることであって、
存在しない差を作ることではない。よって:

- 観測された差は真の差の**保守的な下界**(減衰済み)。**ただし無条件では
  ない — 下記6e-4の条件が要る**。
- 「AのFRRは本当にBより上か」は**対応のある問い** — 同一ペアを同一judgeが
  裁いているので、不一致ペア(片方だけrealized)が全情報を担う。
  **厳密McNemar**で判定する(`scripts/frr_paired_test.py`、stub検証済:
  厳密二項値・gold不定ペアの両側除外を確認)。judgeノイズは不一致ペアに
  既に織り込まれており、ノイズ床の議論は不要。
- この訂正で nano の扱いも変わる: nanoの小さいgap(7.6pt)は「床以下だから
  無意味」ではなく、順位の一致は依然として証拠になる。

### 6e-4. 減衰が成立する条件 = 非差異的誤分類。**その条件は自己一致率で
検証できる**(2026-07-15、サーベイ第3ラウンドで先行研究に接続)

**先行研究**: Chen, Lu, Li, Guo, Li, "Efficient Inference for Noisy
LLM-as-a-Judge Evaluation" (arXiv:2601.05420, 2026-01) が、noisy judgeの
誤分類補正を Rogan–Gladen 型推定量として定式化している。感度
q1 = Pr(Ŷ=1|Y=1)、特異度 q0 = Pr(Ŷ=0|Y=0) として式(5):
**θ = (p + q0 − 1)/(q0 + q1 − 1)**。

**この式を反転すると我々の主張の成否が確定する**(自前で代数確認・数値検証済):

    p = θ·q1 + (1−θ)·(1−q0) = (1−q0) + θ·(q0 + q1 − 1)

観測値pは真値θの**アフィン変換**。よって**2システムの差を取ると切片
(1−q0)が消える**:

    p_A − p_B = (θ_A − θ_B) · (q0 + q1 − 1)

→ **差は (q0+q1−1) 倍されるだけでシフトしない**。judgeがチャンスより良い
(q0+q1>1)なら係数は(0,1)に入り、**差は0方向へ減衰し符号は保たれる**。
**これが「観測された差は真の差の保守的な下界」の厳密な根拠**であり、
ノイズ床の議論より遥かに強い。

**🔴 ただし前提がある — 非差異的誤分類(q0,q1が両システムで同じ)**。
差異的(judgeの誤り率がシステムに依存する)なら保証は消える。数値例:
真の差+0.14に対し、judgeがBで悪い(q=0.80 vs 0.99)と観測差は**+0.2284
= 膨張**、judgeがAで悪い(0.72 vs 0.99)と**−0.0680 = 符号反転**。
我々の設定でこれは絵空事ではない — steerの出力は全面再生成でsrcから遠く、
ef32の出力は最小編集なので、**judgeの難度がシステム間で違いうる**。

**🟢 決定的: その前提は §6e-2 の per-system 自己一致率がそのまま検証する。**
GPT-4o判定で **ef32 0.9860 / routed 0.9926 / steer 0.9781、広がりは
わずか1.45pt**。これを q として最悪ケースを組んでも観測差 +0.1399 に対し
真値 +0.1400 で**減衰が保たれる**。→ **誤分類は近似的に非差異的であり、
減衰の議論は成立する**。

**論文の書き方**: 自己一致率の per-system 列を「judge品質の読み物」で
終わらせず、**対応のある解析の妥当性条件そのものの検査**として提示する。
すなわち「我々が導入した自己一致率測定は、judge選定の根拠であると同時に、
McNemar解析が要求する非差異性の検証でもある」— 1つの測定が2つの役割を負う。

**留保**: 疫学文献には「非差異的誤分類は常に帰無方向とは限らない」という
警告がある(Jurek et al., IJE 2005; "Misconceptions About the Direction of
Bias From Nondifferential Misclassification", PMC9989338)。ただしその反例は
**多値(3値以上)の誤分類・誤差の相関・小標本**に由来するもので、
**2値判定 × 非差異的 × 対応あり**の我々の設定では上の代数がそのまま効く。
この切り分けを1文で書いておくと、統計に厳しい査読者に刺さらない。
なお arXiv:2601.05420 の補正自体(Rogan–Gladen / PPI / EIF)は
**人手ラベルの校正集合を要する**ので我々は使えない。**我々が要るのは符号と
下界だけで、それは非差異性から無料で手に入る** — この点も明記する。

**3 judgeでの検証完了(2026-07-15)**: McNemar実行済(§6cの表)。gemmaの
再判定は §6e-2 の「減衰は正解既知の対比でのみ」の訂正も同時に生んだ
(gemma 0.0488 < nano 0.0757 で ef32−steer gapの単調性が破れた)。
**FRR側の実験は全て完了** — 残るBLEU/chrF(sacrebleu未導入でnan)は
表の空欄埋めのみ。

## 6f-2. P-J実測(2026-07-16、readout delta_local、target-free現象レベル仕様)

**WHERE信号(delta1のtrue fires / random fires)**:

| r | FRC(LinguaLens式) | AUROC(AxBench式) |
|---|---|---|
| 1 | — | **0.06 / 0.07 ← 信号ゼロ** |
| 3 | **0.17 / 0.00** ← 微弱だがtrue専有 | 0.29 / 0.15 |
| 8 | 0.36 / 0.01 | 0.97 / 0.13 |
| 16 | 0.87 / 0.01 | 1.68 / 0.29 |
| 32 | 1.48 / 0.05 | 2.98 / 0.76 |
| 64 | 2.51 / 0.17 | 4.12 / 1.08 |
| (参照: 事例delta、target-peeking) | 5.39 / 1.77 | 同左 |

- **🎯 auroc_r1 = AxBenchプロトコル逐語: 因果WHERE信号が完全にゼロ**
  (true 0.06 ≈ random 0.07)。「単一latentレジームの不足」が**因果経路上の
  実測**になった — 論文の最重要セルの一つ。
- **frc_r3 = LinguaLensプロトコル逐語: 微弱だが実在**(0.17 vs 0.00、
  true専有)。彼らの介入が「効く」ことと整合するが、事例レベルの3%程度。
- **信号はrと単調に成長** — 現象レベルでも r=64 で事例レベルの半分弱まで回復
  (FRC 2.51 / AUROC 4.12 vs 5.39)。**target-freeでもWHEREの因果情報は
  回収可能**(§8の仕様出所限界への部分的回答)。
- **セレクタの質的差**: AUROCは発火が多いがrandom漏れも大きい
  (r=32で 2.98/0.76 ≈ 3.9×)。**FRCは特異性が高い**(1.48/0.05 ≈ 30×)。
  因果基準(FRC)は識別基準(AUROC)より「random対照から分離した」信号を
  選ぶ — AxBench自身の「検出とsteeringは別軸」と同型の乖離が、
  検出(AUROC優位)と因果特異性(FRC優位)の間にもある。
- exactは全セル床(確定済みのWHAT不能と整合)。empty統制は全セル完璧。
- 要確認: `runs/auroc/identified_l12_16k_r1.md` の best-latent AUROC
  (選択器が機能した証拠の sanity gate。~0.9なら auroc_r1 のゼロ信号は
  選択失敗ではなく単一latentの限界)。

## 6h. P-B対照の判定(2026-07-16実測)— **FRCの中身は実在するが、遠く不十分**

probe500_frc_{intersect,pure}_ctrl(両者同一 = **既知**の pool_topk=64 上流
切り詰めによる pure≡intersect。新規バグではない):

| | FRC-true | FRC-random(同数・同大きさ) | 事例delta(参照) |
|---|---|---|---|
| exact | **0.0140** | **0.0000** | 0.210 |
| λ-IoU | **0.5037** | 0.3440 | 0.7449 |
| copy | 0.2305 | 0.4128 | — |

- **true > random が明確**(exact 7ペア vs 0ペア、λ-IoU +0.16)→ P-Bの
  10×崩壊は**分布シフトだけでは説明されない。FRCの中身は実在の信号を運ぶ**。
- **決定的に重要な統制の性質**: true と random は**同等にミスマッチ**
  (どちらも学習が見ていない仕様型)なので、**この比較は train/test
  ミスマッチ批判に免疫**。ミスマッチが汚すのは「事例deltaとの差の大きさ」の
  解釈だけ。
- **P-Bの最終形**: 「現象レベルFRC仕様は、同条件のrandom対照に対して実在の
  WHERE/exact信号を持つ(0.014 vs 0.000、IoU 0.50 vs 0.34)が、事例レベル
  仕様の~7%に留まる。」— 3層分解と完全に整合し、P-I/P-Jの介入側
  (FRC r=16相当でWHERE微弱〜中程度、WHAT不能)とも突き合う。
  **2つの独立な経路(学習済み編集器・無学習readout)が同じ結論**:
  現象レベル同定特徴はWHEREを部分的に運び、WHATを運ばない。

## 6g. P-I v1 判定 — 因果信号はWHEREにあり、WHATに無い(2026-07-16実測)

500ペア、gemma-2-2b層12、readout(反復8・top4発火・INS/DEL/SUB)。

- **empty統制は構成通り完璧**: copy 1.0000 / fires 0.00(全9セル×4構成)。
  exact床 0.0080 = src≈tgt の退化ペア4/500(データ側の床)。
- **🟢 WHERE軸の因果検定は陽性**: true の発火数は random の**2〜3倍**
  (clamp10: 3.27 vs 1.57、delta_local: 3.60 vs 1.13; 同数・同大きさ・ID違い
  のみの対照)。**同定特徴はLM自身の予測に、randomより系統的に強い反対を
  生じさせる** — 介入→LM自身のhead という学習ゼロの経路での初の陽性。
- **🔴 WHAT(argmax p_int)は失敗**: exact 全セル床(delta1のみ+1ペア)、
  **simが copy基準0.6033 を全設定で下回る** = 発火のたびにtargetから遠ざかる。
  原因: argmax p_int はLMの次トークン事前分布に支配され、汎用トークンを返す。
  反復(5〜7発火)が複利で悪化させた。
- **バグ発見**: clamp_local と clamp_all の表が9行完全一致 → pos_mask を
  DeltaHookにしか実装しておらず**clamp経路で未消費**だった(修正済み。
  SaeClampHook.pos_mask 追加、None=B1従来動作)。
- **v2(修正)**: WHAT = **介入が最も昇格させたトークン**
  argmax_{top-50 of p_int}(logits_int − logits_base)(log-softmax差のargmaxと
  同値; 裸Δは「−25→−10」の珍トークンを拾うためtop-K制限)。v1ディレクトリは
  `--what int` のablation腕として保存。
- **読み(論文用)**: 「同定された活性は、現象が**どこ**にあるかをLM自身の
  headに教えるが、**何に**置き換えるべきかまでは指定しない」— detecting≠
  commanding の、介入経路上でのより細かい分解。

### 6g-2. P-I v2 判定(2026-07-16実測)— **WHATの失敗はWHAT規則に依らない。確定**

v2 = PMI WHAT(介入が最も昇格させたトークン、p_int top-50内)+ clampマスク修正。

- **WHERE陽性は再現**: true fires 2〜3× random が v1/v2・clamp/delta・all/local
  の全構成で再現(clamp10: 2.84 vs 1.55、delta_local0.5: 3.57 vs 1.08)。
  **同定活性がLM自身の予測を、同数・同大きさのrandomより系統的に強く動かす**
  という因果結果は頑健。
- **WHATはPMIでも床のまま**: exact 0.0080–0.0100(床=退化ペア4/500)、
  simは全設定でcopy基準0.6033未満。**LM事前分布のargmaxでも、介入の昇格方向
  でも、minimal pairの対応語は復元できない**。
- **clampマスク修正は機能**(clamp_local ≠ clamp_all になった)が、trueでは
  ほぼ無差(2.82 vs 2.84 fires)— 真の抑制特徴はソースの広範な位置で発火して
  おり、局在マスクが実質全域に近い可能性(n_masked はrecordsにあり、要確認)。
  randomでは局在が効く(0.97 vs 1.55)。
- **確定判定**: 学習ゼロの介入readoutは、2種のWHAT規則・2種の介入・2種の
  スコープ・4閾値の全てで exact を復元しない。**「介入は因果性(WHERE)を
  確立するが、正確な編集(WHAT)は条件付け経路(EF/routed)か再生成(B3)を
  要する」**。B3の自由再生成 0.2337 ≫ readout 0.01 という逆転自体が診断的:
  対応語の生成には文脈内での協調(右文脈・複数トークン)が必要で、
  teacher-forced単一位置置換では原理的に届かない — EFの双方向encoderが
  まさにこれを解いていた。
- **論文への位置づけ**: 3層の分解が完成 —
  「**検出できる**(AxBench 0.917)≠ **指令できる**(P-B 10×崩壊)≠
  **介入で書ける**(P-I: WHEREは動くがWHATが出ない)」。介入の因果的
  WHERE信号(true 2-3× random)は、SAE活性が言語現象を担うことの
  **学習フリーな因果証拠**として主張可能。exactの勝負は conditioning
  (routed 0.2839)が担う。

## 6g-3. P-I WHERE統計(2026-07-16実測、delta_local/delta0.5、500ペア)

- **対応のある符号検定: true>random 393 / true<random 31 / tied 76、
  p = 5.6e-81**。平均発火 3.57 vs 1.08(3.30×)。同数・同大きさ・ID違いのみの
  対照なので、**「同定活性はLM自身の予測を因果的に動かす」は確定**。
- **現象別WHERE表が他の表と噛み合う**:
  - **上位 = 発話行為・節タイプ**: interrogative +4.12(7/0)、expressive
    +3.30(10/0)、commisive +3.20(10/0)— 介入がLMの予測に強く反対を生む。
    interrogativeはexactでは**steerの領地**(EF 0.000/steer 0.700)だった現象
    で、因果信号は強いのに離散編集が失敗していた = WHATの協調問題の傍証。
  - **中位 = 形態・時制**: past +1.25(8/0)、third_person_singular、
    past_progressive — EFの領地と一致。
  - **底 = 比喩系**: non_synecdoche_metonymy **−0.17**(5/3、唯一の負)、
    punctual_durative +0.22(3/3)。**比喩系はFRR高・exact 0の
    「方向は実現、正確編集は不可」群だったが、因果WHERE信号すら無い** —
    同定特徴が発火位置でLMの予測を動かしていない。3層分解の最下層
    (検出できるが指令も介入もできない)の実例。
- 論文図: この表 × per-feature exact × per-feature FRR の3面で、現象ごとに
  「検出/指令/介入」のどの層まで到達するかを示せる。

## 6f. P-J — 介入する活性の選択(LinguaLens FRC vs AxBench AUROC、2026-07-16計画)

**問い**: 因果readoutで介入する活性を**現象レベル同定**で選んだとき、
(a) LinguaLensのFRC(r=3が彼らの規範)と AxBenchのAUROC(**r=1が彼らの規範**
— top-k>1は論文に定義がなく、r>1への拡張は我々のもの)のどちらが効くか、
(b) rを3,8,16,32,64に振ると何が起きるか。

**仕様はtarget-free(feature-mode pure)**: 特徴は held-out で同定、大きさは
**ソースの全域プール**から。targetをどこも見ない = §8の「仕様の出所」限界への
直接回答。**readoutは学習ゼロなのでP-Bを壊したtrain/testミスマッチ批判が
原理的に適用されない** — 現象レベル特徴が「介入では効くが学習済み編集器では
効かない」なら、特徴は現象を担っており、編集器の失敗は条件付け分布の問題と
分離できる。

- 実行: `run_readout_selection.sh`(IV/SC は run_b1_improve の勝者で上書き可)
- 参照点: instance-level k=32(target-peeking上限)/ B1 0.1743 / **B3 0.2337**
- `auroc_r1` = AxBenchプロトコル逐語、`frc_r3` = LinguaLensプロトコル逐語。
  これらが低く r=32 が効けば「単一latent/3-vectorレジームの不足」が
  **因果経路上で実測**になる
- `frc_r32_intersect`(delta∩集合)を1セルだけ置き、**仕様出所コスト**と
  **絞り込みコスト**を分解
- FRR: records は judge-ready(mode キー = `{intervention}{value}`、
  例 `delta1`)。gold cache 共有でsystem側のみ課金

**🔴 P-B・P-Jの前提を直した情報漏洩(2026-07-16発見・修正済み)**:
`identify_features_frc.py` の holdout が `random.Random(seed).shuffle`(Python
標準)で、全消費側の `np.default_rng(seed).choice`(numpy)と**別の500ペア**を
除外していた。重なりは実測48/500 → **evalの452ペアがFRC同定に混入**。
集約も `done.values()` 全部を使うためキャッシュ経由でも混入。
**両方修正**(レシピをnumpyに統一 + 集約にeval除外フィルタ。古いキャッシュ行は
自動で除外される)。**P-Bへの影響: 否定的結果なので漏洩はFRC側に有利にしか
働かず、10×崩壊は保守的 = P-Bは無傷むしろ強化**。P-J以降は修正版で生成。
AUROCセレクタ(select_features_auroc.py)は最初からnumpyレシピで漏洩なし。

## 6d. M0 — 条件付け絞り込み仮説の棄却(FRR-by-k、2026-07-15)

同一499ブロック・同一gold: k8 exact 0.060/FRR 0.7753、k16 0.126/0.7895、
k32 0.210/~0.82、k64 0.190/0.7874。**kを絞るとexactもFRRも単調劣化**
(低kはcopy 0.03 = 編集しないのではなく間違った方向に編集する)。
「少ない特徴=綺麗な条件付け→FRR改善」の機構は不在。**k=32は両指標
同時の最適点**であり、rank 17-32の特徴は現象方向と内容指定を分離不能に
符号化している(P-Bと整合)。→ M1(因果pruning)の性能目的は中止、
P-D(使用特徴数×FRC重なり)は付録分析としてのみ残す。M2(soft FRC
重みづけ)は検討・不採用。LinguaLens対比の最終形: 「現象同定1-3特徴は
検証に足るが編集の指令には~32のインスタンス特徴が必要で、それ以下は
exactも彼ら自身の基準(FRR)も落とす」。k32再判定が既存判定と完全一致
= judge決定性の確認も取得。

## 6. Results(確定済み数値 — 出典 README §13.8 S4 verdict)

- 主表(matched 499 / holdout 300):
  - パイプライン exact 0.1102 (0.1033) / sim 0.6681 (0.6744) / copy 0.627
  - **S3 thr0.1 exact 0.1904 (0.1833) = +73% (+77%)**、sign test p<0.0001 / 0.003;
    sim 0.6192 (0.6202)、−0.049 [−0.072, −0.025](実在のコスト)
  - **S3 thr0.5 exact 0.1683 (0.1633)、p=0.002 / 0.020; sim CI両標本で0を含む
    統計的タイ** [−0.036, +0.006] / [−0.041, +0.012] → **バランス王**
  - S2 thr0.5 0.1643 (0.1500)、holdout p=0.076(限界的)→ 本番はS3単一ckpt
  - S3 det sim 0.6816 (0.6854) — sim側もEFが点推定で上回るがCIが0をかすめ ns
- バケット(matched): 1-op **2.1×**(0.394–0.404 vs 0.192)、2-3 +40%
  (0.210 vs 0.148)、**4-8 2.4×(0.068 vs 0.028)= 局所性の配当**、
  9+ はEFのみ命中(0.077、1/13; holdout 0/7)。
- WHERE: λ-IoU true 0.7449 vs empty 0.1539 / random 0.3252(n=499)。
  タガーのcount-oracle 0.7472(200ペア)とパリティ — 優位は「知識」でなく
  「決定経路」(単一の較正可能スカラー vs 多段カスケードの決定損失)。
- 前提保護: empty no_edit 1.0000(全F・全ckpt・500ペア)。
- 判定: 事前登録 (b) フロンティア記録 — (a) を退けたのはholdout simの
  点推定 −0.014 (ns) のみ。

## 7. Analysis

- hazard較正の物語: S1でmean pのt減少(0.19→0.08)を発見 → S2(100k共適応)で
  t安定~0.5に解消、p≥0.5が文字通りの動作点に。
- 局所性の機構: バー低下の実例(隣接編集後にp=0.3のサイトが発火)、
  4-8/9+バケットへの転写、thrモードのsim代償(0.44 vs det 0.51)。
- F 1つでexact↔sim↔前提厳格性のフロンティアを掃ける(thr0.05→0.5の単調性)。
- 決定損失の定量: 両システム~0.74のランキングから、パイプラインは閾値→列挙→
  ゲート→ランカーで、EFは λ≥F·w(t) 1本で決定する差。

## 8. Limitations

- **🔴 最重要: 条件付け仕様の出所(2026-07-15追加。査読で必ず突かれる)**。
  我々の条件付けは `diff_intervention(z_src, z_tgt, k_amp, k_sup)` =
  **`delta = z_tgt − z_src` の top-k** であり、**実際の(src,tgt)ペアから
  計算した事例レベルのSAE特徴差分**である。運用時にtargetは無いので、
  **仕様を概念ラベルから作る問題が残る**。そしてそれは**概念→latentの引き
  当て**であり、AxBench が検出AUROC **0.695**(12手法中11位)で「弱い」と
  測っている当のものである。
  - **主張の射程をこう限定する**: 本研究の貢献は「**特徴レベルの仕様が
    与えられたとき、離散編集がsteeringより良くそれを実現する**」ことであり、
    「**仕様を概念から得る**」ことではない。
  - **P-Bはこの限界の実測**: 概念レベル仕様(FRC同定の現象特徴)に差し替え
    ると編集が10×崩壊する。
  - **🟢 AxBenchのSAE-Aは、P-Bとほぼ同一の実験のsteering版**(2026-07-15、
    PDF実物で確認): 概念ごとに**検出AUROCが最高のlatentを教師ありで選び**
    ("compute AUROC over the dataset given true labels, and select the
    highest-scoring feature by this metric")、**その同じlatentのdecoder方向で
    介入する**("we steer using SAEs by adding their decoder features directly
    to the residual stream")。結果: **検出 0.695→0.917(+32%)なのに
    steering 0.165→0.157(−5%)、winrate 48.8% = より良く検出するlatentが
    負ける**。著者自身の言葉 "better classification does not directly lead to
    better steering"。Probe(検出0.940/steering 0.098)、SSV(0.912/0.026)も
    同型で、Figure 1が両軸を直交にプロット。
    → **「現象を最もよく同定する特徴は、最もよく指令できる特徴ではない」を
    AxBenchはsteeringで、我々はP-Bで離散編集で測った。同じ結論の2つの
    modality**。AxBenchの検出軸は**我々への反論ではなくP-Bの独立再現**として
    引く(§Related Work D.2)。しかもSAE懐疑側の論文・同一ベースモデル。
  - この限定は LinguaLens と同型(彼らも「どの特徴を増減するか」を仕様として
    与えられる)なので、土俵として不当ではない。ただし**明示しなければ
    「targetを見ている」と誤読される**。
- **語順の入れ替え(movement)はDEL+INS分解のまま**: reordering families
  IoU 0.27–0.37、MOVE/pointer op(V7 A2)は将来課題。EDITOR (TACL 2021) が
  reposition の oracle を設計済み(§Related Work W-3)。
- randomのno_editが動作点で0.87とバー0.88をわずかに割る(500ペア);
  randomランキング対照0.33(条件付けへの応答性のコスト)。
- 9+バケットはholdoutで未命中(n=7)。2-3 exactのpilot値0.290は未回復。
- 英語のみ(LinguaLensは中国語もあり)、Gemma-2-2B単一スケール、
  評価はLinguaLens minimal pairsに限定。
- sim計測はembedding類似度であり文法性の直接測定ではない → B1–B3の
  出力にはSLOR(我々のfluency gate指標)を併記して非文性を定量する。

## 9. 残作業(優先順)

**一括実行: `run_paper_todo.sh`(全段done-markerガード+ペア単位resume、
「ALL PAPER-TODO STAGES DONE」まで再投入、~12-16h=2-3回)**

1. ~~B2実装+実行~~ 済(C2成立: 0.1242 vs 0.1904、empty copy 0.477)
2. Stage A: B1実行(忠実クランプ、掃引+clampZ)
3. Stage B: B3実行(steering vector = 指令デルタのW_dec描画をα掃引で加算、
   `eval_clamp_baseline.py --intervention steer`)
4. Stage C: FRR全システム+統制(B1/B3行込み、judge=gemma-2-9b-itパイロット
   → 最終表はJUDGE=openai:gpt-4oで再実行)
5. Stage D: P-A/P-C(`eval_k_sweep.py` — k掃引フロンティア、r95、オラクル
   最小k分布、sae_gain閾値によるdeployable選択)
6. Stage E: P-B(`identify_features_frc.py` — 評価500ペア除外でPS/PN/FRC同定
   → probe `--feature-sets/--feature-mode intersect|pure` で同定集合条件付け)
7. Stage F: SLOR併記(`score_slor.py`、全ベースライン+recon/raw統制)
8. Stage G: compare拡張(`--pipeline-mode`)でEF vs B1/B2/B3の直接ペア統計
9. P-B/P-D比較(FRC同定集合 vs 小k充分集合の重なり)— Stage D/Eのrecordsから
   オフライン集計
10. 投稿前deep-research: 「SAE介入×離散編集」の反例サーベイ("to our
    knowledge"の裏取り)+ 現象タイプ×インベントリ対応表(項目7、別途)

## 引用メモ

- LinguaLens: Jing et al., EMNLP 2025 Main, pp.28232–28251
  (介入手続き・品質劣化・FT劣位の引用は原文確認済み: 定数クランプ10/0、
  "not easily distinguishable by human evaluators"、"affects overall output
  quality"、combined interventionの品質低下)
- Edit Flows: arXiv:2506.09018(付録C.1 = 局所化の出典)
- CRL: arXiv:2602.10437(全生成トークン強制・no-opなし・k=1は原文確認済み)
- SAEdit: arXiv:2510.05081(1トークン手動選択は原文確認済み)
- LEWIS: Reid & Zhong, Findings of ACL 2021 / EdiT5 / Levenshtein Transformer
- Gemma Scope / LLM2Vec / SLOR系(Lau+ 2017, Kann+ 2018)は README §14 参照
