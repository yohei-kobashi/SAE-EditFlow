# 論文構成案 — SAE介入の離散編集化(SAE-LEWIS / Edit Flows)

作成: 2026-07-13。S4判定(README §13.8、EDIT_FLOWS_ZERO §5)完了時点の構成。
数値は全て確定済みの実測(出典を各所に記す)。未実施はベースライン B1–B3 のみ。

## タイトル案

1. *Lifting SAE Interventions into Discrete Edit Operations: Grammaticality-Preserving Minimal-Pair Transformation*
2. *From Feature Clamping to Edit Flows: Structural Interventions on SAE Features*
3. *SAE-LEWIS: Editing as Intervention — SAE-Conditioned Edit Flows for Linguistic Minimal Pairs*

## 中心主張(全実測後に改訂 2026-07-14 — 確認測定済み)

- **C0(ヘッドライン)**: SAE条件付き介入システム(EF λ場が編集regimeを判定:
  ≤1編集なら離散編集、それ以外は指令デルタのW_dec操舵+再生成)が、**全体exactで
  全介入方式に勝つ** — 未接触の確認標本498ペアで 0.2892 vs steer0.5 0.2269
  (+80/−49、p≈0.007)・ef32単独 0.2369。規則は完全教師なし・事前登録・凍結済み。
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
- **目標文(確定稿)**: 本研究の目標は、SAE特徴空間で特定された言語現象への介入を、
  活性への連続バイアスではなく離散的な編集操作の系列として実現することである。
  これにより介入の表現力は語彙的な置換に留まらず、トークンの挿入・削除と複数箇所の
  協調を要する統語レベルの変換にまで拡張され、編集は文法性を保った最小対変換 —
  対象の言語現象のみを反転させ、それ以外を保存する編集 — として評価可能になる。
  - 用語注意: 「言語学的な最小単位」とは書かない(形態素と誤読される)。
    単位は「言語現象」、編集は「最小対変換(minimal-pair transformation)」。
- 貢献リスト:
  (i) SAE介入の2軸分類(WHERE推論 × 作用の離散性)と3欠落の指摘;
  (ii) SAE特徴デルタ条件付きEdit Flow(hazard分解 + 局所化CTMC)— 3欠落に
  一対一対応(WHERE=rate場λ、構造=離散op、文法性=凍結LM頭Q+empty→no-edit);
  (iii) LinguaLens 500ペアのゼロショットOOD評価で、連続介入・テキスト指示・
  多段カスケードに対する系統的比較(matched-pair統計付き)。

## 2. Related Work — 2軸分類が背骨

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

意義: **judge品質が人手ラベルなしで部分的に検証可能になる** — 同一比較で
自分と矛盾するjudgeは、より難しい非exactペアを裁く信頼が無い。
「judge品質は人手ラベル無しには検証不能」という以前の判断は**上方修正**。
実装 `scripts/judge_selfconsistency.py`、実行 `bash run_judge_checks.sh`。

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

- 観測された差は真の差の**保守的な下界**(減衰済み)。
- 「AのFRRは本当にBより上か」は**対応のある問い** — 同一ペアを同一judgeが
  裁いているので、不一致ペア(片方だけrealized)が全情報を担う。
  **厳密McNemar**で判定する(`scripts/frr_paired_test.py`、stub検証済:
  厳密二項値・gold不定ペアの両側除外を確認)。judgeノイズは不一致ペアに
  既に織り込まれており、ノイズ床の議論は不要。
- この訂正で nano の扱いも変わる: nanoの小さいgap(7.6pt)は「床以下だから
  無意味」ではなく、順位の一致は依然として証拠になる。

**3 judgeでの検証完了(2026-07-15)**: McNemar実行済(§6cの表)。gemmaの
再判定は §6e-2 の「減衰は正解既知の対比でのみ」の訂正も同時に生んだ
(gemma 0.0488 < nano 0.0757 で ef32−steer gapの単調性が破れた)。
**FRR側の実験は全て完了** — 残るBLEU/chrF(sacrebleu未導入でnan)は
表の空欄埋めのみ。

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

- **語順の入れ替え(movement)はDEL+INS分解のまま**: reordering families
  IoU 0.27–0.37、MOVE/pointer op(V7 A2)は将来課題。
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
