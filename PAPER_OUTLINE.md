# 論文構成案 — SAE介入の離散編集化(SAE-LEWIS / Edit Flows)

作成: 2026-07-13。S4判定(README §13.8、EDIT_FLOWS_ZERO §5)完了時点の構成。
数値は全て確定済みの実測(出典を各所に記す)。未実施はベースライン B1–B3 のみ。

## タイトル案

1. *Lifting SAE Interventions into Discrete Edit Operations: Grammaticality-Preserving Minimal-Pair Transformation*
2. *From Feature Clamping to Edit Flows: Structural Interventions on SAE Features*
3. *SAE-LEWIS: Editing as Intervention — SAE-Conditioned Edit Flows for Linguistic Minimal Pairs*

## 中心主張(3本)— 各ベースラインはこのどれかを守る

- **C1**: 既存のSAE介入(連続バイアス)では、文法性を保った統語レベルの最小対変換は
  原理的に実現できない(3つの技術基盤の欠落)。→ B1・B3 が守る
- **C2**: SAE特徴デルタによる条件付けは、特徴記述をテキスト指示に描画する方式に
  還元できない(または匹敵しつつ介入としての統制性で優る)。→ B2 が守る
- **C3**: SAE条件付き離散編集の中でも、単一のEdit Flowモデルが多段カスケードを
  exactで有意に上回り(+45–77%)、simは統計的タイ。→ B4(済・S4)が守る

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

1. **B2実装+200ペアpilot**(リスク最大: ここで大差負けなら骨格再考)
2. B1実装(クランプ値掃引込み)+200ペアpilot
3. B3実装(ActAdd / SAE-TS の2変種)
4. B1–B3を500ペア本番 → compare_ef_pipeline.pyで主表完成
5. SLOR併記(B1–B3の非文性定量、§8)
6. 投稿前deep-research: 「SAE介入×離散編集」の反例サーベイ(CRLが2026-02と
   新しく後続の可能性; "to our knowledge"の裏取り)

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
