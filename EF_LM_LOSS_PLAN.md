# 論文コア実験パイプライン(2026-07-18 改訂2、ユーザー指示)

本ファイルはユーザーとの目標統一で確定した計画の記録。変更は必ずユーザー確認。

## 0. 論文のコア表(ユーザー指示 2026-07-18)

**データ = LinguaLens-Data(英語)。全手法を L4 / L12 / L20 の3層で実行し、
featureごとのFIC と exact の2指標で評価する。** 掲載する手法は以下の4つのみ
(それ以外の実測は当面掲載候補から外す):

| 腕 | 手法 | 介入の描画 | プロンプト |
|---|---|---|---|
| A1 | **LinguaLens忠実**(公式github準拠) | set介入(active上書き+min-slot強制挿入、値10/0)+残差をSAE再構成で完全置換、全位置・全ステップ | なし(素テキスト — 彼らのrepoはchat template不使用) |
| A2 | **AxBench忠実 steer** | addition steering: h + factor·max_act·(W_dec方向)、全位置・全ステップ(彼らのAdditionIntervention) | なし(AxBenchのsteering自体は概念情報をプロンプトに入れない。編集タスク化にあたり指示プロンプトも使わない) |
| A3 | **AxBenchのprompting法の移植** | SAE不使用。彼らのprompt-steeringテンプレートを最小改変で移植し、feature記述を言葉で与えて生成 | あり(この手法の定義そのもの) |
| A4 | **本手法**(EF through-LM editor) | 学習editorのΔh(λ場×内容場)をresidual streamに注入 | なし(bare枠) |

- A1/A2/A4 は**同一の事例レベル仕様**(層LのSAEでのz_amp/z_sup、編集局所
  プール+層別blocklist、k64/64)を使う — 差は「描画機構」のみに帰着させる。
- 生成フレームは A3 以外すべて共通: `[BOS]+src(+\n)` の素テキスト継続。
- 統制: random仕様(A1/A2/A4)、empty、raw。FICの定義上randomは必須。

## 0b. 評価の定式化(2026-07-19ユーザー確定 — 反実仮想整合を主軸に)

実現目標 = 「介入により、入力x0の内部表現をx1であったかのように書き換える」。
評価はこの定義に直接従う:

1. **KL/NLL整合(主・judge不要)**: 介入下の継続分布 p(・|x0+Δh) を、
   凍結LM自身の p(・|x1入力) と比較(境界+参照継続のteacher-forcing)。
   ベースライン = p(・|x0入力) との距離。
2. **FIC(主・LinguaLens逐語)**: 裸のsrc→自由継続(\nなし、temp1.0)を
   介入下で生成し特徴顕著性をjudge — 公式example枠と完全一致。
3. **exact(従・我々の指標)**: 編集後の文の復唱。**2026-07-19にフォール
   バックを本採用: src先行復唱プロンプト枠**(復唱率99%実証、A1/A2も同枠で測定済み)。旧`\n`枠は廃止。
   **フォールバック(ユーザー決定)**: 現行系列のexactが振るわない場合、
   明示指示プロンプト「入力文を復唱してください。」枠に切り替える
   (プロンプトは復唱能力のみ供給し編集情報ゼロ。介入が「何を復唱するか」を
   決める)。**その場合はA1(LinguaLens式)・A2(AxBench式)のexact実験も
   同枠で取り直す**(比較の公平性)。

## 1. 指標

- **exact**(+sim/copy 併記): 標準probe(499ペア、seed42)で全腕。
- **FIC(featureごと)**: LinguaLensの介入評価の編集タスク版。
  - enhancement試行: s1(特徴なし側)に +feature 介入して生成 →
    judgeが特徴の存在を判定。ablation試行: s2に −feature 介入 → 不在判定。
  - E_enh / E_abl = targeted成功率 − random対照成功率、FIC = 彼らの
    ペナルティ付き結合(w=0.5、実装は judge_ll_repro.py の検算済み式)。
  - サンプリング: featureごとに層化(全英語featureを対象、ペア数は§4)。
- デコード: **FIC系は彼らのrepo既定(temperature 1.0, do_sample, 100tok)**、
  **exactは同一フレームのgreedy**(別パス)。両方recordsに保存。

## 2. 忠実性の再監査(ユーザー指摘: 論文を再現できていない)

実装前に公式githubと逐語照合し、結果を `reports/audit_ll_axbench.md` に記録:
- LinguaLens (THU-KEG/LinguaLens lingualens/intervener.py):
  set/multiplyの意味論、prompt_only、生成パラメータ、対象latentの平均化
  (FRC top-3を1本ずつ)、randomベースラインの本数の読み。
- AxBench (stanfordnlp/axbench): AdditionIntervention の正確な式
  (factor×max_act×単位方向か生W_decか)、factor格子、prompt-steering法の
  テンプレート全文、彼らのjudge/デコード設定。
- 差分が見つかれば修正してから本番実行。

## 3. 本手法(A4)の学習 — 変更なし(改訂1の内容を維持)

- **層ネイティブEF基盤(2026-07-20)**: 各層でS2構成100k→S3局所化50kのEFチャンピオンを学習し、そのエンコーダをwarm startに使う(L12=既存S3。L4移植劣化の教訓)。
- EFIntervener: LLM2Vecエンコーダ+feature-tokens条件付け、λ_i=σ(rate_head)、
  v_i=content_head(ゼロ初期化)、Δh_i=λ_i·v_i。恒等スタート、steer基底なし。
- 学習: bare枠 `[BOS]+x_t+\n+x1`、through-LM NLL(x1)、中間状態x_t
  サンプリング(t0-prob 0.5)、null教師(empty 0.08 / mismatch 0.12、
  ノルム抑制のみ)、ノルム予算 0.5·||dvec_L||、40k steps、10k時点でprobe100。
- 層ごとに独立学習(L4/L12/L20)。データ=多層zサイドカー(工夫1、実行中)。

## 4. 実行計画とコスト

| 段 | 内容 | 規模 |
|---|---|---|
| P0 | 忠実性監査(§2)+ 統一probe実装(4腕×FIC/exact) | 実装0.5-1日 |
| P1 | サイドカー(実行中)→ L12学習 → probe100ゲート → L4/L20学習 | short-g 3本 |
| P2 | exact probe: 499ペア × 4腕 × 3層(A4はeflm-final) | 層ごとジョブ内 |
| P3 | FIC probe: featureごとn_pairsペア × 両方向 × {targeted, random} × 3腕(A1/A2/A4)+ A3 + raw参照 | GPU生成が支配的 |
| P4 | FIC judge(API)→ feature別 FIC×3層×4腕の表・層プロファイル図 | API課金 |

**FIC規模(ユーザー決定 2026-07-18): featureあたり5試行**(n_pairs=5、
featureごとに異なる5ペア×各1生成。彼らの50は「単一プロンプト×temp1.0の
50リサンプル」なので、実データ5ペアで置き換える)。
生成 ≈ 98feature × (2方向×5ペア×2[targeted/random] + control10) × 4腕 × 3層
≈ 35k(~30-40 GPU時、層別ジョブに分割)。judge呼び出し ≈ 23.5k。
judge = **gpt-4o(公式=LinguaLens論文のjudgeに一致、ユーザー決定 2026-07-18)**。
コスト目安: ~23.5k呼び出し × ~600入力トークン(ペア比較・出力6トークン)
≈ $35-40。
FICの位置づけ(監査で確定): 彼らのFIC原実験は5 feature × 手書きプロンプト
1本 × 50リサンプル(LinguaLens-Data不使用)。我々の全feature×実ペア版は
「彼らの指標を彼らのデータセットへ拡張」と論文に明記する(再現アンカーは
既存の4 feature忠実再現が担う)。

## 4b. T5 — LinguaLens再現の見直し(3層、2026-07-18ユーザー指摘で追加)

- T5-1: 監査で発見した不一致の修正 — 公式はプロンプト込み全文をjudge対象に
  する(我々の再現は継続のみだった)。gen/judgeスクリプトを公式一致に修正。
- T5-2: FRC同定(リーク修正版レシピ)を**L4/L20でも実行**(従来L12のみ)。
- T5-3: 4-feature D.1再現実験(set 10/0、n=50の原設定)を**L4/L12/L20で
  再実行**(L12も修正版で取り直し)。
- T5-4: judge(gpt-4o)再実行 → Table 2アンカー再照合(パターン・レンジ)。
- 注: コア表のA1腕は事例レベル仕様なのでFRC同定に依存しない。FRCが要るのは
  この再現アンカーのみ(論文でも分離を明記)。

## 4c. T6 — LinguaLens実装の同一スタック検証(工程末尾、2026-07-18ユーザー指示)

元論文と同じ **Llama-3.1-8B + OpenSAE**(THU-KEG/OpenSAE-LLaMA-3.1-Layer_XX、
全32層公開済み)で我々の実装を検証する:
- T6-2: **彼らのコード逐語**(TransformerWithSae+Intervener)で
  politeness(156439)/causality(223621 — repo examples記載のindices)×50生成
  → 我々のjudge+FIC式 → Table 2該当行と直接比較(judge側の検証)
- T6-3: 我々の再実装を同一スタック・同一indicesで → T6-2と比較(フック側の検証)
- T6-4(任意): 我々のFRC同定を彼らのスタックで → 同定込みの完全検証
- 前提: meta-llamaのHFゲート承認(ユーザー確認)。実行は7/22-24、interact-g。

## 5. ゲート(変更なし)

L12: A4 exact > A2(同枠steer)、empty/random無害、λ-IoU true専有。
分岐A(複製不成立→norm-alpha 1.0)、分岐B(λ不発→弱直接教師)は発動前に報告。

## 6. 掲載スコープ(ユーザー指示の反映)

- コア表: 4腕×3層×{FIC, exact}(feature別FIC表+層プロファイル)。
- v2 Intervener(プロンプト+steer基底+学習補正、0.2725)は比較・参考行。
- その他の既測定(C1'2×2、P-I/P-J、k掃引、S_min、FRR等)は「追加で載せる
  かもしれないもの」として一旦コア構成から外す(数値・recordsは保全)。
