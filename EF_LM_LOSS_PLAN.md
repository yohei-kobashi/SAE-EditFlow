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

FICのコスト目安(全98feature): n_pairs=10 → 生成 ≈ 98×10×2方向×2(targeted/random)×3腕×3層 ≈ 70k生成(+A3/raw)。
n_pairs=5なら半分。**n_pairsとjudgeモデル(gpt-4o vs gpt-4o-mini)はユーザー決定待ち**。

## 5. ゲート(変更なし)

L12: A4 exact > A2(同枠steer)、empty/random無害、λ-IoU true専有。
分岐A(複製不成立→norm-alpha 1.0)、分岐B(λ不発→弱直接教師)は発動前に報告。

## 6. 掲載スコープ(ユーザー指示の反映)

- コア表: 4腕×3層×{FIC, exact}(feature別FIC表+層プロファイル)。
- v2 Intervener(プロンプト+steer基底+学習補正、0.2725)は比較・参考行。
- その他の既測定(C1'2×2、P-I/P-J、k掃引、S_min、FRR等)は「追加で載せる
  かもしれないもの」として一旦コア構成から外す(数値・recordsは保全)。
