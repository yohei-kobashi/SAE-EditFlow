# 04 — Experiment(コア表の実測記録、2026-07-19時点)

手法定義は reports/03_method.md、計画は EF_LM_LOSS_PLAN.md。
フレーム: src先行復唱プロンプト(exact/KL整合)、greedy。499ペア(seed42)。

## 1. exact(従指標、L12)

| 腕 | true | random | empty | net(t−r) |
|---|---|---|---|---|
| **A4 本手法(v5f)** | **0.2485**(sim 0.731) | 0.0020 | 0.0000(=raw) | **0.2465** |
| A2 steer0.5(同枠) | 0.1804 | 0.0000 | 0.0000 | 0.1804 |
| A3′ 編集prompting | 0.1804 | 0.0882 | 0.0401 | 0.0922 |
| A1 clamp(LL忠実) | 0.1363 | 0.0000 | 0.0000 | 0.1363 |
| A3-literal(付録) | 0.0060 | 0.0080 | 0.0000 | ≈0 |
| raw(素の復唱) | — | — | 0.0000(copy 0.99) | — |

- 本手法は同枠全ベースライン超え、**旧rewrite枠チャンピオン(steer 0.2385)も超過**
- A3′はtrue値でsteerと同値だが特異性が低い(randomでも0.088編集)
- A3-literal(AxBench組み立ての字義移植)は編集タスクに転移しない(付録)
- λ診断(v5f): IoU 0.538(true) vs 0.304(random)、発火 0.318 vs 0.141

## 2. exactの深さプロファイル(ベースライン)

| 層 | steer true | clamp true | 備考 |
|---|---|---|---|
| L4 | 0.1343 | 0.0160 | clampはrecon置換が浅層でテキスト破壊(randomでcopy 0.09) |
| L12 | 0.1804 | 0.1363 | |
| L20 | 0.0060 | 0.0020 | 深層は固定介入でコピーを崩せない |

A4のL20は完了(§5c: true 0.0421)、L4はv5f2学習中。

## 3. KL/NLL反実仮想整合(主指標①、L12、true条件)

KL_red = 1−KL(A‖B)/KL(C‖B)(B=x1入力の参照分布、C=無介入)。

| 腕 | KL_red(true) | NLL(x1) true | KL_red(random) | NLL(x1) random |
|---|---|---|---|---|
| **A4 本手法** | **0.7776** | **0.74** | 0.4608 | 1.91 |
| A1 clamp | 0.5450 | 1.53 | −0.0960 | 3.90 |
| A2 steer | 0.1163 | 2.66 | −0.5631 | 4.85 |
| (C 無介入) | 0 | 3.53 | — | — |

- **指標の注意(要併記)**: KL_redは編集位置で「自信を持って誤るC」に対し
  不確実なヘッジ分布でも下がるため、内容を知らないrandomでも正値になり得る
  (ef random 0.46)。**NLL(x1)は内容を要求する**のでrandom分離が明確
  (1.91 vs 0.74)— 両指標併記が必須。
- 深さ: steerはL4 +0.108 / L12 +0.116 / **L20 −0.207**(深層は逆効果)。
  clamp L4はrecon破壊で解釈不能(empty/randomが正になる異常系、注記)。

## 4. 統制・診断の要点

- 復唱フレーム: 素モデルcopy 0.99(exact 0 vs tgt)— フレーム設計どおり
- 文脈シフト(診断7): reports/context_shift_diag.md — src先行採用で
  spec残存0.59・位置overlap 0.70
- A3′のKL整合セルは「フレーム自体が介入」のため同枠比較が未定義(検討中)

## 5. L4移植実験(層ネイティブ方式の根拠、2026-07-20)

L12ネイティブのS3エンコーダをL4のv5f学習に移植した結果:
ef true 0.0601 < 同枠steer 0.1343(λ-IoU分離も甘い 0.42/0.36)。
**エンコーダのspec読取回路は学習した層のW_dec幾何に特異的**であり、
層間移植は劣化する — この結果を受けL4/L20は層ネイティブEF基盤
(S2構成100k→S3局所化50k)から構築し直し(進行中)。本結果は
「エンコーダ層一致の必要性」のablation証拠として保存。

## 5b. EF基盤(層EFエンコーダwarm start)の寄与 — 記載整理

- **bare枠の実測(確定)**: EF基盤なしの学習(v1恒等/v2編集絞り/v3λ教師)は
  3構成とも**spec分離ゼロ**(true≈random、L12、devでも同様=学習自体の失敗)。
  S3エンコーダ移植(v4)で初めて分離(0.1904/0.0020)。
- **注意**: v3→v4は「EF基盤」と「echo教師」の2要素が同時に入っており、
  系譜だけでは単独帰属不可。**単一要因ablation(v5f − FLOW_INITのみ、
  復唱枠)= Tier1-②としてinteract-gキューで実行中** — 完了後に本節へ
  数値を記載。

## 5c. L20層ネイティブv5f2(2026-07-20完了)

層ネイティブEF基盤(S2 100k→S3 50k、SAE layer_20/width_16k/average_l0_71)
→v5f2(L12と同一レシピ)。probe500(499ペア、復唱枠):

| condition | arm | exact | sim_target |
|---|---|---|---|
| **true** | **ef** | **0.0421** | 0.6225 |
| random | ef | 0.0040 | 0.6013 |
| true | steer | 0.0060 | 0.5160 |
| empty | ef | 0.0000(copy 0.99) | — |

- λ診断: IoU 0.3887(true) vs 0.3056(random) vs 0.1394(empty)
- multi-site: n_ops=1で0.1111、2-3で0.0333
- **所見**: 深層L20は絶対値こそ低い(L12の0.2485比)が、**同枠steer(0.0060)の
  7倍・random分離10倍**で、固定介入が全滅する深層でも学習型介入は
  編集信号を残す。基盤probe段階でdet exactがほぼ0だった層でも
  through-LM学習で復元が立ち上がる点は層ネイティブ方式の追加証拠。
- 基盤単体probe(S3、200ペア)はdet exact≈0(thr0.1のみ n_ops=1で0.20)—
  L4基盤(det 0.25)との対比で「深いほど基盤単体の復元は困難」。

## 6. 計算リソース実測(2026-07-20集計)

環境: 東大/筑波大 Miyabi、Miyabi-Gノード(NVIDIA GH200 ×1/ノード)、
すべて単一ノード・単一GPU、bf16、batch=8。実測時間はチェックポイント
mtime(2000step間隔)とPBSジョブログ(*.o、リポジトリ外・miyabi上に保存)
から再構成。

| ラン | 内容 | steps | 実測時間 |
|---|---|---|---|
| EF基盤 S2(L12) | token出力EFスクラッチ | 100k | ≈6.2h |
| EF基盤 S3(L12) | 局所化warm | 50k | ≈2.5h |
| v4 editor(L12、bare枠) | through-LM | 40k | ≈3.4h |
| **v5f editor(L12、現行)** | through-LM | 40k | **≈3.3h** + probe500 ≈0.4h |
| exact probe(499ペア×腕) | 評価 | — | ≈0.3-0.5h/腕 |
| KL/NLL整合(499×3腕) | 評価 | — | ≈1h |
| FIC生成(L12、~7.9k生成) | 評価 | — | ≈4-5h(interact-g 2hセッション×3) |
| L20 v5f2 editor+probe500 | through-LM | 40k | **実測 3.1h**(19:37–22:43、short-g) |
| L4/L20 EF基盤 S2+S3 | 学習(完了、両層) | 150k | L12実測と同等(≈8.7h/層、walltime分断込み) |
| L4 v5f2 editor | through-LM | 40k | 実行中(完了後追記) |

- **1層あたりの提案手法フルパイプライン ≈ 12h(GH200×1)**
  (基盤8.7h+editor3.3h)。Tier1 ablation学習は各≈3.4h追加。
- FIC judge はAPI(gpt-4o、GPU不使用、≈$35-40 / 23.5k呼び出し見積)。
- L4/L20の実測値は完了後に本表へ追記。

## 7. 実行中・残り(2026-07-20時点)

| 項目 | 状態 |
|---|---|
| L20層ネイティブEF基盤→v5f2 | **完了**(§5c、probe500まで) |
| L4層ネイティブEF基盤→v5f2 | 基盤完了、v5f2学習中(short-g) |
| Ablation Tier2(k掃引・強度掃引・rounds3・scope・blocklist) | **probe完了**(集計待ち) |
| Ablation Tier1×3学習(noS3/noctr/nobudget) | interact-gチェーンで実行中 |
| FIC生成(L12、A1/A2/A3-literal) | **生成完了** → judge(gpt-4o、prepost) |
| feature別集計 | キュー末尾で自動生成(runs/tables/perfeature_*.md) |
| A4のFIC/KL(L4/L20) | v5f2完成後 |
| T6(Llama同一スタック検証) | 未(要HFゲート承認) |
