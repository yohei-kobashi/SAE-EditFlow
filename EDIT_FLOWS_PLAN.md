# Edit Flows 案(SAE-EF)— 具体化計画

位置づけ: V7_PLAN(C→A ハイブリッド)の**並行トラック / 昇格候補**。
Havasi et al., *Edit Flows: Flow Matching with Edit Operations*(arXiv:2506.09018,
NeurIPS 2025)を、SAE 条件付き編集(x′ → x、minimal edit)の regime に適合させる。
レビュー結論(2026-07): 多サイト・可変長の尾部で最有力、主成分(1–3 編集)では
C+A と拮抗見込み。よって**縮小プロトタイプで C1 と直接対決させて判定**する。

---

## 1. 定式化(本プロジェクトへの適合)

### 1.1 編集 regime での flow

- 論文の生成タスクは X₀ = ∅(空列)から X₁ = 完成文への flow。本計画では
  **X₀ = x′(入力文)、X₁ = x(目標文)** — 論文のコード修正デモと同じ編集 regime。
  経路長は最小編集距離(キャッシュの N ≤ 8)なので、生成タスクの数百ステップは不要。
- 演算: 挿入 ins(x,i,a) / 削除 del(x,i) / 置換 sub(x,i,a)。
  モデル出力は位置ごとの λ^ins, λ^del, λ^sub(レート、softplus)と
  Q^ins(a|x,i), Q^sub(a|x,i)(トークン分布)。

### 1.2 カップリングと補助アラインメント

- 教師: **既存 corruption キャッシュのペアがそのまま (x₀, x₁) カップリング**。
  アラインメント z₀, z₁(ブランク ε 込み)は `pair_to_gold` / difflib opcodes から
  決定的に構築(ランダムアラインメントは使わず最小編集距離で固定 — 編集タスクでは
  これが自然)。
  - ε→a = 挿入、tok→ε = 削除、tok→tok′ = 置換。
- 前向き過程: 各未発火演算が時刻 t までに発火する確率 κ_t(論文既定 κ_t = t³)。
  x_t = frm-blanks(z_t)、z_t は各演算を独立に κ_t で z₁ 側へ倒したもの。

### 1.3 学習損失(flow matching / Bregman)

サンプルごとに t ~ U(0,1) を引き、z_t を構築して:

```
L = Σ_pos Σ_op λ̂_op(x_t)                       # 全レートの抑制
  − Σ_{未発火演算 o} w(t) · log û(o の遷移 | x_t)  # 目標演算への重み付き CE
      w(t) = κ̇_t / (1 − κ_t)
```

û は λ̂ と Q̂ の積。実装上は「レート項 = softplus 出力の和」
「CE 項 = 該当位置の λ̂ の log + Q̂ の log」に分解される。

### 1.4 条件付け・ガイダンス・premise protection

- **条件付けプレフィクス [INT_amp, INT_sup] を editor と同一機構で再利用**
  (Proj_A は v6 editor checkpoint から初期化)。t の埋め込みトークンを 1 個追加。
- 学習時 empty-conditioning dropout(既存 `empty_conditioning_prob` = 0.15 相当)を
  必ず入れる → 推論で **CFG: u = u_empty + s·(u_cond − u_empty) を λ と Q に独立適用**
  (論文 §CFG)。
- **premise protection の教師は既存の null レコード**(全 KEEP・ゼロ diff)が
  そのまま「empty 条件では全 λ → 0」を教える。検証: empty 条件の無編集率 ≥ 0.99 を
  ゲートとする。
- **lens バイアスは Q^sub / Q^ins の logits に毎ステップ加算**(V7 C2 と同一部品)。

### 1.5 検証体系の再構築(WHERE/WHAT)

- WHERE: **λ の位置分布 vs gold 編集位置の IoU** を true/empty/random で比較 →
  tagger の span IoU と直接比較可能な「λ-IoU」を定義。
- WHAT: 発火した sub/ins の内容 top-1(gold アラインメント基準)。
- 既存プローブの exact / sim_target / copy_rate はそのまま適用可能
  (テンプレート不要 — モデルが直接 x′ を編集する)。

## 2. アーキテクチャ(凍結 Gemma 上のヘッド構成)

| 部品 | 実装 | 由来 |
|---|---|---|
| バックボーン | 双方向化 Gemma-2-2B + LoRA r=16 | editor と同一(`model._patch_attention_bidirectional`) |
| 条件付け | Proj_A + [INT_amp, INT_sup] プレフィクス | editor checkpoint から初期化 |
| 時刻 t | 学習可能な t 埋め込みトークン 1 個 | 新規(小) |
| λ ヘッド | hidden → Linear(d, 3) → softplus(位置ごと ins/del/sub) | 新規(小) |
| Q^sub | 凍結 lm_head(既存の fill と同じ) | 流用 |
| Q^ins | 位置 i の hidden から「i の直後に挿入するトークン」を凍結 lm_head で予測 | 流用(ギャップ表現のみ新規) |

- 語彙 256k の Q を新規学習しない(凍結 head 流用)ことが、凍結バックボーン予算で
  成立させる鍵。学習対象は LoRA + λ ヘッド + t 埋め込み + Proj_A 補正のみ。
- FlexAttention は不要(最大長 256、パディングで十分)。

## 3. 推論

```
x ← 入力 x′; T ステップ(既定 48、編集距離が小さいため生成用途の 300 は不要):
  各ステップ: forward → λ, Q(CFG 適用、Q に lens バイアス)
    決定的変種(既定): 期待発火数 = h·Σλ に相当する上位レートの演算を確定適用
    確率的変種: 各演算を確率 h·λ で独立発火(温度付き Q サンプル)
  終了: 全 λ の和 < 閾値(モデル自身の「編集完了」宣言)or T 到達
ranker: 異なる s(CFG スケール)/ 温度で K 本サンプル → 既存 directional ranker で選択
```

- minimal pair では発火演算数 ~1–8 なので、実効 forward 数は現行 edit_once
  (テンプレート列挙 ~数十 forward)と同オーダーに収まる見込み。

## 4. 実装ステップ(プロトタイプ)

1. **`editflow_ops.py`**: opcodes → (z₀, z₁) 構築、κ_t サンプリング、z_t / 教師演算の
   生成、λ-IoU 計算。純関数群 — CPU ユニットテスト完備(往復性: 全演算適用で x₁ 復元)。
2. **`editflow.py`**: 上記ヘッド構成のモデル(editor.py の骨格を流用)。
3. **`train_editflow.py`**: キャッシュ→カップリング→損失。dev モニタは
   「seldev ペアの演算復元精度 + empty 無編集率」。30k steps(パイロット同予算 ~1.5h)。
4. **`scripts/editflow_probe.py`**: LinguaLens 190 ペアで C1 と同一プロトコル
   (local 抽出 + lens + true/empty/random、n_edit バケット別 exact/sim/λ-IoU)。
5. **直接対決**: 同一キャッシュ・同一予算(30k steps)で
   C1 継続 FT editor vs SAE-EF を同じ表で比較。

## 5. 判定ゲート(昇格条件 — レビューで特定した3未知の実測)

| 条件 | 指標 | 合格ライン |
|---|---|---|
| (a) λ の OOD localization | λ-IoU vs tagger span IoU(LinguaLens) | tagger 同等以上 |
| (b) 低温度サンプルの品質 | 決定的変種の exact / sim | 確率的変種と同等以上 |
| (c) 凍結バックボーンでの学習安定性 | 学習曲線・empty 無編集率 | 発散なし・≥0.99 |
| 総合 | n_edit バケット別 exact(特に 2-3 / 4-8) | C1 を明確に上回る |

- 3 条件 + 総合を満たせば **v8 本命として昇格**(フル学習 100k + 評価一式)。
- (a) のみ不合格の場合はハイブリッド(tagger の予測サイトで λ を マスクする
  「tagger-gated EF」)を 1 回だけ試す — WHERE は tagger、HOW/WHAT は flow。

## 6. リスクと未知

- 公式実装が確認できないため**論文からの再現実装**(補助過程・重み w(t) の数値安定性、
  κ̇/(1−κ) の t→1 発散はクリップで処置)。
- Q^ins のギャップ表現(位置 i の hidden で「直後挿入」を予測)は簡略化であり、
  論文の実装詳細と異なる可能性。
- minimal-edit regime では経路が短く、flow の利点(長経路の自己修正)が
  発揮されない可能性 — まさにそれを測るのがプロトタイプ。

## 7. スケジュール概算

| 項目 | 規模 |
|---|---|
| 実装(ops + model + train + probe) | 実装セッション 3–4 回、ユニットテスト付き |
| プロトタイプ学習 | miyabi 1 ジョブ(~2h、C1 対決と同居可) |
| 判定 | editflow_probe ~15 分 |

V7_PLAN との関係: C1-0 / C1 の結果が出た時点でプロトタイプ着手を判断。
C1 ゲート成立なら SAE-EF は「尾部特化の追加実験」に格下げ、不成立なら本命昇格。
