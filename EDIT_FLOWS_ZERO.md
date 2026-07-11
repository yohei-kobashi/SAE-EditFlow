# SAE-EF-Z — ゼロベース Edit Flows 計画

位置づけ: **SAE(Gemma Scope 幾何 + 検証プロトコル)/ corruption(データ生成)/
LLM2Vec(双方向エンコーダ)の3要素だけを継承**し、LEWIS 系譜の残りすべて
(tagger/editor 分業、テンプレート機構、Proj_A プール型プレフィクス、ranker、
refine passes)を捨てて Edit Flows(Havasi et al., arXiv:2506.09018)を再設計した場合の計画。
EDIT_FLOWS_PLAN.md(プロトタイプ = warm-start 版)の**上位互換パス**であり、
flow 核(演算・κ・損失・tau-leaping)は実装済みのものを共有する —
「ゼロベース」は実際には**載せ替え可能な差分 Z1–Z5 の集合**である。

パイロット実測(README §13.8)による前提更新: **共有最大未知だった
「λ の OOD localization」は肯定済み**(λ-IoU 0.7337、empty 0.20 /
random 0.30、premise protection は random 指令まで完全)。
recal で判明した重要な訂正: 同一ペアの **tagger count-oracle は 0.7472 =
ランキング能力はパリティ**。e2e の 0.30 は「知識」ではなく「決定較正 +
パイプライン段の損失」であり、EF の優位性の根拠は「決定経路が単一の
較正可能スカラーであること」(パイプラインの多段カスケードとの対比)に
更新される。もう1つの実測: **λ は t に伴う成長を追従しない**
(目標比 0.119@t=0.3 → 0.027@t=0.9、λ≈0.25 で飽和)— t トークン1個では
強度変調が弱い。→ Z1 に t 注入強化を含める。

---

## 0. 論文 v2 精査で確定した設計材料(2026-07)

1. **編集 regime(X₀=実文)は論文未実証** — X₀ は ∅ か一様ランダムのみ。
   最小編集への適用は当プロジェクトの外挿(パイロットで部分実証済み)。
2. **Localized Edit Flows**(付録 C.1): 発火した演算が近傍の発火を誘発する
   補助 CTMC。長系列コード生成で Pass@1 相対 +48%。多サイト協調
   (4-8 編集 exact ≈ 0 問題)への論文側の正面解。
3. **学習済み reverse rates**: 分布を変えない定常修正成分。推論中の
   自己修正 = refine passes の flow ネイティブ版。
4. アラインメントは「ランダム or 最小編集距離」のどちらも可。
   条件付けは**文脈トークン列**(+ t トークン)。tau-leaping ~100 steps。
   CFG は λ / Q に独立。モデルは 280M/1.3B を**フルスクラッチ学習**。

## 1. 縛りの再検討 — 「凍結 Gemma」は SAE 継承の帰結ではない

SAE 特徴の妥当性は**抽出側の凍結 extractor** が担保する。flow 本体の
バックボーンを高 rank LoRA / full FT しても z の意味論は壊れない。
現行の凍結 + r=16 は editor 系譜の予算設計であり、ゼロベースでは
論文(フルスクラッチ)に寄せて **r=64–128 または 2B full FT
(bf16 + 8-bit optimizer で 1 GPU 可)** が正当な選択肢になる。

## 2. 差分モジュール Z1–Z5(効果見込み順)

### Z1 — 特徴トークン条件付け(本命: 課題 #1 feature-bag binding の正面解)

プール済み 2 ベクトル(z_amp/z_sup)を捨て、**指令特徴 1 つにつき 1 トークン**:

```
prefix = [F(f₁,+,v₁), …, F(f_j,−,v_j), …, T(t)]   (最大 2k = 128 トークン)
F(f,±,v) = W_dec[f](凍結) + sign_emb(±) + mag_mlp(log(1+v))
```

- attention が**個々の特徴と個々の編集サイトを直接束縛**できる。
  pool-max による強度情報の損失も条件付け側から消える(課題 #5 の半分)。
- W_dec 行を埋め込み初期化に使う = SAE 幾何の継承(wdec-frozen の後継)。
- CFG ドロップアウトはトークン集合単位(全ドロップ = empty、部分ドロップ =
  部分指令頑健化)。
- **単独で検証可能**: editflow.py の cond_embeds 差し替えのみ。パイロットと
  同予算 30k で再学習し、同じ probe で λ-IoU / exact を直接比較。
  editor からの Proj_A 継承は形が合わないため不可 — ただし LoRA は
  **パイロット EF checkpoint から**温めれば良い。
- **t 注入強化を同梱**(recal 実測: λ が w(t) 成長を追従せず ≈0.25 で飽和 —
  プレフィクス 1 トークンでは強度変調が届かない): t の正弦特徴を
  **λ ヘッドの入力に FiLM(scale/shift)で直接注入** + rate-head の LR を
  小物パラメータ群から分離して増額。
- 実装: editflow.py(cond_embeds を可変長プレフィクスに、N_PREFIX 動的化、
  λ ヘッド FiLM)、train_editflow.py(z_amp/z_sup 疎ベクトル → (f,±,v)
  リスト変換、rate-head LR 群)。

### Z2 — 真アラインメント教師(difflib 事後推定の廃止)

corruption は適用した演算列そのものを知っており、キャッシュの
`tagger_gold` + `editor_input/target_token_ids` に符号化済み
(natural-edit ingest も同経路)。`editflow_ops.cache_slots(rec)` を追加し、
**生成過程と厳密に一致するスロット列**を導出する。同一トークンの重複で
difflib が誤アラインする問題が消える。正則化として共最適アラインメントからの
サンプリング(論文の示唆)も選択肢。実装は小、再学習に相乗り。

### Z3 — Localized 補助 CTMC(多サイト協調)

学習時: fired 集合を独立サンプリングでなく**局所相関つき**でサンプリング
(発火済み演算の近傍窓 w 内の未発火演算の κ を増幅)。推論時: 論文は
CTMC 自体を拡張(発火が近傍レートを押し上げる augmented rate)。
一致・呼応の編集(agreement chain)はこの構造そのもの。cmlm 実験で観測した
「反復は品質の増幅器」力学をレート構造として内蔵する形。
**着手前に付録 C.1 の正確な定式化を再読すること**(本文には要約のみ)。

### Z4 — reverse rates(自己修正の内蔵)

t→1 付近に定常修正成分(逆向き演算のレート)を追加し、途中の誤置換を
分布を変えずに修正可能にする。refine passes(全パイプライン 3 周)の代替。
研究的要素が最も強く、優先度は Z1–Z3 の後。

### Z5 — 学習予算の拡大

論文水準(フルスクラッチ 1.3B)への対応: r=64–128 LoRA または full FT、
100k–200k steps。Z1–Z3 が効果を示した構成に対してのみ投資する(最後)。

## 3. 検証体系の生存(3要素から導出されるため全て残る)

| 部品 | ゼロベースでの扱い |
|---|---|
| true/empty/random プローブ | 生存(条件付けインターフェースのみに依存) |
| λ-IoU(WHERE) | 生存 — tagger span IoU の後継。count-oracle 差は recal probe で解消済み |
| premise protection | 生存 — null レコード教師(corruption 継承)。パイロットで random 指令まで無編集を実証 |
| lens バイアス | 生存 — W_U·W_dec は SAE 固有資産。「後付け」から「Q の事前分布」に再解釈 |
| ranker | 廃止 — best-of-K の選択は**凍結 extractor での指令達成度**(sae_align 相当、SAE ネイティブ)+ CFG/温度に置換。SLOR gate は SAE 外の計測器として必要なら残す |

## 4. 本質的課題マップ

| 課題 | プロトタイプ(warm-start) | SAE-EF-Z |
|---|---|---|
| #1 feature-bag binding | 未対応(プール 2 ベクトル) | **Z1 で正面解** |
| #2 MOVE / l_max | 解決(ins/del ネイティブ) | 同じ |
| #3 硬い決定のカスケード | 解決(単一モデル) | 同じ + Z4 で 1 パス化 |
| #4 共起記憶の学習信号 | 未対応 | **未対応のまま**(データの問題 — corruption 継承の限界。v6 の結論「データが最大レバー」は Z でも不変) |
| #5 プール / スケール | 未対応 | Z1 で条件付け側は解消、バックボーン 2B は Z5 まで据え置き |

## 5. 実施順序と判定ゲート

前段(進行中): recal probe(`run_editflow_recal_v6.sh`)で thr{F} デコードと
レート較正診断。

| 段階 | 内容 | 実装 | 計算 | ゲート |
|---|---|---|---|---|
| Z0 | thr デコード(**済・通過**: thr0.02 exact 0.141 > e2e 0.112、thr0.1 sim margin +0.042、empty 1.00) | 済 | 済 | ✔ exact > 0、sim がベースライン超え |
| Z1 | 特徴トークン条件付け(30k 再学習)【実装済み — 下記 a/b 分割】 | 済 | ~2h×2 | λ-IoU ≥ 0.73 維持 + exact/sim がプロトタイプ超え; 多サイトバケットの伸び |
| Z2 | 真アラインメント教師【実装済み — Z1 に同梱】 | 済 | Z1 と相乗り | dev kind_acc / q_top1 の改善(退行しないこと) |
| Z3 | Localized CTMC | 中(C.1 再読必須) | ~2h | 4-8 バケット exact > 0 へ |
| Z4 | reverse rates | 大 | ~2h | 誤置換の自己修正率; sim 上積み |
| Z5 | r=64–128 / full FT + 100k–200k | 小(設定) | 数ジョブ | 総合: e2e 相当比較で v6 パイプライン(sim 0.6687 / exact 0.114)超え |

- **総合昇格判定**は Z1 + Z3 時点で一度行う: LinguaLens で
  「EF 単体(tagger なし・テンプレートなし・ranker なし)が v6 フルパイプラインを
  bucket-wise exact で上回るか」。上回れば v8 本命として Z5 に投資、
  上回らなければ tagger-gated EF(WHERE は既存 tagger でマスク)に縮退。
- 各段階は probe(~40 分)で独立に判定できる — 因果の切り分けを保つこと。

**Z1 実装(2026-07、`train_editflow_z1.sh`)— a/b 分割で帰属を保つ**:
- **Z1a** = t-FiLM(λ ヘッド入力への FiLM(t)、ゼロ初期化で恒等スタート)+
  rate-head LR 分離(lam_head + lam_film を 3e-3 群に)+ Z2 真アラインメント
  (`editflow_ops.cache_slots` — キャッシュの editor 入出力から生成演算列を
  復元、検算不一致は difflib フォールバック)。プール条件付け維持。
- **Z1b** = Z1a レシピ + `--cond-mode feature-tokens`(指令特徴 1 つ = 1
  プレフィクストークン: 凍結 W_dec 列 + type_emb 符号 + log1p 強度射影
  [ゼロ初期化]、可変長・右パディング)。**両腕とも同じ v6 editor ウォーム
  スタート・同予算 30k** — 差分が特徴トークンの純効果になる。
- probe は `det,thr0.02,thr0.05` — det が「源泉較正だけで発火するか」の判定。
  checkpoint に t_film/cond_mode を保存し probe 側は無変更で両腕を読める。

## 6. リスク

- **編集 regime は依然として論文外挿**(パイロットで λ localization は実証、
  デコード品質は未実証 — Z0 の結果が最初の実データ)。
- Z1 はコールドスタート要素が増える(Proj_A 継承不可)。プレフィクス 128
  トークンは系列長 +50% 相当の計算増。
- Z3 は付録 C.1 の詳細が本文に無く、再現実装リスクが Z の中で最大。
- 課題 #4(WHAT の OOD がデータ共起に縛られる)はアーキテクチャでは動かない。
  Z と並行して corruption 側の内容多様化(v6 の延長)が引き続き必要。
