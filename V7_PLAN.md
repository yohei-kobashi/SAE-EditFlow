# v7 計画 — マスク拡散 editor + 編集演算拡張(C→A ハイブリッド)

対象: v6 の次の大幅改善。§13.6 の測定(WHERE は汎化 / WHAT の3レバー特定 / 多サイト
fill の頭打ち)と、LEWIS 代替調査(README とは別途、2026-07 の検討記録)に基づく。

**中心的な観察**: 現在の editor は「テンプレートの編集位置を双方向文脈で一括予測する
1ステップのマスク拡散モデル」である。したがって C 段階は置き換えではなく**一般化**であり、
確立済みの資産 — tagger(SAE-GROUNDED な localization)、corruption キャッシュ、
条件付けプレフィクス、true/empty/random プローブ、directional ranker、lens バイアス —
は全て無傷で生き残る。

---

## 設計原則(不変条件)

1. **検証体系を壊さない**: すべての段階で true/empty/random プローブが実行可能であること。
   empty → 無編集の構造保証(premise protection)は KEEP 位置のクランプで維持する。
2. **段階ごとに判定ゲート**: gold-template プローブ(fill top-1 / 多サイト bucket /
   true−empty ギャップ)を各段階の GO/NO-GO 指標とする。学習前に推論のみで測れるものは
   先に測る(v6 で実証されたパターン)。
3. **キャッシュ非再生成**: v4–v6 の統合キャッシュ(~1.35M レコード)をそのまま使う。
   学習時の動的変換(マスク率カリキュラム等)は collator / train ループに閉じる。

---

## Stage C1 — editor の多ステップ化(CMLM / マスク拡散学習)

### 学習の変更

現在: テンプレートの全編集位置が [MASK]/[INS]、他は gold — 常に「完全マスク状態」から
の1ステップ予測。これを**部分的に露出した状態(= デノイズ途中)からの予測**に一般化する。

- `data.py CorruptionCollator` に `reveal_ratio_dist` を追加: サンプルごとに
  ρ ~ U(0,1)(または cosine スケジュール)を引き、**編集位置のうち ρ 割合を gold トークンで
  露出**させ、残りをマスクのままにする。loss はマスクが残る位置のみ(-100 で既存機構を流用)。
- ρ=0 が現行挙動なので、混合率(例: 30% は ρ=0 固定)で in-domain 1-step 性能の後退を防ぐ。
- editor.py のアーキテクチャ変更は**なし**(forward は同一)。学習は既存 checkpoint からの
  継続 fine-tune(30–50k steps)で開始し、フル再学習は効果確認後。

変更ファイル: `data.py`(collator)、`train_editor_phaseA.py`(引数 `--reveal-curriculum`)。

### 推論の変更

`evaluate_intervention.py` に `fill_iterative()` を追加(edit_once から呼ぶ):

1. テンプレートの全編集位置をマスクで初期化。T ステップ(既定 4–8)の反復:
   - forward → 全マスク位置の logits(マーカー抑制 + **lens バイアス λ_t を加算** — C2 参照)
   - confidence(softmax max)上位のスケジュール割合(cosine)を確定、残りを再マスク
2. KEEP 位置は常にクランプ(構造的 premise protection)。
3. 既存の fill_topk 変種生成・ranker 選択はそのまま最終ステップに接続。

変更ファイル: `evaluate_intervention.py`、`scripts/lingualens_gold_template_probe.py`
(モード `cmlm{T}` 追加 — 学習前でも現 checkpoint で「反復 Mask-Predict」として測定可能)。

### 判定ゲート C1

| 指標 | v5(local+lens2) | **v6 実測(local+lens1)** | GO ライン |
|---|---|---|---|
| probe 多サイト 2-3 / 4-8 top-1 | 0.35 / 0.25 | **0.488 / 0.367** | +0.05 以上 → **データのみで達成済み** |
| probe 全体 top-1 | 0.37 | **0.471** | 後退しない |
| in-domain editor repl/ins top-1(ceiling) | 0.94 / 0.80 | 0.923 / 0.785 | −0.01 以内 |

**v6 更新(2026-07)**: C1 の GO ライン(+0.05)は v6 データ拡充のみで +0.12–0.14
達成された → **C1 の優先度は低下**(計画内の更新規則どおり)。残る C1 の論点は
「4-8 編集の exact ≈ 0(top-1 0.37 なのに文全体一致に組み上がらない)」のみ。

**C1-0 実測(2026-07、`probe_cmlm`、v6 editor、推論のみ)**: 素の cmlm8 ≈ parallel
(0.297 vs 0.313 — ρ=0 学習の editor に露出 gold は分布外、かつ低品質 fill の
フィードバックはノイズ増幅)。一方 **cmlm8+lens1 は lens1 を全面的に上回る**:
top-1 0.471→0.500、exact 0.416→**0.447**、sim 0.822→0.834、2-3 バケット exact
0.269→**0.343**(+0.075)、4-8 top-1 0.367→0.440。empty は反復で不変(premise
protection 無傷)。**解釈: 反復は per-round fill 品質の増幅器 — lens が正解率を
閾値以上に押し上げたときだけ複利が回る。**
→ 判定: **C1 の推論部(fill_iterative + 毎ステップ lens)は学習ゼロで実装済み**
(`edit_once --fill-iterative`)。C1 の学習部(reveal カリキュラム)は
降格のまま — 残る上振れは「露出文脈を信頼する editor」で増幅器の利得を
さらに上げること。WHERE トラックが停滞した場合のみ再訪。

**e2e 実測(同一 500 ペア)**: sim 0.6687→0.6676 / exact 0.112→0.114 /
iou 0.296→0.312 / copy 0.628→0.604 — ヘッドラインはフラット。gold-site の
利得が乗る多サイトペアは tagger のサイトが既に外れているペアと同じ集合
なので、正しい fill が exact に変換されない(WHERE 希釈の定量確認)。
運用: **論文数値は parallel+lens1 のまま**(同結果で forward 1/2–1/3)。
cmlm は WHERE が改善した瞬間から効き始めるデコードとしてパイプラインに常備。

**先行実験(学習ゼロ)**: 現 checkpoint で `cmlm{T}` モードをプローブ測定。1-step 学習
モデルでも反復 Mask-Predict が多サイトで並列を上回るなら C1 の期待値は高い。
上回らなくても訓練条件との乖離(露出 gold を見たことがない)が原因である可能性が
高いため、NO-GO 判定は学習後に行う。

## Stage C2 — lens ガイダンスの定式化(per-step guidance)

現在の lens バイアスは「1回だけ足すヒューリスティック」。反復デノイズでは
classifier-free guidance と同型の**毎ステップガイダンス**として原理化する。

- λ_t スケジュール: 初期ステップで大きく(内容の方向付け)、終盤で減衰(文法適合を
  editor に委ねる)。既定案: λ_t = λ₀ · (1 − t/T)、**λ₀ = 1(v6 更新: 膝が λ≈2→1 に
  移動 — readout が良くなるほど辞書ステアは少なくてよい)**。
- op 別係数: v6 で実測確定 — **REPL の膝 λ=1(0.543)、INS の膝 λ≈0.5(0.347、
  λ>0.5 で劣化: lens2 0.258 < 無バイアス 0.331)**。位置種別で λ を分ける
  (probe に `--steer-lambda-ins` を追加)。膝の再測定は不要、実装のみ。
- **達成分の減衰**: 各ステップで確定済みトークンが動かした特徴を bias から差し引く
  (refine-recompute の diff 意味論をステップ内に持ち込む)。第2次実装。

変更ファイル: `evaluate_intervention.py`(fill_iterative 内)、probe。

### 判定ゲート C2

- probe で λ 固定 vs スケジュールの比較。REPL/INS 個別膝の同時達成
  (現在は単一 λ でトレードオフ)が改善の主眼。

## Stage A1 — 長さ予測の主経路昇格(可変長の実用解)

列挙(l_max=3、max_templates 256)が課す**挿入長の硬い上限**を、まず既存資産で撤廃する:
リポジトリには length predictor(§4.5、現在は効率化 ablation)が既にある。

- length head を主経路に昇格: ギャップごとに長さを予測し、予測±1 のみ列挙
  (組合せ爆発が消えるため **l_max を 8 に拡張可能**)。
- length head の再学習: v6 キャッシュ(natural edits は長い挿入を含む)で
  `train_length_head.py` を回すだけ。multi-gap 対応(現在は先頭ギャップのみの簡易化)
  を collator 側で解除。

変更ファイル: `train_length_head.py` / `data.py`(multi-gap 教師)/
`evaluate_intervention.py`(列挙を予測±1 に制限、`--use-length-head`)。

### 判定ゲート A1

- LinguaLens の「挿入長 >3 のため到達不能」だったペアの回収数(records.jsonl から
  事前に集計可能)。probe の INS top-1 と e2e exact。

## Stage A2 — MOVE 演算(EdiT5 型 pointing)

並べ替えを DEL+INS 分解から解放する。最も大きい改修なので最後。

- **tagger v3**: op3 ヘッドに加え、EdiT5/EDITOR 型の **pointer ヘッド**(各保持トークンの
  後続位置を指す)を追加。並べ替えは「タグは全 KEEP + pointer の置換」で表現される。
- **gold 導出**: `pair_to_gold` を拡張し、DEL スパンと INS スパンの内容が一致する場合に
  MOVE として再解釈(トークン列マッチング)。**教師データは reordering ファミリー
  (PARTICLE/DATIVE/ADVPLACE/PPFRONT/QUOTINV)が既にキャッシュ内にある** — 再生成不要。
- **editor 入力**: pointer 適用後の順序でテンプレートを構築(移動トークンは gold として
  露出 = C1 の部分露出機構と自然に整合)。
- ranker・条件付けは無変更。

変更ファイル: `lewis_ops.py`(MOVE 導出)、`tagger.py`(pointer ヘッド)、
`train_tagger.py`、`data.py`、`evaluate_intervention.py`(テンプレート構築)。

### 判定ゲート A2

- per-family 評価の reordering 系 span IoU(v6 実測: ADVPLACE 0.330 / PARTICLE 0.271 /
  DATIVE 0.243 — priority pick で供給は回復済みなのに IoU が回復しない =
  DEL+INS 分解そのものが原因と確定、A2 の根拠が強化された)と、
  LinguaLens 語順系現象の exact。

---

## 実施順序と工数見積り

| 段階 | 内容 | 実装 | 計算(miyabi) | ゲート |
|---|---|---|---|---|
| C1-0 | cmlm{T} 推論のみプローブ | 小 | 10 分 | 参考値 |
| C1 | reveal カリキュラム + fill_iterative | 中 | 継続 FT 30–50k(~2h)| 多サイト +0.05 |
| C2 | λ スケジュール / op 別膝 | 小 | プローブのみ | REPL/INS 両立 |
| A1 | length head 主経路化 | 中 | length head 再学習(<1h)| INS/exact |
| A2 | MOVE(pointer tagger) | 大 | tagger 100k(~3.5h)| reordering IoU |

- C1-0 と C2 の膝測定は推論のみ(v6 checkpoint で即実施可能)。
- **v6 の結果による更新(2026-07、README §13.7)**: 多サイト bucket はデータのみで
  大きく動いた(2-3: 0.35→0.488、4-8: 0.25→0.367)→ **C1 は降格、A1/A2 前倒し**。
  tagger OOD iou は跳ねなかった(e2e iou 0.286→0.296)— 一方 gold-site probe は
  exact 0.416 / sim 0.822 に到達しており、e2e(0.112 / 0.669)との差 3.7× は
  **WHERE(localization + enumeration)が独占**。よって次の主戦場は A2 と
  EDIT_FLOWS_PLAN の λ-IoU トラック(ゲート (a) の比較対象 tagger iou ≈ 0.30 は低い壁)。
- 全段階を通過した構成が v7。論文構成上は「v6 = データと抽出の因果分析、
  v7 = 生成機構の一般化」という2部立てになる。

## 温存する代替(この計画が外れた場合)

- **Edit Flows / DiffusER 型への全面移行**: 編集操作の CTMC/拡散。tagger との分離を
  失うため、C1/A2 のゲートが連続して不成立の場合に本命へ昇格。
  **具体化計画は EDIT_FLOWS_PLAN.md**(SAE-EF: 凍結 Gemma + λ/Q ヘッド、
  キャッシュのペアをカップリング教師に流用、CFG を λ/Q に独立適用、
  C1 との同予算直接対決で判定)。
- **Gemma-2-9B + Gemma Scope 65k へのスケール**: 全課題に効くが再構築コスト最大。
  v7 の後の判断。
- **LinguaLens FT(premise protection 付き)**: ベンチマーク特化の最終手段。
