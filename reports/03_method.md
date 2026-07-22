# 03 — Method(現行採用手法の記録、2026-07-19確定)

ユーザーとの目標統一(EF_LM_LOSS_PLAN.md)で確定した手法。以後の変更は
必ずユーザー承認を経て本ファイルを更新する。

## 1. 大前提と全体構成

**SAEへのintervention(仕様)を信号としてeditorに入力し、editorの出力
embedding(Δh)を凍結LMのresidual streamに戻す。テキストは凍結LMの
生成として出る**(トークン出力なし = do(residual)、ReFT類縁)。

```
仕様 z_amp/z_sup(層LのSAE、裸の文から抽出)+ 入力文 x_t
  → editor(LLM2Vec双方向Gemma-2-2B + LoRA r=32、feature-tokens条件付け)
  → λ_i = σ(rate_head(h_i))(編集残存確率)、v_i = content_head(h_i)
  → Δh_i = λ_i · v_i(ゼロ初期化=恒等スタート)
  → 凍結gemma-2-2b-itの層L出力、プロンプト内srcスパンに注入(prefillのみ)
  → 凍結LMが応答を生成(greedy)
```

- **単層注入**(L4/L12/L20の層別モデル)。多層注入は「L12特徴の媒介主張を
  バイパスする」ためユーザー決定で不採用(分析専用の全層oracleも補正の
  二重適用問題で不採用)。
- **エンコーダは各層の層ネイティブEF基盤からwarm start**(2026-07-20
  ユーザー決定 — L12エンコーダのL4移植で性能が劣化したため、全層を
  完全同一パイプラインに統一):
  1. 層Lのサイドカーキャッシュ上で、トークン出力EFチャンピオン構成を
     学習(S2構成スクラッチ100k → S3局所化warm 50k、層LのW_dec)。
     L12は既存のS3チャンピオン(λ-IoU 0.73/0.30)をそのまま使用。
  2. そのspec読取済みエンコーダ(条件付け+LoRA)をeditorのflowに
     strict=Falseで移植。rate/contentヘッドは新品。
  - L12との唯一の差分: L4/L20には層ネイティブなeditorが存在しないため
    S2の`--init-from-editor`を省略(cold条件付け。S2の100k予算がZ1bの
    cold-start問題を解消することは実証済み)— 論文に注記。

## 2. 生成フレーム(v5、2026-07-19ユーザー決定)

**明示復唱指示プロンプト**(chat template、英語):

> Input: {src}\n\nRepeat the input sentence exactly. Never output anything else.

- 採用根拠: 素のgemma-2-2b-it(SAE/editorなし)でLinguaLens src 100文の
  **99%を逐語コピー**(scripts/test_repeat_prompt.py。chat template必須 —
  bareテキスト指示は0%。残り1%は引用符正規化)。
- **src先行の語順**(ユーザー提案、診断7追試で採用): 因果LMではsrcは
  左文脈にしかattendしないため、指示をsrcの後ろに置くと注入位置の状態が
  同定文脈(裸の文)に近づく — spec残存0.52→0.59、位置overlap 0.59→0.70、
  復唱率は99%のまま(reports/context_shift_diag.md)。
- プロンプトは**復唱能力のみ**を供給し、編集情報はΔhだけが運ぶ。
  「何を復唱するか」を介入が決める、という役割分離。
- srcスパンの位置はトークン部分列探索(find_subseq+先頭欠けフォール
  バック)で特定。エンコーダ位置↔プロンプト位置はオフセット対応。

## 3. 仕様(spec)の構築

- **プロンプトなしの裸の文**をベースgemma-2-2b+層LのSAE(Gemma Scope
  16k)でエンコードし、**編集スパン局所max-pool**+層別blocklist+top-k。
- 学習: キャッシュ済みz(層別サイドカー)から diff_to_sparse、
  k_top=32、k_amp/k_sup〜log{1..32}。評価: k=64/64。
- 同定(裸)と注入(プロンプト内)の文脈ずれは診断7で定量し
  reports/context_shift_diag.md に記録(査読対応)。

## 3'. specのfeature化 — 評価プロトコル変更(2026-07-22ユーザー決定・実行中)

> **決定(2026-07-22)**: 評価=正準500(seed-42、従来と同一)、同定=残り
> 4,451ペア。**全アーム**(ef=プール平均spec / LinguaLens=FRC top-3 /
> AxBench=AUROC top-1)が同じ同定プールを使い、評価ペアはどのアームの
> 介入導出にも寄与しない。FICも新プロトコルで再評価(旧prepost判定は
> 停止・破棄)。旧oracle-spec値は**本文から完全に外し付録限定**(仕様追従の上界診断、README規則0')。
> 実装: eval_split.json(正準split)、build_feature_specs.py(1パス3層、
> ef spec+FRC+AUROC+split-half安定性)、eval_ef_bare --feature-spec、
> run_fspec_pilot.sh(L12パイロット)。

動機: 現行のexact/FIC評価はspecを**評価対象ペア自身**(z_tgt−z_src)から
抽出する oracle-spec 設定 — x_tgt を見ており、LinguaLens批判(同定/評価の
データ非分離、audit §5)が我々にも刺さる。評価時のspecを「**同じfeatureの
他のペア**」から構築すれば、(i) 入力独立・オフライン準備というアーキ図の
主張と実装が一致、(ii) 同定/評価の分離が主評価でも成立、(iii) exactが
「他の例示から得たfeature方向で新しい文を正しく編集できるか」という
真の汎化テストになる。

**設計案**:
- 除外集合の正準化(実装済み): runs/tables/eval_split.json =
  評価500(seed-42 stdlib、従来のexact評価と同一)/ 同定プール4,451。
  FIC再生成も評価側はこの500を使う(旧FICベア独自サンプルは廃止)。
  spec構築・同定系スクリプトは全部このファイルを読む(除外レシピ不整合
  48/500の恒久修正を兼ねる)。
- 集約: feature f のspec = mean over pool pairs of (z_s2 − z_s1)
  (裸文エンコード+編集スパン局所max-pool、現行スクリプトと同じ)。
  **符号付き連続量の平均**であって二値top-r選択ではない — LL完全版
  (FRC r3二値クランプ、exact 0.016)/AxBench完全版(AUROC r1、0.070)の
  崩壊はいずれも「粗い二値・極小r」であり、P-J r掃引がr=64まで単調改善
  することから、大きさ保存×k=64は相当回復すると期待。amp方向は符号反転
  (1 spec/featureで両方向)。
- top-k・スケール: 平均は事例固有成分を相殺しノルムが縮むため、
  top-64(±)切り出し後、poolペアのper-pair specノルム中央値(pool内で
  計算 — 洩れなし)に合わせて再スケール。editor入力形式は不変。
- 選択不安定性(audit §5)への防御: 我々の集約はmaxでなくmeanなので
  勝者の呪いを踏まない。spec構築時にsplit-half cosineを副産物として出力し
  「specベクトルの安定性」を数値で示す(top-3一致36-43%のLinguaLensとの
  対比材料)。

**妥当性の論点(正直に書く)**:
- editorは学習時per-pair spec(corruption由来)で訓練済み → feature化
  specはeditorにとって二重にOOD(集約形式×LinguaLens語彙)。exactの
  低下は「feature specが原理的に劣る」と「editorが未適応」の合成であり、
  パイロットで分離不能。低すぎる場合のみspec摂動augmentation付き
  再学習を検討(要ユーザー承認)。
- 語彙的に特異な編集(意味・語用の対事実)はfeature方向だけでは復元
  できない可能性 — 4分類プロファイルが morphology/syntax 優位に反転する
  ことが予想され、それ自体が知見。oracle-spec値の役割は付録での上界
  診断のみ(本文使用禁止、README規則0')。
- prompting腕(A3′)は元からfeatureレベル(用語+gloss)で入力独立 —
  再実行不要、比較の土俵がむしろ揃う。

**強度較正の規約(2026-07-22ユーザー確定)**:
- 本文 = 層別・腕別にプール内dev標本(100ペア、評価500と非交差)で選択した
  スケール。AxBenchの手法別最適強度選択(評価分布の半分で選択)に相当し、
  かつ評価分布に触れない分だけ厳格。
- **正直な記述**: dev標本は同定プールの一部(spec構築への寄与<2.3%)で、
  同定/ハイパラの完全分離ではない。実害の兆候なし(dev選択3.5 vs
  評価500ピーク2.5と、漏れなら一致する方向に出るはずのずれが逆に存在)。
- **付録 = スケール1.0固定の参照行**(較正なし)。L12 supで0.0822
  (dev選択比−36%)と較正の寄与を透明に示す。残セルはfs_s1_*で測定。

**計算コストと削減策**:
1. spec構築: 1 GPUパスで全poolペア(≈8,000文)の残差を**3層同時に**
   フックし3つのSAEでエンコード → 3層分のfeature spec JSONを一度に生成
   (~0.5h、1回きり)。split-half cosineも同パスで出す。
2. パイロット(ゲート): L12・sup方向・ef+steer+clamp のexactのみ
   (~3-4h GPU、1セッション)。oracle 0.3166 との比、steer(feature化)
   との序列が判断材料。
3. 展開(パイロット通過後): 3層×3腕×amp/sup exact + FIC再生成を、
   nb系学習完了後の評価パスに相乗り(既存キュー運用のまま)。FIC judgeは
   新規生成分のAPIコストのみ(キャッシュは効かない)。
4. LOOはやらない: 除外集合を固定した「同定プールspec」1本/featureで
   十分(LOOはevalペア全数を評価に使う場合のみ必要で、計算もN倍)。

データ = corruptionキャッシュ(LinguaLens不使用 = ゼロショットOOD)。
中間状態 x_t(t0-prob 0.5、差分の部分適用)で「残りの編集」を学習。

| 行種 | 確率 | NLL教師 | その他 |
|---|---|---|---|
| true | ~0.80 | x1(編集後の文) | 編集トークン重み1.0/背景0.1、λ BCE(編集位置=1) |
| empty | 0.08 | **x_t(無介入コピー)** | Δhノルム抑制、λ BCE全ゼロ |
| mismatch(P5) | 0.12 | **x_t(コピー)** | 同上(誤spec→無編集の対比教師) |

- 損失 = 編集重み付きNLL + λ BCE(w=0.2)+ ノルム予算(0.5·‖dvec‖、
  超過ペナルティ)+ empty/mismatchのノルム抑制
- 40kステップ、10k時点でprobe100(fail-fast)、best-dev採用、resume対応

## 5. 評価(EF_LM_LOSS_PLAN §0b)

1. **KL/NLL反実仮想整合(主)**: p(・|x0+Δh) vs 凍結LM自身のp(・|x1入力)
2. **FIC(主)**: LinguaLens逐語(裸src→自由継続、temp1.0、gpt-4o judge)
3. **exact(従)**: 復唱枠での編集後文の厳密一致
- 統制: true/empty/random。診断: λ-IoU、ノルム実測/予算比
- コア表: 4腕(A1 LinguaLens忠実set / A2 AxBench忠実steer / A3 AxBench
  prompting移植 / A4 本手法)× 3層 × {FIC, exact}

## 6. 系譜(設計判断の根拠、要約)

- v1(bare枠・恒等初期化): コピー/沈黙に崩壊(true≈random)
- v2(編集トークン絞りloss): 編集は出るがspec盲目のまま
- v3(λ直接教師): 発火が下がるだけで対比が立たず — **through-LM経路では
  条件付けの獲得が困難**と確定(devでも同様=OODではなく学習の問題)
- v4(S3エンコーダwarm start+echo教師): **bare枠で初のspec分離**
  (true 0.1904 / random 0.0020、同枠steerの4倍、oracle残差天井超え)
- **v5f(現行、L12完成)**: v4レシピを**src先行復唱枠**に移植 — 復唱を
  Δhに背負わせず(プロンプトが99%供給)、介入は編集内容に專念。無介入
  コピー行(empty/mismatch→x_t)をNLLで直接教師。
  **L12: exact 0.2485 / KL_red 0.778 / NLL(x1) 0.74 — 同枠全ベースライン
  および旧rewrite枠チャンピオン(0.2385)超え、統制完璧**
- L4でのL12エンコーダ移植は劣化(0.0601 < steer 0.1343)→ **層ネイティブ
  EF基盤方式へ**(§1)。この移植劣化自体が「エンコーダ層一致の必要性」の
  ablation証拠として論文に載る
- 診断群: 線形描画天井(oracle位置でも0.04)/oracle残差0.11(挿入ゼロ・
  復唱不在が主因)/境界Δ寄与+0.01(不採用)/文脈シフト(診断7、
  src先行採用)

## 8. Ablation計画(2026-07-20確定、実行中)

**Tier 1(学習、L12)**: ②S3(層EF)warm startなし / ③対比教師なし
(mismatch=0, empty=0)/ ④ノルム予算なし — v5fレシピから1要素ずつ除去。
①フレーム(bare vs 復唱)はv4/v5f比較として測定済み。
**結果(04§5b)**: ②0.0361(−85%、最重要要素と確定)、③0.2104(−15%+
λ場のスペック特異性喪失: empty IoU 0.10→0.40)、④学習中。

**Tier 2(probe再実行のみ、L12 v5f)**: ⑤推論時top-k掃引(k=1..64、
499ペア、**feature別最適k+oracle-k集計**)/ ⑥反復推論(rounds=3、
λ自己停止)/ ⑦specスコープ(local vs global)/ ⑧blocklist有無 /
**⑨介入強度掃引**(ef-scale 0.5-2.0、best k固定 — steerのα崖に対する
学習較正の実証、ユーザー追加)。

**feature別集計**(ユーザー指示): 全結果をfeature別にも出力
(scripts/aggregate_per_feature.py、最適kのfeature依存性を含む)。

**系譜表(v1→v5f)自体が主ablation表** — 各設計要素の除去が対応する
失敗モード(コピー崩壊・spec盲目・条件付け不獲得)を実測で持つ。

## 7'. 査読防御メモ

- 復唱プロンプトは編集情報ゼロ(全ペア同一文言)。編集の帰属はspec無作為化
  (true/random/empty)が担う
- 同定(裸文)と介入(プロンプト文脈)のずれ: 方向加算型・学習型は動作点
  ずれに頑健、値設定型(A1)は感受 — 診断7の実測を引用
- FICは「指標のデータセット拡張」(原実験は5feature×1プロンプト×50
  リサンプル)、exactは我々の指標としてフレームごと定義
- デコード規約の非対称(exact=greedy / FIC=temp1.0)への防御
  (2026-07-21ユーザー確定): (i) 両指標とも「ペアごと1生成の平均」で
  繰り返し数の非対称はない — 違いはデコード規約と集計粒度のみ。
  (ii) デコードは各指標の出自に従う(FIC=LinguaLens公式repo既定、
  exact=生産的greedy)。各表の内部では全腕同一デコード → 腕間比較は
  内的に妥当。指標間で数値を直接比較する操作はしない。
  (iii) 頑健性: 両フレームFIC(bare=temp1.0 / 復唱=greedy再利用)が
  デコード感度チェックを兼ねる — 腕順位の一致をjudge完了後に確認して
  記載。要求されればexactのサンプリング版(probe1本≈0.4h)を追加可。
