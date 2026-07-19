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
- エンコーダは**S3チャンピオン(トークン出力EF)のspec読取済み重みから
  warm start**(λ-IoU 0.73/0.30の分離を実証済みの条件付け回路を移植)。
  ヘッドは新品。

## 2. 生成フレーム(v5、2026-07-19ユーザー決定)

**明示復唱指示プロンプト**(chat template、英語):

> Input: {src}\n\nRepeat the input sentence exactly. Never output anything else.

- 採用根拠: 素のgemma-2-2b-it(SAE/editorなし)でLinguaLens src 100文の
  **99%を逐語コピー**(scripts/test_repeat_prompt.py。chat template必須 —
  bareテキスト指示は0%。残り1%は引用符正規化)。
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

## 4. 学習(through-LM loss)

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
- v5(現行): v4レシピを復唱枠に移植 — 復唱をΔhに背負わせず(プロンプトが
  99%供給)、介入は編集内容に專念。無介入コピー行をNLLで直接教師
- 診断群: 線形描画天井(oracle位置でも0.04)/oracle残差0.11(挿入ゼロ・
  復唱不在が主因)/境界Δ寄与+0.01(v5から除外)

## 7. 査読防御メモ

- 復唱プロンプトは編集情報ゼロ(全ペア同一文言)。編集の帰属はspec無作為化
  (true/random/empty)が担う
- 同定(裸文)と介入(プロンプト文脈)のずれ: 方向加算型・学習型は動作点
  ずれに頑健、値設定型(A1)は感受 — 診断7の実測を引用
- FICは「指標のデータセット拡張」(原実験は5feature×1プロンプト×50
  リサンプル)、exactは我々の指標としてフレームごと定義
