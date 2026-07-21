# reports/ — 論文執筆用資料(日本語執筆の下敷き)

作成 2026-07-17。**2026-07-18 ユーザー再固定を反映**:

> 🔶 **研究の大前提**: SAEへのintervention(仕様 z_amp/z_sup)を信号として
> **editor**(LEWIS/EF系のエンコーダ)に入力し、**editorの出力embedding
> (Δh)を凍結LMのresidual streamに戻す**。テキストは凍結LMの生成として
> 出る。editorを使わない案(素のsteer/clampを最終形とする枠組み)は
> ユーザー指示により削除した。

論文 = **SAE仕様に条件付けられた学習editor(Intervener)の提案**+
編集可能性による因果評価。steer / clamp は**ベースライン・統制**として
のみ載せる(steerはeditorの自明な特殊ケース=固定描画であり、v2の
初期化基底でもある)。トークンを直接出力するEF系(S系列・ef32・routed・
λ-IoU・M0・P-B)は条件付けのためコード/runsに履歴として残るが
**論文には載せない**(出力インターフェース基準)。

## ファイル構成

| ファイル | 対応する章 | 内容 |
|---|---|---|
| `01_introduction.md` | Introduction | 動機、主張、貢献リスト、冒頭数値、タイトル案 |
| `02_previous_works.md` | Related Work | 冒頭の改訂ガイドに従って読む(embedding出力editorの復帰でD.3の一部が再関連) |
| `03_method.md` | Method | **提案editor(Intervener)**、仕様構築、ベースライン効果器、統制、指標、局在性測定器、判別木、再現 |
| `04_experiment.md` | Experiments | 確定数値の表(トークン出力EFなし)、統計、Intervener欄は学習完了待ち |
| `05_discussion_limitations.md` | Analysis / Limitations | 3層分解、免許規則、attenuation、限界 |
| `06_pipeline_and_theory.md` | 理論とパイプライン状態 | 条件付けvs介入の理論(⚫の根拠)、実験の状態表 |
| `axbench_testdata.md` | 付録/実験 | AxBenchテストデータの正体と再現・相互評価設計 |

数値の一次出典: `PAPER_OUTLINE.md`(🔵/⚫節と§6x台帳)、`runs/tables/`、
`runs/paper_metrics/report.md`、`runs/ll_repro/report.md`、`RELATED_WORK.md`。

## 🔴 全章共通の規則(違反すると主張が壊れる)

0'. **🚫 oracle-spec値を本文に書かない(2026-07-22ユーザー指示)**。
   評価ペア自身からspecを抽出した旧プロトコルの数値(0.3166、0.2485、
   0.2886、0.1804、0.1363、旧FIC全値、下のクイックシート太字値を含む
   一切)は**付録限定**(仕様追従の上界診断)。本文の主表・主claim・
   Abstract・Introductionはすべて**feature-specプロトコル(03§3')の
   再測定値のみ**を使う。ベースライン(FRC-clamp/AUROC-steer/prompting)
   との比較も新プロトコル値同士でのみ行う。
0. **🔶 editorを使わない枠組みで書かない**(2026-07-18ユーザー指示)。
   提案はSAE仕様→editor→Δh→residual streamのシステム。steer/clampの
   数値には必ず「ベースライン」「統制」「固定描画の特殊ケース」の修飾を
   付け、単独で主claimにしない。
1. **⚫ トークン出力EF系の数値・記述を論文に使わない**(routed 0.2839、
   ef32 0.2237、λ-IoU 0.74、M0のk掃引、P-BのFRC条件付け10×崩壊など
   すべて — 出力インターフェースがトークン=条件付けのため)。
   これらの結論はEF非依存の実験が引き継ぐ: P-B → **P-J**、M0 → **B-1
   介入k掃引**、内容混入の証拠 → **S_min組成分析**。
2. **免許規則**: 「事例レベルkで編集成功 → featureの表現はk本」と書かない
   (操作ハンドルには事例内容が混入)。「対応はtop-3より広い/別物」を
   書けるのは (形1) 現象レベルr掃引(P-J、因果的定義への取替を明示)
   (形2) S_min事例横断安定核×FRC3比較、の2形のみ。
3. judge評価は「**reliability**を測る」(品質/validityは禁止 — Norman et al.)。
4. R5の禁止事項: 「介入ベースのSAE評価は初」(SAEBench/RAVEL先行)、
   「因果基準でlatent選択は初」(Beyond Input Activations先行)、
   「言語minimal pairの因果介入評価は初」(CausalGym先行 — 差分は評価器
   [挙動→テキスト編集]と対象[学習特徴化→SAE latent])。
5. 単位は「言語現象」、編集は「最小対変換(minimal-pair transformation)」。
6. LinguaLensのFRCは論文とコードで定義が違う(条件付き vs 周辺)—
   我々はコード側に忠実、と1文明記。

## 数値クイックシート(2026-07-21更新 — コア表はreports/04が正)

> **🚫 注意(07-22)**: 本シートのexact/FIC値はすべて**旧oracle-spec
> プロトコル=付録限定**(規則0')。本文用の数値はfeature-spec再測定
> (進行中)で置き換わる。ここの値を本文に書かないこと。

| 主張 | 数値 | 状態 |
|---|---|---|
| **本手法(v5f EF editor)L12** | **exact 0.2485** / KL_red 0.778 / NLL(x1) 0.74(random 0.002 / empty=raw)— 同枠steer 0.1804・A3′ 0.1804・clamp 0.1363を超え、旧rewrite枠チャンピオン0.2385も超過 | **確定・主claim**(src先行復唱枠) |
| 復唱枠ベースライン(499) | steer0.5: L4 0.134 / L12 0.180 / L20 0.006; clamp: 0.016/0.136/0.002; A3′ prompting 0.180(net 0.092); A3-literal 0.006(付録) | 確定 |
| (旧rewrite枠、参考) C1' 2×2 | steer 0.2337/0.2385、clamp 0.1743 vs LL完全版 0.0160・AxBench完全版 0.0701 | 確定・歴史的ベースライン |
| 床と統制 | raw 0.0601 / recon 0.0100 / random 0.0521 | 確定 |
| 動作点の崖 | steer α 0.25/0.375/0.5/0.75 = 0.098/0.176/**0.2385**/0.18; clamp 5/10/20 = 0.06/0.17/0.03 | 確定 |
| P-N(彼らの指標) | E_abl 我々 +0.631/+0.370 vs 両プロトコル +0.091/+0.115; AxBench score 1.211/1.262 vs 0.570/1.121(raw 1.081) | 確定 |
| P-I WHERE(因果床) | true発火 3.57 vs random 1.08、393/31、p=5.6e-81; WHAT不能 | 確定 |
| P-J r掃引(形1) | FRC r3 0.17/0.00(67/0, p=1.4e-20)→ r64 2.51 単調非飽和; AUROC r1 **ゼロ**(0.86×)なのに検出 mean **0.939** | 確定 |
| FRR(steer行) | FRR 0.7327 / net-FRR 0.4062 / random床 0.3265(GPT-4o); steer自己一致 0.9781 | 確定(EFなし再集計は残) |
| B2(SAE不使用参照) | 0.1242、empty copy 0.4770 | 確定 |
| LinguaLens再現 | FIC 12.0/8.6/13.6/3.0 vs Table2 8.3/22.9/46.9/6.9(弱い行がアンカー) | 確定 |
| 判別木(paper版) | **A 71 / C 21 / B 2 / D 4**(98現象) | 確定 |
| B-1 介入k掃引(操作幅) | k=1..64: 0.068→0.108→0.194→**0.237**→0.239(膝=32、k=1は床)、random平坦~0.05 | 確定 |
| B-2 S_min+安定核(形2) | \|S_min\| median **5**(vs 仕様98.7、S_min=1例あり); 安定核 20/22 非空・核∩FRC3 **8/22**・AUROC1は核に頻出 | 確定 |
| AxBench再現(厳密) | L20: sae **0.162**/sae_a 0.139(アンカー0.151/0.132に一致); L10: sae 0.146/sae_a 0.229(序列逆転は注記) | 確定 |
| AxBench層拡張(sae_a) | L4 0.197 / L10 0.229 / L12 0.179 / L20 0.139 — 浅層ピークの山型 | 確定 |
| LinguaLens再現v2(3層・公式忠実化後) | FICアンカー弱(全セル一桁〜10台 vs Table2)— プロトコルのスタック転移の弱さとして議論 | 確定 |
| **深さプロファイル(層ネイティブv5f2、40k主表)** | ef true = **L4 0.1343 / L12 0.2485 / L20 0.0421** — 全層で同枠steer以上(L20はsteerの7倍) | **確定**(04§5c/5d) |
| 延長80k(収束性分析) | L4 **0.1804**(+34%、steer超え)/ L20 **0.0601**(+43%)/ L12 0.1924(**−23%悪化**)— 未収束層のみ延長有効。dev NLLはexactの弱代理 | 確定(04§5d') |
| KL/NLL深さ(3層×3腕完成) | ef KL_red 0.695/0.778/0.597 — 全層で正かつ最大(steer L20 −0.207 逆効果、clamp L4 異常系) | 確定(04§3) |
| FIC両フレーム(L12、4腕) | bare: ef **0.376**=SAE腕最高(steer 0.284・clamp 0.304; prompting 0.456はE_enh 0.886/E_abl 0.014の偏り)。復唱E_abl: ef **0.940**=4腕トップ。腕順位は両フレーム一致=デコード頑健 | 確定(04§5e) |
| 復唱E_abl 全層×{40k,80k} | 全6セルで ef > steer。judge再判定ノイズ床 ≈1.5pt(steer同一出力の再判定で実測)。L20 80kの−12ptのみ明確な実差 | 確定(04§5e') |
| Tier1 ablation(単一要因) | noS3(EF基盤なし)**0.0361 = −85%**(steer以下に崩壊)/ noctr(対照教師なし)0.2104 = −15%+empty λ-IoU 0.10→0.40(スペック特異性喪失)/ nobudget **L12 40k 0.3166=+27%で確定**(80k 0.2886で40k勝ち、L4/L20は予算あり優位の層依存) | 確定(04§5b) |
| 4大分類別(L4 vs L12) | L12=統語0.253・意味0.307の本拠地。「浅層=形態統語」仮説はexact/FIC両指標で棄却。morphologyは全層最弱(サブワード要因示唆)。pragmaticsのみL4 80kが両指標で逆転(≈2SE) | 確定(04§5f) |
| feature別(99現象) | exact>0が56現象、完全成功6(copular_be・past_tense・s_genitive等)。表=runs/tables/perfeature_ef_l12_combined.md | 確定 |
| FRC同定のsplit-half安定性(99現象×20反復×3層) | FRC値は汎化(top-1 in 0.88-0.90→out 0.84-0.86)だが**top-1特徴のhalf間一致 36-43%・top-3で約50%** — 同定される特徴集合の同一性はデータ依存。LinguaLens原法は全ペアin-sample選択+同一データGPT-4o検証でこれを測れない(262k候補)。表=runs/tables/frc_splithalf_l{4,12,20}.md、詳細=audit_ll_axbench.md §5 | 確定 |

## 進行中・待機(2026-07-22)

- **評価プロトコル変更(ユーザー決定07-22、03§3')**: 同定/評価データ分離 —
  評価500・同定プール4,451、全アーム統一。bspecs(3層同定パス)→
  fspilot(L12パイロット: ef feature-spec / FRC-r3 clamp / AUROC-r1 steer)
  がshort-gで自動連鎖中。**exact/FICの主表数値は全て新プロトコルで
  再測定になる**(旧oracle-spec値=上界診断として保持)。旧FIC判定
  (prepost)は停止・破棄済み
- nobudget学習系(プロトコル非依存なので継続): interact-g nbqチェーン
  (L12 nb 80k→amp probes→nb ablations)、short-g efl4nx(L4 nb 80k)。
  **L20 nb 80k完了: exact 0.0441**(nb 40k 0.0321から+37%だが予算あり80k
  0.0601未満 — L20は予算あり優位の層依存を追認)
- FIC再評価(新プロトコル・全層)— パイロット通過後に生成→prepost判定を再開
- k掃引feature別表(S8のperfeature_ksweep)— キュー末尾で自動生成
- 主表の学習量の判断(40k/80k/層別)は新プロトコル数値が出てから再検討
- LinguaLensとの差別化の検討(ユーザー、後日)
- T6(Llama同一スタック検証、要HFゲート承認)
- 旧参考項目(v2 Intervener 0.2725=プロンプト+基底+補正の比較行、FRR再集計等)は付録候補
