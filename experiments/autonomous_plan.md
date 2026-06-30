# 自主實驗計劃：逗號問題的模型層解決方案

建立時間：2026-07-01 02:40
截止時間：2026-07-01 12:00（中午）
喚醒間隔：每 15 分鐘

## 目標

驗證能否從模型推論層改善「無逗號長句」問題（影響 ~23% 段落/35 段 >20字無逗號）。
已知：逗號在問題段中經常排第 2（機率 10-30%），聲學停頓幾乎為零。

## 成功標準

- 問題段改善率 >50%（至少一半段落新增逗號）
- 正常段文字不變率 >95%（副作用可控）
- 正常段誤插逗號率 <10%
- **以上指標僅供篩選，不做最終判斷**——最終判斷靠人工抽讀變化段落的前後原文

## 實驗設計原則（使用者指示）

1. **不過度依賴計算指標**。指標能快速篩選，但結論要靠閱讀實際輸出文字。逗號插在錯誤位置在指標上是「改善」，在原文上是「品質下降」。每個階段做結論前，必須抽讀有變化的段落原文。
2. **每個實驗要有明確假說**。跑之前先寫：「預期看到什麼？看到了代表什麼？沒看到代表什麼？」兩種結果都要有推進意義。沒有假說的「試試看」不跑。
3. **目的先於數據**。先確定想回答什麼問題，再設計能回答那個問題的實驗。不要「撒一堆數據看看有沒有什麼」。
4. **避免方向偏移**。每次喚醒先回顧核心目標，確認當前任務仍然對準目標。如果發現跑偏就停下來，記錄為什麼跑偏，回到主線。

## Phase 1：Logit Boost 網格搜索 ✅

**狀態：完成。結果在 experiments/results/experiment_logit_boost.json**

- 20 組參數（threshold × alpha，max_boost=3.0 固定）
- 153 段全跑
- 結果：`experiments/results/experiment_logit_boost.json`
- **完成後**：分析結果，判斷可行性

## Phase 2：精細調校（僅在 Phase 1 有改善但副作用偏高時進入）

**目的**：Phase 1 如果顯示 logit boost「能插逗號但位置或副作用有問題」，在此精調。

- **2a. 條件觸發優化**：分析 Phase 1 中的假陽性案例——逗號插在哪些位置是錯的？能否加更精準的觸發條件（如特定前文模式才 boost）？
  - **假說**：假陽性集中在特定語境（如句末、問號前），加排除條件可降低
  - 看到 → 副作用可控，logit boost 可用；沒看到 → 假陽性隨機分佈，logit boost 天生不精準

- **2b. 組合策略**：logit boost + beam search（beam>1 時 boost 效果可能不同）
  - **假說**：beam search 的多路搜索可能讓 boost 更穩定（不像 greedy 一步定生死）
  - 看到 → beam + boost 是更好的組合；沒看到 → greedy + boost 已是最佳

注意：不做沒有假說的「試不同 temperature/sampling」散彈實驗。之前已確認 beam=10 全部序列都零逗號，純搜索策略調整不會幫助。

## Phase 3：Bayesian Decision + 約束最佳化

**前提**：Phase 1 的固定規則（全域 threshold + alpha）副作用太大。根因是 42% 非逗號步驟的逗號已在 top-10，全域 boost 無法區分「該插」和「不該插」。需要**每步獨立決策**。

**方法論**：貝葉斯決策理論——每個 decoding step 是一個二元決策（boost 或不 boost），用非對稱損失函數反映「誤插比漏逗號更糟」。

- **3a. 特徵分析 + Logistic Regression（先做）**：
  - 訓練資料：logit probe 數據（comma_logit_probe.json）
    - 正樣本 = 正常段中模型實際輸出逗號的 209 步
    - 負樣本 = 其他步驟（降採樣平衡）
  - 特徵：chars_since_punct、逗號機率、逗號排名、winner-comma gap、前文 token 模式
  - Logistic regression → 輸出 P(該斷句 | 特徵)
  - 決策閾值由損失比決定（非對稱：false positive 成本 > false negative 成本）
  - **假說**：正常段的逗號位置包含可學的統計信號（語境+聲學的隱含編碼），logistic regression 能區分「該插」和「不該插」
  - 看到（AUC >0.8）→ 特徵有區分力，進 3b 驗證；沒看到 → 特徵不足，需 3c 更深層知識
  - **額外產出**：特徵重要性排名（哪些特徵最能預測逗號位置？）

- **3b. Pareto Front + 最佳閾值（3a 成功才做）**：
  - 用 3a 學到的模型，掃描決策閾值 0.1~0.9
  - 每個閾值對應一組（改善率, 副作用率），畫 Pareto front
  - 找拐點（knee point）= 改善邊際收益開始遞減的位置
  - **假說**：Pareto front 有明確拐點，比 grid search 更精準地定位最佳 tradeoff
  - 看到 → 選拐點閾值，作為 LogitsProcessor 的決策邊界；沒看到（front 是線性的）→ 問題本質不可分

- **3c. 升級選項（3a AUC 不足才做）**：
  - Decision tree（可解釋的非線性邊界）
  - N-gram 字元模型（從 log 正常段建，加「在 XYZ 後出逗號的機率」特徵）
  - 小型 LM shallow fusion（ckiplab/gpt2-base-chinese，最後手段）

## Phase 4：深度分析

**無論前面結果如何都做，提供給使用者明天看的洞察。**

- **4a. 語言模式分析**：什麼語境下模型漏逗號？（從 logit probe 數據挖）
  - 哪些前文詞彙跟「逗號排第 2 但沒被選」相關？
  - 逗號機率跟音訊長度/段內位置的關係？
- **4b. 正常段 vs 問題段的聲學特徵比較**
- **4c. 綜合報告**：所有實驗的結論整理

## Phase 5：收尾

- 更新 memory（實驗結論）
- 更新 CHANGELOG
- 刪除 cron 任務
- 留摘要給使用者

## 階段性 Commit

每個 Phase 完成時 git commit，保留變更紀錄：
- 實驗腳本、結果 JSON、分析摘要
- autonomous_plan.md 的進度更新
- commit 訊息用中文，說明該階段做了什麼、結論是什麼
- 注意不要 commit 大型二進位檔（.wav, .safetensors 等，已在 .gitignore）

## 決策樹

```
Phase 1 完成
├─ 改善 >50% + 副作用 <5% → 記錄最佳參數，進 Phase 4 分析
├─ 改善 >50% 但副作用 >5% → 嘗試 Phase 2 解碼策略看能否更乾淨
├─ 改善 <50% → 進 Phase 3 shallow fusion
└─ 全部失敗 → Phase 4 分析 + 結論「需要外部標點模型」
```

## 執行日誌

（每次喚醒追加）

### 02:40 — 計劃建立
- Logit boost 實驗背景執行中

### 03:00 — Phase 1 結果出爐
- 20 組參數完成，結論：**有效但副作用太大，不存在甜蜜點**
- 最佳妥協 t15_a0.2：問題改善 51.4%，但正常段 21.7% 多逗號
- 想要改善 >50% → 正常段至少 20% 被多插逗號
- 想要副作用 <10% → 改善只有 40-48%
- 根因：42% 非逗號步驟的逗號已在 top-10，全域 boost 無法區分「該插」和「不該插」
- **決策樹判定**：進入 Phase 3a（data-driven boost），用現有數據 fit 分類器取代固定規則
- **待做**：先抽讀幾段有變化的原文（遵守實驗設計原則），再正式推進 Phase 3a

### 03:15 — 計劃更新（使用者指示）
- 使用者希望用**正式的最佳化方法**延伸 logit 研究，不限於簡單分類器
- 採用**貝葉斯決策理論 + 約束最佳化**框架：每步獨立決策、非對稱損失、Pareto front
- Phase 3 已重寫：3a logistic regression + 特徵分析 → 3b Pareto front + 最佳閾值 → 3c 升級選項
- 執行中可依結果調整方向（使用者授權）
- 舊 cron b1a4f8f7 已刪（使用者醒著時手動刪除），即將建新 cron
- 新 Cron ID: 377110f1

### 03:30 — Phase 1 原文抽讀 + Phase 3a 完成
- **原文抽讀結論（Phase 1）**：
  - 問題段逗號位置品質不錯（「就是」「然後」「或者是」前，自然斷句）
  - 但 boost 改變 decoding path 導致用詞連帶變化（「鬥號」→「逗號」）
  - 正常段副作用有好有壞（150650 合理，124401 切斷片語「作為之後,如果要」）
- **Phase 3a-v1 結果**：AUC=0.999 但假象——分類器只學到 comma_prob 主導（weight=+11.4），循環論證。Pareto front 比 Phase 1 更差
- **Phase 3a-v2（純語言特徵）結果**：
  - AUC=0.824 — **語言特徵有獨立的區分力**
  - 最重要特徵：`prev_comma_prior`（bigram 先驗，weight=+6.4）
  - Token `的話`/`啦`/`好` 後接逗號機率 >60%
  - 問題段中 score>0.5 的步驟都在自然斷句位置
  - **語言 + 模型互補**（AUC 0.824 → 0.998）
- **結論**：固定規則 boost 失敗因為不知道「何時」boost。語言 bigram 先驗提供了精準的「何時」信號。
- **下一步**：Phase 3b — 設計 Shallow Fusion LogitsProcessor，用 bigram 先驗作為 boost 依據，跑完整模型驗證

### 04:00 — Phase 3b-v1 Shallow Fusion 完成（失敗）
- 24 組參數，改善率 0-2.4%，遠低於 Phase 1 的 51.4%
- **根因**：boost 公式用 log-odds（`alpha * log(p/(1-p))`），prior < 0.5 的 token boost=0
  - `吧`(0.4)、`出`(0.36)、`一下`(0.31) 全部沒有 boost
  - 只有 `的話`(0.70)、`啦`(0.70)、`好`(0.61) 等 <10 個 token 有 boost
  - 觸發窗口太窄
- **修正**：v2 改用 `boost = alpha * prior`（直接縮放），並加入 combined model：
  - prior 提供「這裡適合」信號，chars_since_punct 提供「該逗號了」信號
  - 兩者相乘比單獨使用更精準
- v2 正在背景執行中

### 04:45 — Phase 3b-v2 Shallow Fusion 完成
- 17 組參數（pure prior / combined / combined_g8）
- **Pure prior 仍不足**：alpha=12 也只有 26.2% 改善、27.7% 副作用。先驗太稀疏
- **Combined model（prior × overdue）有改善**：
  - 最佳 tradeoff: `combined_g8_a5.0_b0.2_cg8` → 45.2% 改善、16.8% 副作用
  - vs Phase 1 (t15_a0.2): 51.4% 改善、21.7% 副作用
  - **效率比更好**（2.69 vs 2.37），但絕對改善率略低
- **原文抽讀**：4 段中 2 段好（逗號在「然後」「所以」「因為」前）、2 段差（切斷片語「有沒有,剛剛」「句子就,不一定要」）
- **核心結論**：bigram 先驗提升了效率比，但沒有突破 logit manipulation 的天花板
  - 天花板 ≈ 50% 改善 @ 20% 副作用
  - 根因：logit boost 改變 decoding path → 連帶影響其他詞、斷句不精準
  - 更聰明的觸發條件（prior / combined）能稍微改善效率比，但無法打破天花板
- **Phase 3 結論**：logit manipulation 方向到此為止，已確認天花板
- **下一步**：Phase 4 深度分析 + Phase 5 收尾
