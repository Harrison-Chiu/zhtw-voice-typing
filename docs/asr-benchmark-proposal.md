# ASR Input 標準化 Benchmark 提案

狀態：提案，尚未取代既有一次性實驗。

## 1. 要回答的決策

Benchmark 不只回答「哪個比較快」，而要能穩定回答：

1. 候選在本機 RTX 4060 的日常短句延遲是否真的更低？
2. 辨識品質的改善是否足以抵銷速度、顯存與維護成本？
3. 哪一類音訊改善或退步：短句、長句、中英混合、低音量、專有名詞、停頓？
4. 改動是否造成漏字、幻覺、重複、簡體輸出或標點退化？
5. 同一輸入能否重現；差異是真實效果還是抽樣／系統噪音？

## 2. 固定測試集

### 2.1 三層資料集

| 層級 | 建議規模 | 用途 | 每個候選預估本機時間 |
|---|---:|---|---:|
| Smoke | 20 段、約 3–5 分鐘音訊 | 開發中快速排除明顯退化 | 3–8 分鐘 |
| Core | 120–200 段、約 25–40 分鐘 | 正式 A/B、品質與效能報告 | 20–45 分鐘 |
| Stress/Long | 20–30 難段 + 3 個長檔 | 幻覺、重複、切段、batch 吞吐 | 15–30 分鐘 |

時間包含載入、暖機、重跑及產生報告；純 GPU 推論通常只占一部分。首次下載模型、
冷開機磁碟讀取與人工校對不計入日常 benchmark 時間，分開報告。

### 2.2 分層抽樣標籤

Core 集至少覆蓋：

- 時長：`<3s`、`3–10s`、`10–20s`、`20–30s`、`>30s`
- 內容：純中文、中英混合、數字/單位、專有名詞、人名、列舉
- 聲學：低 RMS、背景音、遠場、停頓/贅字、句尾漸弱
- 已知風險：fallback、幻覺、重複、漏句、簡繁混雜、歷史誤辨識

資料切分：

- `dev`：可用來調參。
- `test`：凍結，不用來選參數，只做最終比較。
- 新增真實 log 時只追加新版本，例如 `benchmark-v1.1`，不覆寫舊集合。

音訊因含個人語音，不進 Git；Git 只保存匿名 sample ID、標籤、時長、音訊 hash、
gold transcript 與允許的正規化規則。私密 manifest 在本機將 sample ID 對應到 log 路徑。

## 3. Gold transcript 與品質指標

### 3.1 Gold 製作

1. 人工逐字聽寫，不以任何候選輸出直接當答案。
2. 第一遍忠實稿；第二遍對照音訊校正專有名詞、數字與標點。
3. 不清楚處以明確標記，不強猜；含糊樣本可排除在主 CER 外但保留壓力測試。
4. 同時保存兩個 target：
   - `verbatim`：包含贅字與口語重複，用來量模型忠實度。
   - `input_ready`：符合本產品需求的乾淨可貼上文字，用來量實際編輯成本。

### 3.1.1 AI 輔助與人工抽驗

不需要人工聽完所有 log，也不能讓單一 ASR／語言模型直接產生並裁決自己的 gold。
建議採「多來源預標註 + 分歧取樣 + 人工裁決」：

1. 先用 production FW、另一組解碼設定，以及至少一個不同架構/後端產生候選稿。
2. 自動對齊候選，計算逐字分歧、數字、英文、專有名詞、低信心與異常重複標記。
3. 較便宜的文字模型只負責整理候選、標記可疑位置、套用明確格式規則；不能只憑文字
   猜測聽不清楚的內容，也不能把任一候選直接升格為 gold。
4. 多來源完全一致、無風險標記的樣本可進低風險池，只抽驗 10–20%。
5. 有分歧、已知錯誤、數字/英文/專有名詞、低音量與候選採用決策會翻轉的樣本，才由
   使用者聽音訊裁決；必要時只重聽分歧時間窗，不必重聽整段。
6. 抽驗若發現低風險池錯誤率超過門檻（建議 1%），擴大抽驗或將該類標籤全部人工審核。

這能把人工作業集中在約 20–40 段高價值樣本。AI 可大幅節省整理與定位時間，但目前
一般程式代理不能取代音訊 ground truth；它的角色是預標註與 quality-control assistant。

### 3.2 主指標

| 指標 | 定義 | 用途 |
|---|---|---|
| Strict CER | 原始輸出對 verbatim 的字元錯誤率 | 模型本體、標點與字形完整比較 |
| Normalized CER | 統一空白、標點與可接受字形後的 CER | 純內容正確率 |
| MER | CJK 以字、英文/數字以詞計算的混合錯誤率 | 中英混合不被英文拼字長度扭曲 |
| Input-ready CER | 完整 pipeline 對 input_ready target | 本產品最重要的最終品質 |

CER = `(替換 + 刪除 + 插入) / gold 字元數`。所有候選同時報 raw 與 processed，避免
後處理把模型問題遮住，或把後處理收益誤算成模型收益。

### 3.3 診斷指標

- Deletion / insertion / substitution rate：特別辨識「快但漏字」。
- Punctuation F1：逗號、句號、問號、頓號分開報。
- Glossary recall：指定專有名詞命中率，另報 false positive。
- Traditional consistency：簡體殘留率與 OpenCC 改動率。
- Hallucination rate：gold 無對應內容的連續插入，按段數與字數報。
- Repetition degeneration：連續 token 重複、最高單字占比、異常輸出長度比。
- Empty/failed segment rate。
- Determinism：同檔同參數 N 次的 unique output 數與 exact-match 比例。

不建議把單一語意相似度或 LLM 評分當主分數；它們可能忽略數字、否定詞、專有名詞
與實際需要修改的錯誤。可作輔助，但不能取代 CER/MER 與人工錯誤檢視。

## 4. 本機效能規範

### 4.1 固定環境

- 記錄 GPU、driver、Windows build、Python、CUDA、cuDNN、CTranslate2、模型 revision。
- 固定電源模式；正式 tray 路徑保持 EcoQoS opt-out。
- 每輪記錄 benchmark 前 GPU idle memory 與其他 GPU process。
- 同一輪禁止同時跑其他 GPU 工作；保留環境快照於結果 JSON。
- 音訊預先解碼成固定 16kHz mono float32，音訊 decode 時間另報，不混入 ASR latency。

### 4.2 測量項目

- Cold load：重開 process，至少 3 次；「重開機後首次冷讀」另列，不與 warm load 混合。
- Warm latency：暖機一次後每段至少 5 次，報 median、P90、P95、P99。
- End-to-end latency：快捷鍵結束/切句到剪貼簿完成，包含 VAD、ASR、後處理。
- RTF：`推論秒數 / 音訊秒數`；另顯示 realtime-x = `1 / RTF`。
- Throughput：離線 batch 才報總音訊分鐘/牆鐘分鐘。
- Peak VRAM：用 `nvidia-smi` 100ms 取樣；同時保留 total 與扣除 idle 後 delta。
- CPU RAM、CPU 使用率、功耗：第二階段加入，先不阻擋 v1。

### 4.3 降低量測偏差

- 候選執行順序隨機或採 ABBA 交錯，避免溫度與 OS cache 單向偏差。
- 速度比較採 paired samples；報相對差值與 bootstrap 95% confidence interval。
- 單次差異小於約 5–10% 時，不先宣稱優化，除非信賴區間明確且 P95 也改善。
- 每個 backend 在 fresh process 跑，避免前一模型殘留 VRAM/DLL/cache 影響下一個。

## 5. 結果呈現

每次 run 產生 machine-readable JSON、Markdown 摘要與單一 HTML dashboard。

建議圖表：

1. **品質–延遲 Pareto 圖**：X=P95 latency，Y=input-ready CER，泡泡大小=VRAM。
2. **各時長 latency box plot**：看短句是否真的變快，而非被長檔總吞吐掩蓋。
3. **錯誤類型堆疊圖**：替換/刪除/插入/幻覺/重複。
4. **標籤 heatmap**：候選 × 中英混合、低音量、專有名詞等分層 CER。
5. **逐段差值圖**：每段相對 baseline 改善/退步，點擊可看 gold 與兩邊輸出。
6. **重跑控制圖**：P50/P95 latency、CER、版本與日期，抓長期 regression。

Dashboard 預設先顯示結論與 gate，仍可展開逐段全文，避免只看總平均。

## 6. 採用 Gate

針對互動式 tray 候選，建議全部符合才替換預設：

- Input-ready CER 不劣於 baseline 超過 0.5 個百分點，且 95% CI 不顯示實質退步。
- Hallucination、嚴重重複、空輸出不能增加。
- 3–20 秒主力區間 P95 latency 至少改善 10%，或品質有明顯提升可接受等速。
- Peak VRAM 不超過 7 GiB，保留桌面與 VAD 安全餘裕。
- Test 集所有 critical glossary 與數字案例無新嚴重錯誤。
- Windows 啟動、卸載、背景模式與 30 分鐘 soak test 通過。

離線 batch 模式另設 gate：可用較高 VRAM，主看 throughput，但同樣不得以漏段換速度。

## 7. 建置順序

### Phase 1：規格與資料（先做）

- 定義 manifest schema、normalization version、sample hash 與 config fingerprint。
- 從 697 個既有 session 中分層抽 Smoke 20、Core 120；人工製作雙 target gold。
- 把既有 `eval_models_2026_06_29.py`、`test_streaming.py`、codec 與 backend benchmark
  的共通測量邏輯抽成 `experiments/benchmark/` 套件。

### Phase 2：Runner 與指標

- fresh-process backend adapter：FW sequential、FW batched、HF SDPA；之後再加其他引擎。
- CER/MER、標點、glossary、簡繁、幻覺、重複與 determinism。
- 環境快照、VRAM sampler、重跑、隨機/ABBA 排程、JSON schema 驗證。

### Phase 3：報告與回歸

- 產生 Markdown + 可互動 HTML dashboard。
- 保存 baseline release，加入 `--compare-to` 與 gate exit code。
- Smoke 可日常執行；Core/Stress 在引擎、模型、VAD 或解碼參數改動時執行。

## 8. 預估投入

- Runner/資料格式/基本指標：1–2 天。
- 第一版 120 段 gold：多引擎預標註約 30–60 分鐘機器時間；人工集中審核約 1.5–3
  小時。若要求每段雙人完整重聽，才會上升到 4–8 小時。
- HTML dashboard 與 paired statistics：1–2 天。
- 第一輪正式 baseline + 候選報告：半天。

最小可用版本約 2–3 個工作天；完整 v1 約 4–5 個工作天。最大成本不是 GPU 跑的
時間，而是建立可信、凍結且能代表真實使用情境的 gold transcript。
