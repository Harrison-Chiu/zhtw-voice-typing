# ASR Input 主路線圖

本文件是專案未來工作的單一事實來源（SSOT）。它只管理路線、狀態、依賴、責任邊界
與決策；各項實驗的參數和驗收細節只寫在對應規格文件，不在此複製。

## 文件責任

| 文件 | 唯一責任 |
|---|---|
| 本文件 | A–F 路線、優先順序、狀態、跨線依賴與待決策 |
| `TODO.md` | 指向本文件的近期執行入口，不另存一份完整 backlog |
| `docs/asr-benchmark-proposal.md` | A 線資料、gold、指標、runner 與報告規格 |
| `docs/faster-whisper-customization-plan.md` | B 線 Faster Whisper／CT2 修改層級與候選技術 |
| `docs/vad-experiment-plan.md` | C 線 E0–E6 實驗與取代 Silero 的 gate |
| `CLAUDE.md` | 現行架構、已落地能力與已確立的技術決策 |
| `CHANGELOG.md` | 已完成且已提交的變更 |
| `README.md` | 使用者安裝、使用方式與文件入口 |

狀態只有：`ready`（規格足以執行）、`needs-decision`、`exploration`、`deferred`、
`done`。執行結果完成後移入 CHANGELOG；不在多份文件同步勾選。

## 已確立共識

- 互動式預設維持 faster-whisper `large-v3-turbo` sequential。
- batching 先限於離線／佇列；不得用漏段換吞吐。
- 先完成 A 線量尺，再採用 B/C 的任何產品變更。
- `外餐→外參` 是使用者確認的全域個人規則；「表點」使用完整詞組後處理。
- Gold 採多來源預標註與風險分層，只由人工裁決高價值分歧。
- 優先使用 faster-whisper／CT2 公開能力；有最小重現與 benchmark 證據才 fork。
- 大型 UI、模型微調與 CT2 kernel 修改不是近期前置工作。

## 路線與順序

```text
A 評測基礎（共同前置）
├─ B ASR／CT2 優化
└─ C VAD／音訊

D 啟動／Tray／生命週期 ─┐
E Log／回饋／歷史介面 ──┴─ 可在邊界凍結後平行

F 發布工程（確定對外發布時啟動）
```

### A — 評測基礎

狀態：`ready`，第一優先。

近期交付：

1. Smoke/Core/Stress manifest 與 config/environment fingerprint。
2. 多引擎預標註、風險分層與雙 target gold。
3. CER/MER、錯誤類型、決定性、P95 latency、RTF 與 VRAM。
4. JSON schema、Markdown 摘要與互動 HTML dashboard。
5. 保存 baseline，讓 B/C 候選可用同一 gate 比較。

詳細規格：`docs/asr-benchmark-proposal.md`。

### B — ASR／CT2 優化

狀態：`ready after A`。

順序：

1. 解碼參數與 temperature fallback 決定性。
2. compute type（含 `int8_float16`）。
3. VAD 片段 dynamic batching、length bucketing、worker/queue。
4. token/segment diagnostics。
5. 確認 API 缺口才 fork faster-whisper；profiler 證明 kernel 瓶頸才研究 CT2 fork。

延後研究：更多標點 prompt/suppress tokens、台灣語料微調/LoRA、Fun-ASR 繁體鷹架補測、
SenseVoice 等新引擎。既有模型橫評已收斂，沒有新需求或 A 線證據時不重啟模型競賽。

詳細規格：`docs/faster-whisper-customization-plan.md`。

### C — VAD／音訊

狀態：`ready after A fixture/schema`。

包含：

1. E0–E6：固定能量、oracle、自適應 noise tracking、robust statistics、頻譜規則與
   WebRTC，對照 Silero JIT/ONNX。
2. 連續壓力錄音、人工語音區間標註、VAD-only sweep 與 Whisper 下游測試。
3. Codec 嚴格逐字交叉測試；先處理 production fallback 不決定性。
4. 只有通過 non-inferiority gate 才取代 Silero；否則保留 Silero 並優先評估 ONNX。

詳細規格：`docs/vad-experiment-plan.md`。

### D — 啟動／Tray／生命週期

狀態：`needs-decision`。

已同意方向：錄音/VAD 可先開始、模型背景載入、ready 後消化佇列；支援閒置卸載、
開機啟動與單一實例。

待決策：

- 啟動時完全不載、背景預載，或預熱後卸載。
- 閒置預設 15/30/60 分鐘或永不卸載。
- 模型尚未 ready 而錄音提早結束時的等待／取消行為。
- 載入失敗、CUDA 不可用與麥克風錯誤的 UI 狀態。
- 通知靜音的顯式設定，以及直接輸出到游標位置是否解除暫緩。

在這些決策完成前，只做狀態機規格和測試，不直接改產品預設。

### E — Log／回饋／歷史介面

狀態：`exploration`。

建議分期：

1. Log retention、修正版 schema、錯誤標籤與最近結果資料 API。
2. 獨立歷史 viewer：搜尋、播放、raw/processed/final、複製與修正。
3. 從多次修正建議詞表/hotwords；使用者確認後才寫入。
4. 最後才做 Profile、應用程式感知或選取文字語音修改。

延後點子：信心詞高亮、OpenCC 差異標示、浮動狀態/波形、多語言切換、輸出前預覽
編輯與大型 Web UI。hotwords 使用者自訂屬第三期，必須包含整句回歸與明確確認。

待決策：2 GiB 上限的刪除順序、是否保留 metadata、FLAC 轉換時機、發布版預設是否
保存音訊。競品研究在具體 UI 決策前按題目進行，不另立泛用研究專案。

### F — 發布工程

狀態：`deferred`，確定公開或給其他人使用時啟動。

範圍：LICENSE、版本統一、CI、安裝/環境檢查、首次模型下載、portable onedir、
發布/開發設定分離、`docs/index.html` 翻新，以及是否移除 Qwen 和不必要依賴。

## 平行開發邊界

建議本任務負責 A → B/C；另一個獨立 worktree 可負責 D/E 的規格與限界實作。

為避免衝突：

- 路線圖核准後先凍結；平行任務不各自改 `docs/roadmap.md`、`TODO.md`、
  `CHANGELOG.md`，由整合方最後統一更新。
- A 僅修改 `experiments/benchmark/`、benchmark 規格與結果。
- B 主要修改 `src/asr_input/asr/` 與 ASR config；C 主要修改 `src/asr_input/audio/`。
- D 可修改 `tray.py`、啟動腳本與 lifecycle tests。
- E 第一階段只修改 log schema、viewer 與獨立 UI，不改 tray 選單；需要 tray 入口時留到
  整合階段。
- `config.yaml`、`pyproject.toml`、`tray.py`、`TODO.md`、`CHANGELOG.md` 是共享熱點，
  平行分支先不共同修改，或事先指定唯一 owner。

建議各線使用獨立 branch/worktree、小型 commit 與行為測試；先整合 A 的 schema，再整合
C/B，D/E 可獨立審查後合併。
