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

狀態：`ready`，下一步先完成狀態機規格與 lifecycle tests，再修改產品預設。

已同意方向：

- 開機啟動時只載入錄音與 CPU VAD 所需元件，預設不載入 Whisper、不占用模型顯存。
- 第一次按快捷鍵後立即錄音，同時在背景載入 Whisper；不得為等待模型而漏掉開頭音訊。
- 錄音先結束而模型尚未 ready 時，自動保留音訊並排隊，ready 後依序辨識；取消是額外操作，
  不是正常流程必答問題。
- Tray 圖示必須能直接區分未載入、載入中、可用、錄音中、辨識中與錯誤；tooltip/選單文字
  提供同一狀態的文字說明，避免只靠顏色辨識。
- 支援閒置卸載，初版預設為完成最後一次辨識後 30 分鐘；Tray 提供永不卸載、15/30/60
  分鐘及立即卸載。
- 支援開機啟動與單一實例。直接輸出到游標位置維持暫緩，現階段輸出仍以剪貼簿為準。
- 載入失敗、CUDA 不可用與麥克風錯誤都必須進入可辨識且可重試的狀態，不得卡在載入中，
  也不得靜默丟棄已錄音訊。

規格與實作需明確處理並測試：重複快捷鍵、載入中取消、卸載計時與新錄音競態、排隊上限、
程式結束時尚未處理的音訊，以及模型載入失敗後的重試。預熱後卸載僅列為實驗候選；只有實測
證明能穩定縮短下次載入且沒有不合理的記憶體或啟動成本，才考慮加入設定，不作為預設。

### E — Log／回饋／歷史介面

狀態：`exploration`。

建議分期：

1. Log retention、修正版 schema、錯誤標籤與可供後續工具讀取的穩定資料介面。
2. 從多次人工修正提出替換字／完整詞組候選；使用者確認後才寫入替換規則。短詞全域替換
   必須先排除同音異義風險，有歧義時只允許完整詞組，且需做整句回歸。
3. 獨立歷史 viewer：搜尋、播放、raw/processed/final、從任意紀錄複製與修正。現階段不另做
   「複製上一筆」，因最新結果已在剪貼簿。
4. 最後才做 Profile、應用程式感知或選取文字語音修改。

延後點子：信心詞高亮、OpenCC 差異標示、浮動狀態/波形、多語言切換、輸出前預覽
編輯與大型 Web UI。hotwords 是獨立功能，不從修正紀錄自動產生或修改；未來若提供使用者
自訂，仍需明確確認與整句回歸。

已同意開發版音訊 log 暫以 2 GiB 為上限並保留 metadata；metadata 體積小且是離線實驗
不可缺少的上下文。優先採無損 FLAC；Opus 等有損格式只有在嚴格 Whisper 逐字交叉測試全部
通過後才列入候選。公開發布版預設不保存音訊，或只在使用者明確啟用診斷模式後限量保存。

待決策：超過 2 GiB 時按 session、時間或重要性標籤刪除，以及 WAV 何時轉 FLAC、轉換失敗
如何保留原檔。競品研究在具體 UI 決策前按題目進行，不另立泛用研究專案。

### F — 發布工程

狀態：`deferred`，確定公開或給其他人使用時啟動。

範圍：LICENSE、版本統一、CI、乾淨 Windows 安裝 smoke test、Windows/NVIDIA/CUDA 相容性
檢查、首次模型下載與失敗重試、portable onedir、升級時保留使用者設定、發布/開發設定分離，
以及 `docs/index.html` 翻新。

公開版預設只安裝與支援 faster-whisper；Qwen 等已完成比較但未採用的研究引擎不屬於正式
執行依賴，可將 adapter、腳本與結果保留在研究區，並以額外 dependency group 隔離。模型快取、
設定與 log 必須使用穩定的使用者資料目錄，不依賴目前工作目錄；發布版預設關閉音訊 log，
診斷模式與開發版才保留完整資料。是否移除其他依賴，以實際 import/打包稽核結果決定。

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
