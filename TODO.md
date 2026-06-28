# ASR Input — 待辦清單

## MVP（最小可用版本）— 目標：日常語音輸入工具

- [x] Whisper 語音辨識（faster-whisper large-v3-turbo）
- [x] 繁體中文後處理（OpenCC + 台灣用語詞表）
- [x] 麥克風錄音 + 剪貼簿輸出
- [x] System Tray + 全域快捷鍵（Ctrl+Shift+Space toggle）
- [x] VAD 長音訊切段（Silero VAD，自適應切段 800ms→500ms→300ms 遞減）
- [x] 轉錄 log（data/logs/transcripts.jsonl，每筆含時間戳/原始/處理後文字）
- [x] 標點符號正規化 — 上下文感知轉換（CJK 旁轉全形，ASCII 旁保持半形）
- [x] 智慧 OpenCC — 偵測到簡體字才啟用轉換，純繁體跳過（避免過度轉換）
- [x] 詞表清理 — 刪除 no-op 條目、加 OpenCC 反向修正（平臺→平台）、分類整理
- [x] 即時串流辨識 — StreamingVAD 即時切句 + 逐句辨識 + 遞減靜音門檻 + 能量過濾
## 後續改善

### 已知問題（Bug·待驗證/修復）
- [ ] **tray 右鍵選單不隨狀態刷新** — 開機載入完成後，右鍵選單持續顯示「載入模型中」、「卸載模型」反灰；但圖示已轉綠、快捷鍵錄音正常。
  - **現象來源**：使用者實機觀察（2026-06-28）。
  - **為何研判「不是卡在載入」**：使用者表示「啟動計時那行有印出」。該行 `[計時] CUDA 暖機.../載 Whisper.../載 VAD...` 在 `tray.py:_setup()` 的最後、緊接 `_set_state(State.IDLE)` 之前（約 tray.py:195-201）→ 看得到它代表載入已跑完、狀態已切 IDLE。所以是「選單沒更新」，非「模型沒載到」。
  - **機制推定（高信心，尚未實機驗證）**：選單項目用動態 lambda 讀 `self._state`，但 pystray 在 Windows 會快取選單，需呼叫 `icon.update_menu()` 才重建。`_load_model`/`_unload_model` 有呼叫（tray.py:253,260,265），**但 `_set_state()` 沒有**（tray.py:335-339）；圖示是用 `.icon=` 直接設（所以顏色看得到變），選單卻凍在建立時的 LOADING 樣子。
  - **修法方向**：把 `update_menu()` 收進 `_set_state()` 中央處理（所有狀態切換都刷新），移除 `_load_model`/`_unload_model` 裡重複的呼叫。串流回呼 `_on_partial_result`/`_on_transcribing` 走 `.icon=` 不經 `_set_state`，不會造成 update_menu 洗版。
  - **驗收**：開 tray → 等圖示轉綠 → 右鍵應顯示「待機」且「卸載模型」可點。修前先重現一次、修後再看一次才算數。
  - **註**：任務 1/2/3（啟動計時/多色/卸載載入）昨晚只過了 ruff+pytest，pytest 只覆蓋 `processing/`，碰不到 tray GUI 行為，故此類選單刷新 bug 未被攔到。任務 3 嚴格說是「程式碼寫好、未實機驗收」。

### 文字品質
- [ ] 刪節號正規化 — Whisper 輸出 `...`（三個半形點）應轉換為 `…`（U+2026 全形刪節號），在 PunctuationNormalizer 中處理
  - **實作事實（已讀碼確認 2026-06-28）**：現有 `PunctuationNormalizer.process()` 是「逐字元」轉單一標點（`_HALF_TO_FULL` map），多字元併一個的刪節號塞不進那迴圈 → 需另加 `re.sub(r"\.{3,}", "…", text)` 前處理。應比照現有設計只在 CJK 旁轉，否則會誤傷英文 "wait..."。半形句號 `.` 刻意不在 `_HALF_TO_FULL`（怕誤傷小數點），所以動 `...` 不會碰到單一句號。規模：~10 行 + 2 個 pytest case，屬小任務。
- [ ] 標點符號進階研究 — 嘗試更多 initial_prompt 策略、suppress_tokens 微調、或半形句號 `.` 的智慧轉換
- [x] 聲學辨識錯誤 — hotwords 參數已接通，config.yaml 可設定（實驗證實短詞安全、長句有害）
  - 已配置：詞表、待辦、清單、聲學、標點、主分支
  - 已知未修正：磁錶→詞表、代辦→待辦、清淡→清單、組→主、升學→聲學、表點→標點（需對應音檔才能驗證效果）
- [ ] hotwords 使用者自訂機制 — 讓使用者可手動新增常用詞/專有名詞，不預設

### 使用體驗
- [x] 啟動流程拆段計時 — tray._setup() 分段印 CUDA 暖機 / 載 Whisper / 載 VAD
- [x] 狀態列多狀態色 + 串流段數疊加 — 灰/綠/紅/琥珀對比拉開，圖示疊加已完成段數
- [x] 狀態列右鍵手動卸載/載入模型 — 釋放 ~2GB VRAM，UNLOADED 狀態 + 守衛
- [ ] 直接輸出到游標位置 — 模擬鍵盤輸入取代剪貼簿（SendInput/pynput，需處理焦點、速度、中文輸入相容性等問題）
  - **狀態澄清（2026-06-28 查證）**：這項**從未實作、也沒「試過被否決」**，只是 MVP 階段因技術複雜度被「降級延後」（roadmap memory 原文：「技術複雜度高、非 MVP 必要」）。grep `src/` 無 SendInput/pynput 相關碼。→ 不是死路，要做隨時能開。（使用者印象中的「被拒絕」可能混到先前裝套件時 `Stop-Process` 關 tray 被權限分類器擋下，與此無關。）
- [ ] 信心程度標示 — 低信心詞高亮，方便人工確認
- [ ] OpenCC 差異標示 — 只標出替換的詞，不重複整段輸出
- [x] 智慧 OpenCC — 偵測到簡體字才啟用轉換，純繁體跳過（已移入 MVP 完成）
- [ ] 浮動狀態視窗 — 類似輸入法候選框，游標旁顯示錄音/辨識狀態
- [ ] 狀態列波形動畫 / 即時音量視覺化（夜間計劃原任務 5，當時依計劃略過）
  - **技術阻塞（為何沒先做）**：streaming 目前無即時音量回呼，要真波形得在 `streaming.py` 新增 amplitude callback（工程量大，且 streaming.py 含幻覺偵測邏輯，動它要小心）。
  - 系統匣圖示更新受 Windows `Shell_NotifyIcon` 節流，即時性有天花板；要真即時須做獨立浮動視窗 → 與上一條「浮動狀態視窗」合併考量較合理。
- [ ] 辨識結果預覽/編輯 — 輸出前可修改
- [ ] 通知靜音開關 — config.yaml 控制（目前已預設靜音）
- [ ] 多語言切換 — 中英日等語言快速切換

### 技術 / 工具
- [x] 單元測試 — pytest 導入，覆蓋 processing/ 三模組（punct_norm / opencc_conv / tw_terms），38 個 case
  - 已加階段收尾流程到 CLAUDE.md（ruff format/check → pytest → 記錄更新 → commit → memory）
  - 待補：幻覺 RMS 邏輯（streaming.py）需先抽成純函數才好測
- [ ] docs/index.html 翻新 — 目前凍結在 v0.1（引擎寫 Qwen、結構樹缺串流/tray、roadmap 把已完成的當未來）。等正式規劃「後續改善」時連同 roadmap 一次重寫
- [ ] Web UI 測試介面 — 瀏覽器介面，用於測試/展示/設定調整
- [ ] 多引擎擴充 — SenseVoice 等其他引擎
  - [x] whisper-turbo zh-TW 微調評測（4a）→ **不採用**。原生繁體+全形標點，但 transformers 30s 長音檔分塊（官方標 experimental）跑出災難性重複迴圈（「能量量量…」數百字）、專有名詞錯更多、漏併段。baseline+VAD 更穩，候選唯一優勢（全形標點）後處理已補足。腳本留 `scripts/test_hf_whisper_zhtw.py`
  - [x] Fun-ASR-Nano 評測（4b）→ **暫不採用（版本卡關）**。隔離環境 `.venv-funasr`（py3.11）裝 funasr 一次成功（避開 conda py3.13 的 llvmlite 地獄）；但 `pip install funasr`(1.3.14) 載入 Fun-ASR-Nano-2512 缺 ctc_decoder 權重 → 輸出退化成單字重複垃圾。研判 funasr 版本對不上。重試方向：依 FunAudioLLM 官方 GitHub requirements 裝指定版本。腳本 `scripts/test_funasr_nano.py`
  - [ ] **系統化重測（下階段執行）— 先讀「目前評測的事實邊界」再判斷，別把現有結論當定論**：
    - **目前只測過一個 8 分鐘長檔**（`data/test_audio/簡報日.m4a`），**零短句測試**。但日常用途是短語音輸入，短句穩定度才是關鍵指標 → 重測務必含一組 3–10s 短句（可用 `experiments/extract_test_segments.py` 從現有音檔切）。
    - **4a 的對照不公平**：baseline 走 VAD 切段，候選走 transformers experimental 固定 30s 分塊。候選的「能量量量…」重複迴圈**有一部分是分塊方法造成，非純模型缺陷**。→ 上面 4a 那條結論的「災難性重複迴圈」措辭對模型略過度歸因；要對模型下定論，需給候選同樣的 VAD 路徑再比。（CLAUDE.md 同條結論待一併軟化，2026-06-28 標記，尚未改。）
    - **4b 從未測到真實品質**：是「裝錯 funasr 版本」的 packaging 失敗，不是模型判決。模型真實辨識力仍未知。
    - **Qwen3-ASR 完全沒測**：但已整合，`config.yaml` 切 `asr.engine: qwen` 即可當非 Whisper 備援（限制：context 無法引導繁簡，仍靠 OpenCC 後處理，見 CLAUDE.md 設計決策）。
    - **重測規格**：受測 = faster-whisper(基準) / Qwen3-ASR / Fun-ASR-Nano(對版後)（SenseVoice 視時間加碼）；逐項記「原始輸出 + 後處理輸出 + 逐段對齊 + 耗時 + 指標(簡體殘留/全形標點率/重複偵測)」；輸出成 `experiments/results/` 結構化報告 + HTML viewer，讓人能逐段看「輸入長怎樣 → 哪個模型吐了什麼」。
- [ ] **環境修：numpy 2.5 vs numba 衝突** — `-U transformers` 把 numpy 升到 2.5，numba 要 ≤2.4 → librosa 解碼掛掉（`scripts/test_audio_file.py` 受影響；麥克風 app 走 sounddevice 不受影響）。需固定 numpy<2.5 或在 file-decode 改用 ffmpeg。`uv sync` 可還原但會動到 transformers
- [ ] 自動安裝/打包 — exe 或 installer
