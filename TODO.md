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
- [x] tray 右鍵選單不隨狀態刷新（已修，實機驗證通過）— 載入完成後選單卡在「載入模型中」、「卸載模型」反灰，但圖示已轉綠。
  - 根因：選單項目用動態 lambda 讀 `self._state`，pystray(Windows) 需 `icon.update_menu()` 才重建選單。`_set_state()` 只更新 `.icon`/`.title`（走 `.icon=` 會即時反映），漏了選單刷新；初次載入完成走 `_setup()→_set_state(State.IDLE)` 這條路徑因而卡在建立時的 LOADING。
  - 修法：把 `update_menu()` 收進 `_set_state()`，移除 `_load_model`/`_unload_model` 裡重複的呼叫。串流回呼 `_on_partial_result`/`_on_transcribing` 走 `.icon=` 不經 `_set_state`，不受影響。

### 文字品質
- [x] 刪節號正規化 — `...`（3 個以上半形點）→ `…`（U+2026）。在 `punct_norm.py` 加 `_convert_ellipsis()` re 前處理，比照現有設計只在 CJK 旁轉（英文 "wait..." 保留），跑在逐字元迴圈之前。8 個 pytest case 涵蓋 CJK 旁/ASCII/兩點不匹配/六點併一。
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
  - 尚未實作（`src/` 無 SendInput/pynput 相關碼），MVP 階段因技術複雜度延後，非必要功能。
- [ ] 信心程度標示 — 低信心詞高亮，方便人工確認
- [ ] OpenCC 差異標示 — 只標出替換的詞，不重複整段輸出
- [x] 智慧 OpenCC — 偵測到簡體字才啟用轉換，純繁體跳過（已移入 MVP 完成）
- [ ] 浮動狀態視窗 — 類似輸入法候選框，游標旁顯示錄音/辨識狀態
- [ ] 狀態列波形動畫 / 即時音量視覺化
  - streaming 目前無即時音量回呼，要做波形需在 `streaming.py` 新增 amplitude callback（streaming.py 含幻覺偵測邏輯，改動要小心）。
  - 系統匣圖示更新受 Windows `Shell_NotifyIcon` 節流，即時性有天花板；要真即時須做獨立浮動視窗 → 與上一條「浮動狀態視窗」合併考量。
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
  - [x] whisper-turbo zh-TW 微調評測（4a）→ **不採用**。原生繁體+全形標點，但出現大量單字重複、專有名詞錯更多、漏併段；對照非同條件（候選走 transformers 30s 分塊/experimental、baseline 走 VAD），重複部分與分塊方式有關。baseline+VAD 更穩，候選優勢（全形標點）後處理已補足。腳本 `scripts/test_hf_whisper_zhtw.py`
  - [x] Fun-ASR-Nano 評測（4b）→ **暫不採用（版本卡關）**。隔離環境 `.venv-funasr`（py3.11）裝 funasr 一次成功（避開 conda py3.13 的 llvmlite 地獄）；但 `pip install funasr`(1.3.14) 載入 Fun-ASR-Nano-2512 缺 ctc_decoder 權重 → 輸出退化成單字重複垃圾。研判 funasr 版本對不上。重試方向：依 FunAudioLLM 官方 GitHub requirements 裝指定版本。腳本 `scripts/test_funasr_nano.py`
  - [ ] **系統化模型評測（下階段執行）**：
    - **目前只測過一個 8 分鐘長檔**（`data/test_audio/簡報日.m4a`），**零短句測試**。但日常用途是短語音輸入，短句穩定度才是關鍵指標 → 重測務必含一組 3–10s 短句（可用 `experiments/extract_test_segments.py` 從現有音檔切）。
    - 4a 對照非同條件：baseline 走 VAD 切段、候選走 transformers 固定 30s 分塊（experimental），候選的重複現象部分來自分塊方式。要單獨判斷模型品質，須讓候選走相同的 VAD 路徑再比。
    - **4b 從未測到真實品質**：是「裝錯 funasr 版本」的 packaging 失敗，不是模型判決。模型真實辨識力仍未知。
    - **Qwen3-ASR 完全沒測**：但已整合，`config.yaml` 切 `asr.engine: qwen` 即可當非 Whisper 備援（限制：context 無法引導繁簡，仍靠 OpenCC 後處理，見 CLAUDE.md 設計決策）。
    - **重測規格**：受測 = faster-whisper(基準) / Qwen3-ASR / Fun-ASR-Nano(對版後)（SenseVoice 視時間加碼）；逐項記「原始輸出 + 後處理輸出 + 逐段對齊 + 耗時 + 指標(簡體殘留/全形標點率/重複偵測)」；輸出成 `experiments/results/` 結構化報告 + HTML viewer，讓人能逐段看「輸入長怎樣 → 哪個模型吐了什麼」。
- [ ] **環境修：numpy 2.5 vs numba 衝突** — `-U transformers` 把 numpy 升到 2.5，numba 要 ≤2.4 → librosa 解碼掛掉（`scripts/test_audio_file.py` 受影響；麥克風 app 走 sounddevice 不受影響）。需固定 numpy<2.5 或在 file-decode 改用 ffmpeg。`uv sync` 可還原但會動到 transformers
- [x] 桌面捷徑啟動（免打指令）— `scripts/start_tray.vbs`（隱藏視窗）+ `start_tray.bat`（排錯用、有視窗）+ `create_desktop_shortcut.ps1`（建桌面捷徑，需使用者自行在終端跑：MSIX 沙箱下由 Claude 代建的捷徑落不到真實桌面）。vbs 刻意用 python.exe+隱藏視窗而非 pythonw（pythonw 下 sys.stdout 為 None，tray 的 print 會崩）
- [ ] 自動安裝/打包 — exe 或 installer（獨立 .exe，不依賴 .venv/Python；輕量桌面捷徑已先行，見上）
