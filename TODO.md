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
- [x] 多引擎橫向評測 — **2026-06-29 完成，結論：FW 整體最佳、維持預設，模型橫向比較告一段落**。四引擎都跑通、原始輸出存 `experiments/results/*2026-06-29*`，詳細結論見 CLAUDE.md「四引擎評測總結」與「繁體引導能力排名」。
  - [x] whisper-zh-TW 微調（4a）→ **不採用**：短句品質≈baseline，但易掉標點、且 27.7s 單段也會崩潰成「量量量」重複。淘汰理由＝無增益+崩潰風險。腳本 `scripts/test_hf_whisper_zhtw.py`、`experiments/run_4a_segments.py`
  - [x] Qwen3-ASR → **不採用（保留為備援）**：短句品質好、原生全形標點，但原生簡體需 OpenCC。已整合，`config.yaml` 切 `asr.engine: qwen` 可用。context 繁體引導有界（措辭/內容相依），見 CLAUDE.md。
  - [x] Fun-ASR-Nano（4b）→ **翻案、可用但不採用**：先前「版本卡關」是誤判，真因 HF 快照缺 model.py，用 `remote_code=` 指向官方 repo model.py 即跑通、輸出乾淨。原生簡體需 OpenCC，相對 baseline 無增益。腳本 `scripts/test_funasr_nano_v2.py`
  - [ ] （可選·低優先）Fun-ASR 繁體鷹架 02/04 補測 — 已證實把 model.py `get_prompt` 簡體鷹架換繁體能降抗拒段簡體率（seg01 0.296→0.074），但樣本僅 1 段顯著。要寫成設計決策需 `02_技術描述_開發平台.wav`/`04_長段連續描述.wav` 補測再現。腳本 `experiments/probe_funasr_scaffold.py`
  - [ ] （可選）SenseVoice 等其他引擎 — 模型橫向比較已告段落，除非有明確新需求否則暫緩
- [x] **環境：numpy vs numba 衝突已解** — 現況 numpy 2.4.6（符合 numba ≤2.4 需求），`scripts/test_audio_file.py` 路徑可跑；評測腳本一律改用 ffmpeg 解碼（不依賴 librosa），見 `experiments/*2026_06_29*`。

### FW 深化 / 系統底層研究（模型不換，改往「把 FW 做得更好」）
> 背景：四引擎比較後確認 FW 是最佳基底，後續改善聚焦在 FW 與系統管線，而非換模型。以下為「之後想做時可取用」的候選方向，未排程。
- [ ] FW 解碼參數調校 — `beam_size`、`temperature`/fallback、`compression_ratio_threshold`、`no_repeat_ngram_size` 等對「重複/幻覺/漏字」的影響（目前幻覺靠 streaming.py 後段 fallback，未從解碼參數源頭調）
- [ ] FW 微調可行性評估 — 用台灣口音/領域語料對 large-v3-turbo 做 LoRA/全量微調 → 但需轉 CT2 才能進現有 CTranslate2 路徑，且 4a 經驗顯示微調易引入重複退化，先評估成本/風險再決定
- [ ] hotwords/initial_prompt 與後處理的分工再研究 — 哪些錯誤該在模型層（hotwords）修、哪些留給詞表，避免兩邊重工（接「聲學辨識錯誤」「標點符號進階研究」兩條）
- [ ] 系統底層：VAD 切段品質 — seg05 類「講者猶豫/贅字多」段落各引擎都不穩，研究切段點與重疊對長段連續描述的影響

### 本階段產出整理評估（2026-06-29，先記錄不動手）
> 評估結論：**本階段無原始碼變更、無須還原**。新增的全是 `experiments/` 一次性腳本 + `experiments/results/` JSON，符合專案結構規範（根目錄不放 .py），屬實驗存證，建議保留。
- [ ] （可選·輕量）整理 experiments/ 重複腳本 — 本輪新增 8 支腳本，其中 `eval_models_2026_06_29.py`+`eval_qwen_short.py` 為同一評測拆兩段（長檔卡 GPU 才分）、`steer_qwen_traditional.py`+`steer_qwen_variants.py` 同方向。可保留（各有獨立結果 JSON 對應）或合併歸檔，非必要。`experiments/results/eval_2026-06-29.log` 是 stdout log，內容已被同名 JSON 涵蓋，可刪。
- [ ] （提醒）commit 前確認沒夾帶模型權重/音檔；`.venv-funasr` 與 scratchpad 的 Fun-ASR repo clone 在 repo 外，不受影響
- [x] 桌面捷徑啟動（免打指令）— `scripts/start_tray.vbs`（隱藏視窗）+ `start_tray.bat`（排錯用、有視窗）+ `create_desktop_shortcut.ps1`（建桌面捷徑，需使用者自行在終端跑：MSIX 沙箱下由 Claude 代建的捷徑落不到真實桌面）。vbs 刻意用 python.exe+隱藏視窗而非 pythonw（pythonw 下 sys.stdout 為 None，tray 的 print 會崩）
- [ ] 自動安裝/打包 — exe 或 installer（獨立 .exe，不依賴 .venv/Python；輕量桌面捷徑已先行，見上）
