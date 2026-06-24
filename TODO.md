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

### 文字品質
- [ ] 刪節號正規化 — Whisper 輸出 `...`（三個半形點）應轉換為 `…`（U+2026 全形刪節號），在 PunctuationNormalizer 中處理
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
- [ ] 信心程度標示 — 低信心詞高亮，方便人工確認
- [ ] OpenCC 差異標示 — 只標出替換的詞，不重複整段輸出
- [x] 智慧 OpenCC — 偵測到簡體字才啟用轉換，純繁體跳過（已移入 MVP 完成）
- [ ] 浮動狀態視窗 — 類似輸入法候選框，游標旁顯示錄音/辨識狀態
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
- [ ] **環境修：numpy 2.5 vs numba 衝突** — `-U transformers` 把 numpy 升到 2.5，numba 要 ≤2.4 → librosa 解碼掛掉（`scripts/test_audio_file.py` 受影響；麥克風 app 走 sounddevice 不受影響）。需固定 numpy<2.5 或在 file-decode 改用 ffmpeg。`uv sync` 可還原但會動到 transformers
- [ ] 自動安裝/打包 — exe 或 installer
