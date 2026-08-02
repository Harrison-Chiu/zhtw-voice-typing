# ASR Input — 待辦清單

> 本檔只記「還沒做 / 未來想做」。已完成見 `CHANGELOG.md`；
> 現況見 `CLAUDE.md`「目前狀態」；設計理由見「已確立的設計決策」。
>
> 標記：(無)=預計要做　〔探索〕=方向待評估、做不做未定　〔點子〕=未驗證想法、可能不做
> 項目後的「背景：」標出已有成果，避免從零重做。

## 下一步（短期）
- 音訊 log 編碼嚴格交叉測試 — 120 段分層樣本，WAV/FLAC/Opus 16–64k/MP3
  32–64k；每段 raw Faster Whisper 文字（含空格與標點）必須逐字一致。
  背景：初測發現疑難段受 temperature fallback 抽樣影響，WAV 自身會漂；
  強制 temperature=0 雖可重現，卻產生嚴重重複，需先解決基準決定性。
- 〔探索〕VAD 對照實驗 — Silero JIT vs ONNX；Silero vs 自適應相對能量、
  robust 統計、頻譜特徵與 WebRTC VAD 參考組。先用現有 log，之後補錄壓力片段。
- 開機啟動／閒置卸載設計 — 預設啟動但不載模型；快捷鍵先錄音與 VAD，
  模型背景載入，ready 後消化佇列；加入 30 分鐘 app-idle 卸載與 tray 設定。

## 藍圖

### 文字品質與辨識
- 〔探索〕標點符號進階研究 — 更多 initial_prompt 策略、suppress_tokens、半形句號 `.` 智慧轉換。背景：v1–v3 已試 15+ 種 prompt/hotwords/suppress，結論「短 prompt + 後處理」最穩（CLAUDE.md 設計決策、`experiments/results/experiment_punct_v2,v3.json`）→ 既有結論上再探，非從零。
- 〔探索〕hotwords / initial_prompt 與後處理的分工 — 哪些錯誤在模型層修、哪些留後處理，避免重工。背景：hotwords 已接通（短詞安全/長句有害，CLAUDE.md）。
- hotwords 使用者自訂機制 — 讓使用者手動加常用詞/專有名詞。背景：hotwords 機制本身已完成，缺的是使用者自訂介面。
- 已知未修正詞驗證 — 磁錶→詞表、代辦→待辦、清淡→清單、組→主、升學→聲學、表點→標點，需對應音檔才能驗證 hotwords 效果。

### 引擎與底層研究（模型不換，往「把 FW 做更好」）
- 〔探索〕FW 解碼參數調校 — beam_size / temperature fallback /
  no_repeat_ngram_size 對重複、幻覺、漏字與可重現性的影響。背景：同一疑難 WAV
  以 production fallback 重跑 10 次出現 7 種輸出；beam=1 仍會漂；
  temperature=0 可 10/10 一致但退化成長串刪節號，不能直接採用。
- 〔探索〕FW 微調可行性評估 — 台灣口音/領域語料 LoRA。背景：需轉 CT2 才進現有路徑，且 4a 微調出現重複退化（CHANGELOG 06-25、CLAUDE.md）→ 先評成本/風險。
- 〔探索〕VAD 切段品質 — 講者猶豫/贅字多的難段各引擎都不穩。背景：自適應遞減切段已實作（CHANGELOG 06-17/18），本條是難段深化。
- 〔探索·低優先〕Fun-ASR 繁體鷹架 02/04 補測 — 已證實改 get_prompt 簡體鷹架降抗拒段簡體率（seg01 0.296→0.074），樣本僅 1 段，需 02/04 再現才寫成設計決策。腳本 `experiments/probe_funasr_scaffold.py`。
- 〔點子〕SenseVoice 等其他引擎 — 模型橫向比較已告段落，無明確新需求則暫緩。

### 互動體驗
- 直接輸出到游標位置（暫緩，非當前重點）— 模擬鍵盤取代剪貼簿（SendInput/pynput，需處理焦點、輸入速度、中文輸入法相容性）。原 MVP 最後一塊。
- 〔點子〕信心程度標示 — 低信心詞高亮，方便人工確認。
- 〔點子〕OpenCC 差異標示 — 只標替換的詞，不重複整段。
- 〔點子〕浮動狀態視窗 + 即時音量/波形 — 游標旁顯示狀態；要真即時須獨立視窗（系統匣圖示受 Shell_NotifyIcon 節流）。
- 〔點子〕辨識結果預覽/編輯 — 輸出前可修改。
- 〔點子〕多語言切換 — 中英日等快速切換。
- 通知靜音開關 — config.yaml 控制（目前已預設靜音，僅缺顯式開關）。

### 工程 / 打包 / 文件
- 自動安裝/打包 — 獨立 .exe，不依賴 .venv/Python（輕量桌面捷徑已先行）。
- 發布版／開發版設定分離 — 發布版預設不存音訊或只留極小額度；開發版使用
  lossless FLAC、2 GiB 上限；確認有損格式嚴格通過後才考慮取代。
- 自動測試補強 — streaming 佇列、模型 ready gate、閒置卸載狀態機、log
  容量輪替與 tray 設定。
- 補測 streaming.py 幻覺 RMS 邏輯 — 需先抽成純函數才好測（目前 pytest 只覆蓋 processing/）。
- docs/index.html 翻新 — 凍結在 v0.1（引擎寫 Qwen、結構樹缺串流/tray、roadmap 過時）。
- 〔點子〕Web UI 測試介面 — 瀏覽器介面，用於測試/展示/設定調整。
