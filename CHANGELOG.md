# 變更紀錄

倒序，一條一行；括號內 `(hash)` 可用 `git show <hash>` 看細節。
本檔只記「已完成」；待辦見 `TODO.md`。

## 2026-06-29
- session log 系統：每段存 raw + processed + wav + config 快照，供離線實驗 (pending)
- log 品質掃描工具 + HTML 檢視器：確定性標籤掃描模型/後處理問題 (pending)
- 四引擎評測收尾：FW 整體最佳、維持預設，模型橫向比較告一段落 (62f6744)
- 桌面捷徑啟動 tray，免打指令（含有視窗排錯版 + README）(3bef354)

## 2026-06-28
- 刪節號正規化：CJK 旁 `...` → `…` (e447208)
- CUDA 暖機改 config 可選、預設關 + tray 首載 ~27s 歸因分析 (7bdce24)
- 修 tray 選單狀態不刷新（update_menu 收進 _set_state）(2054a1e)
- CLAUDE.md 加「記錄寫法」原則（記錄寫給零背景讀者）(c9f0067)

## 2026-06-25
- 系統匣右鍵手動卸載/載入模型，釋放 ~2GB VRAM (8c36bc4)
- 狀態列配色拉開對比 + 串流時圖示疊加段數 (cb37514)
- 啟動流程拆段計時（CUDA 暖機 / 載 Whisper / 載 VAD）(96d3df1)
- 模型評測 4a（whisper-zh-TW）/ 4b（Fun-ASR-Nano）初評（結論 06-29 收斂）(b7617c1, c0e69bf)

## 2026-06-20
- 文件去重與職責釐清（單一真相來源）(459ae07)

## 2026-06-19
- 導入 pytest + 明文化階段收尾流程 (fe85b5d)

## 2026-06-18
- 幻覺 fallback：偵測轉錄 >1.5s 自動重切 → 逐步遞減 → RMS 正規化 + 音訊時長下限 (d4ff3d9)
- 整理專案結構，根目錄不再散落 .py (f1cfe16)

## 2026-06-17
- 即時串流辨識：StreamingVAD 即時切句 + 逐句辨識 (167c369)
- 遞減靜音門檻 + 能量過濾防幻覺 (5f2d079)
- 智慧 OpenCC（偵測到簡體才轉換）+ 詞表清理 (94434bd)
- hotwords 偏置常見辨識錯誤詞 (6863e9f)
- 上下文感知標點正規化（CJK 旁半形→全形）(5f4e1ac)
- JSONL 轉錄 log（data/logs/transcripts.jsonl）(0a39d3d)
- 串流測試 JSON 報告 + HTML 檢視器 (e9f3a81)

## 2026-06-16（MVP 基礎）
- System Tray app + 全域快捷鍵（最終 Ctrl+Shift+Space）(33a4bc3)
- Silero VAD 長音訊切段 (c537305)
- 基礎鏈路：faster-whisper 辨識 + OpenCC 簡轉繁 + 麥克風錄音 + 剪貼簿輸出
