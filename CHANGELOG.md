# 變更紀錄

倒序，一條一行；括號內 `(hash)` 可用 `git show <hash>` 看細節。
本檔只記「已完成」；待辦見 `TODO.md`。

## 2026-09-03
- FLAC 無損保存驗證（只驗證，未切換保存格式）：`experiments/verify_flac_roundtrip.py`
  對 `data/logs/audio/` 全量 542 檔做 PCM16 → FLAC → PCM16 往返，逐 sample 比對
  **539/539 bit-exact**（另 3 檔是 44 bytes、0 frames 的空 wav，libsndfile 無法為零長度
  串流開啟 FLAC 寫入——單變因確認 0 frames 失敗、1 frames 正常，屬邊界條件而非無損性失敗）。
  容量比 **0.522**（602.4 → 314.7 MiB），跨時長區間 0.45–0.53 幾乎持平；編碼 539 檔共 11.1s。
  另新增 `src/asr_input/storage/flac_archive.py`：先寫 `.tmp`、讀回逐 sample 驗證、通過才
  `os.replace()` 就位、最後才選擇性刪來源，任何前段失敗都不動原檔；**production 尚未呼叫它**。
  新增 9 項測試（共 245 passed），含四個失敗階段的參數化測試 `(pending)`
- E 線錯誤候選篩選：新增 `src/asr_input/eval/signals.py`（18 條確定性訊號）、
  `scripts/scan_error_candidates.py`（掃 `history.sqlite3` 出候選 + 同時長分布對照組）與
  `scripts/build_review_queue.py`（可播音訊、三態標記的審核頁）。門檻取自當時 1425 段的實測
  分位數而非猜測；首掃 87 段候選（6.1%）+ 22 段對照。審核頁刻意隱藏「候選／對照」分組直到
  該筆被標記，否則審核者對已知被標記的段會判得更嚴，對照組就量不到訊號的漏抓率。
  幻覺訊號維持**絕對轉錄時間**，並補一條測試把「改成 transcribe/audio 比值」的回歸釘死
  （25.2s 音訊 3.5s 轉錄的已知幻覺，比值 0.14 反而比健康短段更好看）。
  新增 24 項測試（共 234 passed）`(4d70af0)`
- macOS 移植 Phase 0：建立 `src/asr_input/platform/` 平台抽象層，處理
  `docs/macos-port-plan.md` 的 A1–A3。`pyproject.toml` 的 torch／torchaudio CUDA index 加
  `sys_platform != 'darwin'` marker（Windows 解析結果不變，uv.lock 只多出 darwin 分支）；
  剪貼簿抽成後端協定，Windows 保留原本的 PowerShell `Set-Clipboard` 行為、macOS 走 `pbcopy`
  並以 stdin 餵入（文字不再進命令列，順帶消掉引號跳脫的脆弱點）；device／compute_type 改由
  `platform/device.py` 統一解析，config 明確指定優先、未指定才偵測 CUDA，且探測維持惰性
  （`import ctranslate2` 會連帶載入 torch）。另新增 `paths.py`，把 history／session log 的
  相對路徑改以套件位置解析——這是為 macOS LaunchAgent 不保證 cwd 做的**預防性**修正，
  Windows 兩條啟動路徑（`.vbs` 設 CurrentDirectory、`.bat` 有 `cd /d`）本來就正確，
  不是修既有故障。新增 22 項測試（共 210 passed）；不碰 `tray.py` `(1d02dfe)`
- 撤回文件中「幻覺門檻改用 RTF 相對比」的提案並收斂到單一決策點：該提案與 `docs/roadmap.md`
  E 線的明文禁令衝突，卻仍被反覆提起。追查後確認不是記憶問題而是文件問題——決策只寫在
  不會自動載入的 roadmap，而 `docs/macos-port-plan.md` 把 Phase 2 直接命名為「門檻 RTF
  相對化」、roadmap G 線的跨線依賴又照抄該階段名，前一次修正只改了論述段、漏掉階段名。
  三處已改為「門檻的裝置校準」並附撤回理由與驗算，決策本身寫進 `CLAUDE.md` `(1ab671f, 3777261)`

## 2026-09-01
- README 補「為什麼有這個專案」一節與四條技術功能：動機以一段改寫後的對話引言帶出
  （本地執行、繁體輸出、現成方案不夠好），後接四項實際耗時的問題與數據（短音訊幻覺、
  標點半形全形、冷啟動 18 秒、EcoQoS 慢 2.4 倍），並說明只轉字形不改詞彙的預設。
  功能段維持原有五條不動，新增分段串流辨識、非阻塞生命週期、幻覺防護、狀態圖示四條，
  沿用「名詞 — 技術說明」格式。串流一項明確寫成 segment-level pipelining
  （非逐字即時、段落內不輸出增量假設），避免被誤讀為 online ASR。
  開頭加一句英文摘要供搜尋 `(fc2944f)`
- 桌面捷徑腳本改成「預設一個、其餘選配」：`create_desktop_shortcut.ps1` 預設只建無視窗的
  「ASR Input」，有視窗的排錯版改由 `-WithDebugShortcut` 選配（不建捷徑也能直接雙擊
  `scripts\start_tray.bat` 排錯）。新增 `-Autostart` 在「啟動」資料夾放一份開機自動常駐的
  捷徑、`-Remove` 移除本腳本建立過的捷徑，並在缺 `.venv` 時提前警告而非等雙擊才跳錯誤框。
  `start_tray.vbs` 配合支援 `/autostart` 具名參數（未帶時仍為 `manual` 預載，行為不變），
  開機啟動因此走 lazy 模式、不在開機時占顯存；順帶正規化該檔混用的換行 `(e83d635)`
- 新增 `docs/macos-port-plan.md`：macOS 支援的評估正本——A 硬阻斷（torch 的 CUDA index、
  `device: cuda`、PowerShell 剪貼簿、pystray 私有 win32 API、Windows mutex）、B 行為差異
  （絕對秒數的幻覺門檻在不同硬體上會誤判並靜默丟棄短段、pystray 與 pynput 在 macOS 主
  執行緒可能衝突、TCC 權限、App Nap）、C 工具鏈，以及 Phase 0–4 順序與四項待實測未知數。
  每項判斷標注「讀碼確認」或「依套件行為推測（未在 macOS 實測）」。
  決策：單一 repo + 平台抽象層，不拆成兩個 repo `(ae6cb9f)`
- `docs/roadmap.md` 新增 G 線並標注跨線依賴（G-Phase 2 與 B 線同動 `streaming.py` 品質
  判斷路徑、G-Phase 3 與 D 線同動 `tray.py`、G-Phase 4 與 F 線範圍重疊）；
  `TODO.md` 加入 G-Phase 0 執行入口與「尚未建立 remote」的待辦 `(ae6cb9f)`

- 測試音檔定名為 `data/test_audio/隊伍簡報日_8分鐘.m4a`：保留原本的辨識度（機器人競賽的
  隊伍簡報日）與長度標示，且不含公司名。音檔本身不進版控（`data/test_audio/` 已 gitignore），
  但檔名字串會出現在被追蹤的文件與腳本中，故檔名仍需去識別化。去識別化已在全歷史完成，
  此次只動工作區與本機檔案 `(843bf0e)`
- 目錄職責歸位：`scripts/` 只留可重複使用的工具（測試驅動、log 工具、Windows 啟動器），
  三支一次性模型評測 `test_funasr_nano.py`／`test_funasr_nano_v2.py`／`test_hf_whisper_zhtw.py`
  移入 `experiments/`。原本同一個實驗的腳本被切在兩個目錄（CLAUDE.md 的決策條目就同時引用
  `scripts/` 與 `experiments/` 下的檔案），引用已一併更新 `(0939a0d)`
- 刪除 `scripts/test_full.py`：傳 `prompt=` 給 `QwenASREngine`，但該類別的參數是 `context`，
  執行即 `TypeError`；另寫死 Qwen 1.7B 與已改掉的 `s2twp`，功能由 `scripts/test_audio_file.py`
  取代（它驅動真正的 `main()`），且無任何文件引用 `(0939a0d)`
- 修死設定 `processing.tw_dict_path`：先前從未被讀，路徑寫死在 `tw_terms.py`。
  改由 `build_pipeline()` 傳入（相對專案根目錄，未設定則用內建預設），
  並補兩個測試釘住這條線，避免又退化成沒人驗證的設定 `(0939a0d)`
- 新增 `.gitattributes`：repo 內統一存 LF，`.bat`／`.vbs`／`.ps1` 強制 checkout 成 CRLF。
  先前換行只靠各機器的 `core.autocrlf`，在只有 Windows 時可行；為 macOS 支援預先寫死規則 `(0939a0d)`
- 修正 `scripts/test_streaming.py` docstring 的輸出路徑（`data/` → `experiments/results/`）`(0939a0d)`

- 準備公開發布：改寫 git 歷史移除 `experiments/results/`（及其早期路徑 `data/experiment_*.json`、
  `data/streaming_*`）——該目錄存的是作者本人語音的逐字稿語料，單一實驗檔含數千段轉錄文字。
  目錄改列 `.gitignore`、本機檔案保留，緣由寫在新增的 `experiments/README.md`；
  CHANGELOG 與 `experiments/autonomous_plan.md` 內 39 個 commit hash 依 filter-repo 的
  commit-map 自動更新為改寫後的值 `(909199d)`
- 準備公開發布：測試音檔檔名去識別化，移除檔名中的公司名（工作區與全歷史一併替換，
  本機檔案同步改名）；最終定名見下方條目 `(909199d)`
- 新增 `LICENSE`（MIT）與 README 授權章節；依賴授權讀自安裝後套件 metadata，
  其中 pystray／pynput 為 LGPLv3，以原始碼形式散布不影響本專案授權，
  打包成單一執行檔前需另行確認 `(909199d)`
- 修正短段幻覺直接進剪貼簿：`min_hallucination_audio_sec`（3.0s）原本讓 3 秒以下的段跳過整個
  幻覺偵測，實際 log 出現 0.26s 音訊輸出 60 字亂碼、0.29s 子段輸出「作詞・作曲・編曲 男高等部分」
  等文字被原樣貼給使用者。改為此門檻只切分「還救得回來／救不回來」：短段仍跑 RMS 正規化，
  救不回來就捨棄文字並在 log 保留原音訊、各次嘗試與 `hallucination-rejected` 標籤 `(780a35d)`
- 修正 fallback 切分用盡後仍採用已知偏慢子段：`fallback_succeeded=False` 時，仍超過 1.5s 且
  短於 `min_hallucination_audio_sec` 的子段不再併入輸出（`adopted=False` + `rejected=True`）；
  這正是 job 364bdc50 剪貼簿出現「作詞・作曲・編曲，男高等部分方向政府，強行固定中文名字。」
  的來源 `(780a35d)`
- 修正 fallback 段在 `segments` 表 `processed_text` 為 NULL：stat 補上 `raw`/`processed`，
  log 檢視器不再顯示「30 秒音訊、文字空白」的假象 `(780a35d)`
- 回歸集：`tests/data/short_segment_cases.json` 凍結 143 筆真實 <3s 辨識嘗試（人工標註），
  產生器 `experiments/extract_short_segment_cases.py`，測試 `tests/test_short_segment_hallucination.py`
  （修正前 8 個失敗、修正後全過）`(780a35d)`
- 重現：`experiments/replay_short_segment_hallucination.py` 用存檔音訊重跑 job 364bdc50 seg3，
  0.29s→3.36s、0.52s→3.54s 皆輸出無關文字；文字跨 session 不同、延遲訊號穩定重現
  `(experiments/results/short_segment_hallucination_2026-09-01.md)` `(780a35d)`
- 移除 `startup.cuda_warmup`：暖機呼叫本身早已被拿掉，只剩 `tray.py` 把設定讀進 `self._cuda_warmup`
  卻無人使用，是死設定；2026-09-01 VAD 改走 ONNX、錄音路徑不再 import torch 後，「留給未來 PyTorch
  引擎或 GPU VAD」的理由也更弱。config.yaml 的 `startup:` 區塊與說明一併刪除 `(0a80935)`
- 啟動延遲：VAD 改用 ONNX Runtime 後端（`vad.backend`，預設 `onnx`），錄音路徑完全不 import torch；
  tray 從啟動到「可錄音」由數秒～數十秒降到約 0.4s（開發機、autostart 模式實測）`(pending)`
- 等價性驗證：`experiments/compare_vad_backends.py` 在 498 秒實際測試音檔（含 8 分鐘長檔）比對
  torch JIT 與 ONNX，逐窗機率最大差 4.7e-06、門檻決策零翻轉、`StreamingVAD` 切段邊界完全相同
  （長檔 32/32 段），推論本身也快約 2.2×；ONNX 模型載入 0.13s vs torch hub 2.28s `(pending)`
- 卸載路徑：`torch.cuda.empty_cache()` 改為只在 torch 已載入時才呼叫，避免 onnx 後端為了釋放
  「從未配置過的 PyTorch 記憶體」而付出完整 torch import 成本 `(pending)`
- 啟動可用性：快捷鍵監聽移到 `import torch` 與 VAD 載入之前註冊，啟動期間按下不再靜默無反應，
  改回報「啟動中（已 N 秒）」或既有啟動錯誤（3 秒節流）；錄音路徑仍需 VAD，故此階段只回報進度、
  不錄音 `(pending)`
- 啟動計時：新增帶時間戳的啟動橫幅、`import torch` 單獨計時、「可錄音就緒」總耗時與每次 Whisper
  載入／載入失敗的精確耗時，補上 roadmap D「記錄每次模型載入精確耗時」`(pending)`
- 啟動診斷：實測 `import torch` 為冷啟動主要成本且高度依賴 OS 檔案快取（2026-09-01 開發機：
  長時間閒置後首次 18.1s、快取熱時 1.7s）；`main.py` 匯入訊息由「正在載入 PyTorch」改為
  「正在載入語音模組」，因 tray 是先 `import torch` 才匯入 `main`，原訊息出現時 torch 早已載完 `(pending)`

## 2026-08-18
- 雲端實驗：加入預設停用的 OpenRouter STT adapter，key 由環境變數或 Windows Credential Manager 安全讀取，預設要求 ZDR，並記錄 usage／cost `(pending)`
- 模型評測：補跑 Qwen3-ASR 0.6B／1.7B 與 faster-whisper 五段同場比較；FW 維持預設，1.7B 保留研究選項，0.6B 不採用 `(pending)`
- 後處理：簡轉繁預設改為忠實字形 `s2t`；台灣詞彙本地化拆成獨立 `localize_tw_terms` 選項並預設關閉 `(pending)`
- 錄音診斷：PortAudio `input overflow` 通知改為說明可能短暫遺失音訊的中文訊息 `(pending)`

## 2026-08-17
- VibeVoice-ASR-BitNet standalone smoke 收尾：依序測試中英、機器人展示、隊伍簡報日，
  再測隊伍簡報日切出的五個 segments；只有機器人展示相對可用，其餘出現系統性錯詞／漏字／重複，
  長檔另有 RTF > 1 與 repetition degeneration，確認不採用、不整合主流程
  (`experiments/results/vibevoice_bitnet_smoke_2026-08-17.md`)

## 2026-08-02
- 計畫治理：建立 A–F 主路線圖作為 SSOT，TODO 降為執行入口；補齊 VAD E0–E6 正式規格、non-inferiority gate 與 A/B/C、D/E 平行開發邊界 (71f2bf1)
- Whisper 後端初測：加入可重跑的 sequential／batched／HF SDPA benchmark；RTX 4060 上 batched 對 10–13 秒短句僅快約 5–7%，長檔吞吐雖快約 2.4 倍但發生漏段與重複，不改 production 預設 (e5fc48e)
- 評測規劃：新增標準 benchmark 提案，定義分層資料集、雙 gold transcript、CER/MER、效能規範、採用 gate 與視覺報告 (e5fc48e)
- 底層研究：整理 Faster Whisper／CTranslate2 可修改層級、候選技術 backlog 與先上層調度後底層 runtime 的實施順序 (02f245a)

## 2026-07-31
- 後處理：依既有語音 log 加入「外餐→外參」單詞修正；「表點」仍採完整詞組規則，避免誤傷報表點選等正常用法；15 筆既有錯誤語境全數涵蓋 (3c86259)
- 實驗工具：新增音訊 codec 嚴格逐字回歸與錄音／模型並行載入探針；首輪確認 Faster Whisper temperature fallback 會讓疑難 WAV 本身不具完全決定性，完整 codec 結論留待基準問題處理後再跑 (c0dca3f)

## 2026-07-07
- fix(tray)：啟動時退出 Win11 EcoQoS（`SetProcessInformation` ProcessPowerThrottling）(pending)。start_tray.vbs 隱藏視窗啟動被 Win11 判為背景並節流，實測 faster-whisper 解碼中位數 ~1.1s（抖 0.7–1.4s）→ opt-out 後 ~0.46s（快 2.4 倍、變異幾乎歸零）。`timeBeginPeriod(1)` 實測無效已排除；單執行緒 CPU 探針測不到（節流打的是驅動 GPU 的多執行緒）

## 2026-07-01
- 後處理改善：CJK 間空格→逗號（8% 段受惠）、ASR 同音詞修正（辦形→半形、轉入→轉錄、以有→已有）(e019a4b)
- 實驗：prompt/decode 參數對逗號無效（5 組參數×7 段，逗號密度完全不變）；FW 內建品質門檻從未觸發（門檻太鬆）；8 分鐘測試音檔零幻覺（VAD 改善已從源頭消除）
- 實驗：logit manipulation 逗號研究（fb59801）— logit probe 診斷根因（逗號 rank=2 佔 9.1%、gap 極小）→ 固定規則 boost 天花板 51% 改善 @ 22% 副作用 → Bayesian decision + bigram prior 改善效率比但不改天花板 → 結論：模型層 logit 操作有根本上限，需外部標點模型
- 實驗：MCTS-lite 逗號搜索（pending）— Phase 0 診斷：45 個 near-miss 位置逗號路徑 log-prob 全部劣於 winner（0/45），模型 probability landscape 裡逗號就是 inferior path。Phase 1：外部標點獎勵可強制改善（9/9 問題段），但本質同 logit boost、副作用同等級（正常段 30-40% 內容變化）。結論：純搜索策略（MCTS/MCMC/best-of-N）對此問題理論上無效
- 問題段判定修正：原 23.5%（35/149, cjk>20）灌水嚴重 → 嚴格篩選（cjk>40 + 無任何標點）後為 9/149（6%），14 段邊界案例多數有句號/問號已足夠分句

## 2026-06-29
- viewer 改版：三態標籤篩選（包含/排除/不篩選）、預設全顯示、搜尋目標切換、fetch+blob 音訊播放、`--serve` 一鍵啟動 (pending)
- session log 系統：每段存 raw + processed + wav + config 快照，供離線實驗 (pending)
- log 品質掃描工具 + HTML 檢視器：確定性標籤掃描模型/後處理問題 (pending)
- 四引擎評測收尾：FW 整體最佳、維持預設，模型橫向比較告一段落 (6deef38)
- 桌面捷徑啟動 tray，免打指令（含有視窗排錯版 + README）(b3aad88)

## 2026-06-28
- 刪節號正規化：CJK 旁 `...` → `…` (fce4e50)
- CUDA 暖機改 config 可選、預設關 + tray 首載 ~27s 歸因分析 (917aed7)
- 修 tray 選單狀態不刷新（update_menu 收進 _set_state）(cd6e6bc)
- CLAUDE.md 加「記錄寫法」原則（記錄寫給零背景讀者）(b9c4e87)

## 2026-06-25
- 系統匣右鍵手動卸載/載入模型，釋放 ~2GB VRAM (3b34930)
- 狀態列配色拉開對比 + 串流時圖示疊加段數 (68fdf75)
- 啟動流程拆段計時（CUDA 暖機 / 載 Whisper / 載 VAD）(70a5e9a)
- 模型評測 4a（whisper-zh-TW）/ 4b（Fun-ASR-Nano）初評（結論 06-29 收斂）(f447b3d, 5a3f741)

## 2026-06-20
- 文件去重與職責釐清（單一真相來源）(13bdebb)

## 2026-06-19
- 導入 pytest + 明文化階段收尾流程 (b3d662a)

## 2026-06-18
- 幻覺 fallback：偵測轉錄 >1.5s 自動重切 → 逐步遞減 → RMS 正規化 + 音訊時長下限 (9cb053a)
- 整理專案結構，根目錄不再散落 .py (783d5b4)

## 2026-06-17
- 即時串流辨識：StreamingVAD 即時切句 + 逐句辨識 (901e2cb)
- 遞減靜音門檻 + 能量過濾防幻覺 (6d1e344)
- 智慧 OpenCC（偵測到簡體才轉換）+ 詞表清理 (79a0e25)
- hotwords 偏置常見辨識錯誤詞 (4c75a05)
- 上下文感知標點正規化（CJK 旁半形→全形）(5f4e1ac)
- JSONL 轉錄 log（data/logs/transcripts.jsonl）(0a39d3d)
- 串流測試 JSON 報告 + HTML 檢視器 (aaa3ac0)

## 2026-06-16（MVP 基礎）
- System Tray app + 全域快捷鍵（最終 Ctrl+Shift+Space）(33a4bc3)
- Silero VAD 長音訊切段 (c537305)
- 基礎鏈路：faster-whisper 辨識 + OpenCC 簡轉繁 + 麥克風錄音 + 剪貼簿輸出
