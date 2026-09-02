# 夜間自動執行計畫 — 2026-09-03

本文件是這一輪夜間無人執行的**工作正本與進度紀錄**。

寫它的理由：夜間工作量可能超過單一 session 的上下文，compact 或 session 中斷會讓對話裡的
決定與進度消失。任何接手的人或 agent 只要讀這一份，就能知道「做到哪、為什麼這樣做、
下一步是什麼」，不需要回頭翻對話。

**接手方式**：先讀「執行守則」與「進度紀錄」，找到第一個未完成任務，從那裡繼續。
每完成一個任務就更新「進度紀錄」，不要等到最後一次補寫。

- 建立時間：2026-09-03 晚間
- 起始 commit：`ca44b84`（已 push 到 `origin/main`）
- 起始基準：`pytest` 188 passed、`ruff check src/` clean
- 預定結束：2026-09-04 06:30，或清單全部完成，或發生重大錯誤

---

## 1. 目標

把 `docs/roadmap.md` 中**現階段不需要使用者在場、不需要新硬體**的工作盡量往前推。

使用者已明確授權：可自行解決過程中的問題，不必等回覆；只要不是不可逆的大錯誤都可代為
執行。用量充足，不需要為節省 token 而縮減工作。

## 2. 執行守則（不可違反）

1. **不可逆動作要克制**：不 `git push --force`、不 `reset --hard`、不刪除 `data/logs/`
   底下任何音訊或 DB、不改使用者的 `config.yaml` 既有值（新增有預設值的鍵可以）。
   回退一律用 `git revert`。
2. **repo 是公開的**。`experiments/results/` 含個人逐字稿，**不得 `git add`**。
   任何新產出若含逐字稿內容，一律放在已被 gitignore 的路徑。提交前跑 `git status`
   確認沒有夾帶 `.m4a` / `.wav` / 模型權重 / log。
3. **每個任務結束都要**：`ruff format src/` → `ruff check src/` → `pytest` 全綠 → commit。
   測試沒過就不 commit，先修。
4. **不要為了讓測試過而弱化測試**。改不動就記進「阻塞與待決」，跳到下一個任務。
5. **一次只改一個變因**。特別是實驗類任務，不可同時改兩個條件再宣稱某一個有效。
6. **證據等級要標**：`observed`（實測）／`derived`（推導）／`hypothesis`（待驗證）。
   不得把推測寫成已驗證結論。
7. 使用者不在時，**有疑義就選保守做法並記錄**，不要停下來等答案。

## 3. 分支策略

- `T1`（G Phase 0 平台抽象層）在分支 `feat/platform-layer` 上做，完成後
  `git merge --no-ff` 回 `main`。理由：跨多 commit、動到多個模組，符合 CLAUDE.md
  的開分支條件。
- 其餘任務（T2–T5）性質是新增腳本／新增模組，彼此獨立，直接小步 commit 到 `main`。
- push 時機見「9. 待使用者確認」。

---

## 4. 進度紀錄

> 每完成一項就在這裡加一行，格式：`- [x] T?.? 一句話結果 (commit)`。
> 失敗或跳過也要記，寫明原因。

- [x] T0.1 取得基準：pytest 188 passed、ruff clean、起始 commit `ca44b84`
- [x] T0.2 排除 `.vbs` cwd 疑慮（見「5. 已查證的事實更正」）
- [x] T0.3 追查「RTF 相對比」提案反覆復發的來源並根除 (`3777261`)：
      不是記憶問題。`docs/macos-port-plan.md` 有兩處（B1 論述 + 分階段順序的 Phase 2
      名稱），`docs/roadmap.md` G 線跨線依賴又照抄了階段名，造成 roadmap 自我矛盾。
      上一輪只改 B1 論述段、漏掉階段名，而階段名才是「下一步做什麼」的必讀處。
      三處已全部改名為「門檻的裝置校準」；並把這條決策補進 `CLAUDE.md`
      （原本只在不會自動載入的 roadmap 裡，是復發的根本原因）。
- [x] T1.1–T1.6 平台抽象層落地 (`1d02dfe`)：torch 加 darwin marker、
      `platform/clipboard_backends.py`＋`platform/device.py`＋`paths.py`、
      新增 22 項測試（共 210 passed）、`scripts/test_audio_file.py` 實跑 27 段正常。
      過程中抓到一個真 bug：device 解析原用單一 note 變數，macOS 的 cuda→cpu 說明會被
      後續 float16→int8 的說明覆蓋，使用者只看得到第二個原因；已改為累積後合併。
      另記錄一項實測：`import ctranslate2` 冷啟 ~73s 且會把 torch 拉進 `sys.modules`
      （observed），因此 CUDA 探測必須惰性、macOS 路徑在探測前就短路。
- [x] T1.7 文件更新 (`4a98b0b`)：macos-port-plan A 表加狀態欄、Phase 0 標完成並註明
      「未在 macOS 上驗證」、roadmap G 線狀態、CLAUDE.md 架構節、CHANGELOG 兩條。
      依 §5.1，路徑修正在 CHANGELOG 寫成預防性修正，不寫成修既有故障。
- [x] T1.8 `git merge --no-ff` 回 main (`8cfd51a`)，hash 回填 (`0f023ef`)。
- [x] T2.1 盤點 DB（2026-09-03 當下）：542 jobs 全為 completed、全部
      whisper/large-v3-turbo/beam 5/cuda；1425 segments、1473 attempts、38 corpus_labels
      （slow-transcription 19、fallback 10、capture-warning 4、hallucination-rejected 4、
      fallback-exhausted 1）；2344 個 wav、1301.8 MB、無缺檔；資料期間 2026-08-12～09-02。
      關鍵發現：1425 段裡只有 19 段轉錄時間 > 1.5s，**單靠延遲訊號候選數遠遠不夠**，
      高 recall 得靠其他訊號撐；另有 6 段 raw 很長但 processed 為 NULL（13–30s 音訊），
      因此新增 `processed-missing` 訊號。
- [x] T2.2–T2.3 訊號模組與掃描腳本 (`e1516f2`)：`src/asr_input/eval/signals.py` 18 條
      確定性訊號 + `scripts/scan_error_candidates.py`（唯讀開 DB、同時長分布對照組、
      seed 可重現）、24 項新測試（共 234 passed）。掃描結果 1425 段中 87 段（6.1%）為候選、
      對照組 22 筆。門檻取自當下 1425 段的實測分位數，記在模組 docstring。
      `no-comma-long` 由 30 字調到 60 字：30 字會打中 128 段、其中 127 段只有這個訊號打中，
      會淹掉佇列（段長 p50 38 字）。從未命中的訊號有 4 條（dominant-char、empty-output、
      cjk-spacing、near-silent），留著但要在審核後檢討是否過嚴。
- [x] T2.4 審核佇列 viewer (`2216831`)：`scripts/build_review_queue.py` →
      `data/logs/review/queue.html`。已在瀏覽器實測：109 筆載入、音訊 GET 200 並播放、
      點選與快捷鍵標記都寫進 marks.json、重建 HTML 後標記正確帶回。設計上**標記前不顯示
      候選／對照分組**，避免審核者看到標籤後對候選判得較嚴，讓對照組失去意義。
- [ ] （以下待執行）

---

### T2.5 便宜模型初審 — 暫緩，改列待決（`observed` 環境限制）

決策 #3 是「做」，但夜間執行時停在前置條件：初審必須把候選段的逐字稿送到外部模型，
而本機是否備有可用的 API 金鑰無法確認——查詢環境變數與 Windows Credential Manager 的
指令被 auto mode 分類器擋下（不是金鑰不存在，是查不到）。

在使用者不在場時，把個人逐字稿送往外部服務屬於不可逆的對外行為（送出即可能被快取），
依守則 7 選保守做法：不送、不猜金鑰來源，把這項移到「8. 阻塞與待決」。

**恢復方式**：使用者確認金鑰可用、且同意候選逐字稿送往該服務之後，再實作
`scripts/prereview_candidates.py`——只讀 `data/logs/review/candidates_*.json`，
輸出附加的排序分數檔，不寫 `corpus_labels`、不產生 gold（roadmap 明文禁止）。

### T3 FLAC 無損保存驗證 — 完成（`observed`）

全量 542 檔往返：**539/539 有內容的檔 bit-exact**，容量比 **0.522**（602.4 → 314.7 MiB），
各時長區間 0.45–0.53 幾乎持平，編碼 539 檔共 11.1s／解碼 20.9s。

3 個「失敗」檔查清楚了：44 bytes、0 frames 的空 wav。單獨用合成資料驗證，
0 frames 開檔即失敗、1 frames 正常 → 是 FLAC 無法表示零長度串流，不是無損性問題。
遷移時必須當 `skipped-empty` 略過。

T3.4 產出 `src/asr_input/storage/flac_archive.py` + `tests/test_flac_archive.py`（9 項），
**沒有接上任何 production 路徑**，符合「只寫測試與設計」。安全性靠順序保證：
`.tmp` → 讀回逐 sample 驗證 → `os.replace` → 才選擇性刪來源。
四個失敗階段各有參數化測試，斷言原檔 bytes 不變且無殘留 `.tmp`。

明細 `experiments/results/flac_roundtrip_2026-09-03.md`（不進 git）。
**未切換保存格式**，切換仍是待決策。

### T4 評測指標與 runner 骨架 — 完成（`observed`）

四個新模組（全部在 `src/asr_input/eval/`）＋ 62 項新測試，全套 **315 passed**：

- `metrics.py`（T4.1）：所有指標都建在同一份 Levenshtein 對齊上，
  CER／錯誤型態分解／Punctuation F1／插入串因此描述的是同一個對齊結果。
  對齊的 tie-break 固定為 sub → del → ins，否則同一組 pair 的
  substitution／deletion 數會隨實作細節漂移。
  Punctuation F1 **按位置計分**（比對齊後的位置，不是比數量）——標點放錯子句
  在數量比對下會被算成命中。
- `test_metrics.py`（T4.2）：29 項，每個期望值手算。過程中抓到的三個錯都是
  **我自己的手算錯誤而非程式錯誤**（等長 pair 逼出 3 個 substitution、
  假 s2t 表漏了兩個字、`这是简体` 只改 3 字不是 4 字）——這正是 T4.2 的用途。
- `manifest.py`（T4.3）：manifest 只放匿名 ID／標籤／時長／音訊 hash／split，
  私密對應檔（音訊路徑＋gold 逐字稿）放 `data/logs/` 底下。
  `validate_manifest()` 會**主動拒絕**帶 `audio_path`／`transcript`／`gold` 等鍵的
  項目，`save_manifest()` 先驗證自己的輸出再寫檔——repo 是公開的，
  逐字稿進了歷史就很難清乾淨，寧可在這裡直接失敗。
- `environment.py`（T4.4）：GPU／driver／OS build／Python／套件版本／
  benchmark 前的 GPU idle memory。走 `nvidia-smi` CLI 而不是 torch，
  因為錄音路徑現在完全不 import torch，指紋不該把 torch 拖回進程。
  每個探針失敗都記錄原因而非丟例外。不收集主機名、使用者名或家目錄路徑，
  所以指紋可以直接貼進公開 issue。
- `runner.py` + `scripts/run_benchmark.py`（T4.5）：runner 收一個
  `transcribe(path) -> str` callable，本身不 import 任何 ASR 引擎（無 GPU 也能測）。
  彙總採 **micro**（總錯誤 ÷ 總參考長度），macro 平均會讓 2 秒的段和 60 秒的段等重。
  單一樣本失敗只記錄不中斷整輪。結果 JSON **只存「輸出是否為空」而不存輸出文字**。

**已跑通 Smoke 層（`observed`）**：用 55 個候選 job 中的 5 個單段短音訊
（3–8s，共 28.05s）臨時組了一份 manifest 跑完整條路徑，產出 JSON + Markdown 到
`data/logs/benchmark/runs/`（gitignored）。這份 gold 是**既有的已採用輸出**，
所以量到的是「重現性」不是「準確率」，真正的 gold 需要人聽音訊。

跑的過程中發現一個會誤導人的預設：runner 一開始只評分**引擎原始輸出**，
但 log 裡的既有文字是**後處理過的**。同一批音訊，raw 對 processed gold 是
Strict CER 7.96%／標點 recall 0.556；補上後處理後是 **1.77%／recall 1.0**。
差額幾乎全是後處理本來就負責的標點與字形。因此 `run_benchmark.py` 加了
`--stage {processed,raw}`，預設 `processed`（使用者實際拿到的文字），
並把 stage 記進結果的 engine 欄位——混用兩邊會憑空製造 6 個百分點的 CER。

另外，`normalized_cer`（1.92%）可能高於 `strict_cer`（1.77%）：正規化會移除標點，
分母跟著變小，錯誤數更少但比率反而升高。這是定義使然，不是 bug。

`speed.realtime_factor` 只是吞吐量報告，模組 docstring 已明寫它**不參與**任何品質或
幻覺判斷（幻覺判斷用絕對轉錄時間）。

## 5. 已查證的事實更正

這一節記錄夜間查證推翻既有文件敘述的地方，避免錯誤敘述繼續被引用。

### 5.1 `.vbs` 啟動的 cwd 不是問題（`observed`，2026-09-03）

`docs/macos-port-plan.md` C 節寫：`history_store.py` 的 `DEFAULT_DB_PATH` 是相對 cwd 的
路徑，「這在 Windows 用 `.vbs` 啟動時可能已是潛在問題，值得一併查」。

**查證結果：不是問題。** `scripts/start_tray.vbs` 有 `sh.CurrentDirectory = repoDir`，
明確把 cwd 設成 repo 根目錄再啟動 Python；`scripts/start_tray.bat` 也有 `cd /d "%~dp0.."`。
兩條 Windows 啟動路徑的 cwd 都是對的。

相對路徑本身仍值得改成以套件位置解析（macOS LaunchAgent 不保證 cwd、使用者從別的目錄
直接跑 `python -m asr_input.tray` 也會建錯位置），但它是**預防性修正**，不是修現行 bug。
修的時候不要在 CHANGELOG 寫成「修好了一個既有的資料遺失問題」。

### 5.2 `manual`／`autostart` 啟動模式已實作（`observed`，2026-09-03）

`docs/roadmap.md` 的「D/E/F 後續交接清單」仍把「實作 `manual`／`autostart` 啟動模式」
列為 D 線待辦，但 `src/asr_input/tray.py:191` 已有 `startup_mode` 參數與驗證、
`:1477` 已有對應 CLI 選項、`:437`/`:450` 有行為分支。這條該畫掉。

### 5.3 roadmap 狀態欄混用兩條軸（`derived`，2026-09-03）

roadmap 宣告狀態只有 `ready`／`needs-decision`／`exploration`／`deferred`／`done`，
其中 `ready` 指「規格夠完整可以開工」，是**規格成熟度**；但 D 標 `validation`、
E 標 `in_progress`、G 標 `planned`，這些是**執行進度**。同一欄兩種語意，讀者會誤以為
「ready 的線卻有東西完成了」是矛盾。建議拆成兩欄（見 T5.2）。

---

## 6. 任務清單

排序即優先序。清單刻意開得比預期能完成的量更多，**做不完是正常的**，不要為了清空清單
而降低每一項的品質。

### T1 — G Phase 0：平台抽象層（分支 `feat/platform-layer`）

規格正本：`docs/macos-port-plan.md` 的 Phase 0（處理 A1、A2、A3 並收攏 `platform/`）。
驗收：`pytest` 全綠 + `scripts/test_audio_file.py` 能跑出結果。**不碰 `tray.py`**
（A4／A5 屬 Phase 3，會與 D 線衝突）。

- [x] **T1.1** `pyproject.toml`：torch／torchaudio 的 `[tool.uv.sources]` 加
  `marker = "sys_platform != 'darwin'"`。注意不要動到 `uv.lock` 的其他部分；改完確認
  `uv sync` 或既有 `.venv` 仍可用。若 lock 檔會大幅重寫，先只改 pyproject 並記錄。
- [x] **T1.2** 建 `src/asr_input/platform/` 套件，第一個成員是剪貼簿：
  定義 `ClipboardBackend` 協定，`WindowsClipboard`（現行 PowerShell 行為，語意不變）、
  `MacClipboard`（`pbcopy`，**以 stdin 餵入**，順帶消掉現行 PowerShell 單引號跳脫的
  脆弱點）、`NullClipboard`（其他平台，只警告不 crash）。
  `output/clipboard.py` 改成薄包裝，對外 API 不變。
- [x] **T1.3** `asr/__init__.py:build_engine()` 加 device／compute_type 自動偵測：
  config 明確指定就尊重；未指定或指定不可用時，Windows/Linux 有 CUDA → `cuda`+`float16`，
  否則 `cpu`+`int8`。**不改使用者 `config.yaml` 的既有值**，改用「未設定才推導」的語意。
- [x] **T1.4** 相對路徑改以套件位置解析：`output/history_store.py:20`、
  `output/session_log.py:18`、`output/transcript_log.py:7`。保留可用參數覆寫。
  注意：現有 DB 在 `data/logs/history.sqlite3`，解析結果必須指向同一個檔，
  **不可讓既有資料看起來消失**。改完實際確認能讀到 537 筆 jobs。
- [x] **T1.5** 新增 `tests/test_platform_layer.py`：剪貼簿後端選擇、`pbcopy` 參數組成
  （用假的 runner，不真的呼叫）、device 推導矩陣、路徑解析在不同 cwd 下一致。
- [x] **T1.6** 驗收：`pytest` 全綠、`ruff` clean、`scripts/test_audio_file.py` 實跑一次。
- [x] **T1.7** 更新 `docs/macos-port-plan.md`（Phase 0 標完成、A1–A3 標解法已落地）、
  `docs/roadmap.md` G 線狀態、`CHANGELOG.md`、`CLAUDE.md` 架構節新增 `platform/`。
- [x] **T1.8** `git merge --no-ff` 回 `main`。

### T2 — E 線：真實 log 高 recall 錯誤候選與審核佇列

目的（roadmap E 線）：從真實使用資料建立可重現的錯誤案例，供後續實驗驗證。
現有資料：`data/logs/history.sqlite3` — 537 jobs／1410 segments／1458 asr_attempts／
38 corpus_labels，另有 2339 個 wav。

**重點是高 recall**：寧可初篩多抓（false positive 可接受），不可漏掉真正的錯誤。

- [x] **T2.1** 盤點 DB：各表 schema、既有 corpus_labels 的 38 筆是什麼、
  `asr_attempts` 的 adopted／rejected 分布、時長分布。輸出一份現況摘要（不含逐字稿內容
  的統計可以進 repo；含逐字稿的一律留在 gitignore 路徑）。
- [x] **T2.2** 實作候選規則腳本 `scripts/scan_error_candidates.py`。訊號至少涵蓋：
  絕對轉錄時間 > 1.5s（已知高相關 heuristic，**不得弱化或改成 ratio**，見 roadmap）、
  fallback 觸發與 fallback exhausted、極短音訊異常、重複退化（連續 token 重複、
  最高單字占比）、輸出長度與音訊時長比異常、簡體殘留、半形標點、空輸出、
  capture warning／near-zero RMS。每個候選要標明**是哪個訊號打中的**，方便事後檢討訊號
  本身的有效性。
- [x] **T2.3** 抽正常對照組：與候選同時長分布、未被任何訊號打中的樣本，數量約候選的
  20–30%。沒有對照組就無法判斷訊號的 precision。
- [x] **T2.4** 產出審核佇列：擴充或新寫 viewer，支援「逐筆播放音訊 + 看 raw/processed +
  三鍵標記（真錯誤／正常／不確定）+ 記錄是哪個訊號打中」。輸出到 gitignore 路徑。
  這是早上使用者第一件可以直接開始做的事，**優先把它做到能用**。
- [ ] **T2.5** 若時間允許：用便宜模型對候選做初審，只標「可疑位置」與「明顯無關語音的
  幻覺」，**不得讓模型直接裁決成 gold**（roadmap 明文禁止）。初審結果只作為排序依據。

### T3 — E 線：FLAC 無損保存驗證

roadmap：FLAC 是第一候選，先以 PCM round-trip checksum、寫入失敗保留原檔及實際容量比
驗證；通過後才決定直接寫 FLAC 或背景轉換。**本輪只驗證，不切換保存格式。**

- [x] **T3.1** 確認 `soundfile` 的 FLAC 寫入可用（已是既有相依）。
- [x] **T3.2** `experiments/verify_flac_roundtrip.py`：對現有 wav 全量或大樣本做
  PCM16 → FLAC → PCM16，逐 sample 比對（必須 **bit-exact**，不是近似）。
  任何一筆不符就是不通過，記錄該檔特徵。
- [x] **T3.3** 實測容量比：總 bytes、各時長區間的壓縮率、壓縮與解壓耗時。
  這關係到 2 GiB 音訊護欄能多存多少，是決策的關鍵數字。
- [x] **T3.4** 寫入失敗保留原檔的行為設計與測試（只寫測試與設計，不改 production 寫入路徑）。
- [x] **T3.5** 結果寫成 `experiments/results/flac_roundtrip_2026-09-03.md`
  （**注意：`experiments/results/` 不進 git**；結論摘要另寫進 CHANGELOG／roadmap）。

### T4 — A 線：評測指標 runner 骨架

規格正本：`docs/asr-benchmark-proposal.md` 第 3 節。
**本輪不碰 gold transcript**（需要人聽音訊），只做不依賴 gold 的部分：指標實作與 schema。

- [x] **T4.1** `src/asr_input/eval/metrics.py`：Strict CER、Normalized CER、
  MER（CJK 以字、英數以詞）、Punctuation F1（逗號／句號／問號／頓號分開報）、
  deletion／insertion／substitution 分解、Traditional consistency（簡體殘留率、
  OpenCC 改動率）、Repetition degeneration、Determinism（N 次的 unique output 數與
  exact-match 比例）。純函數，好測。
- [x] **T4.2** 用**合成資料**寫單元測試：手工構造已知編輯距離的 pair，驗證每個指標。
  這是本任務的主要價值——指標本身算錯的話後面所有結論都不能信。
- [x] **T4.3** manifest／結果的 JSON schema：sample ID（匿名）、標籤、時長、音訊 hash、
  dev/test 切分。**音訊與逐字稿不進 git，只進 manifest 的私密對應檔。**
- [x] **T4.4** environment fingerprint 收集：GPU、driver、Windows build、Python、CUDA、
  cuDNN、CTranslate2 版本、模型 revision、benchmark 前 GPU idle memory。
- [x] **T4.5** runner 骨架 + Markdown 摘要輸出。可以先只跑得動 Smoke 層。

### T5 — 延伸（前面做完才做，或前面卡住時的替代）

- [x] **T5.1** ~~B1 門檻 RTF 相對化~~ **已取消（2026-09-03，使用者裁決）。**
  提案本身是錯的：我從 `docs/macos-port-plan.md` B1 抄來，但那份文件與 `docs/roadmap.md`
  E 線的既定決策衝突——roadmap 明文「不得只因理論上可能受負載影響，就改用
  transcribe/audio ratio、弱化或移除這個訊號」。使用者亦已多次表態絕對時間較準。
  以 roadmap 記載的實測點驗算，比例訊號會漏抓長段幻覺（21.8s 音訊／2.5s 轉錄 → 比例
  0.11；25.2s／3.5s → 0.14）並誤殺正常短段（固定開銷讓 0.3s 音訊的比例達 1.3），
  兩個方向都比絕對時間差。機制上，beam search 跑掉的解碼時間由 max token 長度決定，
  與音訊長度幾乎無關，除以音訊長度等於把訊號 normalize 掉。
  **已改為**：修掉兩份文件的矛盾（`macos-port-plan.md` B1 已加註撤回，改推 (b) device
  profile），並寫入記憶 `absolute-time-beats-rtf-ratio`。
  **後續不要重提此提案。** macOS CPU int8 讓正常段普遍超過門檻是真問題，但解法是
  門檻進 device profile，不是改用比例。
- [ ] **T5.2** roadmap 狀態欄拆成「規格狀態」與「執行進度」兩欄（見 5.3），
  順便畫掉 5.2 那條已完成的 D 線待辦。
- [ ] **T5.3** 盤點 `docs/index.html` 與現況的落差，列出翻新清單（F 線範圍，只盤點不動工）。
- [ ] **T5.4** A 線多來源預標註：用 production FW + 另一組解碼設定 + 至少一個不同後端
  對同一批音訊產生候選稿，自動對齊算逐字分歧。**需要 GPU，使用者已確認今晚 GPU 閒置。**
  產出的是「哪些段需要人裁決」的排序，不是 gold。
- [ ] **T5.5** determinism 實測：固定 `temperature=0` 對難段重跑 N 次，量 unique output 數，
  對照 roadmap 記載的 2026-07-31 前置實驗（某疑難段在預設 fallback 下 10 次產生 7 種輸出，
  固定 temperature 後 10 次一致）。擴大樣本，**不得沿用原實驗那個帶
  `known_asr_error` 標籤的排序截斷取樣**。

---

## 7. 明確不做（本輪範圍外）

- **任何需要聽音訊判斷的事**：gold 逐字稿、VAD 語音區間標註、corpus 升格裁決。
  可以把工作**準備好排隊**，但不可代為裁決。
- **D 線 7 項 Windows 實機驗收**：要拔麥克風、開系統靜音、等真實 30 分鐘 idle、
  強制中斷。需要人在機器前。
- **C 線 E0–E6**：需要人工標註的語音區間當 oracle。
- **G Phase 1–4**：需要 mac。
- **F 線發布工程**：何時對外、公開版預設值是使用者的決定。
- **切換音訊保存格式**：T3 只驗證，切換要另外決策。
- **改 production 幻覺門檻**：除非 T5.1 的前置證明做完。

## 8. 阻塞與待決（夜間遇到就往這裡加）

> 格式：問題、卡在哪、我採取的保守做法、需要使用者決定什麼。

- **T2.5 便宜模型初審**：卡在「無法確認本機有可用的 API 金鑰」（查環境變數與 Credential
  Manager 的指令被 auto mode 擋下），且送出候選逐字稿是不可逆的對外行為。保守做法＝不送、
  不實作呼叫端。**需要使用者決定**：要用哪個服務／金鑰，以及是否同意個人逐字稿送往該服務。
  詳見「4. 進度紀錄」同名段落。

## 9. 使用者已裁決（2026-09-03 夜間開始前）

1. **不 push**。夜間完成的 commit 只留本機，早上使用者看過再推。
2. **不動 `config.yaml` 既有值**。device 自動偵測只在鍵未設定時推導。
3. **T2.5 便宜模型初審：做**，但只作審核佇列的排序訊號，不寫入任何 corpus 標籤，
   不裁決 gold。
4. **T5.1 取消**，見該項說明。
5. **優先順序由執行者自行判斷**，使用者無指定偏好。採用的順序與理由見下。

### 執行順序與理由

`T1 → T2 → T3 → T4`。

- **T1 先做**，因為它是唯一「上下文品質會直接影響成果」的任務：跨模組重構需要對現有
  程式碼有完整掌握，趁 session 上下文還完整時做最安全。它也有最明確的驗收（pytest
  全綠），失敗會立刻知道。
- **T2 第二**，因為它的產出是**唯一能把夜間工時轉換成使用者工時**的東西。整條 E 線的
  真正瓶頸是人工裁決，把待審案例排好隊，使用者早上就能直接開工。單項價值最高，也最大。
- **T3 第三**，它短（純確定性驗證），插在兩個大任務之間當節奏調節，並產出一個決策關鍵
  數字（FLAC 容量比 → 2 GiB 護欄能多存多少）。
- **T4 最後**，價值高但不急：它是 A/B/C 線的基礎，而那幾條線本來就還沒開工。純函數
  加單元測試，即使上下文已經退化也不容易做錯。

## 10. 收尾條件

達成以下任一即停止並解除 cron：

1. T1–T4 全部完成（T5 視為加分，不列入停止條件）
2. 發生重大錯誤：測試無法恢復全綠、資料疑似損毀、git 狀態異常
3. 2026-09-04 06:30 到達

停止時必須留下：更新完的「4. 進度紀錄」、「8. 阻塞與待決」，以及一段給使用者的
早晨摘要（做完什麼、卡在哪、第一件建議接手的事）。
