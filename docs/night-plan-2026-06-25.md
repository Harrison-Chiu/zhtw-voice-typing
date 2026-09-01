# 夜間自動執行計劃 — 2026-06-25

> 給「半夜被 cron 喚醒自動執行」的 Claude 看。本檔是**單一真相來源**,即使對話被壓縮也以此為準。
> 收尾流程依 CLAUDE.md「階段收尾流程」:每個任務做完 `ruff format/check` → `pytest` → 視情況更新 TODO/CLAUDE.md → commit。

## 全域原則（重要）

- **環境**:專案用 `.venv`（**Python 3.12**,路徑 `D:\Harrison\code_test\asr-input\.venv\Scripts\python.exe`）。conda base 是 Python 3.13、與本專案無關,**別用**。
- **安裝套件的分離問題**:沙箱與使用者環境分離,我這邊裝的使用者看不到。我**可以**為了自己驗證而裝,但裝完**必須明確留下「使用者若要實作需自行執行的指令」**。
  - 使用者睡前狀態:**`transformers` 已裝進 `.venv`、whisper-zh-TW + Fun-ASR-Nano 權重已下載**;**`funasr` 未裝(依賴地獄,延後)**。
  - **若 runtime `import` 失敗(套件缺) → 不要在使用者環境硬裝**。記錄「缺哪個套件 + 使用者該跑的指令」到 commit message / TODO,跳過該任務、繼續下一個。
  - 安裝務必指定 `.venv`:`uv pip install -U <pkg> --python "D:\Harrison\code_test\asr-input\.venv\Scripts\python.exe"`,否則會裝到 conda base。
- **可逆安全網**:模型測試若把環境弄亂,`uv sync` 可還原到 `uv.lock` 狀態。
- **模型測試零污染原則**:新模型一律「能跑就跑測試音檔、眼睛驗、不行就丟」。丟 = 刪掉新增的 adapter 檔 + 移除 `build_engine()` 分支 + config 還原,**不留痕跡**。
- **測試音檔**:`data/test_audio/簡報錄音_8分鐘.m4a`（8 分鐘,含已知幻覺段）。
- **不要動** `streaming.py` 的幻覺偵測（為 Whisper 調的,本階段不重構;模型測試階段把它當背景）。

## 執行前必讀（給冷啟動 / cron 喚醒的自己）

- **必須在本機執行**:任務 4a 需要本機 RTX 4060 GPU、`.venv`、已下載的權重。**不要在雲端環境跑**(雲端沒有 GPU 和這些檔案)。任務 1–3 改本機 `tray.py` 也只有本機驗得了。
- **斷點續跑(冪等)**:每次被喚醒,**先 `git log --oneline -15` 看哪些任務已 commit**,從下一個未完成的接著做,不要重做已完成的。每個任務做完**立刻 commit**,進度才不會因下次喚醒重來或中斷而遺失。
- **同一時間只跑一個**:任務會改同一批檔(`tray.py`)並 commit,若上一輪還在跑就**不要重複開工**(避免衝突)。
- **截止與解除**:做到使用者指定的截止時間(預設 06:05)或 1→2→3→4a 全部完成,就**解除排程**並留總結。

## 環境現況快照（2026-06-25 睡前驗證）

- `.venv` Python 3.12.11;`transformers 5.12.1`、`numpy 2.5.0`(被 `-U transformers` 帶上來,已偏離 uv.lock)、`faster_whisper` OK、`torch 2.6.0+cu124` CUDA 可用。
- `pytest` 基準:**38 passed**(改完任何東西後應維持綠燈)。
- `funasr` **未裝**(4b 延後)。兩個模型權重皆已在 HF cache。
- 若環境出怪事,`uv sync` 可還原(但會把 numpy 降回鎖檔版,可能影響 transformers，視情況取捨)。

## 執行順序

1. 啟動拆段計時（零依賴）
2. 狀態列多狀態色 — core（零依賴）
3. 狀態列右鍵手動卸載/載入模型（零依賴）
4a. whisper-turbo zh-TW 評測（transformers 已裝進 .venv + 權重已下載）
4b. Fun-ASR-Nano 評測（今晚自行嘗試裝 funasr，**用隔離環境**，見任務 4b）
5. 〔選配·有時間才做·排最後〕狀態列波形動畫 / 浮動 UI

---

## 任務 1 — 啟動流程拆段計時

- **目標**:量出 CUDA 暖機 vs 載權重各花多久。
- **檔案**:`src/asr_input/tray.py` `_setup()`
- **做法**:
  1. `engine.load()` 前插 CUDA 暖機計時:`torch.zeros(1).to(device); torch.cuda.synchronize()`，`device` 讀 config（cpu 時跳過 synchronize）。
  2. 分別 `time.perf_counter()` 計時:CUDA 暖機 / `engine.load()` / VAD `torch.hub.load`。
  3. 印 `[計時] CUDA 暖機: X.Xs / 載 Whisper: X.Xs / 載 VAD: X.Xs`。
- **驗收**:啟動印出三行,總和 ≈ 體感等待。
- **錯誤訊號 + 方向**:
  - 若三段相加遠小於體感總時間 → 漏算了某段（可能 `import torch` 本身,在 module top）。方向:在 `main()` 最前面補一個「程式啟動到進 `_setup` 」的粗計時。
  - CUDA 暖機數字抖動大（多跑幾次差很多）→ 屬正常,退回只留「載 Whisper / 載 VAD」兩段。不算失敗。

## 任務 2 — 狀態列多狀態色（core）

- **目標**:狀態更易辨識。
- **檔案**:`src/asr_input/tray.py` `COLORS` / `State` / `_make_icon`
- **做法**:現有 4 狀態配色拉開對比;`_make_icon` 可把段數數字 render 進圖示（PIL `ImageDraw.text`）。
- **驗收**:四狀態肉眼可區分。
- **錯誤訊號 + 方向**:
  - 圖示色沒更新 → pystray 需重設 `icon.icon`；若選單標籤沒刷新需 `icon.update_menu()`。
  - 數字字型缺 → PIL 預設字型即可,不要依賴系統字型路徑。

## 任務 3 — 狀態列右鍵手動卸載/載入模型

- **目標**:不關程式即可釋放 ~2GB VRAM。
- **檔案**:`src/asr_input/tray.py`（選單、load/unload handler、新增 `UNLOADED` 狀態）
- **做法**:
  1. 選單加項,標籤隨狀態切「卸載模型(釋放顯卡)」/「載入模型」,切換後 `icon.update_menu()`。
  2. 卸載:`engine.unload()` + `torch.cuda.empty_cache()`,狀態設 `UNLOADED`(灰)。
  3. 守衛 `_toggle()`:`UNLOADED` 按快捷鍵 → 通知「請先載入模型」,不錄音。
  4. 守衛:`STREAMING`/`TRANSCRIBING` 禁止卸載。
  5. 載入在背景執行緒（30–60s），完成回 `IDLE`。
- **驗收**:右鍵卸載 → `nvidia-smi` VRAM 掉 ~2GB;載入 → 回升、錄音恢復;串流中卸載項為灰/停用。
- **錯誤訊號 + 方向**:
  - VRAM 沒完全歸零 → caching allocator 保留 reserved 屬正常,`empty_cache()` 後仍殘留可接受,不算失敗,文件註明即可。
  - 卸載後又被觸發錄音導致 crash → 代表守衛沒擋住,優先補 `_toggle()` 的 `UNLOADED` 檢查。
  - 重載卡死 → 確認舊 session 已 `stop()`、舊 engine 已釋放再 `load()`。

## 任務 4a — whisper-turbo zh-TW 評測

- **接法（重要）**:**不要走 faster-whisper**（它只吃 CTranslate2 格式,要轉檔）。本評測用 `transformers` pipeline 直接跑,目的只是「看品質值不值得」。
- **檔案**:新增 `scripts/test_hf_whisper_zhtw.py`（一次性測試腳本,放 scripts/）
- **做法**:
  1. 用現有 `FileAudioSource` 解碼 `data/test_audio/簡報錄音_8分鐘.m4a` 成 16k np.ndarray（重用現成解碼,不要自己寫 ffmpeg）。
  2. `transformers` ASR pipeline,`model="JacobLinCool/whisper-large-v3-turbo-common_voice_19_0-zh-TW"`,**務必 `chunk_length_s=30`**（8 分鐘音訊不切會 OOM）。
  3. 印原始輸出;另跑一份過現有 `build_pipeline` 後處理的輸出對照。
  4. **同時跑一份現況基準**:用現有 config（large-v3-turbo + faster-whisper）跑同一支音檔,兩者並列,才能比較。
- **驗收（眼睛判斷,給自動執行的明確準則）**:
  - 是否**原生繁體**（掃有無簡體字,如「软/视/启/发」）。
  - 標點是否正常且全形,**沒有**大量 `…` 連續、**沒有**逗號被空白取代。
  - 內容沒有幻覺式重複。
  - **短句穩定度 ≥ 現況基準**（這是換它的主要動機）。
- **錯誤訊號 + 方向**:
  - `import transformers` 失敗 → 使用者沒裝成功,**不要自行安裝**,記錄後跳到 4b。
  - m4a 解不開 → `FileAudioSource` 已能解（既有測試就用 .m4a）,若仍失敗檢查是否被改動。
  - OOM → 確認有設 `chunk_length_s`,並用 `torch_dtype=float16`、`device=0`。
  - 輸出簡體 → 該微調未達預期,屬「結論」非「錯誤」,照實記錄,判定不如現況 → 丟。
  - **判定丟棄時**:刪 `scripts/test_hf_whisper_zhtw.py`?→ 可保留腳本（在 scripts/ 不污染主程式）,但 config 不要改。結論寫進 TODO/CLAUDE.md。

## 任務 4b — Fun-ASR-Nano 評測（今晚自行嘗試安裝）

> **安裝授權**:使用者已授權「夜間自動執行時自行嘗試裝 funasr」。**鐵則:用隔離環境,絕不裝進主 `.venv`**（funasr 會升 numpy/torch，會弄壞現在能跑的 app）。
>
> **背景**:先前在 conda base(Python 3.13)裝失敗——umap-learn→pynndescent→llvmlite 0.36 只支援 py<3.10。推測獨立 Python 3.11 環境 resolver 會挑到正常的 llvmlite/numba。

### 步驟 0 — 建隔離環境並裝 funasr（先做這步，失敗就放棄 4b）

```
cd D:\Harrison\code_test\asr-input
uv venv .venv-funasr --python 3.11        # >=3.12 的警告無妨，這是測試環境
uv pip install funasr --python ".venv-funasr\Scripts\python.exe"
```

- **`.venv-funasr` 記得加進 .gitignore**（或確認 `.venv*` 已被排除），別 commit。
- 若還要 GPU 版 torch（CPU 跑 8 分鐘音檔太慢）：裝完 funasr 後再 `uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --python ".venv-funasr\Scripts\python.exe"`。先看裝不裝得起來，再煩惱 GPU。
- **錯誤訊號 + 方向**：
  - 又卡 llvmlite/numba 編譯 → 試 `uv pip install funasr --python ...`（讓 resolver 自由挑新版）；仍不行就試 Python 3.10。
  - 三次以內裝不起來 → **放棄 4b**，把錯誤摘要寫進 TODO，清掉 `.venv-funasr`，繼續其餘任務。**不要花整晚跟它纏鬥。**

### 步驟 1 — 評測（裝成功才做）

- **檔案**:新增 `src/asr_input/asr/funasr_nano.py`（繼承 `ASREngine`）+ `build_engine()` 加 `elif engine_name == "funasr_nano"` 分支。
- **注意**:funasr 在 `.venv-funasr`，不在主 `.venv`。評測腳本要用 `.venv-funasr\Scripts\python.exe` 跑（adapter 與測試走隔離環境；確認可用、值得留，才談整合進主環境）。
- **做法**:
  1. 用 `funasr` 的 `AutoModel`,`hub="hf"`,model 指向 `FunAudioLLM/Fun-ASR-Nano-2512`（API 細節寫 adapter 時上網/讀 repo 確認）。
  2. `transcribe()`:吃 16k np.ndarray → 回字串。
  3. **重點驗引導性**:在它的 prompt/context（讀 repo 確認參數名）餵「使用台灣繁體中文」,看會不會吐繁體。這是它對比 Whisper 的關鍵賣點。
  4. config 暫切 `engine: funasr_nano`,跑 `scripts/test_audio_file.py` 簡報音檔。
- **驗收**:能被引導出繁體 + 品質 ≥ 現況基準。
- **錯誤訊號 + 方向**:
  - `import funasr` 失敗 → 使用者沒裝,**不要自行安裝**,記錄後結束模型階段。
  - `AutoModel` 載入報錯（hub/路徑）→ 先試 `hub="hf"`,再試預設(modelscope);確認權重已在 HF cache。
  - 找不到 prompt/引導參數 → 讀 `FunAudioLLM/Fun-ASR-Nano-2512` 的 HF README 或 Fun-ASR GitHub 範例;若確實無引導機制,結論記為「不可引導」,看原生輸出是繁是簡再判。
  - 依賴衝突弄壞環境 → `uv sync` 還原。
  - **判定丟棄時**:刪 `src/asr_input/asr/funasr_nano.py` + 移除 `build_engine()` 分支 + config 還原成 `whisper`。

## 任務 5 〔選配·排最後·有時間才做〕— 狀態列波形動畫 / 浮動 UI

- **前置**:streaming 目前**無即時音量回呼**,真波形要新增 callback,工程量大。
- **方向**:若評估要動 `streaming.py` 才拿得到即時振幅 → **本任務直接略過**,只在計劃留記錄。系統匣本身有節流,即時性有天花板（要真即時須做獨立浮動視窗,屬更大的 TODO）。

---

## 收尾與回報

- 每完成一個任務就 commit（中文訊息,講清楚改了什麼）。
- 模型評測結論（4a/4b 是繁是簡、品質如何、是否採用）寫進 `CLAUDE.md`「已確立的設計決策」+ `TODO.md`。
- 全部跑完,在最後留一則總結:哪些做完、哪些因套件缺/品質不足而跳過、後續建議。
