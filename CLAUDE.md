# CLAUDE.md — ASR Input 專案引導

## 專案概述

本地語音輸入系統，在 RTX 4060（8GB VRAM）上以本地 ASR 引擎（預設 faster-whisper，可切換 Qwen3-ASR）辨識語音，
經 OpenCC + 自訂詞表後處理輸出**台灣繁體中文**。目標是日常語音輸入工具。

## 快速指令

```bash
uv sync                              # 安裝依賴
uv run python -m asr_input.main      # 啟動 CLI（麥克風錄音→辨識）
uv run python -m asr_input.tray      # 啟動 System Tray 版（全域快捷鍵）
uv run python scripts/test_audio_file.py  # 用音檔測試（不需要麥克風）
uv run ruff check src/               # lint 檢查
uv run ruff format src/              # 格式化
uv run pytest                        # 跑單元測試（processing/，毫秒級不碰硬體）
```

注意：首次 `import torch` 需要 30-60 秒載入 CUDA，這是正常的。
模型首次載入會從 HuggingFace 下載約 3.4GB，之後有快取。

## 階段收尾流程

每個階段做完，依序執行（B 組視這次做了什麼挑需要的，不必每次硬湊）：

**A. 程式碼乾淨**（動到程式碼才需要）
1. `ruff format src/` — 自動排版
2. `ruff check src/` — 檢查，有報錯就修
3. `pytest` — 跑測試（每次都跑，僅 ~0.3s；目前只覆蓋 `processing/`）

**B. 記錄更新**（repo 內檔案，會進同一個 commit）
4. `TODO.md` — 完成項打勾、補新待辦
5. `CLAUDE.md`：
   - 「已確立的設計決策」— 有實驗結論／決定不重做的做法
   - 「目前狀態」— 功能落地、版本推進

**C. 封板**
6. `git status` 確認沒夾帶 `.m4a` / `.wav` / 模型權重
7. `git commit`（中文訊息，講清楚改了什麼）

**D. 記憶**（在 repo 外 `~/.claude/...`，獨立於 commit）
8. 學到非顯而易見的事才更新 memory（設計取捨、踩坑、慣例），純程式結構不記

## 分支與回退政策

- **預設：直接 commit 到主分支 `main`**（小步、漸進：bugfix、調參、文件、單一明確功能）。沿用「每階段就 commit」的習慣，線性歷史方便比較各階段 diff
- **開分支（符合任一才開）**：① 可能整包丟掉的探索（試新引擎、大重構）② 跨多 commit、要做完才有意義的大功能 ③ 想並排比較兩做法 → 完成用 `git merge --no-ff` 合回 `main`
- **回退優先 `git revert`（非破壞、加反向 commit），少用 `reset --hard`**（會丟掉可能已被參照的工作）
- 此政策覆蓋 harness「在預設分支就先開分支」的反射

## 文件職責分工（單一真相來源，不重複）

每類資訊只在**一處**維護，其他地方用連結指過去，避免多份抄寫各自漂移：

| 檔案 | 負責 | 不該放什麼 |
|------|------|-----------|
| `README.md` | 人類上手：是什麼、怎麼裝、怎麼跑 | 會漂移的細節（架構、config 範例、功能清單）→ 改放連結 |
| `CLAUDE.md`（本檔） | 架構與設計決策**正本** + 給 Claude 的工作慣例（含上方收尾流程） | — |
| `docs/index.html` | 對外展示的完整互動版（**待翻新**，見 TODO） | — |
| `TODO.md` | 待辦與藍圖 | — |
| `.gitignore` | 「不該 commit 的清單」的**強制執行處** | — |

**原則（正本位置）**：新增/修改文件前，先確認該資訊的「正本」在哪。已有正本就只改正本 + 他處放連結，**不要複製內容**。這是判斷型規範，靠遵守、無自動強制。

**原則（記錄寫法）**：寫記錄時要預設讀者對本次對話一無所知，所以以客觀事實與現狀為主。交代邏輯判斷時必須掌握的上下文資訊，讓讀者可以理解判斷的邏輯，並針對當下情境去調整。不必要寫當下才懂的辯解、給讀者不理解的強硬指令、或反駁沒人提過的誤解，這些在沒有上下文的狀況下會製造困惑。

## 架構

模組化 pipeline，每個環節可獨立抽換：

```
麥克風 → AudioCapture → ASREngine → TextProcessing → Output
              │              │              │            │
        audio/capture.py  asr/qwen.py  processing/*  output/clipboard.py
```

串流模式（Tray 預設）：
```
麥克風 → StreamingVAD（即時切句）→ ASREngine（逐句辨識）→ Pipeline → 累積 → 一次複製
              │                          │                      │
    audio/streaming_vad.py          asr/whisper_fw.py     processing/*
                        ↑ 協調器: streaming.py ↑
```

### 關鍵檔案

- `src/asr_input/main.py` — CLI 進入點，串接整條 pipeline
- `src/asr_input/tray.py` — System Tray 進入點（全域快捷鍵 + 串流辨識）
- `src/asr_input/streaming.py` — `StreamingSession` 協調器（VAD + ASR + pipeline 串流整合）
- `src/asr_input/asr/base.py` — ASR 引擎抽象介面（`ASREngine`）
- `src/asr_input/asr/__init__.py` — `build_engine()` 工廠，依 config `engine` 切換
- `src/asr_input/asr/qwen.py` — Qwen3-ASR 實作，用 `qwen-asr` 套件
- `src/asr_input/asr/whisper_fw.py` — faster-whisper 實作（**目前預設引擎**）
- `src/asr_input/processing/pipeline.py` — `TextProcessor` 抽象介面 + `ProcessingPipeline` 串接器
- `src/asr_input/processing/punct_norm.py` — 上下文感知標點正規化（CJK 旁半形→全形）
- `src/asr_input/processing/opencc_conv.py` — 智慧 OpenCC：偵測到簡體字才跑 s2twp 轉換，純繁體跳過
- `src/asr_input/processing/tw_terms.py` — 自訂台灣用語替換（讀 `data/tw_dict.yaml`）
- `src/asr_input/audio/capture.py` — `AudioSource` 抽象介面 + `MicrophoneCapture` 實作
- `src/asr_input/audio/streaming_vad.py` — `StreamingVAD`：即時逐 chunk 餵入 Silero VAD，靜音觸發切句
- `src/asr_input/output/clipboard.py` — 剪貼簿輸出
- `config.yaml` — 使用者設定（模型、裝置、語言、後處理選項、串流參數）
- `data/tw_dict.yaml` — 台灣用語替換詞表
- `docs/index.html` — 互動式專案文件頁面
- `scripts/` — 手動執行的開發/測試腳本（`test_audio_file.py`、`test_streaming.py`、`transcribe_file.py`）
- `experiments/` — 一次性實驗腳本，結果在 `experiments/results/`

### 擴展模式

新增 ASR 引擎：繼承 `asr/base.py:ASREngine`，實作 `load()`、`transcribe()`、`unload()`，再到 `asr/__init__.py:build_engine()` 加一個分支。
新增後處理步驟：繼承 `processing/pipeline.py:TextProcessor`，實作 `process()`，然後在 pipeline 中 `.add()` 即可。
新增輸出方式：同樣繼承 `TextProcessor`（目前設計如此，之後可能獨立介面）。

## 已確立的設計決策（不要重做）

- **ASR 引擎**：**預設改用 faster-whisper large-v3-turbo**（~2GB VRAM、辨識 ~0.5s）。Qwen3-ASR 1.7B 保留為備用引擎。可在 `config.yaml` 的 `asr.engine` 切換（`whisper` / `qwen`）
- **whisper-turbo zh-TW 微調評測（不採用）**：測 `JacobLinCool/whisper-large-v3-turbo-common_voice_19_0-zh-TW`，走 transformers pipeline `chunk_length_s=30`。原生繁體+原生全形標點（贏 baseline 的 raw），但出現大量單字重複、專有名詞錯更多、句界漏併。**2026-06-29 重測 segments 釐清**：短句（~10-16s）品質其實≈baseline，唯一明顯弱點是易掉句末標點；但**單段 27.7s（<30s，不觸發分塊）也會崩潰成「量量量」重複數百次** → 重複退化**不純是分塊造成**（先前「與分塊有關」的假設只對一半，模型本身在難段也會塌）。淘汰理由正確版＝**相對 baseline 無增益 + 崩潰風險**，非模型本體品質差。baseline（faster-whisper + VAD + 短 prompt + 後處理）更穩，不值得為它做 CT2 轉檔或另建 VAD 路徑。腳本 `scripts/test_hf_whisper_zhtw.py`、`experiments/run_4a_segments.py`，輸出 `experiments/results/whisper_zhtw_segments_2026-06-29.json`
- **Whisper initial_prompt 能引導繁體+標點（已證實）**：短繁體句+全形標點（`繁體中文，台灣用語。`）→ 輸出原生 0% 簡體 + 帶標點。prompt 字體決定輸出字體、prompt 標點決定輸出標點，兩者獨立。詳見 `experiments/experiment_whisper_prompt.py`。這跟 Qwen 的 context 完全相反
- **標點全形化靠後處理，不靠 prompt（已證實，勿重試）**：三輪實驗（v1-v3）測試了 15+ 種 prompt、hotwords、suppress_tokens。結論：長 prompt 可提高全形率但引入亂碼/幻覺；hotwords 對多 token 標點無效；suppress 半形逗號會被句號取代。最穩方案是短 prompt + `PunctuationNormalizer` 後處理（看前後字元判斷中英文語境）。實驗結果見 `experiments/results/experiment_punct_v2.json`、`experiments/results/experiment_punct_v3.json`、`experiments/results/experiment_viewer.html`
- **繁中轉換策略**：簡轉繁**以** OpenCC s2twp + 自訂詞表後處理**為主**（最穩、跨內容一致）。模型層 prompt 引導只能當輔助（見下「繁體引導」條），不足以單獨依賴。OpenCC 設有智慧偵測：白名單排除「台」等台灣常用異體，其餘有任何簡體字才觸發轉換
- **Fun-ASR-Nano 評測結論（已對版實測·不採用，2026-06-29 翻案）**：先前（2026-06-25）以為「funasr 版本卡關」是誤判。真因是 **Fun-ASR-Nano 是 LLM-ASR（SenseVoice encoder + Qwen3-0.6B backbone），HF 模型快照裡沒有 `model.py`**，通用 `pip install funasr`（1.3.14，版本其實符合官方要求 `funasr>=1.3.0`）的 AutoModel 找不到 `FunASRNano` 類別 → 退化用 CTC 路徑硬載 model.pt → 噴 `miss key ctc_decoder.*` 並輸出單字重複垃圾。**那些 `miss key` 警告是無害紅鯡魚**（LLM-ASR 前向不用 CTC head，正常能跑的版本一樣會印）。**修法**：clone 官方 repo `github.com/FunAudioLLM/Fun-ASR`，用 `.venv-funasr` + `AutoModel(..., remote_code="<repo>/model.py")` 指向 repo 的 model.py（內含 FunASRNano 實作）。修好後輸出乾淨、短句辨識最準、RTF~0.23。限制：**原生簡體**需 OpenCC、`language`/`hotword` 引導繁體不穩健（見下）。結論：可用但相對 baseline 無增益，不採用。腳本 `scripts/test_funasr_nano_v2.py`、`experiments/probe_funasr_scaffold.py`
- **Hotwords 實驗結論（已證實）**：短詞 hotwords（`"詞表 待辦 清單"`）最安全不影響品質；長句 hotwords 會導致標點全變句號、重複句、幻覺。擴充 initial_prompt 也有副作用（如「開發平台→開發平臺」）。詳見 `experiments/results/experiment_hotwords.json`
- **Qwen3-ASR API**：引導文字用 `context` 參數（不是 `prompt`），音訊可傳 `(np.ndarray, sample_rate)` tuple
- **Qwen context 對繁簡引導：有界、措辭/內容相依（2026-06-29 更正舊「零效果」結論）**：舊結論（`experiment_context.py` 11 組探針 byte 相同、簡體 23.7%）只對了一半——那 11 組探針正好都落在無效措辭。重測發現 `context="請以繁體中文輸出"`（system prompt）能把簡體率從 ~0.24-0.37 壓到 ~0（兩短檔一致、確定性）。但**極度依賴確切措辭**：語意等價的「請輸出繁體中文」「裸關鍵詞」「正體」「英文指令」皆失效，「以繁體中文輸出」「請用繁體中文」只半效。且**內容相依**：某些音訊段（如自我介紹段）頑固抗拒、維持簡體。另證實 **context 不是 prefix 槽**（塞半句前文完全不被覆誦 echo=False）、不做台灣用語在地化、領域熱詞偏置只有邊際效果。`language` 參數限固定 30 種、只有 `Chinese` 無繁體。→ 可引導但不穩健，簡轉繁仍以 OpenCC 為主。腳本 `experiments/steer_qwen_traditional.py`、`steer_qwen_variants.py`、`probe_context_purpose.py`
- **繁體引導能力排名（2026-06-29，跨模型實驗）**：就「讓模型原生輸出繁體」這軸，**FW initial_prompt（最穩，跨段全 0% 簡體）≫ Qwen context（措辭+內容相依）> Fun-ASR language（內容相依，最佳 probe 也僅部分檔）**。跨模型收斂觀察（兩個獨立 LLM-ASR 同向）：① 都有**內容/聲學相依的字體先驗**，prompt 只能部分扭轉，有些段頑固抗拒（但抗拒的段因模型而異，別把某段標成普世難段）；② 引導指令**要用繁體字寫「繁體」**才可能觸發（「正體」「繁体」「英文」弱/無效）。**Fun-ASR 內部 scaffolding 字體有影響**（單變因 A/B 證實）：把 model.py `get_prompt` 的簡體鷹架（`语音转写成`/`热词列表`）換繁體，能讓 language 值搞不定的抗拒段 seg01 簡體率 0.296→0.074，但非萬靈丹（seg03 仍全抗拒）。**侷限**：樣本小（每實驗 2-3 段）、單一指標（OpenCC s2t 簡體率）、未量字準確度，勿外推到任意音訊
- **四引擎評測總結（2026-06-29，模型測試告一段落）**：受測 baseline faster-whisper / whisper-zh-TW(4a) / Qwen3-ASR / Fun-ASR-Nano(4b)。**結論：FW 整體最佳，維持預設，模型橫向比較到此為止**。理由（憑原文判讀，限這幾段）：FW 原生繁體、自動清掉講者贅字/重複（語音輸入要的乾淨）、英文詞最準（ArduCopter/PWM；Fun-ASR 會拆成「P W N」「DO CAPTURE」）、從不崩潰。候選只有**小贏**：Qwen/Fun-ASR 人名列舉用頓號「、」較正確、逐字忠實（保留「呃」與單位「伏」）——但這對「語音輸入要乾淨可用」反是缺點。4a 因崩潰風險出局。沒有任一候選夠強到值得換。完整原始輸出見 `experiments/results/*2026-06-29*`，記憶 `~/.claude/.../memory/model-eval-2026-06-29.md`
- **CUDA 暖機預設關閉（空轉，2026-06-28）**：tray 啟動原有 `torch.zeros(1).to(cuda)` 暖機，本意是提前觸發 CUDA context 初始化以縮短首次推論延遲。但 faster-whisper 推論走 CTranslate2（自帶獨立 CUDA 初始化，不共用 PyTorch context）、Silero VAD 跑在 CPU（`tray.py` / `streaming_vad.py` 都未把 model/tensor `.to(device)`）→ 進程內沒有任何 PyTorch GPU 工作會用到這個 context，暖機純空轉（實測 ~0.1s）。改由 config `startup.cuda_warmup` 控制，**預設 `false`**，保留開關給未來（改用 PyTorch-based ASR 引擎、或讓 VAD 上 GPU 時再開）
- **tray 首次載入 ~27s 的歸因（已測，2026-06-28）**：啟動計時顯示瓶頸全在「載 Whisper」（暖機僅 0.1s、載 VAD ~1.2s）。大宗是**首次冷讀磁碟**——CTranslate2/cuDNN 等原生 DLL（隨 `import faster_whisper` 載入）+ ~2GB 權重檔（`WhisperModel()` 建構時讀進 GPU）。卸載→重載快很多，因 DLL 已常駐進程、權重已進 OS 檔案快取、CT2 已初始化，只剩「把已快取權重再讀進 GPU」。細分腳本 `experiments/profile_cold_start.py`（熱快取下各階段總和 ~9.5s，遠低於冷啟的 27.7s；要量真冷啟須重開機後第一件事就跑）
- **Python 環境**：uv 管理（鎖檔 `uv.lock`），Python 3.12，PyTorch CUDA 12.4 透過 `[tool.uv.sources]` 從 pytorch-cu124 index 安裝
- **src layout**：程式碼在 `src/asr_input/` 下，hatchling build backend
- **程式碼風格**：ruff（設定在 `pyproject.toml`，取代 black+flake8），line-length 100，規則集 E/F/I/UP/B/SIM；測試用 pytest

## 台灣繁中轉換說明

ASR 模型輸出簡體中文 + 中國用語，經兩層後處理：
1. **OpenCC s2twp** — 簡體→台灣繁體 + 慣用詞轉換（如 视频→影片、软件→軟體）
2. **自訂詞表** `data/tw_dict.yaml` — 補 OpenCC 沒覆蓋的（如 人工智能→人工智慧、代碼→程式碼）

已知問題：OpenCC 會過度轉換某些詞（如 平台→平臺），需要在詞表中加反向修正。

## 目前狀態

v0.3 — MVP 完成，進入後續改善階段。核心鏈路全數驗證通過：辨識／簡轉繁／台灣用語／串流辨識／VAD 切段／標點正規化／全域快捷鍵／轉錄 log／單元測試。
功能細節見「架構」與「已確立的設計決策」；待辦與藍圖見 `TODO.md`。

**模型橫向比較告一段落（2026-06-29）**：四引擎（baseline FW / whisper-zh-TW / Qwen3-ASR / Fun-ASR-Nano）評測完成，結論 **FW 整體最佳、維持預設**（見上「四引擎評測總結」）。三個候選都跑通且原始輸出已存檔，但無一值得替換。下一步改善方向轉向「**把 FW 做得更好**」而非換模型——藍圖見 `TODO.md`。

## 注意事項

- 開發環境在 Windows 11，RTX 4060，Claude Code 跑在虛擬機中
- `uv run` 在虛擬機中可能有路徑問題，可直接用 `.venv\Scripts\python.exe` 替代
- `data/test_audio/` 裡有測試音檔（.m4a / .wav），不要 commit 到 git（.gitignore 已整個資料夾排除 `data/test_audio/`）
- 模型權重不進 git（.gitignore 已排除 *.safetensors 等）
