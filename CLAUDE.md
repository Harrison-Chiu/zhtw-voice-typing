# CLAUDE.md — ASR Input 專案引導

## 專案概述

本地語音輸入系統，在 RTX 4060（8GB VRAM）上用 Qwen3-ASR 1.7B 辨識語音，
經 OpenCC + 自訂詞表後處理輸出**台灣繁體中文**。目標是日常語音輸入工具。

## 快速指令

```bash
uv sync                              # 安裝依賴
uv run python -m asr_input.main      # 啟動 CLI（麥克風錄音→辨識）
uv run python -m asr_input.tray      # 啟動 System Tray 版（全域快捷鍵）
uv run python test_audio_file.py     # 用音檔測試（不需要麥克風）
uv run ruff check src/               # lint 檢查
uv run ruff format src/              # 格式化
```

注意：首次 `import torch` 需要 30-60 秒載入 CUDA，這是正常的。
模型首次載入會從 HuggingFace 下載約 3.4GB，之後有快取。

## 架構

模組化 pipeline，每個環節可獨立抽換：

```
麥克風 → AudioCapture → ASREngine → TextProcessing → Output
              │              │              │            │
        audio/capture.py  asr/qwen.py  processing/*  output/clipboard.py
```

### 關鍵檔案

- `src/asr_input/main.py` — CLI 進入點，串接整條 pipeline
- `src/asr_input/tray.py` — System Tray 進入點（全域快捷鍵 + 常駐 tray）
- `src/asr_input/asr/base.py` — ASR 引擎抽象介面（`ASREngine`）
- `src/asr_input/asr/__init__.py` — `build_engine()` 工廠，依 config `engine` 切換
- `src/asr_input/asr/qwen.py` — Qwen3-ASR 實作，用 `qwen-asr` 套件
- `src/asr_input/asr/whisper_fw.py` — faster-whisper 實作（**目前預設引擎**）
- `src/asr_input/processing/pipeline.py` — `TextProcessor` 抽象介面 + `ProcessingPipeline` 串接器
- `src/asr_input/processing/punct_norm.py` — 上下文感知標點正規化（CJK 旁半形→全形）
- `src/asr_input/processing/opencc_conv.py` — 智慧 OpenCC：偵測到簡體字才跑 s2twp 轉換，純繁體跳過
- `src/asr_input/processing/tw_terms.py` — 自訂台灣用語替換（讀 `data/tw_dict.yaml`）
- `src/asr_input/audio/capture.py` — `AudioSource` 抽象介面 + `MicrophoneCapture` 實作
- `src/asr_input/output/clipboard.py` — 剪貼簿輸出
- `config.yaml` — 使用者設定（模型、裝置、語言、後處理選項）
- `data/tw_dict.yaml` — 台灣用語替換詞表
- `docs/index.html` — 互動式專案文件頁面

### 擴展模式

新增 ASR 引擎：繼承 `asr/base.py:ASREngine`，實作 `load()`、`transcribe()`、`unload()`，再到 `asr/__init__.py:build_engine()` 加一個分支。
新增後處理步驟：繼承 `processing/pipeline.py:TextProcessor`，實作 `process()`，然後在 pipeline 中 `.add()` 即可。
新增輸出方式：同樣繼承 `TextProcessor`（目前設計如此，之後可能獨立介面）。

## 已確立的設計決策（不要重做）

- **ASR 引擎**：**預設改用 faster-whisper large-v3-turbo**（~2GB VRAM、辨識 ~0.5s）。Qwen3-ASR 1.7B 保留為備用引擎。可在 `config.yaml` 的 `asr.engine` 切換（`whisper` / `qwen`）
- **Whisper initial_prompt 能引導繁體+標點（已證實）**：短繁體句+全形標點（`繁體中文，台灣用語。`）→ 輸出原生 0% 簡體 + 帶標點。prompt 字體決定輸出字體、prompt 標點決定輸出標點，兩者獨立。詳見 `experiment_whisper_prompt.py`。這跟 Qwen 的 context 完全相反
- **標點全形化靠後處理，不靠 prompt（已證實，勿重試）**：三輪實驗（v1-v3）測試了 15+ 種 prompt、hotwords、suppress_tokens。結論：長 prompt 可提高全形率但引入亂碼/幻覺；hotwords 對多 token 標點無效；suppress 半形逗號會被句號取代。最穩方案是短 prompt + `PunctuationNormalizer` 後處理（看前後字元判斷中英文語境）。實驗結果見 `data/experiment_punct_v2.json`、`data/experiment_punct_v3.json`、`data/experiment_viewer.html`
- **繁中轉換策略**：簡轉繁**只能靠** OpenCC s2twp + 自訂詞表後處理。context 引導已實驗證明**零效果**（見下）
- **Qwen3-ASR API**：引導文字用 `context` 參數（不是 `prompt`），音訊可傳 `(np.ndarray, sample_rate)` tuple
- **context 不能引導簡繁（已證實，勿重試）**：`experiment_context.py` 跑過 11 組探針（指令/關鍵詞/繁體前文/簡體前文/英文/否定/熱詞），輸出 byte 完全相同，簡體比例全 23.7%。context 進到了 prompt 的 system 訊息、模型也收到，但對「輸出簡體還是繁體」無作用；它的用途是罕見專有名詞的熱詞偏置。`language` 參數也只支援 `Chinese`，無繁體選項
- **Python 環境**：uv 管理，Python 3.12，PyTorch CUDA 12.4 從專用 index 安裝
- **src layout**：程式碼在 `src/asr_input/` 下，hatchling build backend
- **程式碼風格**：ruff，line-length 100，規則集 E/F/I/UP/B/SIM

## 工具鏈

- **uv** — 套件管理（取代 pip + venv），鎖檔在 `uv.lock`
- **ruff** — linter + formatter（取代 black + flake8），設定在 `pyproject.toml`
- **PyTorch CUDA 12.4** — 透過 `[tool.uv.sources]` 從 pytorch-cu124 index 安裝

## 台灣繁中轉換說明

ASR 模型輸出簡體中文 + 中國用語，經兩層後處理：
1. **OpenCC s2twp** — 簡體→台灣繁體 + 慣用詞轉換（如 视频→影片、软件→軟體）
2. **自訂詞表** `data/tw_dict.yaml` — 補 OpenCC 沒覆蓋的（如 人工智能→人工智慧、代碼→程式碼）

已知問題：OpenCC 會過度轉換某些詞（如 平台→平臺），需要在詞表中加反向修正。

## 目前狀態

v0.2 — System Tray 版可用。已驗證：
- 模型載入 ✓、音檔辨識 ✓、麥克風辨識 ✓、簡轉繁 ✓、台灣用語替換 ✓
- 多引擎切換 ✓（faster-whisper / Qwen），whisper 已設為預設、繁體+標點原生輸出 ✓
- **全域快捷鍵 + System Tray** ✓ — `pynput` 熱鍵 + `pystray` 常駐 tray，Toggle 模式
- **VAD 切段** ✓ — Silero VAD 自適應切段（800ms→500ms→300ms 遞減），裝飾器模式包裝引擎
- **標點正規化** ✓ — 上下文感知半形→全形轉換，Pipeline 第一步（標點→OpenCC→詞表→輸出）
- **轉錄 log** ✓ — JSONL 格式，每筆含時間戳/原始/處理後文字
- **智慧 OpenCC** ✓ — 偵測簡體字才跑轉換，純繁體跳過（避免項目→專案、台→臺等過度轉換）
- **詞表清理** ✓ — 刪除 no-op、加 OpenCC 反向修正（平臺→平台）、分類整理

## 後續方向
- **即時串流辨識** — 邊講邊出字
- **直接輸出到游標位置** — 模擬鍵盤輸入取代剪貼簿
- **多引擎支援** — ✓ Whisper/Qwen 已可切換；SenseVoice 待加
- **Web UI 測試介面** — 瀏覽器介面，用於測試/展示/設定調整

## 注意事項

- 開發環境在 Windows 11，RTX 4060，Claude Code 跑在虛擬機中
- `uv run` 在虛擬機中可能有路徑問題，可直接用 `.venv\Scripts\python.exe` 替代
- `data/test_audio/` 裡有測試音檔（.m4a），不要 commit 到 git（已在 .gitignore 排除 *.m4a 的上層目錄）
- 模型權重不進 git（.gitignore 已排除 *.safetensors 等）
