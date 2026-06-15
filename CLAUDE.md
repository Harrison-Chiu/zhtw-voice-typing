# CLAUDE.md — ASR Input 專案引導

## 專案概述

本地語音輸入系統，在 RTX 4060（8GB VRAM）上用 Qwen3-ASR 1.7B 辨識語音，
經 OpenCC + 自訂詞表後處理輸出**台灣繁體中文**。目標是日常語音輸入工具。

## 快速指令

```bash
uv sync                              # 安裝依賴
uv run python -m asr_input.main      # 啟動 CLI（麥克風錄音→辨識）
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
- `src/asr_input/asr/base.py` — ASR 引擎抽象介面（`ASREngine`）
- `src/asr_input/asr/qwen.py` — Qwen3-ASR 實作，用 `qwen-asr` 套件
- `src/asr_input/processing/pipeline.py` — `TextProcessor` 抽象介面 + `ProcessingPipeline` 串接器
- `src/asr_input/processing/opencc_conv.py` — OpenCC s2twp 簡轉繁
- `src/asr_input/processing/tw_terms.py` — 自訂台灣用語替換（讀 `data/tw_dict.yaml`）
- `src/asr_input/audio/capture.py` — `AudioSource` 抽象介面 + `MicrophoneCapture` 實作
- `src/asr_input/output/clipboard.py` — 剪貼簿輸出
- `config.yaml` — 使用者設定（模型、裝置、語言、後處理選項）
- `data/tw_dict.yaml` — 台灣用語替換詞表
- `docs/index.html` — 互動式專案文件頁面

### 擴展模式

新增 ASR 引擎：繼承 `asr/base.py:ASREngine`，實作 `load()`、`transcribe()`、`unload()`。
新增後處理步驟：繼承 `processing/pipeline.py:TextProcessor`，實作 `process()`，然後在 pipeline 中 `.add()` 即可。
新增輸出方式：同樣繼承 `TextProcessor`（目前設計如此，之後可能獨立介面）。

## 已確立的設計決策（不要重做）

- **ASR 引擎**：選定 Qwen3-ASR 1.7B 作為 MVP 引擎，佔 ~3.9GB VRAM
- **繁中轉換策略**：ASR prompt 引導效果有限，主要靠 OpenCC s2twp + 自訂詞表後處理
- **Qwen3-ASR API**：引導文字用 `context` 參數（不是 `prompt`），音訊可傳 `(np.ndarray, sample_rate)` tuple
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

v0.1 — MVP 完成，CLI 可用。已驗證：
- 模型載入 ✓、音檔辨識 ✓、麥克風辨識 ✓、簡轉繁 ✓、台灣用語替換 ✓

## 後續方向（尚未開始）

- **VAD 切段** — 用 Silero VAD 將長音訊切成短段逐段辨識，改善長音訊速度
- **全域快捷鍵 + System Tray** — 不用開終端機，按快捷鍵直接錄音
- **即時串流辨識** — 邊講邊出字
- **多引擎支援** — SenseVoice、Whisper 可在 config 切換
- **Web UI 測試介面** — 瀏覽器介面，用於測試/展示/設定調整
- **詞表擴充** — 領域專用詞、OpenCC 過度轉換修正

## 注意事項

- 開發環境在 Windows 11，RTX 4060，Claude Code 跑在虛擬機中
- `uv run` 在虛擬機中可能有路徑問題，可直接用 `.venv\Scripts\python.exe` 替代
- `data/test_audio/` 裡有測試音檔（.m4a），不要 commit 到 git（已在 .gitignore 排除 *.m4a 的上層目錄）
- 模型權重不進 git（.gitignore 已排除 *.safetensors 等）
