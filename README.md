# 🎤 ASR Input

本地語音輸入系統 — 台灣繁體中文

在 NVIDIA GPU 上運行 Qwen3-ASR 語音辨識模型，自動將辨識結果轉換為台灣繁體中文用語。

## 功能

- **語音辨識** — Qwen3-ASR 1.7B，支援中文（含方言）
- **台灣繁中轉換** — OpenCC 簡轉繁 + 自訂台灣用語詞表
- **剪貼簿輸出** — 辨識結果自動複製到剪貼簿，隨處貼上
- **模組化設計** — ASR 引擎、後處理、輸出方式皆可抽換

## 系統需求

- Windows 10/11
- NVIDIA GPU（建議 8GB+ VRAM，已測試 RTX 4060）
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 套件管理工具

## 安裝

```bash
# 1. Clone 專案
git clone <repo-url>
cd asr-input

# 2. 安裝依賴（uv 會自動建立虛擬環境）
uv sync

# 3. 首次執行會自動從 HuggingFace 下載模型（~3.4GB）
```

## 使用方式

```bash
uv run python -m asr_input.main
```

啟動後：

1. 等待模型載入（首次約 30-60 秒）
2. 按 **Enter** → 開始錄音 🎤
3. 講完話 → 按 **Enter** 停止
4. 辨識結果自動轉為台灣繁中，複製到剪貼簿 📋
5. 輸入 `q` + Enter 退出

### 用音檔測試（不需要麥克風）

```bash
# 把 .m4a/.wav 檔放進 data/test_audio/
uv run python test_audio_file.py
```

## 架構

```
麥克風 → AudioCapture → ASREngine → TextProcessing → Output
```

| 模組 | 檔案 | 說明 |
|------|------|------|
| Audio Capture | `audio/capture.py` | 麥克風收音（16kHz） |
| ASR Engine | `asr/qwen.py` | Qwen3-ASR 1.7B 推理 |
| Text Processing | `processing/` | OpenCC s2twp + 台灣用語替換 |
| Output | `output/clipboard.py` | 剪貼簿輸出 |

每個模組有抽象介面，可獨立替換。詳細架構說明見 `docs/index.html`。

## 設定

編輯 `config.yaml`：

```yaml
asr:
  engine: qwen
  model_id: "Qwen/Qwen3-ASR-1.7B"
  device: cuda
  language: "Chinese"
  context: "以下是台灣繁體中文的語音轉錄。"

processing:
  opencc_config: s2twp
  tw_dict_path: data/tw_dict.yaml
```

### 自訂台灣用語詞表

編輯 `data/tw_dict.yaml` 新增替換規則：

```yaml
"中國用語": "台灣用語"
"人工智能": "人工智慧"
"代碼": "程式碼"
```

## 開發

```bash
uv run ruff check src/    # 程式碼檢查
uv run ruff format src/   # 自動格式化
```

## 技術棧

- **ASR**: [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 1.7B
- **繁中轉換**: [OpenCC](https://github.com/BYVoid/OpenCC) s2twp
- **環境管理**: [uv](https://docs.astral.sh/uv/)
- **程式碼品質**: [ruff](https://docs.astral.sh/ruff/)
- **GPU**: PyTorch CUDA 12.4
