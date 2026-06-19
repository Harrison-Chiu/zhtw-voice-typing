# 🎤 ASR Input

本地語音輸入系統 — 台灣繁體中文。

在本機 GPU 上做語音辨識，自動把結果轉成台灣繁體中文用語，複製到剪貼簿隨處貼上。
全程離線、不上傳雲端。

## 功能

- **語音辨識** — 本地 ASR 引擎，可在 `config.yaml` 切換（預設 faster-whisper）
- **台灣繁中轉換** — OpenCC 簡轉繁 + 自訂台灣用語詞表
- **全域快捷鍵** — System Tray 常駐，一鍵開始/停止錄音
- **剪貼簿輸出** — 辨識結果自動複製，隨處貼上

## 系統需求

- Windows 10/11
- NVIDIA GPU（建議 8GB+ VRAM，已測試 RTX 4060）
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 套件管理工具

## 安裝

```bash
git clone <repo-url>
cd asr-input
uv sync          # 自動建立虛擬環境並安裝依賴
```

首次執行會自動從 HuggingFace 下載模型（數 GB），之後有快取。

## 使用

```bash
uv run python -m asr_input.tray   # System Tray 版（全域快捷鍵，日常推薦）
uv run python -m asr_input.main   # CLI 版（終端機互動）
```

設定（引擎、裝置、語言、後處理、串流參數）都在 `config.yaml`，
自訂台灣用語詞表在 `data/tw_dict.yaml`。

## 更多文件

- **架構與設計決策** — [`CLAUDE.md`](CLAUDE.md)（含模組職責、擴展方式、已確立決策）
- **待辦與藍圖** — [`TODO.md`](TODO.md)

## 技術棧

faster-whisper · OpenCC s2twp · PyTorch CUDA · uv · ruff
