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
uv run python -m asr_input.tray   # System Tray 版（主動啟動，背景預載 Whisper）
uv run python -m asr_input.tray --startup-mode autostart  # 未來開機啟動用，首次錄音才載模型
uv run python -m asr_input.main   # CLI 版（終端機互動）
```

### 桌面捷徑啟動（免打指令）

不想每次開終端機打指令，可建桌面捷徑，雙擊即啟動：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1
```

會在桌面建兩個捷徑：

- **ASR Input** — 走 [`start_tray.vbs`](scripts/start_tray.vbs)，背景常駐、無視窗（日常用；結束請用系統匣圖示右鍵）
- **ASR Input (視窗)** — 走 [`start_tray.bat`](scripts/start_tray.bat)，有視窗看得到 log（排錯用；關視窗即結束）

改程式內容不需重建捷徑——捷徑只負責啟動，python 每次都讀最新程式碼。只有專案資料夾搬移/改名時才需重跑此腳本。

設定（引擎、裝置、語言、後處理、串流參數）都在 `config.yaml`，
自訂台灣用語詞表在 `data/tw_dict.yaml`。

## 更多文件

- **架構與設計決策** — [`CLAUDE.md`](CLAUDE.md)（含模組職責、擴展方式、已確立決策）
- **主路線圖（SSOT）** — [`docs/roadmap.md`](docs/roadmap.md)
- **近期執行入口** — [`TODO.md`](TODO.md)
- **標準 Benchmark** — [`docs/asr-benchmark-proposal.md`](docs/asr-benchmark-proposal.md)
- **Faster Whisper／CT2 計畫** — [`docs/faster-whisper-customization-plan.md`](docs/faster-whisper-customization-plan.md)
- **VAD E0–E6 實驗** — [`docs/vad-experiment-plan.md`](docs/vad-experiment-plan.md)
- **變更紀錄** — [`CHANGELOG.md`](CHANGELOG.md)

## 技術棧

faster-whisper · OpenCC s2twp · PyTorch CUDA · uv · ruff
