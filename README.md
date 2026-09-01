# 🎤 ASR Input

本地語音輸入系統 — 台灣繁體中文。

預設在本機 GPU 上做語音辨識，把簡體字形轉成繁體後複製到剪貼簿隨處貼上。
本地引擎全程離線；另有預設停用的 OpenRouter 實驗引擎可手動開啟。

## 功能

- **語音辨識** — 本地 ASR 引擎，可在 `config.yaml` 切換（預設 faster-whisper）
- **繁中字形轉換** — OpenCC `s2t`，不擅自改寫使用者說出的地區詞彙
- **台灣詞彙本地化** — 獨立選項，預設關閉
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

### OpenRouter（選用）

OpenRouter adapter 預設停用，也不從 `config.yaml` 讀明文 key。若要啟用，先將 key 存入
Windows Credential Manager：

```powershell
.venv\Scripts\python.exe -c "import getpass,keyring; keyring.set_password('asr-input/openrouter','api-key',getpass.getpass('OpenRouter key: '))"
```

再參考 `config.yaml` 內的 OpenRouter 區塊切換引擎。也可暫時使用
`OPENROUTER_API_KEY` 環境變數；請勿把 key 寫入 repo。雲端音訊會傳給 OpenRouter 與實際模型
provider，adapter 預設要求 Zero Data Retention。

## 更多文件

- **架構與設計決策** — [`CLAUDE.md`](CLAUDE.md)（含模組職責、擴展方式、已確立決策）
- **主路線圖（SSOT）** — [`docs/roadmap.md`](docs/roadmap.md)
- **近期執行入口** — [`TODO.md`](TODO.md)
- **標準 Benchmark** — [`docs/asr-benchmark-proposal.md`](docs/asr-benchmark-proposal.md)
- **Faster Whisper／CT2 計畫** — [`docs/faster-whisper-customization-plan.md`](docs/faster-whisper-customization-plan.md)
- **VAD E0–E6 實驗** — [`docs/vad-experiment-plan.md`](docs/vad-experiment-plan.md)
- **變更紀錄** — [`CHANGELOG.md`](CHANGELOG.md)

## 技術棧

faster-whisper · Qwen3-ASR · OpenCC · PyTorch CUDA · uv · ruff

## 授權

本專案以 [MIT License](LICENSE) 釋出。

依賴套件的授權（讀自安裝後的套件 metadata，2026-09-01）：faster-whisper、CTranslate2、
silero-vad、sounddevice、onnxruntime、PyYAML、keyring、Pillow 為 MIT 系；torch、soundfile、
numpy 為 BSD-3；opencc-python-reimplemented、qwen-asr 為 Apache-2.0。

其中 **pystray 與 pynput 為 LGPLv3**。本專案以 pip 安裝、未修改其原始碼、且以原始碼形式散布，
因此不影響本專案自身的授權；若日後打包成單一執行檔散布，需另行確認 LGPL 的相應義務。

模型權重不隨本 repo 散布，各自適用其上游授權。
