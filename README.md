# 🎤 ASR Input

本地語音輸入系統 — 台灣繁體中文。

*A local, offline voice typing tool for Taiwanese Traditional Chinese (zh-TW).
Runs faster-whisper on your own GPU, global hotkey, output to clipboard.*

預設在本機 GPU 上做語音辨識，把簡體字形轉成繁體後複製到剪貼簿隨處貼上。
本地引擎全程離線；另有預設停用的 OpenRouter 實驗引擎可手動開啟。

## 為什麼有這個專案

一開始的問題大概是這樣：

> 「我這台電腦明明就有顯卡，有沒有辦法直接在自己電腦上跑語音輸入？
> Windows 內建的那個、還有另外裝的那幾套，用起來都沒有想像中好。
> 而且我要的是繁體中文。」

真的做下去之後，時間多半花在這幾件事上：

- **短音訊會生出無關的字。** 0.26 秒的雜音轉出 3.33 秒的文字，內容是設定檔裡的 hotwords。
  幻覺偵測的閾值以一份凍結回歸集校準：143 筆真實短音訊，人工標註。
- **標點半形全形混用。** 試過 15 種以上的 prompt 與解碼參數組合都無效，改用依前後字元
  判斷語境的後處理才穩定。
- **冷啟動 18 秒。** 瓶頸是 VAD 為了跑 Silero 而載入 PyTorch。改用 ONNX Runtime 後，
  錄音路徑不再 import torch，啟動到可錄音約 0.4 秒。
- **無視窗啟動時解碼慢 2.4 倍。** Windows 11 對背景行程套用 EcoQoS。單執行緒 CPU 探針
  量不到，需要真實解碼負載才觀察得到。

預設只做字形轉換（OpenCC `s2t`），不改寫說出口的詞彙——「接口」不會變成「介面」。
詞彙在地化是獨立選項，預設關閉。

實驗數據、否定掉的假設與淘汰的路線都在 [`CLAUDE.md`](CLAUDE.md)。
作者每天用它跟 Claude Code 講話，這份 README 也有段落是講出來的。

## 功能

- **語音辨識** — 本地 ASR 引擎，可在 `config.yaml` 切換（預設 faster-whisper）
- **繁中字形轉換** — OpenCC `s2t`，不擅自改寫使用者說出的地區詞彙
- **台灣詞彙本地化** — 獨立選項，預設關閉
- **全域快捷鍵** — System Tray 常駐，一鍵開始/停止錄音
- **剪貼簿輸出** — 辨識結果自動複製，隨處貼上
- **分段串流辨識** — VAD 依靜音門檻切段，ASR worker 與錄音執行緒並行消化佇列。
  非逐字即時辨識，段落內不輸出增量假設，停止後的延遲下限為最後一段的解碼時間
- **非阻塞生命週期** — 錄音不等模型載入完成，未辨識工作進 FIFO 佇列；閒置 30 分鐘
  卸載模型釋放顯存，下次錄音自動重載
- **幻覺防護** — 轉錄耗時超過門檻時觸發 RMS 正規化與遞減靜音門檻重切；短於
  `min_hallucination_audio_sec` 且重切無效的段落捨棄文字並保留音訊與各次嘗試
- **狀態圖示** — Tray 圖示以色相區分模型與錄音狀態，中央數字為已切出段數，
  右下角點表示佇列長度非零

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

預設只在桌面建一個 **ASR Input**，走 [`start_tray.vbs`](scripts/start_tray.vbs)：背景常駐、
無視窗，結束請用系統匣圖示右鍵。

選配參數：

| 參數 | 作用 |
|---|---|
| `-Autostart` | 另外在「啟動」資料夾放一份，開機自動常駐。走 `autostart` 模式：只載錄音與 CPU VAD，第一次錄音才載 Whisper，開機後不占顯存 |
| `-WithDebugShortcut` | 另外建「ASR Input (視窗)」，走 [`start_tray.bat`](scripts/start_tray.bat)，有視窗看得到 log（關視窗即結束） |
| `-Remove` | 移除本腳本建立過的捷徑 |

不加 `-WithDebugShortcut` 也能排錯——直接雙擊 `scripts\start_tray.bat` 即可，捷徑只是省一次
資料夾切換。無視窗版的輸出寫在 `data/logs/launcher.log`。

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
