# macOS 支援評估與移植計畫（G 線）

本文件是 macOS 支援的專題規格正本：障礙清單、解法路線、分階段順序與待驗證的未知數。
路線狀態與跨線依賴見 [`docs/roadmap.md`](roadmap.md) 的 G 線。

## 證據等級說明

本文件的每項判斷標注來源，方便日後重新評估可信度：

- **[碼]** — 讀本 repo 的原始碼確認，2026-09-01。
- **[推測]** — 依套件的已知行為推論，**未在 macOS 上實測**（撰寫時沒有 mac 可用）。
  這類判斷在拿到機器後應優先驗證。

## 結論

核心 pipeline（VAD → ASR → processing → history）幾乎與平台無關，可移植。障礙集中在三處：

1. 依賴宣告與裝置設定（硬阻斷，但好修）
2. tray／全域快捷鍵／剪貼簿（需要一層平台抽象，工程量最大）
3. **以絕對秒數為單位的幻覺門檻**（會靜默改變辨識行為，不只是變慢——最容易被忽略）

**不建議拆成 Windows 與 macOS 兩個 repo。** 真正平台相依的只有下列少數幾處，其餘
（VAD、ASR、processing、streaming 狀態機、history、幻覺偵測）是共用的，而共用的部分
才是持續在改的部分。拆兩個 repo 等於每次改 streaming 邏輯都要同步兩邊。
採單一 repo + 平台抽象層，Windows 現有行為包成 `Windows*` 實作，語意不變。

## A. 硬阻斷：不改就裝不起來或跑不起來

| # | 位置 | 問題 | 解法路線 |
|---|------|------|---------|
| A1 | `pyproject.toml` 的 `[tool.uv.sources]` | torch／torchaudio 釘在 `pytorch-cu124` index，macOS 無此 wheel，`uv sync` 直接失敗 **[碼]** | 加 marker：`marker = "sys_platform != 'darwin'"`。更乾淨的做法是把 torch 降為 optional extra——VAD 已走 ONNX，torch 目前只剩 qwen 引擎與 torch VAD 後端在用 **[碼]** |
| A2 | `config.yaml` 的 `asr.device: cuda`／`compute_type: float16` | macOS 無 CUDA。faster-whisper 底層的 CTranslate2 在 macOS 只有 CPU 後端，沒有 Metal／MPS **[推測]** | 短期 `device: cpu` + `compute_type: int8`，並在 `asr/__init__.py:build_engine()` 加自動偵測。長期見 Phase 1 |
| A3 | `src/asr_input/output/clipboard.py` | 直接 `subprocess.run(["powershell", ..., "Set-Clipboard"])` **[碼]** | 抽平台層：macOS 用 `pbcopy`（以 stdin 餵入，順帶消掉目前那個引號跳脫的脆弱點） |
| A4 | `src/asr_input/tray.py` 的 `_silent_notify()` | `from pystray._util import win32` 並呼叫私有的 `icon._message`；在 macOS 上 import 即失敗 **[碼]** | 平台分支：macOS 走 `icon.notify()` 或 `osascript -e 'display notification'` |
| A5 | `src/asr_input/single_instance.py` | 用 `CreateMutexW`；非 win32 已 early-return 視為取得，**不會 crash，但 macOS 等於沒有單一實例保護** **[碼]** | 加 POSIX 分支：lock file + `fcntl.flock` |

`tray.py` 的 `_optout_ecoqos()` 已有 `sys.platform != "win32"` 保護，不需修改 **[碼]**。

## B. 裝得起來，但行為會不同

### B1. 時間門檻校準失效（優先度最高）

`streaming.hallucination_threshold_sec: 1.5` 與 `min_hallucination_audio_sec: 3.0` 是
**絕對秒數**門檻，而支撐它們的凍結回歸集 `tests/data/short_segment_cases.json`
是在 RTX 4060 + CUDA float16 上蒐集的（見 CLAUDE.md「短段幻覺改成分級門檻」）**[碼]**。

在 macOS CPU int8 下 RTF 預期差一個量級，正常段的轉錄時間會普遍超過 1.5s → 被判為疑似
幻覺 → 短於 3s 的段依現行分級規則會**直接被捨棄**。使用者觀察到的現象是「講短句沒反應」，
而不是「變慢」**[推測]**。

解法路線，**建議第一項為 (b)**：

- ~~**(a) 門檻從絕對秒改成 RTF 相對比**：判斷式改為 `dt / audio_sec > k × baseline_rtf`~~
  **2026-09-03 撤回。** 本文件原本把這列為建議第一項，與 `docs/roadmap.md` E 線的既定
  決策直接衝突——該處明文寫「不得只因理論上可能受負載影響，就改用 transcribe/audio
  ratio、弱化或移除這個訊號」。以 roadmap 記載的實測點驗算，比例訊號在兩個方向都比
  絕對時間差 **[derived，資料來源為 roadmap E 線既有實測記錄]**：

  | 段 | 音訊 | 轉錄 | 絕對 >1.5s | 比例 |
  |---|---|---|---|---|
  | 正常段中位數 | — | ~0.5s | 過 | — |
  | 錯誤密集段 | 21.8s | 2.5s | 抓到 | 0.11 → 漏抓 |
  | 嚴重重複幻覺 | 25.2s | 3.5s | 抓到 | 0.14 → 漏抓 |
  | 短段幻覺 | 0.26s | 18.32s | 抓到 | 70 → 抓到 |

  機制上的理由：失敗模式是 beam search 卡進重複迴圈，**跑掉的解碼時間由 max token
  長度決定，與輸入音訊長度幾乎無關**。除以音訊長度等於把訊號本身 normalize 掉——長段
  幻覺被大分母稀釋成看似正常，正常短段又因固定開銷（0.3s 音訊花 0.4s，比例 1.3）被
  誤殺。
- **(b) 保留絕對秒但移進 device profile**，macOS 上另外量測重建回歸集。成本確實較高，
  每換一台機器就要重量一次基準，但它保留了絕對時間這個已驗證有效的訊號，是目前唯一
  與 roadmap 決策相容的路線。
- (c) 引擎級關閉：`transcription_latency_is_quality_signal` 這個旗標已經存在
  （`src/asr_input/asr/base.py` 為 `True`、`src/asr_input/asr/openrouter.py` 為 `False`）**[碼]**，若 macOS 換引擎可
  沿用此機制關掉延遲訊號。代價是失去這層防護。

### B2. tray 與 hotkey 的主執行緒衝突（工程風險最大）

macOS 上 pystray 的 `Icon.run()` 需在主執行緒跑 NSApplication run loop；pynput 的 macOS
listener 同樣依賴 CFRunLoop。目前架構是在 tray ready callback 裡才啟動 hotkey listener
（`tray.py` 的 `_listen_hotkey()`）**[碼]**，兩者能否共存**無法從碼上斷定，必須實測**。

若衝突，路線是把 `TrayApp` 的狀態機與 UI 層拆開（目前混在同一個 class），macOS 端 UI 改用
rumps 或直接 PyObjC，hotkey 改用 Carbon `RegisterEventHotKey` 或
`NSEvent.addGlobalMonitorForEvents`。這是唯一可能需要動到 `tray.py` 結構的項目。

附帶：`tray.py` 的 `_KEY_MAP` 沒有 `cmd` 鍵名（只有 `"win": Key.cmd`）**[碼]**，且預設的
`ctrl+shift+space` 在 macOS 與輸入法切換衝突機率高 **[推測]**。

### B3. 權限（TCC）

麥克風、輔助使用、輸入監控三項都需授權。從終端執行會繼承終端 app 的授權；打包後需要
`NSMicrophoneUsageDescription` 等 plist 欄位 **[推測]**。麥克風被拒時會拿到全零音訊，
正好會被現有的 `microphone.near_zero_rms` 偵測到 **[碼]**。

### B4. App Nap

macOS 沒有 EcoQoS，但背景 app 有 App Nap／QoS 降級。是否會重演 Windows 上「隱藏視窗被
判背景、GPU 解碼慢 2.4×」的問題屬推測；若有，對應解法是
`NSProcessInfo.beginActivityWithOptions` **[推測]**。

啟動延遲的既有結論（`import torch` 18.1s vs ONNX 後 0.4s）是 Windows 實測值，方向大概率
成立但數字須重測。

## C. 次要與工具鏈

- `scripts/start_tray.bat`、`.vbs`、`create_desktop_shortcut.ps1` 無 macOS 對應，需
  `.command` 或 LaunchAgent plist **[碼]**。
- `src/asr_input/output/history_store.py` 的 `DEFAULT_DB_PATH = Path("data/logs/history.sqlite3")` 是
  **相對 cwd** 的路徑 **[碼]**。用 LaunchAgent 啟動時 cwd 不是專案目錄，DB 會建在別處。
  （這在 Windows 用 `.vbs` 啟動時可能已是潛在問題，值得一併查。）
- 相依無 macOS 障礙：opencc-python-reimplemented 是純 Python，onnxruntime／soundfile／
  numpy 都有 arm64 wheel，keyring 在 macOS 走 Keychain **[推測]**。
- 全域開發慣例中的 MSIX 虛擬化、Big5 編碼、`.vbs`／`.ps1` 編碼問題在 macOS 不存在。

## D. 分階段順序

**Phase 0 — 讓 CLI 路徑通**（不需要 mac 就能完成，不影響 Windows 現行行為）
處理 A1、A2、A3，並把平台相依收攏成 `platform/` 一層。
驗收：`scripts/test_audio_file.py` 能跑出結果、`pytest` 全綠。不碰 tray。

**Phase 1 — macOS 引擎決策**
實測 faster-whisper CPU int8 的 RTF。若不可接受，候選是 mlx-whisper（Apple Silicon GPU，
最省事）或 whisper.cpp（Metal + Core ML）。
**注意**：CLAUDE.md 的「Whisper initial_prompt 能引導繁體+標點」是在 faster-whisper 上
驗證的，換引擎必須重驗——whisper.cpp 與 mlx-whisper 都有語意對應的 prompt 參數，但輸出
機率分布不同，不能直接沿用結論。

**Phase 2 — B1 門檻 RTF 相對化**
建議即使不移植 macOS 也做，它修的是一類通病。

**Phase 3 — tray／hotkey macOS 化**（B2、A4、A5）
最貴，待前面都通了再評估。

**Phase 4 — 打包**：權限 plist、簽章、LaunchAgent。與 roadmap F 線（發布工程）有重疊，
啟動前先確認邊界。

## E. 必須實測才能定案的未知數

1. pystray + pynput 在 macOS 主執行緒能否共存（決定 Phase 3 是「加分支」還是「重寫 UI 層」）。
2. faster-whisper CPU int8 在目標 mac 上的實際 RTF（決定要不要換引擎）。
3. App Nap 對背景 tray 的實際影響。
4. 換引擎後繁體 prompt 引導是否仍成立。
