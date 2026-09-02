# `docs/index.html` 落差清單（2026-09-03 盤點）

`docs/index.html` 最後一次更新是 2026-06-18（`783d5b4`），內容停在 v0.1.0；
專案現況是 v0.3。翻新本身屬 F 線（發布工程）範圍，本文件只盤點落差，不做改版，
避免翻新時再從頭讀一次 563 行 HTML。

證據等級：以下每條都是把頁面內容與現行 `config.yaml`／`CLAUDE.md`／原始碼比對得到
（`observed`）；「建議」欄是判斷（`derived`）。

## 1. 已經與事實不符（會誤導讀者）

| 位置 | 頁面寫的 | 現況 |
|---|---|---|
| L282、L371、L389、L550 | 預設引擎是 Qwen3-ASR-1.7B（~3.9GB VRAM） | 預設是 faster-whisper `large-v3-turbo`；Qwen 保留為可選研究引擎（2026-08-18 benchmark） |
| L284、L376、L445、L458-459 | OpenCC 用 `s2twp`（字形＋慣用詞一起轉） | 預設改為純字形 `s2t`；台灣詞彙本地化拆成 `localize_tw_terms`，**預設關閉**（2026-08-18 決策） |
| L435-446 | `config.yaml` 範例只有 asr／audio／processing／output 四段 | 現行有 12 個頂層區塊（另有 `vad`／`microphone`／`streaming`／`logging`／`recovery`／`history`／`hotkey`／`lifecycle`）；`engine: qwen` 與 `prompt:` 欄位名也已不符 |
| L389-392「ASR 引擎比較」 | Whisper 標「可替換」、SenseVoice／FireRedASR「待評估」 | 四引擎橫評已於 2026-06-29 收斂：FW 勝出並成為預設，Fun-ASR-Nano 與 whisper-zh-TW 微調版皆評測後不採用 |
| L494-530「後續計畫」 | v0.2 VAD、v0.3 快捷鍵／Tray、v0.4 串流「下一步／未來」 | 三者都已上線（VAD 切段、全域快捷鍵 + Tray、串流逐句辨識）；路線圖已改為 A–G 七條線 |
| L550 footer | `v0.1.0` | v0.3 |

## 2. 頁面完全沒提到的現有能力

- 串流模式（`streaming.py` + `audio/streaming_vad.py`）：Tray 的預設路徑。
- VAD 走 ONNX Runtime、錄音路徑不 import torch（2026-09-01）。
- 幻覺偵測與三層 fallback，以及短段的分級門檻。
- SQLite history（`data/logs/history.sqlite3`）、log 檢視器、審核佇列。
- 評測基礎（`src/asr_input/eval/`、`scripts/run_benchmark.py`，2026-09-03）。
- macOS 支援已立項（G 線，Phase 0 完成）。

## 3. 結構性問題（翻新時要一併決定）

- **頁面是第二份架構說明**，與 `CLAUDE.md`「架構」章節重複。依文件職責分工，
  正本在 `CLAUDE.md`；頁面若要保留模組樹與設計原則，會產生第二份需要同步的內容。
  建議翻新時把「會漂移的細節」換成連結，頁面只留對外的定位、能力概觀與安裝入口。
- **README 與頁面的讀者重疊**。若兩者都寫「怎麼裝怎麼跑」，等於再多一份會漂移的抄本。
  建議頁面定位為「對外展示」（截圖／流程動畫／設計取捨敘事），安裝步驟只連 README。
- **內容手寫在 HTML 裡**。目前每次改設定或架構都要手改 HTML，因此才漂了兩個半月。
  若翻新後仍要維持同步，值得考慮由 Markdown 產生，或至少把 config 範例改成
  「連到 repo 的 `config.yaml`」而不是抄一份。

## 4. 沒有的問題

頁面內沒有任何個人逐字稿、音訊路徑或使用者資料，示例文字都是通用句子；
repo 公開後不需要為它做額外的隱私處理。
