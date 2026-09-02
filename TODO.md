# ASR Input — 執行入口

專案主路線、狀態、依賴與待決策的單一事實來源是
[`docs/roadmap.md`](docs/roadmap.md)。本檔不另存一份完整 backlog，避免計畫分叉。

## 下一個執行批次

1. **A 評測基礎**：manifest/schema、指標與 runner 已於 2026-09-03 落地
   （`src/asr_input/eval/` + `scripts/run_benchmark.py`，Smoke 層已跑通）。
   剩下的是 **gold transcript**——需要人聽音訊，與下方 E 線的人工審核是同一個瓶頸；
   dashboard 與 P95／VRAM 收集尚未做。規格見
   [`docs/asr-benchmark-proposal.md`](docs/asr-benchmark-proposal.md)。
2. **B ASR／CT2**：A baseline 凍結後，先跑解碼參數與 fallback 決定性，再評估
   compute type、dynamic batching 與 diagnostics。
3. **C VAD／音訊**：與 A 共用 fixture/schema，依
   [`docs/vad-experiment-plan.md`](docs/vad-experiment-plan.md) 執行 E0–E6、Silero
   JIT/ONNX 與 codec 決定性實驗。
4. ~~**G-Phase 0 macOS 前置**~~：2026-09-03 完成（torch 平台 marker、
   `device`／`compute_type` 解析、剪貼簿平台層、`paths.py`）。Phase 1 起需要實體 mac。
   規格見 [`docs/macos-port-plan.md`](docs/macos-port-plan.md)。
5. **人工審核（A／E 共同瓶頸）**：`scripts/build_review_queue.py` 產出的審核佇列
   （87 段候選 + 22 段對照）一旦有人聽過，同時解開 E 線訊號的 precision／recall
   與 A 線的 gold。這是目前擋住最多下游工作的單一項目。

D/E 可在獨立 worktree 平行準備，但需遵守 roadmap 的共享檔案與整合邊界；F 暫緩。
G-Phase 3 會動 `tray.py`，與 D 線衝突，不可同時進行。

## 待辦（非程式碼）

- **remote 已建立並公開**：`https://github.com/Harrison-Chiu/asr-input`（2026-09-01 推送，
  已用匿名 `ls-remote` 驗證為 public，遠端 121 檔無音訊／權重／log）。認證走既有的
  Git Credential Manager（HTTPS），本機未安裝 `gh`、`~/.ssh` 也沒有 GitHub 設定。
- **待做（需在 GitHub 網頁操作，本機憑證只夠 push）**：
  1. repo 改名為 `zhtw-voice-typing`（已決定；舊網址會 301 轉址），改完更新本機 remote URL
  2. 設定 About 的 description 與 topics（`speech-to-text` `whisper` `faster-whisper`
     `traditional-chinese` `zh-tw` `taiwan` `voice-input` `voice-typing` `asr` `offline`
     `local-first` `windows` `python`）——repo 搜尋比對名稱／description／topics，
     README 內文預設不在比對範圍
  3. 推送本機領先的 commit

## 維護規則

- 未來方向只改 `docs/roadmap.md`。
- 專題參數與驗收只改其規格文件。
- 完成結果寫入 `CHANGELOG.md` 並從 roadmap 更新狀態。
- 現行架構或已確立決策變更時才更新 `CLAUDE.md`。
