# ASR Input — 執行入口

專案主路線、狀態、依賴與待決策的單一事實來源是
[`docs/roadmap.md`](docs/roadmap.md)。本檔不另存一份完整 backlog，避免計畫分叉。

## 下一個執行批次

1. **A 評測基礎**：依 [`docs/asr-benchmark-proposal.md`](docs/asr-benchmark-proposal.md)
   建立 manifest/schema、Smoke/Core 樣本、指標 runner 與第一版 dashboard。
2. **B ASR／CT2**：A baseline 凍結後，先跑解碼參數與 fallback 決定性，再評估
   compute type、dynamic batching 與 diagnostics。
3. **C VAD／音訊**：與 A 共用 fixture/schema，依
   [`docs/vad-experiment-plan.md`](docs/vad-experiment-plan.md) 執行 E0–E6、Silero
   JIT/ONNX 與 codec 決定性實驗。
4. **G-Phase 0 macOS 前置**：依 [`docs/macos-port-plan.md`](docs/macos-port-plan.md)
   的 Phase 0 處理 torch 的平台 marker、`device`/`compute_type` 自動偵測、剪貼簿抽平台層，
   並把平台相依收攏成 `platform/`。不需要 mac、不影響 Windows 現行行為。

D/E 可在獨立 worktree 平行準備，但需遵守 roadmap 的共享檔案與整合邊界；F 暫緩。
G-Phase 3 會動 `tray.py`，與 D 線衝突，不可同時進行。

## 待辦（非程式碼）

- 尚未建立 git remote。公開發布前置（移除逐字稿語料、去識別化、LICENSE）已於
  2026-09-01 完成，repo 可推送。建立 GitHub public repo 後：
  `git remote add origin <url>` 再 `git push -u origin main`。
  認證走既有的 Git Credential Manager（HTTPS，首次推送會開瀏覽器登入），
  本機未安裝 `gh`、`~/.ssh` 也沒有 GitHub 設定。

## 維護規則

- 未來方向只改 `docs/roadmap.md`。
- 專題參數與驗收只改其規格文件。
- 完成結果寫入 `CHANGELOG.md` 並從 roadmap 更新狀態。
- 現行架構或已確立決策變更時才更新 `CLAUDE.md`。
