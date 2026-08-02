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

D/E 可在獨立 worktree 平行準備，但需遵守 roadmap 的共享檔案與整合邊界；F 暫緩。

## 維護規則

- 未來方向只改 `docs/roadmap.md`。
- 專題參數與驗收只改其規格文件。
- 完成結果寫入 `CHANGELOG.md` 並從 roadmap 更新狀態。
- 現行架構或已確立決策變更時才更新 `CLAUDE.md`。
