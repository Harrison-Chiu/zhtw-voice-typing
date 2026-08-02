# VAD E0–E6 標準實驗規格

## 決策問題

確認非神經網路 VAD 能否在本專案的真實使用情境中可靠取代 Silero；若不能，評估
Silero ONNX 是否能在維持品質下移除 PyTorch 依賴。Silero 是 baseline，不是真值。

本實驗依賴 `docs/asr-benchmark-proposal.md` 的 sample ID、gold、環境快照與報告格式。

## 共同比較原則

- 輸入：16 kHz、mono、float32；WebRTC adapter 另轉 16-bit PCM。
- 主比較採 32 ms frame、32 ms hop，與目前 Silero 一致；16 ms hop/50% overlap 只作
  後續 ablation，不混入主排名。
- DC removal 納入共同基線；80–120 Hz high-pass 預設關閉，另作 ablation，避免把
  濾波收益誤算給 speech score。
- 所有方案輸出帶時間戳的 frame score/decision，再交給同一個以毫秒為單位的狀態機。
- 共用 onset、hysteresis、pre-roll、hangover、最短語音、30 秒上限與既有 silence ramp。
- WebRTC 使用原生 10/20/30 ms frame；adapter 保持相同牆鐘時間軸，不補成 32 ms。
- 只在 dev set 選參數；凍結後只跑一次 test set。禁止依 test 結果回頭調參。

## 候選

| ID | 方法 | 角色 | 是否可產品化 |
|---|---|---|---|
| Baseline-JIT | 現況 Silero JIT | 品質/行為基準 | 是 |
| Baseline-ONNX | 同版 Silero ONNX | 依賴與效能候選 | 是 |
| E0 | 固定 RMS threshold | 無適應能力下限 | 是，但預期淘汰 |
| E1 | Otsu / k-means / 2-GMM 全段能量群聚 | non-causal oracle，測能量可分上限 | 否 |
| E2 | rolling percentile noise floor + relative dB | 第一個簡單 causal 候選 | 是 |
| E3 | rolling median + MAD + 最小 relative dB | robust causal 候選 | 是 |
| E4 | fast/slow asymmetric noise tracking | 環境變動候選 | 是 |
| E5 | E2/E4 + voice-band、低頻、flatness、ZCR 等規則 | 最強自製非模型候選 | 是 |
| E6 | WebRTC VAD mode 0–3 | 成熟非神經參考 | 是 |

原始提案中的參數範圍保留為候選庫，但不做所有 score × state-machine 的笛卡兒積。

## 資料與標註

### 第一輪：既有資料

從 session log 分層挑選低 RMS、短/長句、fallback、長停頓、贅字、中英混合、已知錯誤、
背景聲與 Silero 異常切段。既有檔多為 Silero 已切出的語音，只能用於句首/尾、增益與
下游 Whisper 診斷，不能單獨證明 noise false-trigger 表現。

### 必要 fixture：連續壓力錄音

在任何「取代 Silero」決策前，補一段至少 3–5 分鐘的連續錄音，包含：

- 正常、小聲、遠距、一開始立即講話、長停頓與長句。
- 風扇開關、鍵盤、滑鼠、桌面碰撞、說話與鍵盤同時發生。
- 背景音量途中改變與麥克風增益變化。

人工插入靜音可作合成 regression，但不能取代真實 noise-only 與環境轉換錄音。

人工標註至少包含 speech interval、允許不確定區間、事件類型；重大候選分歧以音訊
裁決，不能以 Silero decision 當 gold。

## 分階段搜尋

### Phase 1 — Score 可視化

輸出 waveform、log RMS、noise floor、relative/robust score、Silero probability、各候選
decision 與切段邊界。先確認 score 是否能分離 speech/noise。

### Phase 2 — Score 參數粗掃

先固定一組合理的共同狀態機，只掃各 E 方法本身的少量參數。淘汰漏語音、誤觸、對
增益敏感或參數稍動就崩潰的設定。每個方法只保留少數 Pareto 候選。

### Phase 3 — 狀態機微調

只對留下的 Pareto 候選掃 onset 3/5 frames、pre-roll 100/150/200 ms、hangover
300/500/800/1000 ms 與 min speech 150/250/400 ms。使用 successive halving 或逐階淘汰，
不跑無界全排列。

### Phase 4 — 分歧人工檢查

HTML 集中顯示 Silero/候選不一致、邊界差 >200/500 ms、合併/拆分差異、疑似鍵盤/風扇
誤觸；可播放分歧時間窗並記錄裁決。

### Phase 5 — Whisper 下游

只有 Pareto front 候選進入 ASR。比較 raw/processed transcript、句首尾漏字、幻覺、
空輸出、標點、切段上下文損失與 end-to-end latency。

### Phase 6 — 壓力與不變性

加入 -18/-12/-6/+6/+12 dB、白噪音/實錄風扇、鍵盤/碰撞、clipping、遠距與小聲。
合成變換另列，不與原始真實錄音平均成單一分數。

## 指標

- Frame-level precision/recall 只作診斷；主要報 event/speech-duration recall/precision。
- Missed speech duration、false-trigger duration、noise-only segment count。
- 句首/句尾裁切 P50/P95、開始偵測與結束觸發 latency。
- Segment count、合併/拆分、過短/過長比例。
- 下游 CER/MER、deletion、hallucination、empty rate。
- VAD 每小時音訊 CPU time、P95 callback time、RAM、依賴/安裝容量。
- 不同增益、背景與事件標籤的分層結果，不只報總平均。

## 取代 Gate

Gate 在第一輪人工標註後，以 baseline 分布凍結具體 non-inferiority margin；至少符合：

1. Speech-duration recall 與句首/尾裁切對 Silero 不構成實質退步，95% CI 通過預先定義
   的 non-inferiority margin。
2. False-trigger duration 與 noise-only segments 不高於 Silero 的允許 margin。
3. 下游 CER/MER、deletion、hallucination 與 empty rate無系統性退化。
4. 原始與增益/背景壓力條件皆不需逐錄音重調門檻。
5. CPU、啟動時間或依賴容量至少一項有事先定義的實質改善，且其他資源沒有不可接受
   的退步。

若 E5/E6 無法通過，停止增加手工規則，保留 Silero並比較 JIT/ONNX；E1 永遠不進產品。

## 交付物

- `experiments/vad/` runner、candidate adapters、manifest 與結果 schema。
- VAD-only JSON、可播放 HTML、Pareto 與分層圖表。
- Pareto 候選的完整 ASR benchmark 報告。
- 採用 Silero JIT、Silero ONNX、E2–E6 或維持現況的決策紀錄。

