# ASR Input 主路線圖

本文件是專案未來工作的單一事實來源（SSOT）。它只管理路線、狀態、依賴、責任邊界
與決策；各項實驗的參數和驗收細節只寫在對應規格文件，不在此複製。

## 文件責任

| 文件 | 唯一責任 |
|---|---|
| 本文件 | A–F 路線、優先順序、狀態、跨線依賴與待決策 |
| `TODO.md` | 指向本文件的近期執行入口，不另存一份完整 backlog |
| `docs/asr-benchmark-proposal.md` | A 線資料、gold、指標、runner 與報告規格 |
| `docs/faster-whisper-customization-plan.md` | B 線 Faster Whisper／CT2 修改層級與候選技術 |
| `docs/vad-experiment-plan.md` | C 線 E0–E6 實驗與取代 Silero 的 gate |
| `CLAUDE.md` | 現行架構、已落地能力與已確立的技術決策 |
| `CHANGELOG.md` | 已完成且已提交的變更 |
| `README.md` | 使用者安裝、使用方式與文件入口 |

狀態只有：`ready`（規格足以執行）、`needs-decision`、`exploration`、`deferred`、
`done`。執行結果完成後移入 CHANGELOG；不在多份文件同步勾選。

## 已確立共識

- 互動式預設維持 faster-whisper `large-v3-turbo` sequential。
- batching 先限於離線／佇列；不得用漏段換吞吐。
- 先完成 A 線量尺，再採用 B/C 的任何產品變更。
- `外餐→外參` 是使用者確認的全域個人規則；「表點」使用完整詞組後處理。
- Gold 採多來源預標註與風險分層，只由人工裁決高價值分歧。
- 優先使用 faster-whisper／CT2 公開能力；有最小重現與 benchmark 證據才 fork。
- 大型 UI、模型微調與 CT2 kernel 修改不是近期前置工作。

## 路線與順序

```text
A 評測基礎（共同前置）
├─ B ASR／CT2 優化
└─ C VAD／音訊

D 啟動／Tray／生命週期 ─┐
E Log／回饋／歷史介面 ──┴─ 可在邊界凍結後平行

F 發布工程（確定對外發布時啟動）
```

### A — 評測基礎

狀態：`ready`，第一優先。

近期交付：

1. Smoke/Core/Stress manifest 與 config/environment fingerprint。
2. 多引擎預標註、風險分層與雙 target gold。
3. CER/MER、錯誤類型、決定性、P95 latency、RTF 與 VRAM。
4. JSON schema、Markdown 摘要與互動 HTML dashboard。
5. 保存 baseline，讓 B/C 候選可用同一 gate 比較。

詳細規格：`docs/asr-benchmark-proposal.md`。

### B — ASR／CT2 優化

狀態：`ready after A`。

順序：

1. 解碼參數與 temperature fallback 決定性。
2. compute type（含 `int8_float16`）。
3. VAD 片段 dynamic batching、length bucketing、worker/queue。
4. token/segment diagnostics。
5. 確認 API 缺口才 fork faster-whisper；profiler 證明 kernel 瓶頸才研究 CT2 fork。

延後研究：更多標點 prompt/suppress tokens、台灣語料微調/LoRA、Fun-ASR 繁體鷹架補測、
SenseVoice 等新引擎。既有模型橫評已收斂，沒有新需求或 A 線證據時不重啟模型競賽。

詳細規格：`docs/faster-whisper-customization-plan.md`。

### C — VAD／音訊

狀態：`ready after A fixture/schema`。

包含：

1. E0–E6：固定能量、oracle、自適應 noise tracking、robust statistics、頻譜規則與
   WebRTC，對照 Silero JIT/ONNX。
2. 連續壓力錄音、人工語音區間標註、VAD-only sweep 與 Whisper 下游測試。
3. Codec 嚴格逐字交叉測試；先處理 production fallback 不決定性。
4. 只有通過 non-inferiority gate 才取代 Silero；否則保留 Silero 並優先評估 ONNX。

2026-09-01：ONNX 評估完成並已上線為預設（`vad.backend`）。這是同一份 Silero v5 權重換 runtime，
不是換 VAD，因此不套用上述 non-inferiority gate；改用的理由是啟動延遲（錄音路徑不再 import
torch，tray 到「可錄音」約 0.4s）。等價性證據見 `experiments/compare_vad_backends.py` 與
`experiments/results/vad_backend_compare_*.json`：498 秒實際音檔、逐窗機率最大差 4.7e-06、
門檻決策零翻轉、切段邊界完全相同。E0–E6 的替代演算法評估不受影響，仍待執行。

詳細規格：`docs/vad-experiment-plan.md`。

### D — 啟動／Tray／生命週期

狀態：`validation`。狀態機、主要生命週期實作與自動化測試已落地，正在做 Windows 實機情境
驗收與 16×16 Tray 可讀性收斂；未完成項目仍保留在本節，不以單元測試通過冒充整條路線完成。

已同意方向：

- 桌面捷徑啟動後應先建立 Tray 圖示與基本狀態，再於背景 import／初始化重量級元件；不得讓
  PyTorch、OpenCC、Whisper 或 VAD 的載入阻塞第一個可見回饋。啟動前段失敗也要留下可辨識
  的錯誤狀態，避免使用者因看不到反應而重複啟動。
- 啟動策略必須區分使用者意圖，不以「有沒有終端視窗」猜測。使用者從終端、桌面一般捷徑或
  除錯捷徑主動啟動時，Tray 與 CPU VAD ready 後立即在背景預載 Whisper；使用者已主動開程式，
  不應再把約數十秒的冷載延遲推到第一次錄音。未來由 Windows 開機自動啟動的項目則明確傳入
  `autostart`／lazy 模式，只載錄音與 CPU VAD，不占用模型顯存。兩種模式都必須先顯示 Tray，
  且模型尚未 ready 時仍可立即錄音、完整保留開頭。實際建立 Windows 開機啟動項目屬 F 線；
  D 線先提供可測試的明確啟動模式與桌面／終端的一般預載行為。
- 第一次按快捷鍵後立即錄音，同時在背景載入 Whisper；不得為等待模型而漏掉開頭音訊。
- 錄音先結束而模型尚未 ready 時，自動保留音訊並排隊，ready 後依序辨識；取消是額外操作，
  不是正常流程必答問題。
- 快捷鍵採按鍵鎖存／邊緣觸發：同一次實體按住期間，即使作業系統重複送出 key-down 也只觸發
  一次；按鍵放開後再次按下仍立即切換開始／停止，不使用會吞掉正常第二次操作的時間式 debounce。
- 錄音與辨識是獨立工作。模型載入中或前一段正在辨識時，仍可開始並完成下一段錄音；每次
  start→stop 形成獨立 FIFO job，ASR 單 worker 依序處理，完成一段就更新剪貼簿與通知。模型
  載入只允許一個共享中的 future，重複 job 不得重複載入模型。
- 獨立不代表延後辨識：模型已 ready 且沒有更早 job 時，每個主要 VAD 片段一切出就立即送入
  ASR；模型在錄音途中才 ready 時，先依序追趕已累積片段，再銜接後續即時片段。停止快捷鍵只
  flush 尾段並等待尚未完成工作，不得把整次錄音重新送入或等到停止後才開始逐段辨識。
- 多個 job 依序完成時，現有 `ClipboardOutput` 只會覆寫目前剪貼簿，不能假設每位使用者都已
  啟用 Windows 剪貼簿歷程。應用程式另保存文字型最近結果（預設 20 筆，不含音訊），Tray 只
  增加一個巢狀「最近結果」選單顯示最近 5 筆摘要，點選即複製；不把 5 筆平鋪在主選單，也不
  因此提前實作完整歷史 UI。筆數與停用選項保留為設定。
- Tray 圖示必須能直接區分未載入、載入中、可用、錄音中、辨識中與錯誤；tooltip/選單文字
  提供同一狀態的文字說明，避免只靠顏色辨識。
- 支援閒置卸載，初版預設為完成最後一次辨識後 30 分鐘；Tray 提供永不卸載、15/30/60
  分鐘及立即卸載。卸載後的下一次快捷鍵沿用相同的「立即錄音 + `ensure_model_loaded()` + FIFO」
  路徑自動重載，不要求使用者先進選單，讓釋放顯存前後的操作方式維持一致。
- 支援開機啟動與單一實例。直接輸出到游標位置維持暫緩，現階段輸出仍以剪貼簿為準。
- 載入失敗、CUDA 不可用與麥克風錯誤都必須進入可辨識且可重試的狀態，不得卡在載入中，
  也不得靜默丟棄已錄音訊。
- 啟動復原時，`pending` 與前次 `failed` 工作都自動排入 FIFO 並各重試一次，不要求先進 Tray
  手動按重試；同一次執行中新失敗的工作仍保留明確錯誤與手動重試入口，避免無限重試迴圈。
  工作日後重試成功時也必須保留原始 error history，不能因目前狀態改為 completed 就抹除診斷
  證據。
- 錄音中以低頻率動態圖示表示 callback 持續收到輸入，讓使用者不必開設定即可確認麥克風仍
  在工作。裝置 API 回報斷線或 stream error 時立即切到錯誤狀態；只有長時間近零但 callback
  正常時，顯示「未偵測到聲音／可能靜音」，不可把安靜直接誤判為斷線。callback status、連續
  frame 數與缺口也要送進開發版 log，避免 UI 只變色卻沒有可追查的 capture 證據。
- Windows Tray handle 的建立／替換／釋放必須序列化；音量、ASR partial、模型載入計時等純 UI
  observer 的例外不得穿透 PortAudio callback 或使 ASR worker 失敗。若即時 segment observer
  暫時失敗，停止時需由完整 capture 的主要 segment 清單補送，避免 UI 故障演變成漏辨識。
- 圖示各層只負責一種訊號：主圓以同一灰藍色系表示未載入／載入中、綠色表示 ready、紅色表示
  錄音；live ASR 開始處理片段時，即使 capture 仍開啟也短暫切成橘色，完成後回紅色，整次工作
  完成後回綠色。這些立即變化用來降低主觀等待感，但不得改變實際錄音、queue 或 worker 行為。
- 中央數字表示 VAD 已切出的主要語音片段數，讓使用者在 ASR 完成前就能確認錄音正在前進；
  pending backlog 另以單一小型 queue 點表示，精確數量留在 tooltip／選單，不在 16px 圖示塞
  第二組數字。中央 `!`／`×` 分別表示近零警告與硬體／stream error，不得只靠顏色。
- Tray 主圓以約 15×15 實際像素、單字元字級至少相當於 64 px 畫布的 44 px 為基準。麥克風
  輸入強弱以同一紅／橘色系的外緣亮度／脈動表示，只使用主圓外側原已保留的抗鋸齒空間，不得
  縮小主圓或中央文字，也不使用與紅／橘主狀態衝突的跳色藍環。
- 錄音停止後白圈立即出現，保留到該次錄音的尾段辨識完成；開始下一次錄音時立即清除，即使
  前次仍有 backlog 也不阻礙新的停止回饋。白圈、queue 點與錯誤符號是獨立訊號，可在語意同時
  成立時疊加。圖示仍須在實際 Windows 96 DPI、16×16 工作列與深／淺背景驗證；近零警告未通過
  麥克風情境驗收前，不得寫成「麥克風斷線」。
- 待辨識音訊的故障復原與完整診斷 log 是兩件事。可提供獨立開關，將尚未成功辨識的工作寫入
  暫存檔，成功後立即清除；異常結束後可於下次啟動恢復或刪除。初版預設開啟，但不設定自動
  到期時間；正常完成不留下復原音訊，關閉時只保留於記憶體，程式結束即放棄。
- 背景載入、模型 ready 後消化、閒置卸載後重載及異常重啟恢復共用同一個 job repository；
  復原關閉時 payload 在記憶體，開啟時使用故障安全暫存。queue 容量上限與接近上限的警告門檻
  暫不設定，留待 FLAC／音訊容量與 retention 實驗一起決定；無論最後門檻為何，都不得靜默刪除
  舊 job。

規格與實作需明確處理並測試：重複快捷鍵、載入中取消、卸載計時與新錄音競態、未來容量上限、
程式結束時尚未處理的音訊、麥克風中途斷線、啟動／轉錄／後處理／log 寫入執行緒例外，以及
模型載入失敗後的重試。任何 worker 失敗都要收斂到可見狀態並保留可恢復工作，不得只結束執行緒
後回傳部分結果。預熱後卸載僅列為實驗候選；只有實測證明能穩定縮短下次載入且沒有不合理的
記憶體或啟動成本，才考慮加入設定，不作為預設。

D 線除單元測試外，固定保留以下操作情境作 Windows 實機驗收；每次 Tray／生命週期大改都依序
走完並記錄結果，避免只驗個別函式而漏掉使用流程：

1. 冷啟動立即錄音：圖示先出現；模型未 ready 仍完整錄到開頭，途中 ready 後追趕並銜接即時
   辨識，停止只等待尾段。
2. ready 後長錄音：多次 VAD 切句在停止前完成辨識；紅／橘狀態、中央主要切句數、RMS 外環及
   停止白圈依實際流程切換，UI 更新錯誤不得使錄音或 ASR 失敗。
3. 連續工作：前一段辨識中再錄下一段，結果依原始錄音時間 FIFO 寫入最近結果與剪貼簿，沒有
   重複、漏段或同時使用兩個 ASR worker。
4. 卸載重載：立即卸載及 30 分鐘 idle 卸載後，下一次快捷鍵操作不變，自動重載並處理錄音。
5. 異常復原：分別在模型載入、錄音停止後寫入、ASR 中途強制結束；重啟後 pending／failed 依
   原始建立時間自動重試一次，成功後仍留 error history，持續失敗不無限循環。
6. 麥克風狀態：正常語音、安靜、Windows 靜音、拔除／停用裝置及 callback 中斷；近零只能提示
   可能靜音，硬錯誤需停止並保存，無效 ADC time 不得製造假 gap。
7. 關閉與單一實例：錄音中／辨識中／待命時分別退出，不留 listener、不重複啟動，也不再出現
   pystray setup timeout 或 CFFI callback 視窗。

2026-09-01 補充（啟動延遲歸因）：「模型未 ready 仍可立即錄音」只延後了 Whisper，未延後
PyTorch 與 Silero VAD；快捷鍵原本要等兩者載完才註冊，實測冷啟動主要成本在 `import torch`
（開發機長時間閒置後首次 18.1s、OS 檔案快取熱時 1.7s）。已先讓快捷鍵提前註冊並在啟動期間
回報進度，並補齊各階段計時；要真正縮短「按下就在錄」的等待，需把錄音路徑與 torch 解耦
（Silero ONNX，見 C 線）或改為先緩衝音訊、VAD ready 後再接上切句器。

目前進度（2026-08-07）：Tray-first 啟動、lazy model、即時錄音與途中接回 live ASR、單一 FIFO
worker、最近結果、手動／idle 卸載、單一實例、pending／failed 重啟復原、麥克風事件記錄，以及
Tray／observer 例外隔離均已實作並有分層測試。實機已確認快速出圖、冷啟動可立即錄音、live
切句、卸載後自動重載、最近結果與正常結束；曾觀察到的 pystray `WinError 1402`、UI 例外造成
假性辨識失敗及無效 ADC time 造成假 gap 已有針對性修正。仍需完成：不同麥克風硬錯誤／靜音
情境、強制中斷後自動復原、連續多 job FIFO、實際等待 30 分鐘 idle，以及最新主圓、粗體數字與
RMS 外環在 Windows 深／淺背景的最終視覺驗收。queue 容量與 codec 依前述決策延後，不算 D 線
本輪缺漏。

### E — Log／回饋／歷史介面

狀態：`in_progress`。Phase 1 的資料底座已落地；真實 log 篩選、凍結回歸集、codec 實驗、修正
學習與完整 viewer 尚未完成。

核心目的不是保存一般使用歷史，而是從真實使用資料建立可重現的錯誤案例與回歸集，供後續
ASR、VAD、後處理及 codec 實驗驗證。錯誤型態、案例價值與保留規則不得只靠想像設計；需要
大量分析時，另開獨立工作使用便宜模型掃描實際 log，再由人工抽查重要問題與令人困擾的小問題。
所有新規則與 schema 欄位應標明來源層級：`observed`（實際 log／實驗）、`derived`（由證據推導）
或 `hypothesis`（尚待驗證的設計）。後續不得把 AI 提案或直覺假設改寫成既有實驗結論。

版本與資料邊界：

- 現階段只維護一套程式與設定，不為「開發版／公開版」建立分叉邏輯。以
  `logging.mode: off/errors/full` 控制音訊與診斷資料；目前個人開發環境使用 `full`，真正發布時
  再由 F 線決定新安裝的初始預設，既有使用者設定不被覆寫。D 線未完成工作暫存與文字型最近
  結果都不等同於完整診斷 log。
- metadata 體積小且包含實驗必要上下文，原則上保留；音訊 retention 另行管理。開發版音訊預算
  暫定 2 GiB，但這只是容量護欄，不代表超限時一律按時間刪除。
- 音訊應優先保留經實際輸出、AI 初審或人工複核支持的辨識錯誤、能重現問題的片段與必要的正常
  對照組。哪些自動標籤真正有效、各類比例及淘汰順序，必須先讀取現有大量 log 後再決定；
  現階段不寫死「錯誤率」公式。

舊版 session log 並不是一次快捷鍵開始到停止的完整原始錄音。舊路徑的麥克風 callback 只把 chunk 送入
StreamingVAD；只有通過最短時長與 RMS 門檻、經過前後 padding／靜音裁切並送進 ASR 的片段，
才會在辨識後寫成 `seg_NNN.wav`。未進入 speech、被判定過短／低能量、segment 間靜音與尚未
寫 log 就異常結束的音訊都無法由舊版 session log 還原；fallback log 雖保留 fallback 前的
ASR 片段，仍不是整次錄音，也沒有保存完整的 VAD 決策與每次 intervention。

新 SQLite history 已以「完整 capture 音訊 + 派生事件」為可重現基礎：

- 每次開始到停止保留一份連續 session audio；segment、被淘汰候選與 fallback 子段以 sample
  offset／長度參照同一來源，避免重複存放多份音訊。
- 記錄 VAD 邊界、emit／discard 原因、RMS、必要的 probability 摘要、capture status／frame
  coverage，以及每次 ASR 嘗試使用的設定、輸出、耗時與採用結果。
- 完整 capture 只屬開發／明確診斷模式；公開版維持預設不保存。先量測增加的容量，再與 FLAC
  及 2 GiB retention 一起決定保存粒度。

Metadata 已採 SQLite 作為新 history SSOT，音訊維持外部檔案：

- 使用 Python 內建 `sqlite3`、schema version、foreign keys、WAL、busy timeout 與單一 writer，
  讓 session、segment、ASR attempt、correction、label、recent result 及 audio asset 可用 transaction
  一致寫入與查詢，不需安裝或維護 SQL server。
- 舊 JSONL／WAV 不搬移、不覆寫、不刪除；提供 idempotent importer，以來源路徑與穩定識別避免
  重複匯入。過渡期工具統一經 repository API 讀 SQLite，新資料不再雙寫兩套 SSOT。
- SQLite 不適合直接用文字編輯器閱讀，因此保留 JSONL／JSON 匯出與備份工具；資料庫損壞、
  migration 中斷與舊 log 相容必須有測試。個人 transcript、資料庫與音訊繼續位於 gitignore 的
  使用者資料目錄。

目前進度（2026-08-07）：已建立具 schema version、foreign keys、WAL、busy timeout 與寫入鎖的
SQLite repository；完整 capture 以外部 PCM16 WAV 原子寫入，job、segment、ASR attempts、
VAD／capture／gap metadata、error history、correction candidate、corpus label 與最近文字結果均
可稽核。已提供舊 JSONL／WAV idempotent importer、repository-based 搜尋／viewer 讀取、JSON
匯出與一致備份；2 GiB rolling audio 回收會保留 metadata 並避開已升格 corpus。自動候選目前只
標示實際 fallback／fallback exhausted、絕對轉錄時間超標及 capture warning，尚未證明足以涵蓋
真實錯誤，因此不視為 corpus 已完成。尚待工作是讀取真實大量 log、建立高 recall 候選與正常
對照、人工／AI 複核流程、凍結 engine-specific／cross-model／capture-VAD 回歸集、codec 驗收、
修正規則確認流程及最後期 viewer。

資料保留分成一個循環池與三種可升格的固定 corpus，避免用同一套標籤回答不同問題：

- rolling capture/log：開發版先完整收集，受 2 GiB 音訊容量護欄限制，是候選來源而非永久集。
- engine-specific corpus：針對目前 faster-whisper 模型、版本與解碼設定，保存 >1.5 秒、
  fallback、重試／切分結果等可重現故障；更換模型或設定時仍保留其 provenance，不冒充跨模型
  gold。
- cross-model acoustic/content corpus：保存本身難辨識的詞、口音、噪音、中英混合及人工／AI
  確認的 target，供不同模型公平比較，不以某一模型的耗時 heuristic 當唯一選樣依據。
- capture/VAD corpus：保存完整錄音、麥克風狀態與 VAD 邊界，用來驗證漏收、誤觸、錯切及
  callback continuity；它與單純 ASR 錯詞案例的資料需求不同。

目前 `scripts/search_logs.py` 與 viewer 的分類不是常駐系統，只在手動執行時依半形標點、簡體、
空格、重複、長段無逗號、短輸出及後處理差異產生標籤；它不會自動升格或永久保留案例。下一版
應保留這些 deterministic tags 作候選訊號，但 corpus promotion 必須是明確、可稽核的動作。

既有 1.5 秒判定是 production 上已反覆觀察到高相關性的有效 heuristic，不是尚未驗證的直覺：
早期實測正常段轉錄中位數約 0.5 秒，21.8 秒的錯誤密集段耗時 2.5 秒，25.2 秒的嚴重重複幻覺
耗時 3.5 秒；當時結論是非短段的「絕對轉錄時間 > 1.5 秒」都值得檢查。後續不得只因理論上
可能受負載影響，就改用 transcribe/audio ratio、弱化或移除這個訊號；若要變更，必須先以凍結
錯誤集與正常對照組證明不退化。它是高相關偵測訊號，不等於對所有錯誤型態的完整定義。

`min_hallucination_audio_sec=3.0` 的歷史理由則較窄：32 段實驗中一個 2.15 秒的正常「呃……」
花 2.22 秒，為避免該假陽性而加入短音訊排除。但後續真實 log 已出現 0.26 秒音訊轉錄 18.32
秒並輸出 132 字等明顯異常，因此「3 秒以下一律不檢查」必須重驗。

**2026-09-01 已處理。** 依上述要求先建測試再改規則：`tests/data/short_segment_cases.json`
凍結 history log 中全部 143 筆 < 3 秒的辨識嘗試並人工標註（產生器
`experiments/extract_short_segment_cases.py`）。這份資料上，判為可用輸出的段最慢 1.24 秒，
7 筆明顯幻覺中 6 筆 >= 1.57 秒，既有 1.5 秒門檻零誤殺、單一漏抓是 0.80 秒→1.03 秒的
`Duh.`（延遲落在正常範圍，需另一種訊號才抓得到，暫列已知限制）。因此採分級門檻而非新訊號：
`min_hallucination_audio_sec` 不再關閉偵測，改為切分「還救得回來／救不回來」——短於它的段
仍跑 RMS 正規化，但沒有第二個靜音間隙可切，重切救不了，仍偏慢就捨棄文字並保留證據。
以真實音訊重跑（`experiments/replay_short_segment_hallucination.py`）：0.29 秒子段 3.36 秒、
0.52 秒子段 3.54 秒，輸出皆與語音無關；幻覺文字跨 session 不同，延遲訊號穩定重現。

現有 fallback 在所有切分門檻用盡後，即使最後仍有子段超過 1.5 秒，也會回傳並採用最後一組
結果。下一版必須把 `fallback_succeeded` 與 `fallback_exhausted` 分開；後者保留原始音訊、
各次嘗試與可見警告，不可把「已執行 fallback」誤記成「問題已解決」。

**2026-09-01 已處理。** 兩個旗標本就分開記錄，缺的是「用盡之後怎麼辦」：仍偏慢且短於
`min_hallucination_audio_sec` 的子段不再併入輸出（`adopted=False` + `rejected=True`），
原音訊、每次嘗試與 CLI 警告照舊保留，並寫入 `hallucination-rejected` corpus label
（有標籤的 job 不受 2 GiB 容量護欄淘汰，證據不會被回收）。較長的偏慢子段仍然採用——
丟掉數十秒真實內容的代價高於留下一段可疑文字，這條界線刻意只畫在「沒有其他救法」的短段。
順帶修正 fallback 段在 `segments` 表 `processed_text` 為 NULL、log 檢視器顯示成
「30 秒音訊、文字空白」的問題。

音訊格式現況與下一步：

- Codec 驗收排在 D 線即時轉錄與 Tray 修復之後，本輪不切換保存格式。
- 現況仍保存 PCM WAV。無損 FLAC 是第一候選，先以 PCM round-trip checksum、寫入失敗保留
  原檔及實際容量比驗證；通過後才決定直接寫 FLAC 或背景轉換。
- Opus／MP3 等有損格式只有在固定且可重現的 Whisper 解碼設定下，對真實錯誤案例做嚴格逐字
  交叉測試全部通過後才列入候選。
- 2026-07-31 前置實驗從 120 段中抽 16 段各重跑 3 次，只有一個疑難 WAV 出現兩種文字；該段
  在預設 temperature fallback 下重跑 10 次產生 7 種近似輸出，beam size 1 仍未穩定；固定
  `temperature=0` 後同一段 10 次一致。這只證明特定疑難段在 fallback 設定下不具逐字決定性，
  不代表一般 WAV 無法重現同類錯誤，也尚不足以得出任何 codec 通過結論。下一次 codec 實驗需
  擴大固定解碼條件下的穩定樣本，再比較 FLAC、Opus 與 MP3。原實驗的前 16 段全帶有硬編碼
  `known_asr_error` 標籤，只有一段同時是 fallback；下一輪不得沿用這個排序截斷取樣。

Codec／fallback 回歸集應先從真實 log 的 fallback、絕對轉錄時間 >1.5 秒、極短音訊異常、
嚴重重複／輸出長度異常與已知問題挑選，再依序驗證：(1) production 設定重跑時是否重現同類
錯誤；(2) 單純重試、固定 temperature、RMS 正規化與重切各自是否改善；(3) FLAC／Opus／MP3
是否改變逐字結果、錯誤重現率或 intervention 成功率。每次只改一個因素；RMS 已達 target 時
再次轉錄屬「單純重試」對照，不可記成 RMS 正規化成功。

建議分期：

1. 收斂 log schema、故障安全寫入、2 GiB 容量統計及可供分析工具讀取的穩定資料介面。
2. 依真實 log 建立高 recall 的候選集，優先降低漏掉真正錯誤的 false negative，容許初篩有較多
   false positive；可用便宜模型批次初審，再由人工複核有爭議或準備升格的案例。人工待審超過
   三段時應使用批次 viewer／審核佇列，不以逐檔手工作業作長期流程。
3. 將確認後的代表性錯誤與少量正常對照升格為獨立、凍結的回歸集；rotating log 受 2 GiB
   限制照常回收，不靠「永遠不要刪特定 live log」維持測試資料。再依實際分析結果決定 retention
   標籤與淘汰策略。
4. 從多次人工修正提出替換字／完整詞組候選；使用者確認後才寫入替換規則。短詞全域替換
   必須先排除同音異義風險，有歧義時只允許完整詞組，且需做整句回歸。hotwords 是獨立功能，
   不從修正紀錄自動產生或修改。
5. 最後才做完整歷史 viewer：搜尋、播放、raw/processed/final、從任意紀錄複製與修正。D 線
   只先提供最近 5 筆的 Tray 巢狀選單作多 job 安全網，不另做「複製上一筆」快捷鍵，也不提前
   擴張成大型 Web UI。

延後點子：Profile、應用程式感知、選取文字語音修改、信心詞高亮、OpenCC 差異標示、浮動
狀態／波形、多語言切換與輸出前預覽編輯。競品研究在具體 UI 決策前按題目進行，不另立泛用
研究專案。

#### D/E/F 後續交接清單

以下項目已刻意延後或切到其他路線；後續任務應直接以本清單與上文規格為準，不從舊對話重新
猜測需求：

- **留在 D 線本輪收尾**：實作 `manual`／`autostart` 啟動模式；終端與桌面捷徑一般啟動預載
  Whisper，lazy 只由未來 Windows autostart 明確指定；~~記錄每次模型載入精確耗時~~（2026-09-01
  完成：啟動橫幅時間戳、`import torch`／VAD／可錄音就緒／Whisper 載入各自計時）；完成接近
  15×15 的滿版主圓、同色系麥克風動態、大型中央符號、queue 點、白圈／警告／錯誤獨立疊加與
  所有狀態組合測試；
  再做多 job FIFO、麥克風靜音／拔除與異常關閉復原的受控 Windows 驗收。
- **E 線資料／實驗後續**：先用真實大量 log 建立高 recall 錯誤候選與少量正常對照，再進行
  人工／便宜模型複核，升格並凍結 engine-specific、cross-model 與 capture/VAD corpus。不得只靠
  AI 想像錯誤類型，也不得把 >1.5 秒高相關 heuristic 誤寫成錯誤的完整定義。
- **E 線 codec／容量後續**：目前維持 PCM16 WAV。FLAC 先做 lossless round-trip、失敗保留原檔
  與容量實測；Opus／MP3 必須通過固定解碼設定下的真實錯誤案例逐字交叉測試。queue 容量上限、
  接近上限警告與更細 retention 淘汰順序，待 codec 與實際容量資料一起決定；不得靜默刪除工作。
- **E 線修正／介面後續**：從已確認案例提出替換字或完整詞組，使用者確認後才寫規則並做整句
  回歸；不從修正紀錄自動修改 hotwords。待審超過三段時建立批次審核介面；完整歷史 viewer、
  任意紀錄複製、播放、raw／processed／final 與修正功能最後才做，Tray 近期結果維持輕量安全網。
- **F 線發布後續**：真正建立 Windows 開機自動啟動項目並傳入 `autostart` lazy 模式；處理乾淨
  Windows 安裝、打包、模型下載／快取、升級保留設定、公開版 logging 預設與發布／開發設定。
  使用者主動開啟的桌面捷徑仍屬 `manual` 預載，不得因它同樣是無視窗啟動而誤判為 autostart。

### F — 發布工程

狀態：`deferred`，確定公開或給其他人使用時啟動。

範圍：LICENSE、版本統一、CI、乾淨 Windows 安裝 smoke test、Windows/NVIDIA/CUDA 相容性
檢查、首次模型下載與失敗重試、portable onedir、升級時保留使用者設定、發布/開發設定分離，
以及 `docs/index.html` 翻新。

公開版仍以 faster-whisper 為預設；Qwen 與預設停用的 OpenRouter adapter 是否納入正式安裝，
發布前依實際 import/打包稽核決定，或以額外 dependency group 隔離。模型快取、
設定與 log 必須使用穩定的使用者資料目錄，不依賴目前工作目錄；發布版預設關閉音訊 log，
診斷模式與開發版才保留完整資料。是否移除其他依賴，以實際 import/打包稽核結果決定。

## 平行開發邊界

建議本任務負責 A → B/C；另一個獨立 worktree 可負責 D/E 的規格與限界實作。

為避免衝突：

- 路線圖核准後先凍結；平行任務不各自改 `docs/roadmap.md`、`TODO.md`、
  `CHANGELOG.md`，由整合方最後統一更新。
- A 僅修改 `experiments/benchmark/`、benchmark 規格與結果。
- B 主要修改 `src/asr_input/asr/` 與 ASR config；C 主要修改 `src/asr_input/audio/`。
- D 可修改 `tray.py`、啟動腳本與 lifecycle tests。
- E 第一階段只修改 log schema、viewer 與獨立 UI，不改 tray 選單；需要 tray 入口時留到
  整合階段。
- `config.yaml`、`pyproject.toml`、`tray.py`、`TODO.md`、`CHANGELOG.md` 是共享熱點，
  平行分支先不共同修改，或事先指定唯一 owner。

建議各線使用獨立 branch/worktree、小型 commit 與行為測試；先整合 A 的 schema，再整合
C/B，D/E 可獨立審查後合併。
