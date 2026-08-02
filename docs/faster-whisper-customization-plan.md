# Faster Whisper 客製化與優化計畫

## 目前到底是怎麼使用的

目前不是呼叫雲端 API，也不是只能使用不可修改的黑盒服務。實際堆疊是：

```text
ASR Input 自有程式
  └─ faster-whisper Python 套件（轉錄流程、解碼策略、VAD 介接）
       └─ CTranslate2 原生 runtime（C++/CUDA 高效推論）
            └─ large-v3-turbo CT2 模型權重（本機 Hugging Face cache）
```

- `faster-whisper`、`ctranslate2` 目前是安裝在 `.venv/Lib/site-packages` 的套件版本。
- `WhisperModel("large-v3-turbo")` 首次會下載已轉成 CTranslate2 格式的模型，之後從本機
  cache 載入；推論不需要雲端服務。
- 我們自己的 `WhisperFWEngine` 已直接控制 model、compute type、language、beam、
  temperature、initial prompt 與 hotwords，但只暴露上游參數的一小部分。
- faster-whisper 與 CTranslate2 都是可 fork、修改與重新安裝的開源專案。沒有必要為了
  「能修改」就先把原始碼複製進本 repo；應先用固定 Git revision dependency 或 editable
  fork，避免 vendor 後失去上游更新與 bug fix。

## 建議的修改層級

### L1：自有 adapter 與 config（優先）

不改上游原始碼，只在 `src/asr_input/asr/whisper_fw.py` 暴露更多參數與策略。

候選：

- `beam_size`、`patience`、`length_penalty`
- `temperature` fallback schedule
- `compression_ratio_threshold`、`log_prob_threshold`、`no_speech_threshold`
- `repetition_penalty`、`no_repeat_ngram_size`
- `condition_on_previous_text`、`prompt_reset_on_temperature`
- `hallucination_silence_threshold`
- `suppress_tokens`、`max_new_tokens`
- prompt/hotword 依內容或 segment tag 動態組合

門檻低、升級風險小，而且足以驗證多數解碼假設。所有參數必須經標準 benchmark，
不能依一兩段輸出手調。

### L2：自有調度與切段（最可能有產品收益）

仍使用原版 faster-whisper/CTranslate2，但修改我們餵資料的方式：

- 互動式單句維持 sequential，離線/佇列積壓才 dynamic batching。
- 以 VAD 後的獨立片段形成 batch，避免直接整個長檔 batching 造成漏段。
- adaptive batch size：依片段長度、目前 VRAM 與等待時間決定 1/2/4。
- length bucketing：長度相近的片段合批，減少 padding 浪費。
- ready gate：錄音與 VAD 先工作，模型背景載入完成後消化佇列。
- app idle unload、預載、模型 load state machine。
- ASR 與音訊解碼/後處理 pipeline 化，避免 GPU 等 CPU。

這一層比換推論 runtime 更符合目前短句 tray 的瓶頸與產品行為。

### L3：維護 faster-whisper fork（有明確缺口才做）

做法：fork 上游 repo，鎖定 commit，在獨立環境 `uv pip install -e <fork>`；本專案只保留
dependency revision 與 patch 說明，不直接複製整包 source。

可能理由：

- batched pipeline 與 sequential 的 prompt/hotword/切段行為不一致。
- 需要取得上游未暴露的 token-level log probability 或 decoder diagnostics。
- 自訂 fallback 決策、segment merge、timestamp restoration。
- upstream bug 已定位且 patch 小，可同時送 PR 降低長期維護成本。

採用前要求：有最小重現案例、benchmark 證明收益、patch 可與上游定期 rebase。

### L4：修改 CTranslate2 C++/CUDA（高門檻、低優先）

只有 profiler 證明瓶頸在 runtime kernel 才進入：

- compute type / INT8-FP16 組合
- worker、queue、allocator 與非同步執行
- encoder/decoder kernel、fusion、CUDA graph 或記憶體配置
- 自編 CTranslate2 wheel 與 CUDA/cuDNN 相容矩陣

這層需要 C++、CUDA、CMake、wheel packaging 與跨版本驗證。RTX 4060 上可能有收益，但
維護成本遠高於調度與解碼參數，不應先做。

### L5：模型層（獨立研究線）

- 台灣中文/領域資料 fine-tuning 或 LoRA。
- glossary-aware fine-tuning，降低靠全域字典造成的誤傷。
- fine-tuned Transformers checkpoint 轉成 CTranslate2 格式後再進現有 runtime。
- 蒸餾、量化或不同模型 checkpoint。

最大門檻是可信訓練/驗證資料與退化測試，不是單純下載模型。之前 zh-TW 微調模型曾有
難段重複崩潰，因此這條必須最後用凍結 test set 驗證。

## 候選技術 Backlog

| 優先 | 技術 | 預期收益 | 主要風險 | Benchmark 重點 |
|---:|---|---|---|---|
| P0 | 解碼參數矩陣 | 降幻覺/重複、可能降延遲 | 漏字或不可重現 | CER、deletion、determinism、P95 |
| P0 | VAD 片段 dynamic batching | 離線吞吐提高 | 等待時間、漏段 | 短句 latency 與長檔 completeness |
| P0 | 詞彙規則改成上下文安全 | 避免正常中文誤傷 | 規則覆蓋不足 | glossary recall + false positive |
| P1 | int8_float16 / compute type | 降 VRAM、可能加速 | 品質或 kernel 差異 | CER、P95、VRAM |
| P1 | length bucketing | batch 效率提高 | scheduler 複雜度 | throughput、queue delay |
| P1 | prompt/hotword ablation | 專有詞、繁體與標點 | prompt 污染/重複 | glossary、CER、崩潰率 |
| P1 | fallback policy | 決定性與難段品質 | temperature=0 退化 | exact repeat、stress set |
| P1 | background load + ready gate | 首次使用體感改善 | 狀態競態、記憶體 | end-to-end、soak test |
| P2 | faster-whisper fork diagnostics | 可定位 batched 漏段 | fork 維護 | token/segment trace |
| P2 | HF SDPA/FlashAttention 對照 | 長檔吞吐 | Windows 安裝、8GB OOM | Pareto 圖、短句 P95 |
| P2 | fine-tune + CT2 conversion | 台灣詞彙/口音品質 | 資料、重複退化 | frozen test CER/MER |
| P3 | CTranslate2 runtime fork | kernel 級效能 | 高維護成本 | profiler + 全套回歸 |

## 建議執行順序

1. 先完成標準 benchmark v1 與人工 gold；沒有量尺不開始大規模調參。
2. 跑 baseline 解碼參數矩陣，優先解決 temperature fallback 不決定性。
3. 做 VAD 後片段 dynamic batching prototype，只作離線模式，不動 tray 預設。
4. 測 compute type、HF SDPA；只有本機 Pareto 明顯勝出才測 FlashAttention。
5. 若 batched 漏段仍存在，建立 faster-whisper fork，加 diagnostics 並定位最小 patch。
6. 模型 fine-tuning 與 CTranslate2 runtime 修改維持獨立研究線，等前四步顯示上層已到瓶頸。

## 第一輪可驗收交付

- Benchmark v1：Smoke/Core manifest、雙 target gold、CER/MER 與 HTML dashboard。
- 參數實驗：baseline 加 8–12 組有理據的解碼設定，不做無界 grid search。
- Dynamic batching prototype：單句立即解碼，離線片段 batch 1/2/4，自動降級避免 OOM。
- 決策報告：品質–延遲–VRAM Pareto、逐段 regression、採用或不採用理由。

