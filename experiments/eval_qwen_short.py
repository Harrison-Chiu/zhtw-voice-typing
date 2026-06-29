"""Qwen3-ASR 短檔快測（2026-06-29）。

主評測腳本卡在「長檔不切 VAD 一次性丟給 Qwen LLM」（475s 音訊推論極慢、且與另一
份 eval 並跑造成 GPU 超載），本腳本只測兩個 ~10s 短檔（日常用途），原生 context 空。
結果併入 model_eval_2026-06-29.json 的 qwen3_asr 區塊。
"""

import gc
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_models_2026_06_29 import (  # noqa: E402
    AUDIO_DIR,
    OUT_JSON,
    decode_audio,
    metrics,
    build_engine,
    load_config,
    build_pipeline,
)

SHORT_FILES = ["中英錄音測試.m4a", "機器人展示，語音轉錄測試.m4a"]


def main():
    config = load_config()
    sr = config["audio"]["sample_rate"]
    post = build_pipeline(config, include_output=False)

    qwen_cfg = {
        "engine": "qwen",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "device": "cuda",
        "language": "Chinese",
        "context": "",
    }
    print("載入 Qwen3-ASR-1.7B ...")
    engine = build_engine(qwen_cfg, vad_cfg=None)
    engine.load()

    files = {}
    for fn in SHORT_FILES:
        a = decode_audio(AUDIO_DIR / fn, sr)
        t0 = time.time()
        raw = engine.transcribe(a, sr)
        sec = round(time.time() - t0, 2)
        processed = post.run(raw)
        m = metrics(raw)
        files[fn] = {
            "transcribe_sec": sec,
            "raw": raw,
            "processed": processed,
            "metrics_raw": m,
            "metrics_processed": metrics(processed),
        }
        print(f"\n--- {fn} ({round(len(a) / sr, 1)}s, 轉錄{sec}s) ---")
        print(f"[原始]   {raw}")
        print(f"[後處理] {processed}")
        print(
            f"[指標]   簡體={m['simplified_residual'] or '無'} 全逗={m['comma_full']} "
            f"半逗={m['comma_half']} 句號全={m['period_full']} "
            f"最長重複={m['max_consecutive_run']}x{m['run_token']!r}"
        )
    engine.unload()
    gc.collect()
    torch.cuda.empty_cache()

    # 併回主 JSON
    data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    data["engines"]["qwen3_asr_short"] = {
        "status": "ok",
        "note": "僅短檔；長檔不切 VAD 一次性推論不可行（極慢/GPU 超載）",
        "files": files,
    }
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已併入 {OUT_JSON.name}")


if __name__ == "__main__":
    main()
