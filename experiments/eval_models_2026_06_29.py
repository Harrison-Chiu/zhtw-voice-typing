"""系統化模型評測（2026-06-29）：baseline / whisper-zh-TW(4a) / Qwen3-ASR。

回應需求：把候選模型在「不加任何額外引導」下的原生輸出落地存檔，讓人能逐檔看
「輸入長怎樣 → 哪個模型吐了什麼」，據此判斷是模型本身品質問題還是可避免的系統問題。

關鍵設計：
- 首次納入兩個 ~10s 短檔（日常用途是短語音輸入）。短檔 <30s，4a 候選的
  chunk_length_s=30 不會觸發分塊 → 給出「無分塊干擾」的乾淨 4a 品質視角。
- baseline = 現行 config（faster-whisper + VAD + initial_prompt），當對照基準。
- 候選 4a / Qwen 都跑「原生」：不給 prompt、不給 hotword，看模型本體輸出。
- 全部結果存 experiments/results/model_eval_2026-06-29.json + 同步 print。

用法：.venv\\Scripts\\python.exe experiments/eval_models_2026_06_29.py
"""

import gc
import io
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", line_buffering=True, write_through=True
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from asr_input.asr import build_engine  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data/test_audio"
OUT_JSON = PROJECT_ROOT / "experiments/results/model_eval_2026-06-29.json"
CANDIDATE_MODEL = "JacobLinCool/whisper-large-v3-turbo-common_voice_19_0-zh-TW"

# 短檔優先（日常用途）；長檔放最後重現之前的重複現象
AUDIO_FILES = [
    "中英錄音測試.m4a",
    "機器人展示，語音轉錄測試.m4a",
    "簡報日.m4a",
]

SIMPLIFIED_SAMPLE = set("软视启发这说应话时题问为业产权术语种类别样东马达机现们个")


def scan_simplified(text: str) -> list[str]:
    return sorted({c for c in text if c in SIMPLIFIED_SAMPLE})


def repetition_stats(text: str) -> dict:
    """偵測退化型重複。回傳最長連續重複 token 的長度與內容、單字最高佔比。"""
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return {"max_consecutive_run": 0, "run_token": "", "top_char_ratio": 0.0}
    # 最長「同一 1-3 字 token 連續重複」次數
    best_run, best_tok = 0, ""
    for tok_len in (1, 2, 3):
        i = 0
        n = len(stripped)
        while i < n - tok_len:
            tok = stripped[i : i + tok_len]
            run = 1
            j = i + tok_len
            while stripped[j : j + tok_len] == tok:
                run += 1
                j += tok_len
            if run > best_run:
                best_run, best_tok = run, tok
            i += tok_len if run == 1 else (j - i)
    cnt = Counter(stripped)
    top_char, top_n = cnt.most_common(1)[0]
    return {
        "max_consecutive_run": best_run,
        "run_token": best_tok,
        "top_char": top_char,
        "top_char_ratio": round(top_n / len(stripped), 3),
    }


def metrics(text: str) -> dict:
    return {
        "len": len(text),
        "simplified_residual": scan_simplified(text),
        "comma_full": text.count("，"),
        "comma_half": text.count(","),
        "period_full": text.count("。"),
        "period_half": text.count("."),
        "ellipsis": text.count("…"),
        **repetition_stats(text),
    }


def decode_audio(path: Path, sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_baseline(audios, config):
    """現行 config：faster-whisper large-v3-turbo + VAD + initial_prompt。"""
    engine = build_engine(config["asr"], vad_cfg=config.get("vad"))
    engine.load()
    out = {}
    for name, audio in audios.items():
        t0 = time.time()
        raw = engine.transcribe(audio, config["audio"]["sample_rate"])
        out[name] = {"raw": raw, "sec": round(time.time() - t0, 2)}
    engine.unload()
    _free()
    return out


def transcribe_candidate(audios, sr):
    """4a：whisper-large-v3-turbo zh-TW 微調，transformers pipeline，原生（無 prompt）。"""
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=CANDIDATE_MODEL,
        torch_dtype=dtype,
        device=device,
    )
    out = {}
    for name, audio in audios.items():
        t0 = time.time()
        res = asr(
            {"raw": audio, "sampling_rate": sr},
            chunk_length_s=30,
            generate_kwargs={"language": "zh", "task": "transcribe"},
        )
        out[name] = {"raw": res["text"], "sec": round(time.time() - t0, 2)}
    del asr
    _free()
    return out


def transcribe_qwen(audios, config):
    """Qwen3-ASR-1.7B，原生（context 空），無 VAD。長檔可能 512-token 截斷。"""
    qwen_cfg = {
        "engine": "qwen",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "device": "cuda",
        "language": "Chinese",
        "context": "",
    }
    engine = build_engine(qwen_cfg, vad_cfg=None)  # 原生，不包 VAD
    engine.load()
    out = {}
    for name, audio in audios.items():
        t0 = time.time()
        raw = engine.transcribe(audio, config["audio"]["sample_rate"])
        out[name] = {"raw": raw, "sec": round(time.time() - t0, 2)}
    engine.unload()
    _free()
    return out


def main():
    config = load_config()
    sr = config["audio"]["sample_rate"]
    post = build_pipeline(config, include_output=False)

    print("解碼音檔（ffmpeg）...")
    audios, durations = {}, {}
    for fn in AUDIO_FILES:
        p = AUDIO_DIR / fn
        if not p.exists():
            print(f"  ! 找不到 {fn}，跳過")
            continue
        a = decode_audio(p, sr)
        audios[fn] = a
        durations[fn] = round(len(a) / sr, 1)
        print(f"  {fn}  {durations[fn]}s")

    engines = [
        ("baseline_whisper_fw", lambda: transcribe_baseline(audios, config)),
        ("candidate_whisper_zhtw_4a", lambda: transcribe_candidate(audios, sr)),
        ("qwen3_asr", lambda: transcribe_qwen(audios, config)),
    ]

    results = {"durations_sec": durations, "engines": {}}
    for eng_name, fn in engines:
        print(f"\n{'=' * 60}\n載入並執行：{eng_name}\n{'=' * 60}")
        try:
            t0 = time.time()
            raw_out = fn()
            load_run_sec = round(time.time() - t0, 1)
            eng_block = {"status": "ok", "total_sec": load_run_sec, "files": {}}
            for name, item in raw_out.items():
                raw = item["raw"]
                processed = post.run(raw)
                eng_block["files"][name] = {
                    "transcribe_sec": item["sec"],
                    "raw": raw,
                    "processed": processed,
                    "metrics_raw": metrics(raw),
                    "metrics_processed": metrics(processed),
                }
                print(
                    f"\n----- [{eng_name}] {name} ({durations.get(name, '?')}s, "
                    f"轉錄 {item['sec']}s) -----"
                )
                print(f"[原始]   {raw}")
                print(f"[後處理] {processed}")
                m = metrics(raw)
                print(
                    f"[指標]   簡體殘留={m['simplified_residual'] or '無'} "
                    f"全形逗號={m['comma_full']} 半形逗號={m['comma_half']} "
                    f"最長重複={m['max_consecutive_run']}x'{m['run_token']}' "
                    f"單字最高佔比={m['top_char_ratio']}"
                )
            results["engines"][eng_name] = eng_block
        except Exception as e:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc()
            print(f"!! {eng_name} 失敗：{type(e).__name__}: {e}")
            print(tb)
            results["engines"][eng_name] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            }
        # 每跑完一個引擎就存檔（避免中途崩潰丟全部）
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n（已存檔 → {OUT_JSON.name}）")

    print(f"\n完成。結果：{OUT_JSON}")


if __name__ == "__main__":
    main()
