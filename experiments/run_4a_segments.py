import io, json, subprocess, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, torch
from transformers import pipeline as hf_pipeline

ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "data/test_audio/segments"
OUT = ROOT / "experiments/results/whisper_zhtw_segments_2026-06-29.json"
MODEL = "JacobLinCool/whisper-large-v3-turbo-common_voice_19_0-zh-TW"
FILES = [
    "01_純中文長句_自我介紹.wav",
    "02_技術描述_開發平台.wav",
    "03_中英混合_Arduino.wav",
    "04_長段連續描述.wav",
    "05_數字混合_電池規格.wav",
]


def dec(p, sr=16000):
    r = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(p),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(r, dtype=np.float32).copy()


asr = hf_pipeline("automatic-speech-recognition", model=MODEL, torch_dtype=torch.float16, device=0)
res = {}
for f in FILES:
    a = dec(SEG / f)
    t0 = time.time()
    out = asr(
        {"raw": a, "sampling_rate": 16000},
        chunk_length_s=30,
        generate_kwargs={"language": "zh", "task": "transcribe"},
    )
    res[f] = {"raw": out["text"], "sec": round(time.time() - t0, 2)}
    print(f"[{f}] ({res[f]['sec']}s)\n  {out['text']}\n")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
