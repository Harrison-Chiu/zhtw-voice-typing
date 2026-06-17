"""Diagnostic: show each VAD segment's duration and transcription.

Usage:
    .venv\Scripts\python.exe experiment_vad_segments.py <audio_path>
"""

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa
import numpy as np

from asr_input.asr import build_engine
from asr_input.audio.vad import SileroVAD
from asr_input.config import load_config


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python experiment_vad_segments.py <audio_path>")
        return

    path = sys.argv[1]
    config = load_config()
    asr_cfg = config["asr"]
    sample_rate = config["audio"]["sample_rate"]

    print("載入模型...", flush=True)
    engine = build_engine(asr_cfg)  # no VAD wrapper
    engine.load()

    vad = SileroVAD(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
    )
    vad.load()

    print(f"載入音檔: {path}", flush=True)
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    total_sec = len(audio) / sample_rate
    print(f"音檔長度: {total_sec:.1f}s\n", flush=True)

    segments = vad.detect(audio, sample_rate)
    print(f"VAD 切出 {len(segments)} 段\n", flush=True)

    short_count = 0
    for i, seg in enumerate(segments):
        chunk = audio[seg.start_sample : seg.end_sample]
        chunk_sec = len(chunk) / sample_rate
        start_sec = seg.start_sample / sample_rate
        end_sec = seg.end_sample / sample_rate

        t0 = time.time()
        text = engine.transcribe(chunk, sample_rate)
        dt = time.time() - t0

        flag = " ⚠️短" if chunk_sec < 2.0 else ""
        print(
            f"[{i+1:>3}/{len(segments)}] "
            f"{start_sec:>6.1f}s-{end_sec:>6.1f}s "
            f"({chunk_sec:>5.1f}s) "
            f"→ {dt:.1f}s{flag}"
        )
        print(f"    {text.strip()}")
        print()

        if chunk_sec < 2.0:
            short_count += 1

    print(f"\n=== 統計 ===")
    print(f"總段數: {len(segments)}")
    print(f"短段 (<2s): {short_count} ({short_count/len(segments)*100:.0f}%)")
    print(f"平均段長: {sum(s.end_sample - s.start_sample for s in segments) / len(segments) / sample_rate:.1f}s")

    engine.unload()


if __name__ == "__main__":
    main()
