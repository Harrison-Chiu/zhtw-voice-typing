"""Test adaptive VAD segmentation (800ms + fallback for long segments).

Usage:
    .venv\\Scripts\\python.exe experiment_vad_adaptive.py <audio_path>
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa
import numpy as np

from asr_input.audio.vad import SileroVAD


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_vad_adaptive.py <audio_path>")
        return

    path = sys.argv[1]
    sample_rate = 16000

    print(f"載入音檔: {path}")
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    total_sec = len(audio) / sample_rate
    print(f"音檔長度: {total_sec:.1f}s\n")

    vad = SileroVAD(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=800,
        speech_pad_ms=100,
        max_segment_sec=60.0,
        fallback_silence_ms=(500, 300),
    )
    vad.load()

    segments = vad.detect(audio, sample_rate)
    durations = [(s.end_sample - s.start_sample) / sample_rate for s in segments]

    print(f"自適應 VAD: {len(segments)} 段\n")
    for i, (seg, dur) in enumerate(zip(segments, durations)):
        start = seg.start_sample / sample_rate
        end = seg.end_sample / sample_rate
        flag = " ⚠️短" if dur < 2.0 else (" ⚠️長" if dur > 60.0 else "")
        print(f"  [{i+1:>3}] {start:>6.1f}s - {end:>6.1f}s ({dur:>5.1f}s){flag}")

    short = sum(1 for d in durations if d < 2.0)
    long = sum(1 for d in durations if d > 60.0)
    print(f"\n=== 統計 ===")
    print(f"總段數: {len(segments)}")
    print(f"平均段長: {np.mean(durations):.1f}s")
    print(f"最短: {min(durations):.1f}s | 最長: {max(durations):.1f}s")
    print(f"短段(<2s): {short} ({short/len(segments)*100:.0f}%)")
    print(f"長段(>60s): {long}")


if __name__ == "__main__":
    main()
