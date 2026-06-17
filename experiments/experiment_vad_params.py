"""Compare VAD segmentation with different min_silence_duration_ms values.

Usage:
    .venv\\Scripts\\python.exe experiment_vad_params.py <audio_path>
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa
import numpy as np

from asr_input.audio.vad import SileroVAD


def analyze(segments, sample_rate, label):
    if not segments:
        print(f"  {label}: 0 段")
        return
    durations = [(s.end_sample - s.start_sample) / sample_rate for s in segments]
    short = sum(1 for d in durations if d < 2.0)
    long = sum(1 for d in durations if d > 30.0)
    print(
        f"  {label}: {len(segments):>3} 段 | "
        f"平均 {np.mean(durations):>5.1f}s | "
        f"最短 {min(durations):>4.1f}s | "
        f"最長 {max(durations):>5.1f}s | "
        f"短段(<2s) {short:>2} ({short/len(segments)*100:>2.0f}%) | "
        f"長段(>30s) {long}"
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment_vad_params.py <audio_path>")
        return

    path = sys.argv[1]
    sample_rate = 16000

    print(f"載入音檔: {path}")
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    print(f"音檔長度: {len(audio)/sample_rate:.1f}s\n")

    test_values = [300, 500, 800, 1000, 1500, 2000, 3000]

    print("=== min_silence_duration_ms 比較 ===\n")
    for ms in test_values:
        vad = SileroVAD(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=ms,
            speech_pad_ms=100,
        )
        vad.load()
        segments = vad.detect(audio, sample_rate)
        analyze(segments, sample_rate, f"{ms:>4}ms")
        vad.unload()


if __name__ == "__main__":
    main()
