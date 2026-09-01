"""Extract test segments from the long audio file for experiments.

Usage:
    .venv\\Scripts\\python.exe extract_test_segments.py
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "data/test_audio/隊伍簡報日_8分鐘.m4a"
OUT_DIR = PROJECT_ROOT / "data/test_audio/segments"
SR = 16000

SEGMENTS = [
    {
        "name": "01_純中文長句_自我介紹",
        "start": 0.1,
        "end": 16.5,
        "purpose": "標點基準測試：一般中文長句，多個逗號",
    },
    {
        "name": "02_技術描述_開發平台",
        "start": 24.7,
        "end": 40.5,
        "purpose": "技術詞+標點：馬達、雷射、感測器等專有名詞",
    },
    {
        "name": "03_中英混合_Arduino",
        "start": 83.4,
        "end": 112.8,
        "purpose": "中英混合標點：Arduino, Arducaptor, PWM 等英文詞穿插中文",
    },
    {
        "name": "04_長段連續描述",
        "start": 113.8,
        "end": 168.3,
        "purpose": "長段品質：54秒連續描述，測長段標點一致性",
    },
    {
        "name": "05_數字混合_電池規格",
        "start": 207.8,
        "end": 235.5,
        "purpose": "數字旁標點：14.8V 等數字，測半形句點不被誤轉",
    },
]


def main():
    print(f"載入音檔: {SOURCE}")
    audio, _ = librosa.load(str(SOURCE), sr=SR, mono=True)
    audio = audio.astype(np.float32)
    print(f"音檔長度: {len(audio)/SR:.1f}s\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seg in SEGMENTS:
        start_sample = int(seg["start"] * SR)
        end_sample = int(seg["end"] * SR)
        chunk = audio[start_sample:end_sample]
        duration = len(chunk) / SR

        out_path = OUT_DIR / f"{seg['name']}.wav"
        sf.write(str(out_path), chunk, SR)
        print(f"  {seg['name']}")
        print(f"    {seg['start']:.1f}s - {seg['end']:.1f}s ({duration:.1f}s)")
        print(f"    目的: {seg['purpose']}")
        print(f"    → {out_path}")
        print()

    print(f"完成，共 {len(SEGMENTS)} 個測試片段存放於 {OUT_DIR}/")


if __name__ == "__main__":
    main()
