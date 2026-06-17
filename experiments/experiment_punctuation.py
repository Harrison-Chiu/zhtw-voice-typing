"""Experiment: test different initial_prompt values for punctuation control.

Runs each test segment with different prompts and counts half-width vs
full-width punctuation to find the best prompt for consistent full-width
Chinese punctuation while keeping half-width for English/numbers.

Usage:
    .venv\\Scripts\\python.exe experiment_punctuation.py
"""

import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

import librosa
import numpy as np
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENT_DIR = PROJECT_ROOT / "data/test_audio/segments"
SR = 16000

PROMPTS = {
    "A_目前": "繁體中文，台灣用語。",
    "B_加長全形": "繁體中文，台灣用語。這是一段語音辨識的測試，請使用全形標點符號！好的，沒問題。",
    "C_純標點範例": "，。？！：；、",
    "D_指令式": "請使用全形標點符號。繁體中文，台灣用語。",
    "E_中英混合": "繁體中文，台灣用語。English words use half-width: 3.14, hello.",
    "F_無prompt": None,
}

HALF_WIDTH = re.compile(r"[,\.\?!:;]")
FULL_WIDTH = re.compile(r"[，。？！：；、]")


def count_punct(text: str) -> dict:
    half = HALF_WIDTH.findall(text)
    full = FULL_WIDTH.findall(text)
    return {
        "half_total": len(half),
        "full_total": len(full),
        "half_detail": {c: half.count(c) for c in set(half)},
        "full_detail": {c: full.count(c) for c in set(full)},
    }


def main():
    wav_files = sorted(SEGMENT_DIR.glob("*.wav"))
    if not wav_files:
        print(f"找不到測試片段，請先執行 extract_test_segments.py")
        return

    print("載入模型...", flush=True)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("模型載入完成。\n", flush=True)

    segments: dict[str, np.ndarray] = {}
    for f in wav_files:
        audio, _ = librosa.load(str(f), sr=SR, mono=True)
        segments[f.stem] = audio.astype(np.float32)
        print(f"  載入: {f.stem} ({len(audio)/SR:.1f}s)")
    print()

    results: dict[str, dict[str, dict]] = {}

    for prompt_name, prompt_val in PROMPTS.items():
        print(f"{'='*70}")
        print(f"Prompt: {prompt_name}")
        print(f"  值: {prompt_val!r}")
        print(f"{'='*70}")

        results[prompt_name] = {}

        for seg_name, audio in segments.items():
            segs, _ = model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                initial_prompt=prompt_val,
            )
            text = "".join(s.text for s in segs).strip()
            stats = count_punct(text)
            results[prompt_name][seg_name] = {"text": text, "stats": stats}

            half_str = " ".join(f"{k}:{v}" for k, v in stats["half_detail"].items()) or "(無)"
            full_str = " ".join(f"{k}:{v}" for k, v in stats["full_detail"].items()) or "(無)"

            print(f"\n  [{seg_name}]")
            print(f"    半形 {stats['half_total']:>2} 個: {half_str}")
            print(f"    全形 {stats['full_total']:>2} 個: {full_str}")
            print(f"    文字: {text[:120]}{'...' if len(text) > 120 else ''}")

        print()

    print(f"\n{'='*70}")
    print("=== 總結：各 prompt 的半形/全形標點比例 ===")
    print(f"{'='*70}\n")

    for prompt_name in PROMPTS:
        total_half = sum(r["stats"]["half_total"] for r in results[prompt_name].values())
        total_full = sum(r["stats"]["full_total"] for r in results[prompt_name].values())
        total = total_half + total_full
        pct = total_full / total * 100 if total else 0
        print(f"  {prompt_name:<20} 半形:{total_half:>3}  全形:{total_full:>3}  全形率:{pct:>5.1f}%")


if __name__ == "__main__":
    main()
