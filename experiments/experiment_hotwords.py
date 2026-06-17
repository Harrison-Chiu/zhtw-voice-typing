"""Experiment: test hotwords effect on recognition accuracy.

Tests different hotwords configurations against the same audio segments
to see which format works best for correcting recognition errors.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = PROJECT_ROOT / "data/test_audio/segments"
RESULTS_FILE = Path(__file__).resolve().parent / "results/experiment_hotwords.json"


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def run_experiment():
    from faster_whisper import WhisperModel

    print("載入模型...", flush=True)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("模型載入完成!", flush=True)

    segments = sorted(SEGMENTS_DIR.glob("*.wav"))
    if not segments:
        print("ERROR: No wav segments found in", SEGMENTS_DIR)
        return

    print(f"找到 {len(segments)} 個測試片段\n")

    # First pass: baseline to find what errors exist
    print("=" * 60)
    print("Phase 1: Baseline — 找出現有辨識錯誤")
    print("=" * 60)

    base_prompt = "繁體中文，台灣用語。"
    baseline_results = {}

    for seg_path in segments:
        audio, sr = load_audio(seg_path)
        name = seg_path.stem
        print(f"\n--- {name} ({len(audio)/sr:.1f}s) ---")

        segs, _ = model.transcribe(
            audio,
            language="zh",
            beam_size=5,
            initial_prompt=base_prompt,
        )
        text = "".join(s.text for s in segs).strip()
        print(f"  結果: {text}")
        baseline_results[name] = text

    print("\n\n" + "=" * 60)
    print("Phase 2: Hotwords 實驗")
    print("=" * 60)

    # Different hotwords configurations to test
    hotword_configs = {
        "A_no_hotwords": None,
        "B_single_words": "詞表 待辦 清單 聲學 標點 預填充",
        "C_two_char_pairs": "詞表 待辦 清單 聲學 標點",
        "D_short_phrases": "台灣用語詞表，待辦事項清單，聲學辨識錯誤，標點符號正規化",
        "E_full_sentences": "我們的台灣用語詞表需要擴充。待辦事項清單上還有很多工作。聲學辨識的錯誤需要用hotwords修正。標點符號的正規化已經完成。",
        "F_in_initial_prompt": None,  # uses extended initial_prompt instead
    }

    extended_prompt = "繁體中文，台灣用語。詞表、待辦事項清單、聲學辨識、標點符號。"

    all_results = {"baseline_prompt": base_prompt, "configs": {}}

    for config_name, hotwords in hotword_configs.items():
        print(f"\n{'─' * 40}")
        print(f"Config: {config_name}")
        if config_name == "F_in_initial_prompt":
            print(f"  initial_prompt: {extended_prompt}")
            print(f"  hotwords: None")
        else:
            print(f"  initial_prompt: {base_prompt}")
            print(f"  hotwords: {hotwords}")
        print(f"{'─' * 40}")

        config_results = {}

        for seg_path in segments:
            audio, sr = load_audio(seg_path)
            name = seg_path.stem

            if config_name == "F_in_initial_prompt":
                segs, _ = model.transcribe(
                    audio,
                    language="zh",
                    beam_size=5,
                    initial_prompt=extended_prompt,
                )
            else:
                kwargs = {
                    "language": "zh",
                    "beam_size": 5,
                    "initial_prompt": base_prompt,
                }
                if hotwords:
                    kwargs["hotwords"] = hotwords
                segs, _ = model.transcribe(audio, **kwargs)

            text = "".join(s.text for s in segs).strip()
            changed = text != baseline_results[name]
            marker = " [CHANGED]" if changed else ""
            print(f"  {name}: {text}{marker}")
            config_results[name] = text

        all_results["configs"][config_name] = {
            "hotwords": hotwords if config_name != "F_in_initial_prompt" else None,
            "initial_prompt": extended_prompt if config_name == "F_in_initial_prompt" else base_prompt,
            "results": config_results,
        }

    # Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存到 {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()
