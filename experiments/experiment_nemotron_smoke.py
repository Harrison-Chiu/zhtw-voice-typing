r"""Minimal Nemotron 3.5 ASR smoke test.

This is intentionally separate from the production ASR factory.  It exercises
the official Transformers RNNT path on the existing short fixture and records
both the current post-processing behavior and an unconditional s2twp
diagnostic.

Run with the isolated environment:

    .venv-nemotron\Scripts\python.exe experiments\experiment_nemotron_smoke.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "test_audio" / "中英錄音測試.m4a"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    raw = subprocess.run(
        [
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
            str(sample_rate),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def build_processors() -> tuple[object, object, object, object]:
    from opencc import OpenCC
    from asr_input.processing.opencc_conv import OpenCCConverter
    from asr_input.processing.punct_norm import PunctuationNormalizer
    from asr_input.processing.tw_terms import TaiwanTermReplacer

    return PunctuationNormalizer(), OpenCCConverter("s2twp"), TaiwanTermReplacer(), OpenCC(
        "s2twp"
    )


def process_current(raw: str, punctuation: object, opencc: object, terms: object) -> str:
    text = punctuation.process(raw)
    text = opencc.process(text)
    return terms.process(text)


def process_forced_s2twp(
    raw: str, punctuation: object, forced_opencc: object, terms: object
) -> str:
    text = punctuation.process(raw)
    text = forced_opencc.convert(text)
    return terms.process(text)


def decode_text(processor: object, sequences: object, *, skip_special_tokens: bool) -> str:
    decoded = processor.decode(sequences, skip_special_tokens=skip_special_tokens)
    if isinstance(decoded, list):
        return "".join(str(item) for item in decoded).strip()
    return str(decoded).strip()


def run(audio_path: Path, languages: list[str]) -> dict:
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor

    punctuation, current_opencc, terms, forced_opencc = build_processors()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    sample_rate = processor.feature_extractor.sampling_rate
    audio = decode_audio(audio_path, sample_rate)

    load_start = time.perf_counter()
    model = AutoModelForRNNT.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    load_sec = time.perf_counter() - load_start

    runs = []
    for language in languages:
        inputs = processor(audio, sampling_rate=sample_rate, language=language)
        inputs = inputs.to(model.device, dtype=model.dtype)
        start = time.perf_counter()
        output = model.generate(**inputs, return_dict_in_generate=True)
        infer_sec = time.perf_counter() - start
        raw = decode_text(processor, output.sequences, skip_special_tokens=True)
        tagged = decode_text(processor, output.sequences, skip_special_tokens=False)
        runs.append(
            {
                "language": language,
                "infer_sec": round(infer_sec, 4),
                "rtf": round(infer_sec / (len(audio) / sample_rate), 5),
                "raw": raw,
                "decoded_with_special_tokens": tagged,
                "processed_current": process_current(
                    raw, punctuation, current_opencc, terms
                ),
                "processed_forced_s2twp": process_forced_s2twp(
                    raw, punctuation, forced_opencc, terms
                ),
            }
        )

    result = {
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": MODEL_ID,
            "audio": audio_path.name,
            "audio_sec": round(len(audio) / sample_rate, 3),
            "sample_rate": sample_rate,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "load_sec": round(load_sec, 4),
        "runs": runs,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        choices=["zh-CN", "auto"],
        help="Repeat to compare language conditioning modes (default: zh-CN and auto)",
    )
    args = parser.parse_args()
    if not args.audio.exists():
        parser.error(f"audio file not found: {args.audio}")

    result = run(args.audio, args.languages or ["zh-CN", "auto"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS_DIR / f"nemotron_smoke_{stamp}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
