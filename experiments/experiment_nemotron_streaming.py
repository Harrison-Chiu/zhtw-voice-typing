r"""Quick native streaming check for Nemotron 3.5 ASR.

This intentionally measures only a small set of existing fixtures and the two
extreme supported lookahead settings (80 ms and 1120 ms). It is a go/no-go
check before investing in the full benchmark matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_nemotron_smoke import (  # noqa: E402
    MODEL_ID,
    build_processors,
    process_current,
)

DEFAULT_AUDIO = [
    PROJECT_ROOT / "data" / "test_audio" / "中英錄音測試.m4a",
    PROJECT_ROOT / "data" / "test_audio" / "機器人展示，語音轉錄測試.m4a",
]
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


def stream_once(model, processor, audio: np.ndarray, language: str, lookahead: int) -> dict:
    import torch
    from transformers import TextIteratorStreamer

    processor.set_num_lookahead_tokens(lookahead)
    sample_rate = processor.feature_extractor.sampling_rate
    prep_start = time.perf_counter()
    first_chunk_inputs = processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sample_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=language,
        return_tensors="pt",
    )
    first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)
    prep_sec = time.perf_counter() - prep_start

    def input_features_generator():
        yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        start_idx = mel_frame_idx * hop_length - n_fft // 2
        while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=sample_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=language,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            yield inputs.input_features
            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    generate_kwargs = {
        **first_chunk_inputs,
        "input_features": input_features_generator(),
        "streamer": streamer,
        "num_lookahead_tokens": lookahead,
    }
    error: list[BaseException] = []

    def generate() -> None:
        try:
            model.generate(**generate_kwargs)
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    start = time.perf_counter()
    thread = threading.Thread(target=generate)
    thread.start()
    pieces: list[str] = []
    first_text_sec = None
    for piece in streamer:
        pieces.append(piece)
        if first_text_sec is None and piece:
            first_text_sec = time.perf_counter() - start
    thread.join()
    if error:
        raise error[0]

    raw = "".join(pieces).strip()
    punctuation, current_opencc, terms, _forced_opencc = build_processors()
    processed = process_current(raw, punctuation, current_opencc, terms)
    total_sec = time.perf_counter() - start
    audio_sec = len(audio) / sample_rate
    return {
        "language": language,
        "lookahead": lookahead,
        "streaming_latency_ms": processor.streaming_latency_ms,
        "audio_sec": round(audio_sec, 3),
        "prep_sec": round(prep_sec, 4),
        "first_text_sec": round(first_text_sec, 4) if first_text_sec is not None else None,
        "final_text_sec": round(total_sec, 4),
        "rtf": round(total_sec / audio_sec, 5),
        "raw": raw,
        "processed_current": processed,
    }


def run(paths: list[Path], languages: list[str], lookaheads: list[int]) -> dict:
    import torch
    from transformers import AutoModelForRNNT, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    sample_rate = processor.feature_extractor.sampling_rate
    model = AutoModelForRNNT.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    results = []
    for path in paths:
        audio = decode_audio(path, sample_rate)
        for language in languages:
            for lookahead in lookaheads:
                processor.set_num_lookahead_tokens(lookahead)
                print(
                    f"Running {path.name} / {language} / {processor.streaming_latency_ms}ms",
                    flush=True,
                )
                item = stream_once(model, processor, audio, language, lookahead)
                item["audio"] = path.name
                results.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": MODEL_ID,
            "languages": languages,
            "lookaheads": lookaheads,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "runs": results,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", action="append", type=Path)
    parser.add_argument("--language", action="append", choices=["zh-CN", "auto"])
    parser.add_argument("--lookahead", action="append", type=int, choices=[0, 3, 6, 13])
    args = parser.parse_args()
    paths = args.audio or DEFAULT_AUDIO
    if any(not path.exists() for path in paths):
        parser.error("one or more audio files do not exist")
    result = run(paths, args.language or ["zh-CN", "auto"], args.lookahead or [0, 13])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"nemotron_streaming_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
