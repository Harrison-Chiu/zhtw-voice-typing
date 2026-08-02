r"""Benchmark Whisper inference backends on this project's real audio samples.

This is deliberately an experiment, not a new production engine.  It keeps the
model, language, prompt and audio inputs as close as each backend allows, and
records latency plus transcripts so speed is never evaluated without quality.

Examples (run each backend in a fresh process for clean GPU-memory numbers):

    .venv\Scripts\python.exe experiments\benchmark_whisper_backends.py --backend fw
    .venv\Scripts\python.exe experiments\benchmark_whisper_backends.py --backend fw-batched
    .venv\Scripts\python.exe experiments\benchmark_whisper_backends.py --backend hf-sdpa

The Hugging Face run may download ``openai/whisper-large-v3-turbo``.  Flash
Attention 2 is intentionally not installed by this script; on Windows it is a
separate, comparatively fragile environment experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import librosa
import numpy as np

from asr_input.config import load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_DIR = ROOT / "data" / "test_audio"
RESULTS_DIR = ROOT / "experiments" / "results"


class GPUMemorySampler:
    """Poll total GPU memory use; useful for CT2, which torch cannot observe."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read_mib() -> int | None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            return int(output.splitlines()[0].strip())
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            return None

    def __enter__(self) -> GPUMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._read_mib()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)


def load_audio(paths: list[Path], sample_rate: int) -> list[tuple[Path, np.ndarray]]:
    loaded = []
    for path in paths:
        audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        loaded.append((path, audio.astype(np.float32)))
    return loaded


def transcribe_fw(model: object, audio: np.ndarray, config: dict) -> str:
    segments, _ = model.transcribe(
        audio,
        language=config.get("language", "zh"),
        beam_size=config.get("beam_size", 5),
        temperature=config.get("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        initial_prompt=config.get("initial_prompt") or None,
        hotwords=config.get("hotwords") or None,
    )
    return "".join(segment.text for segment in segments).strip()


def build_backend(name: str, config: dict, batch_size: int) -> tuple[object, object]:
    if name in {"fw", "fw-batched"}:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model = WhisperModel(
            config["model_id"],
            device=config.get("device", "cuda"),
            compute_type=config.get("compute_type", "float16"),
        )
        if name == "fw":
            return model, lambda audio: transcribe_fw(model, audio, config)
        batched = BatchedInferencePipeline(model=model)

        def run_batched(audio: np.ndarray) -> str:
            segments, _ = batched.transcribe(
                audio,
                batch_size=batch_size,
                language=config.get("language", "zh"),
                beam_size=config.get("beam_size", 5),
                initial_prompt=config.get("initial_prompt") or None,
                hotwords=config.get("hotwords") or None,
            )
            return "".join(segment.text for segment in segments).strip()

        return batched, run_batched

    if name == "hf-sdpa":
        import torch
        from transformers import pipeline

        model_id = config.get("hf_model_id", "openai/whisper-large-v3-turbo")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch.float16,
            device="cuda:0",
            model_kwargs={"attn_implementation": "sdpa"},
        )

        def run_hf(audio: np.ndarray) -> str:
            result = pipe(
                audio,
                chunk_length_s=30,
                batch_size=batch_size,
                return_timestamps=True,
                generate_kwargs={"language": "zh", "task": "transcribe"},
            )
            return result["text"].strip()

        return pipe, run_hf

    raise ValueError(name)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["fw", "fw-batched", "hf-sdpa"], required=True)
    parser.add_argument("--audio", action="append", type=Path, help="Repeat for multiple files")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.repeats < 1 or args.batch_size < 1:
        parser.error("--repeats and --batch-size must be positive")

    paths = args.audio or sorted(DEFAULT_AUDIO_DIR.glob("*.m4a"))
    if not paths:
        parser.error(f"no audio files found under {DEFAULT_AUDIO_DIR}")

    project_config = load_config()
    asr_config = project_config["asr"]
    sample_rate = project_config["audio"]["sample_rate"]
    audio_files = load_audio(paths, sample_rate)
    idle_mib = GPUMemorySampler._read_mib()

    with GPUMemorySampler() as gpu:
        load_start = time.perf_counter()
        backend, transcribe = build_backend(args.backend, asr_config, args.batch_size)
        load_sec = time.perf_counter() - load_start

        # One unreported warm-up removes lazy CUDA/kernel initialization from timing.
        transcribe(audio_files[0][1])

        runs = []
        for path, audio in audio_files:
            timings = []
            transcripts = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                transcripts.append(transcribe(audio))
                timings.append(time.perf_counter() - start)
            audio_sec = len(audio) / sample_rate
            runs.append(
                {
                    "file": path.name,
                    "audio_sec": round(audio_sec, 3),
                    "latency_sec": [round(value, 4) for value in timings],
                    "median_sec": round(statistics.median(timings), 4),
                    "rtf": round(statistics.median(timings) / audio_sec, 5),
                    "deterministic": len(set(transcripts)) == 1,
                    "transcript": transcripts[-1],
                }
            )

        # Keep a reference alive until sampling has captured the fully loaded state.
        _ = backend
        time.sleep(0.2)

    total_audio = sum(run["audio_sec"] for run in runs)
    total_median = sum(run["median_sec"] for run in runs)
    report = {
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(),
            "backend": args.backend,
            "model": (
                asr_config["model_id"]
                if args.backend.startswith("fw")
                else asr_config.get("hf_model_id", "openai/whisper-large-v3-turbo")
            ),
            "batch_size": args.batch_size if args.backend != "fw" else 1,
            "repeats": args.repeats,
            "beam_size": asr_config.get("beam_size", 5),
            "load_sec": round(load_sec, 3),
            "gpu_idle_mib": idle_mib,
            "gpu_peak_total_mib": max(gpu.samples) if gpu.samples else None,
            "gpu_peak_delta_mib": (
                max(gpu.samples) - idle_mib if gpu.samples and idle_mib is not None else None
            ),
        },
        "summary": {
            "audio_sec": round(total_audio, 3),
            "sum_median_sec": round(total_median, 4),
            "aggregate_rtf": round(total_median / total_audio, 5),
            "realtime_x": round(total_audio / total_median, 2),
        },
        "runs": runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS_DIR / f"whisper_backend_{args.backend}_{stamp}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
