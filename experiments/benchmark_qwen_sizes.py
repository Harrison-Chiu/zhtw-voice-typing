r"""Compare faster-whisper with Qwen3-ASR 0.6B and 1.7B.

The parent process runs every engine in an isolated child so GPU memory is
released between candidates. Results include raw text and two OpenCC policies:
character-only s2t and phrase-aware s2twp.

    .venv\Scripts\python.exe experiments\benchmark_qwen_sizes.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from opencc import OpenCC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "test_audio" / "segments"
ENGINES = {
    "faster-whisper": "large-v3-turbo",
    "qwen-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen-1.7b": "Qwen/Qwen3-ASR-1.7B",
}


class GPUMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_mib: list[int] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                ).stdout
                self.samples_mib.append(int(output.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError):
                return


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), sample_rate


def conversions(raw: str) -> dict[str, str]:
    from asr_input.processing.punct_norm import PunctuationNormalizer
    from asr_input.processing.tw_terms import TaiwanTermReplacer

    punctuated = PunctuationNormalizer().process(raw)
    return {
        "raw": raw,
        "character_s2t": OpenCC("s2t").convert(punctuated),
        "phrase_s2twp": OpenCC("s2twp").convert(punctuated),
        "current_pipeline": TaiwanTermReplacer().process(OpenCC("s2twp").convert(punctuated)),
    }


def build_engine(name: str):
    from asr_input.asr.qwen import QwenASREngine
    from asr_input.asr.whisper_fw import WhisperFWEngine

    if name == "faster-whisper":
        return WhisperFWEngine(
            model_id=ENGINES[name],
            device="cuda",
            compute_type="float16",
            language="zh",
            initial_prompt="使用繁體中文，台灣用語。",
            hotwords="詞表 待辦 清單 聲學 標點 主分支 逗號",
            beam_size=5,
        )
    return QwenASREngine(
        model_id=ENGINES[name],
        device="cuda",
        language="Chinese",
        context="請以繁體中文輸出。",
    )


def run_worker(engine_name: str, paths: list[Path]) -> dict:
    monitor = GPUMonitor()
    monitor.start()
    baseline_mib = None
    while baseline_mib is None and len(monitor.samples_mib) < 1:
        time.sleep(0.12)
    if monitor.samples_mib:
        baseline_mib = monitor.samples_mib[-1]

    engine = build_engine(engine_name)
    load_start = time.perf_counter()
    engine.load()
    load_sec = time.perf_counter() - load_start
    runs = []
    for path in paths:
        audio, sample_rate = read_audio(path)
        start = time.perf_counter()
        raw = engine.transcribe(audio, sample_rate)
        infer_sec = time.perf_counter() - start
        audio_sec = len(audio) / sample_rate
        runs.append(
            {
                "audio": path.name,
                "audio_sec": round(audio_sec, 3),
                "infer_sec": round(infer_sec, 4),
                "rtf": round(infer_sec / audio_sec, 5),
                **conversions(raw),
            }
        )
    engine.unload()
    monitor.stop()
    peak_mib = max(monitor.samples_mib) if monitor.samples_mib else None
    return {
        "engine": engine_name,
        "model": ENGINES[engine_name],
        "status": "ok",
        "load_sec": round(load_sec, 4),
        "gpu_baseline_mib": baseline_mib,
        "gpu_peak_mib": peak_mib,
        "gpu_increment_mib": peak_mib - baseline_mib if peak_mib and baseline_mib else None,
        "runs": runs,
    }


def run_parent(paths: list[Path], engines: list[str]) -> dict:
    results = []
    for engine in engines:
        print(f"\n=== {engine} ===", flush=True)
        command = [sys.executable, __file__, "--worker", engine]
        for path in paths:
            command.extend(["--audio", str(path)])
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        if completed.returncode:
            print(completed.stdout, end="")
            print(completed.stderr, file=sys.stderr, end="")
            results.append(
                {
                    "engine": engine,
                    "model": ENGINES[engine],
                    "status": "error",
                    "returncode": completed.returncode,
                    "error": completed.stderr[-4000:],
                }
            )
            continue
        item = json.loads(completed.stdout)
        results.append(item)
        print(
            f"load={item['load_sec']}s, GPU +{item['gpu_increment_mib']} MiB, "
            f"mean RTF={sum(run['rtf'] for run in item['runs']) / len(item['runs']):.3f}",
            flush=True,
        )
    return {
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(),
            "audio": [path.name for path in paths],
            "note": "GPU increment is global nvidia-smi peak minus pre-load global baseline.",
        },
        "engines": results,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", action="append", type=Path)
    parser.add_argument("--engine", action="append", choices=ENGINES)
    parser.add_argument("--worker", choices=ENGINES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    paths = args.audio or sorted(DEFAULT_AUDIO_DIR.glob("*.wav"))
    if not paths or any(not path.exists() for path in paths):
        parser.error("benchmark audio is missing")
    if args.worker:
        print(json.dumps(run_worker(args.worker, paths), ensure_ascii=False))
        return

    result = run_parent(paths, args.engine or list(ENGINES))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"qwen_size_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
