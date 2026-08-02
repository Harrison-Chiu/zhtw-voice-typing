"""Measure whether microphone callbacks keep running while Whisper loads.

No audio samples are retained. The result only contains callback counts,
timing gaps, PortAudio status messages, and aggregate RMS range.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml

from asr_input.asr import build_engine

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments" / "results"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024


def main() -> None:
    callback_times: list[float] = []
    callback_frames: list[int] = []
    callback_rms: list[float] = []
    statuses: list[str] = []
    lock = threading.Lock()

    def callback(indata, frames, time_info, status) -> None:
        del time_info
        with lock:
            callback_times.append(time.perf_counter())
            callback_frames.append(frames)
            callback_rms.append(float(np.sqrt(np.mean(indata**2))))
            if status:
                statuses.append(str(status))

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    engine = build_engine(config["asr"], vad_cfg=None)

    requested_device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else SAMPLE_RATE
    device_info = sd.query_devices(requested_device, "input")
    started = time.perf_counter()
    with sd.InputStream(
        device=requested_device,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        stream_started = time.perf_counter()
        time.sleep(0.25)
        load_started = time.perf_counter()
        engine.load()
        load_finished = time.perf_counter()
        time.sleep(0.5)
        stream_finished = time.perf_counter()

    intervals = np.diff(callback_times)
    stream_sec = stream_finished - stream_started
    expected_frames = stream_sec * sample_rate
    actual_frames = sum(callback_frames)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "privacy": "audio content was not retained",
        "input_device": {
            "requested_index": requested_device,
            "name": device_info["name"],
            "hostapi": sd.query_hostapis(device_info["hostapi"])["name"],
        },
        "sample_rate": sample_rate,
        "block_size": BLOCK_SIZE,
        "stream_sec": stream_sec,
        "model_load_sec": load_finished - load_started,
        "callback_count": len(callback_times),
        "expected_frames_from_wall_clock": expected_frames,
        "actual_frames": actual_frames,
        "frame_coverage_ratio": actual_frames / expected_frames if expected_frames else 0,
        "callback_interval_sec": {
            "expected": BLOCK_SIZE / sample_rate,
            "mean": float(np.mean(intervals)) if len(intervals) else None,
            "p95": float(np.percentile(intervals, 95)) if len(intervals) else None,
            "max": float(np.max(intervals)) if len(intervals) else None,
        },
        "status_messages": statuses,
        "rms_range": {
            "min": min(callback_rms, default=None),
            "max": max(callback_rms, default=None),
        },
        "total_experiment_sec": time.perf_counter() - started,
    }
    payload["passed"] = (
        not statuses
        and payload["frame_coverage_ratio"] >= 0.95
        and payload["callback_interval_sec"]["max"] is not None
        and payload["callback_interval_sec"]["max"] <= BLOCK_SIZE / sample_rate * 2.5
    )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    output = RESULT_ROOT / f"capture_while_loading_{stamp}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
