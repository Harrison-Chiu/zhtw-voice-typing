"""Compare the torch JIT and ONNX Silero VAD backends on real audio.

Question this answers: can the recording path drop torch (and its cold-start
cost) without changing what gets sent to ASR?

Two things are measured, because they fail independently:

1. Per-window probability agreement.  Both backends run the same Silero v5
   weights, so large divergence would mean the ONNX state handling is wrong,
   not that the model is "slightly different".
2. Segment agreement through the real `StreamingVAD`.  Probabilities near the
   threshold can flip a decision, so identical-looking probabilities still need
   the downstream segment boundaries checked.

Run:
    .venv\\Scripts\\python.exe experiments/compare_vad_backends.py <audio...>

Without arguments it uses every wav/m4a in data/test_audio/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from asr_input.audio.streaming_vad import StreamingVAD  # noqa: E402
from asr_input.audio.vad_backend import SileroOnnxVAD, load_torch_vad  # noqa: E402

SAMPLE_RATE = 16000
CHUNK = StreamingVAD.SILERO_CHUNK_SAMPLES


def load_audio(path: Path) -> np.ndarray:
    """Decode to 16 kHz mono float32 via ffmpeg (librosa is broken by numpy 2.5)."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def probabilities(model, audio: np.ndarray) -> tuple[list[float], float]:
    model.reset_states()
    probs: list[float] = []
    t0 = time.perf_counter()
    for start in range(0, len(audio) - CHUNK + 1, CHUNK):
        probs.append(float(model(audio[start : start + CHUNK], SAMPLE_RATE)))
    return probs, time.perf_counter() - t0


def segments(model, audio: np.ndarray) -> list[tuple[int, int]]:
    """Run the production segmenter and report each emitted segment's length."""
    emitted: list[tuple[int, int]] = []

    def on_segment(segment: np.ndarray, probs: list[float]) -> None:
        emitted.append((len(segment), len(probs)))

    vad = StreamingVAD(on_speech_segment=on_segment, sample_rate=SAMPLE_RATE, verbose=False)
    vad.load(model)
    vad.reset()
    # Feed in microphone-sized blocks so buffering behaves as it does live.
    for start in range(0, len(audio), 1024):
        vad.feed(audio[start : start + 1024])
    vad.flush()
    return emitted


def compare(path: Path, torch_model, onnx_model) -> dict:
    audio = load_audio(path)
    torch_probs, torch_sec = probabilities(torch_model, audio)
    onnx_probs, onnx_sec = probabilities(onnx_model, audio)

    n = min(len(torch_probs), len(onnx_probs))
    diff = np.abs(np.array(torch_probs[:n]) - np.array(onnx_probs[:n]))
    torch_speech = np.array(torch_probs[:n]) >= 0.5
    onnx_speech = np.array(onnx_probs[:n]) >= 0.5

    torch_segments = segments(torch_model, audio)
    onnx_segments = segments(onnx_model, audio)

    return {
        "file": path.name,
        "duration_sec": round(len(audio) / SAMPLE_RATE, 2),
        "windows": n,
        "prob_max_abs_diff": float(diff.max()) if n else None,
        "prob_mean_abs_diff": float(diff.mean()) if n else None,
        "decision_flips": int((torch_speech != onnx_speech).sum()),
        "torch_segments": torch_segments,
        "onnx_segments": onnx_segments,
        "segments_identical": torch_segments == onnx_segments,
        "torch_infer_sec": round(torch_sec, 3),
        "onnx_infer_sec": round(onnx_sec, 3),
    }


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        audio_dir = REPO / "data" / "test_audio"
        paths = sorted(p for p in audio_dir.iterdir() if p.suffix in {".wav", ".m4a"})
    if not paths:
        raise SystemExit("no audio files given and data/test_audio/ is empty")

    t0 = time.perf_counter()
    torch_model = load_torch_vad()
    torch_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    onnx_model = SileroOnnxVAD()
    onnx_load = time.perf_counter() - t0

    results = []
    for path in paths:
        print(f"--- {path.name}", flush=True)
        result = compare(path, torch_model, onnx_model)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        results.append(result)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "torch_model_load_sec": round(torch_load, 2),
        "onnx_model_load_sec": round(onnx_load, 2),
        "files": results,
    }
    out = (
        REPO
        / "experiments"
        / "results"
        / f"vad_backend_compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()
