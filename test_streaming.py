"""Test streaming ASR by simulating real-time audio input from a file.

Feeds an audio file through the full StreamingSession pipeline (the same
code path used by TrayApp), using start_from_file() to simulate microphone
input. This verifies VAD + ASR + post-processing end-to-end.

Outputs a JSON report to data/streaming_test_results.json.

Usage:
    .venv\\Scripts\\python.exe test_streaming.py [audio_path]
    (no args -> first .m4a in data/test_audio/)
"""

import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from asr_input.asr import build_engine  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline  # noqa: E402
from asr_input.streaming import StreamingSession  # noqa: E402

TEST_DIR = Path("data/test_audio")
OUTPUT_PATH = Path("data/streaming_test_results.json")


def main() -> None:
    config = load_config()
    asr_cfg = config["asr"]
    sample_rate = config["audio"]["sample_rate"]
    streaming_cfg = config.get("streaming", {})
    vad_cfg = config.get("vad", {})

    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        files = sorted(TEST_DIR.glob("*.m4a"))
        if not files:
            print(f"No audio files found in {TEST_DIR}")
            return
        audio_path = files[0]

    silence_trigger_ms = streaming_cfg.get("silence_trigger_ms", 1000)
    print(f"=== 串流辨識測試（完整 pipeline）: {audio_path.name} ===")
    print(f"引擎: {asr_cfg.get('engine', 'qwen')} / {asr_cfg['model_id']}")
    print(f"靜音切句門檻: {silence_trigger_ms}ms")
    print()

    # Load audio file
    print("載入音檔...", flush=True)
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    audio_sec = len(audio) / sample_rate
    print(f"音檔長度: {audio_sec:.1f}s")

    # Load models
    print("載入 ASR 模型...", flush=True)
    engine = build_engine(asr_cfg, vad_cfg=None)
    engine.load()
    pipeline = build_pipeline(config, include_output=False)

    print("載入 VAD 模型...", flush=True)
    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    print("模型載入完成!\n")

    # Run full streaming pipeline (same code path as TrayApp)
    session = StreamingSession(
        engine=engine,
        pipeline=pipeline,
        vad_model=vad_model,
        sample_rate=sample_rate,
        silence_trigger_ms=silence_trigger_ms,
        silence_min_ms=streaming_cfg.get("silence_min_ms", 300),
        ramp_start_sec=streaming_cfg.get("ramp_start_sec", 10.0),
        ramp_end_sec=streaming_cfg.get("ramp_end_sec", 25.0),
        vad_threshold=vad_cfg.get("threshold", 0.5),
        min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
        speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
        max_segment_sec=streaming_cfg.get("max_segment_sec", 30.0),
        min_energy=streaming_cfg.get("min_energy", 0.005),
    )

    print("模擬串流輸入中...", flush=True)
    t0 = time.time()
    session.start_from_file(audio)
    full_text = session.stop()
    total_time = time.time() - t0

    stats = session.segment_stats
    audio_durations = [s["audio_sec"] for s in stats]
    transcribe_times = [s["transcribe_sec"] for s in stats]

    print(f"\n=== 完整結果（{session.segment_count} 段, {total_time:.1f}s）===")
    print(full_text or "（沒有辨識到文字）")

    # Build JSON report
    report = {
        "meta": {
            "ts": datetime.now(UTC).isoformat(),
            "audio_file": audio_path.name,
            "audio_total_sec": round(audio_sec, 1),
            "engine": f"{asr_cfg.get('engine', 'qwen')} / {asr_cfg['model_id']}",
            "silence_trigger_ms": silence_trigger_ms,
            "silence_min_ms": streaming_cfg.get("silence_min_ms", 300),
            "ramp_start_sec": streaming_cfg.get("ramp_start_sec", 10.0),
            "ramp_end_sec": streaming_cfg.get("ramp_end_sec", 25.0),
            "max_segment_sec": streaming_cfg.get("max_segment_sec", 30.0),
            "min_energy": streaming_cfg.get("min_energy", 0.005),
            "total_transcribe_sec": round(total_time, 1),
        },
        "summary": {
            "total_segments": len(stats),
            "empty_segments": sum(1 for s in stats if s["empty"]),
            "audio_sec_min": round(min(audio_durations), 2) if audio_durations else 0,
            "audio_sec_max": round(max(audio_durations), 2) if audio_durations else 0,
            "audio_sec_avg": round(np.mean(audio_durations), 2) if audio_durations else 0,
            "audio_sec_median": round(float(np.median(audio_durations)), 2)
            if audio_durations
            else 0,
            "transcribe_sec_min": round(min(transcribe_times), 2)
            if transcribe_times
            else 0,
            "transcribe_sec_max": round(max(transcribe_times), 2)
            if transcribe_times
            else 0,
            "transcribe_sec_avg": round(np.mean(transcribe_times), 2)
            if transcribe_times
            else 0,
            "realtime_factor": round(audio_sec / total_time, 1) if total_time > 0 else 0,
        },
        "segments": [{"index": i + 1, **s} for i, s in enumerate(stats)],
        "full_text": full_text,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n報告已輸出: {OUTPUT_PATH}")

    engine.unload()
    print("Done!")


if __name__ == "__main__":
    main()
