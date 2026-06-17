"""Test streaming ASR by simulating real-time audio input from a file.

Loads an audio file, then feeds it to StreamingSession in small chunks
(simulating microphone input) to verify VAD segmentation + per-sentence
transcription works correctly.

Usage:
    .venv\\Scripts\\python.exe test_streaming.py [audio_path]
    (no args -> first .m4a in data/test_audio/)
"""

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from asr_input.asr import build_engine  # noqa: E402
from asr_input.audio.streaming_vad import StreamingVAD  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline  # noqa: E402

TEST_DIR = Path("data/test_audio")
CHUNK_SIZE = 1024  # samples per chunk, same as sounddevice default


def main() -> None:
    config = load_config()
    asr_cfg = config["asr"]
    sample_rate = config["audio"]["sample_rate"]
    streaming_cfg = config.get("streaming", {})
    silence_trigger_ms = streaming_cfg.get("silence_trigger_ms", 1000)
    vad_cfg = config.get("vad", {})

    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        files = sorted(TEST_DIR.glob("*.m4a"))
        if not files:
            print(f"No audio files found in {TEST_DIR}")
            return
        audio_path = files[0]

    print(f"=== 串流辨識測試: {audio_path.name} ===")
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

    # Collect speech segments via StreamingVAD
    segments: list[np.ndarray] = []

    def on_segment(seg_audio: np.ndarray) -> None:
        segments.append(seg_audio)

    vad = StreamingVAD(
        on_speech_segment=on_segment,
        sample_rate=sample_rate,
        threshold=vad_cfg.get("threshold", 0.5),
        min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
        silence_trigger_ms=silence_trigger_ms,
        speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
    )
    vad.load(vad_model)

    # Simulate real-time: feed audio in chunks
    print(f"模擬即時輸入（{CHUNK_SIZE} samples/chunk = {CHUNK_SIZE/sample_rate*1000:.0f}ms）...")
    t0 = time.time()
    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i : i + CHUNK_SIZE]
        vad.feed(chunk)
    vad.flush()
    vad_time = time.time() - t0
    print(f"VAD 完成: {len(segments)} 段語音（{vad_time:.2f}s）\n")

    # Transcribe each segment
    results: list[str] = []
    for i, seg_audio in enumerate(segments):
        seg_sec = len(seg_audio) / sample_rate
        t0 = time.time()
        raw = engine.transcribe(seg_audio, sample_rate)
        dt = time.time() - t0

        if not raw.strip():
            print(f"  段 {i+1}: {seg_sec:.1f}s → (空)")
            continue

        processed = pipeline.run(raw)
        results.append(processed)
        print(f"  段 {i+1}: {seg_sec:.1f}s → {dt:.1f}s 辨識")
        print(f"    原始: {raw}")
        print(f"    處理: {processed}")

    print("\n=== 完整結果 ===")
    full_text = "".join(results)
    print(full_text)
    print(f"\n共 {len(results)} 段")

    engine.unload()


if __name__ == "__main__":
    main()
