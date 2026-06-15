"""Test ASR pipeline with audio files (no mic needed)."""

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch  # noqa: E402

from asr_input.asr.qwen import QwenASREngine  # noqa: E402
from asr_input.processing.opencc_conv import OpenCCConverter  # noqa: E402
from asr_input.processing.pipeline import ProcessingPipeline  # noqa: E402
from asr_input.processing.tw_terms import TaiwanTermReplacer  # noqa: E402

TEST_DIR = Path("data/test_audio")


def main() -> None:
    pipeline = ProcessingPipeline()
    pipeline.add(OpenCCConverter("s2twp"))
    pipeline.add(TaiwanTermReplacer())

    print("Loading Qwen3-ASR-1.7B...")
    engine = QwenASREngine(
        model_id="Qwen/Qwen3-ASR-1.7B",
        device="cuda",
        language="Chinese",
        prompt="以下是台灣繁體中文的語音轉錄。",
    )
    engine.load()
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print()

    audio_files = sorted(TEST_DIR.glob("*.m4a"))
    if not audio_files:
        print(f"No .m4a files found in {TEST_DIR}")
        return

    for audio_path in audio_files:
        print(f"=== {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB) ===")

        start = time.time()
        results = engine._model.transcribe(audio=str(audio_path), language="Chinese")
        elapsed = time.time() - start

        raw_text = results[0].text if results else "(no output)"
        processed = pipeline.run(raw_text)

        print(f"  Time    : {elapsed:.1f}s")
        print(f"  Raw     : {raw_text}")
        print(f"  Processed: {processed}")
        print()

    engine.unload()
    print("Done!")


if __name__ == "__main__":
    main()
