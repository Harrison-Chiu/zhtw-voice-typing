"""Test the full mic pipeline with audio files (no real microphone needed).

Uses FileAudioSource as a fake microphone, so this drives the EXACT path that
main.py uses for real mic input: AudioSource -> engine.transcribe() -> pipeline.
Engine + post-processing come from config.yaml, so it always tests the current
default engine (whisper).
"""

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from asr_input.asr import build_engine  # noqa: E402
from asr_input.audio.capture import FileAudioSource  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.processing.opencc_conv import OpenCCConverter  # noqa: E402
from asr_input.processing.pipeline import ProcessingPipeline  # noqa: E402
from asr_input.processing.tw_terms import TaiwanTermReplacer  # noqa: E402

TEST_DIR = Path("data/test_audio")


def main() -> None:
    config = load_config()
    asr_cfg = config["asr"]
    sample_rate = config["audio"]["sample_rate"]

    pipeline = ProcessingPipeline()
    pipeline.add(OpenCCConverter(config["processing"]["opencc_config"]))
    pipeline.add(TaiwanTermReplacer())

    print(f"引擎: {asr_cfg.get('engine', 'qwen')} / {asr_cfg['model_id']}")
    print("載入模型中...")
    t0 = time.time()
    engine = build_engine(asr_cfg)
    engine.load()
    print(f"模型載入完成 ({time.time() - t0:.1f}s)\n")

    audio_files = sorted(TEST_DIR.glob("*.m4a"))
    if not audio_files:
        print(f"No .m4a files found in {TEST_DIR}")
        return

    for audio_path in audio_files:
        print(f"=== {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB) ===")

        mic = FileAudioSource(audio_path, sample_rate=sample_rate)
        mic.start()
        audio = mic.stop()

        start = time.time()
        raw_text = engine.transcribe(audio, sample_rate)
        elapsed = time.time() - start
        processed = pipeline.run(raw_text)

        print(f"  Time     : {elapsed:.1f}s ({len(audio) / sample_rate:.1f}s audio)")
        print(f"  Raw      : {raw_text}")
        print(f"  Processed: {processed}\n")

    engine.unload()
    print("Done!")


if __name__ == "__main__":
    main()
