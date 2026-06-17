"""Entry B — test that main() works, using a fake microphone (no hardware).

Drives main's real core path (`capture_and_transcribe`) with a FileAudioSource
standing in for the mic, so this verifies the same code the live-mic CLI runs —
not a re-implementation of it. Engine + post-processing come from config.yaml.

Usage:
    .venv\\Scripts\\python.exe test_audio_file.py [audio_path ...]
    (no args -> all .m4a in data/test_audio/)
"""

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from asr_input.asr import build_engine  # noqa: E402
from asr_input.audio.capture import FileAudioSource  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline, capture_and_transcribe  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data/test_audio"


def main() -> None:
    config = load_config()
    asr_cfg = config["asr"]
    sample_rate = config["audio"]["sample_rate"]

    print(f"引擎: {asr_cfg.get('engine', 'qwen')} / {asr_cfg['model_id']}")
    print("載入模型中...")
    t0 = time.time()
    engine = build_engine(asr_cfg, vad_cfg=config.get("vad"))
    engine.load()
    pipeline = build_pipeline(config)
    print(f"模型載入完成 ({time.time() - t0:.1f}s)\n")

    if len(sys.argv) > 1:
        audio_files = [Path(p) for p in sys.argv[1:]]
    else:
        audio_files = sorted(TEST_DIR.glob("*.m4a"))
    if not audio_files:
        print(f"No audio files found (TEST_DIR={TEST_DIR})")
        return

    for audio_path in audio_files:
        print(f"=== {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB) ===")
        source = FileAudioSource(audio_path, sample_rate=sample_rate)

        start = time.time()
        result = capture_and_transcribe(source, engine, pipeline, sample_rate)
        elapsed = time.time() - start

        if result is None:
            print("  (沒有收到音訊或沒有辨識到文字)\n")
            continue
        raw_text, processed = result
        print(f"  Time     : {elapsed:.1f}s")
        print(f"  Raw      : {raw_text}")
        print(f"  Processed: {processed}\n")

    engine.unload()
    print("Done!")


if __name__ == "__main__":
    main()
