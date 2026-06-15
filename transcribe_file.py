"""Entry A — transcribe an audio file (the simple path).

Just file -> engine -> post-processing. No fake microphone, no main() loop; this is
the minimal path for "I have a file, give me the text".

Usage:
    .venv\\Scripts\\python.exe transcribe_file.py <audio_path>
"""

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa  # noqa: E402

from asr_input.asr import build_engine  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline  # noqa: E402


def transcribe_file(path: str) -> str:
    config = load_config()
    sample_rate = config["audio"]["sample_rate"]

    engine = build_engine(config["asr"])
    engine.load()
    try:
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
        raw = engine.transcribe(audio, sample_rate)
        return build_pipeline(config).run(raw)
    finally:
        engine.unload()


def main() -> None:
    if len(sys.argv) != 2:
        print("用法: python transcribe_file.py <音檔路徑>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"轉錄: {path}\n載入模型中...")
    t0 = time.time()
    text = transcribe_file(path)
    print(f"({time.time() - t0:.1f}s)\n{text}")


if __name__ == "__main__":
    main()
