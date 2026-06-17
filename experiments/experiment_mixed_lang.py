"""Experiment: best language + initial_prompt for Chinese-English code-switching.

Audio: data/test_audio/中英錄音測試.m4a (中英夾雜，含 Whisper / ZH / Initial Prompt 等英文詞).
Matrix: language {zh, auto} x initial_prompt {none, 純中, 中英示範, 混合描述}.

We print full text for human eval, plus: English alpha count (是否保留英文),
Simplified-char ratio (中文是否繁體), punctuation count.

Run:
    .venv\\Scripts\\python.exe experiment_mixed_lang.py
"""

import io
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from opencc import OpenCC  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO = str(PROJECT_ROOT / "data/test_audio/中英錄音測試.m4a")
_s2t = OpenCC("s2t")
_PUNCT = "，。、！？；：「」（）"


def simplified_count(text: str) -> int:
    return sum(1 for ch in text if ch.strip() and _s2t.convert(ch) != ch)


def english_count(text: str) -> int:
    return sum(len(w) for w in re.findall(r"[A-Za-z]+", text))


LANGS = {"zh": "zh", "auto": None}
PROMPTS = {
    "P0_none": None,
    "P1_zhonly": "繁體中文，台灣用語。",
    "P2_bilingual": "繁體中文，會中英夾雜，例如 Whisper、prompt、ZH 模式。",
    "P3_describe": "以下是中英混合的台灣繁體中文逐字稿。",
}


def main() -> None:
    audio, sr = librosa.load(AUDIO, sr=16000, mono=True)
    print(f"audio: {len(audio) / sr:.1f}s\nloading model...")
    m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print()

    for lang_tag, lang in LANGS.items():
        for p_tag, prompt in PROMPTS.items():
            t = time.time()
            segs, info = m.transcribe(audio, language=lang, beam_size=5, initial_prompt=prompt)
            txt = "".join(s.text for s in segs).strip()
            dt = time.time() - t
            han = sum(1 for c in txt if "一" <= c <= "鿿")
            ns = simplified_count(txt)
            ratio = ns / han if han else 0.0
            print(
                f"[{lang_tag:4s} | {p_tag:12s}] "
                f"簡{ratio:4.0%} 英{english_count(txt):2d} 標{sum(c in _PUNCT for c in txt):2d} "
                f"{dt:.1f}s det={info.language}"
            )
            print(f"    {txt}\n")


if __name__ == "__main__":
    main()
