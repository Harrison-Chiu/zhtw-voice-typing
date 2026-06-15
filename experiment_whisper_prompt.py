"""Experiment: can faster-whisper's `initial_prompt` steer script (繁/簡) and punctuation?

Unlike Qwen3-ASR's inert `context`, Whisper's initial_prompt is prepended as preceding
text and is known to bias style. We test two things at once:
  1. Script   — does a Traditional prompt push output to Traditional? (simp% should drop)
  2. Punctuation — does a prompt WITH full-width punctuation make output punctuated?

Paired reads:
  A(none) vs B(繁+標點)  -> does prompt do anything
  B(繁+標點) vs C(繁無標點) -> is punctuation in prompt what drives output punctuation
  B(繁) vs D(簡)         -> does prompt SCRIPT decide output script

Run:
    .venv\\Scripts\\python.exe experiment_whisper_prompt.py
"""

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from faster_whisper import WhisperModel  # noqa: E402
from opencc import OpenCC  # noqa: E402

AUDIO = "data/test_audio/機器人展示，語音轉錄測試.m4a"
_s2t = OpenCC("s2t")
_PUNCT = "，。、！？；：「」（）"


def simplified_count(text: str) -> int:
    return sum(1 for ch in text if ch.strip() and _s2t.convert(ch) != ch)


PROMPTS = {
    "A_none": None,
    "B_tw_punct": "以下是台灣的演講逐字稿，使用繁體中文，並加上標點符號。",
    "C_tw_nopunct": "以下是台灣的演講逐字稿 使用繁體中文",
    "D_cn_punct": "以下是中国大陆的演讲逐字稿，使用简体中文，并加上标点符号。",
    "E_short_tw": "繁體中文，台灣用語。",
    "F_domain_tw": "今天的機器人展示，我們先做了簡報，介紹控制系統與視覺感知，並播放成果影片。",
}


def main() -> None:
    t0 = time.time()
    print("Loading large-v3-turbo (cached)...")
    m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print(f"loaded: {time.time() - t0:.1f}s\n")

    rows = []
    for label, prompt in PROMPTS.items():
        t1 = time.time()
        segs, _ = m.transcribe(AUDIO, language="zh", beam_size=5, initial_prompt=prompt)
        text = "".join(s.text for s in segs).strip()
        dt = time.time() - t1
        han = sum(1 for c in text if "一" <= c <= "鿿")
        ns = simplified_count(text)
        npunct = sum(1 for c in text if c in _PUNCT)
        rows.append((label, ns, han, npunct, dt, text))
        print(f"[{label:13s}] simp {ns:2d}/{han:2d}={ns / han:5.1%}  punct={npunct:2d}  {dt:4.1f}s")
        print(f"               prompt: {prompt}")
        print(f"               out: {text}\n")

    print("=" * 70)
    print(f"{'probe':15s}{'simp%':>8s}{'punct':>7s}{'sec':>7s}")
    for label, ns, han, npunct, dt, _ in rows:
        print(f"{label:15s}{ns / han:>7.1%}{npunct:>7d}{dt:>7.1f}")

    d = {r[0]: (r[1] / r[2], r[3]) for r in rows}
    print("\nPaired reads:")
    print(f"  A(none)    simp {d['A_none'][0]:.1%}  punct {d['A_none'][1]}")
    print(f"  B(繁+標點) simp {d['B_tw_punct'][0]:.1%}  punct {d['B_tw_punct'][1]}  <- prompt 有沒有用")
    print(f"  C(繁無標點) simp {d['C_tw_nopunct'][0]:.1%}  punct {d['C_tw_nopunct'][1]}  <- 標點靠 prompt 的標點?")
    print(f"  D(簡+標點) simp {d['D_cn_punct'][0]:.1%}  punct {d['D_cn_punct'][1]}  <- prompt 字體決定輸出字體?")


if __name__ == "__main__":
    main()
