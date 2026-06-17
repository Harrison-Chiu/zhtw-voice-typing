"""Experiment: reverse-engineer how Qwen3-ASR's `context` param actually works.

We don't assume the operating logic. Instead each probe isolates one variable, and
PAIRED comparisons (B<->C, D<->E, F<->G) tell us which hypothesis the data supports:

  H1 instruction-following | H2 prefix-continuation | H3 lexical hotword bias
  H4 script-of-context only | H5 no effect (null)

Metric = Simplified-char ratio in the RAW output (before OpenCC). Output is
deterministic, so one run per probe is enough. Small audio file only.

Run:
    .venv\\Scripts\\python.exe experiment_context.py
"""

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch  # noqa: E402
from opencc import OpenCC  # noqa: E402

from asr_input.asr.qwen import QwenASREngine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO = str(PROJECT_ROOT / "data/test_audio/機器人展示，語音轉錄測試.m4a")

# Same-domain paragraph, written once in Traditional (F) and once in Simplified (G)
# so the ONLY difference is the script -> cleanest test of the continuation hypothesis.
_PRIME_TW = (
    "今天的機器人展示非常順利，我們先做了簡報，介紹控制系統與視覺感知模組，"
    "接著播放成果影片，展示機器人的行走步態。"
)
_PRIME_CN = (
    "今天的机器人展示非常顺利，我们先做了简报，介绍控制系统与视觉感知模块，"
    "接着播放成果视频，展示机器人的行走步态。"
)

PROBES = {
    "A_empty": "",
    "B_instr_tw": "請用台灣繁體中文（正體字）輸出。",
    "C_instr_cn": "请用台湾繁体中文（正体字）输出。",
    "D_keywords_tw": "繁體字 台灣 軟體 影片 程式 網路 機器人",
    "E_keywords_cn": "繁体字 台湾 软件 视频 程序 网络 机器人",
    "F_prime_tw": _PRIME_TW,
    "G_prime_cn": _PRIME_CN,
    "H_offtopic_tw": "台灣的天氣四季分明，許多人喜歡在週末到郊外踏青與品嚐美食。",
    "I_negative": "不要輸出簡體字。",
    "J_english": "Transcribe in Traditional Chinese (Taiwan).",
    "K_hotwords": "成果影片 機器人 控制 步態 視覺感知 行走",
}

_s2t = OpenCC("s2t")
_s2twp = OpenCC("s2twp")


def simplified_count(text: str) -> int:
    return sum(1 for ch in text if ch.strip() and _s2t.convert(ch) != ch)


def main() -> None:
    print("Loading Qwen3-ASR-1.7B...")
    engine = QwenASREngine(device="cuda", language="Chinese")
    engine.load()
    print(f"loaded, GPU {torch.cuda.memory_allocated() / 1024**2:.0f}MB\n")

    rows = []
    for label, ctx in PROBES.items():
        t0 = time.time()
        r = engine._model.transcribe(audio=AUDIO, context=ctx, language="Chinese")
        dt = time.time() - t0
        raw = r[0].text if r else "(none)"
        han = sum(1 for c in raw if "一" <= c <= "鿿")
        ns = simplified_count(raw)
        rows.append((label, ns, han, dt, raw))
        print(f"[{label:14s}] simp {ns:2d}/{han:2d} = {ns / han:5.1%}  {dt:4.1f}s")
        print(f"               ctx: {ctx or '(none)'}")
        print(f"               raw: {raw}\n")

    print("=" * 70)
    print(f"{'probe':16s}{'simp%':>8s}{'simp/han':>10s}{'sec':>7s}")
    for label, ns, han, dt, _ in rows:
        print(f"{label:16s}{ns / han:>7.1%}{f'{ns}/{han}':>10s}{dt:>7.1f}")

    print("\nPaired reads:")
    d = {r[0]: r[1] / r[2] for r in rows}
    print(f"  B(繁指令) {d['B_instr_tw']:.1%}  vs  C(簡指令) {d['C_instr_cn']:.1%}  -> 字體 vs 語意")
    print(f"  D(繁詞)   {d['D_keywords_tw']:.1%}  vs  E(簡詞)  {d['E_keywords_cn']:.1%}  -> 餵簡詞會否拖簡")
    print(f"  F(繁前文) {d['F_prime_tw']:.1%}  vs  G(簡前文){d['G_prime_cn']:.1%}  -> 接續假設(關鍵)")
    print(f"  A(基準)   {d['A_empty']:.1%}")

    engine.unload()
    print("\nDone!")


if __name__ == "__main__":
    main()
