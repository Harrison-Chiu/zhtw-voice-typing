"""Deterministic error-candidate signals for offline review (roadmap E line).

These rules are a *high-recall first pass*: they select segments worth a human
listening to, not segments that are certainly wrong. False positives are the
expected cost; a missed real error is the failure mode we care about. Nothing
here writes labels or changes production behaviour.

Two design constraints carried over from earlier decisions:

- The latency signal uses the **absolute** transcription time
  (`transcribe_sec > 1.5`), never `transcribe_sec / audio_sec`. The failure
  mode is beam search stuck in a repetition loop, whose runaway decode time is
  set by the max-token limit and is largely independent of the input length, so
  dividing by audio length normalises the signal away. See CLAUDE.md
  「幻覺門檻用絕對轉錄時間」.
- Thresholds are calibrated on RTX 4060 + CUDA float16 data. On another device
  the latency threshold has to be re-measured rather than relativised.

Threshold provenance (measured 2026-09-03 over the 1425 segments then in
`data/logs/history.sqlite3`, all from faster-whisper large-v3-turbo / beam 5):

    transcribe_sec  p50 0.62  p95 1.04  p99 2.37  max 4.04
    audio_sec       p50 9.0   p90 25.6  min 0.26
    chars/second    p10 3.05  p50 4.21  p90 5.35  p99 6.82  (audio > 0.5s)
    segment rms     p1  0.013 min 0.0099
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Characters that OpenCC s2t would rewrite but which are standard in Taiwan usage.
# Same list as scripts/search_logs.py — kept in sync deliberately, not imported,
# because that script also serves the legacy jsonl path.
SIMPLIFIED_WHITELIST = {"台", "里", "面", "干", "后", "才"}

SLOW_TRANSCRIBE_SEC = 1.5
SHORT_AUDIO_SEC = 3.0
NEAR_SILENT_RMS = 0.005
# chars/sec bounds are deliberately far outside the observed p10–p99 band so that
# only gross mismatches (dropped speech, runaway output) are flagged.
CPS_LOW = 1.5
CPS_HIGH = 8.0
CPS_MIN_AUDIO_SEC = 2.0
# 逗號缺失是已知的獨立問題（見 CLAUDE.md 逗號實驗），這個訊號主要用於歸因而非驅動佇列。
# 門檻取 60 字是為了不淹沒審核佇列：在 2026-09-03 的 1425 段上，>=30 字命中 128 段
# （其中 127 段只有這一個訊號打中）、>=60 字命中 57 段。段落字數 p50 為 38、p75 為 76，
# 所以 30 字的「長段」其實是中位數等級的一般語句。
NO_COMMA_MIN_CHARS = 60

SIGNAL_LABELS = {
    "slow-transcription": "轉錄時間 > 1.5s（絕對值，幻覺高相關）",
    "short-audio-slow": "短音訊（< 3s）卻轉錄偏慢",
    "fallback-triggered": "觸發 fallback 重切",
    "fallback-exhausted": "fallback 門檻用盡",
    "rejected-segment": "已被判為幻覺並捨棄",
    "char-repeat": "同字連續重複 ≥ 5 次",
    "phrase-repeat": "同一短語連續重複 ≥ 3 次",
    "dominant-char": "單一字占比過高",
    "cps-low": "輸出字數相對音訊過少（疑似漏聽）",
    "cps-high": "輸出字數相對音訊過多（疑似贅生）",
    "empty-output": "有音訊但輸出為空",
    "simplified-residual": "processed 仍含簡體字",
    "halfwidth-punct": "CJK 旁的半形標點",
    "cjk-spacing": "CJK 字間有空格",
    "no-comma-long": "長段完全沒有逗號",
    "processed-missing": "有 raw 但 processed 空白（後處理或紀錄斷點）",
    "truncated": "音訊 ≥ 2s 但輸出 ≤ 5 字",
    "prompt-echo": "輸出疑似覆誦 hotwords／initial_prompt",
    "near-silent": "音訊 RMS 接近靜音",
}


@dataclass
class SignalHit:
    """One rule firing on one segment."""

    name: str
    detail: str = ""

    @property
    def label(self) -> str:
        return SIGNAL_LABELS.get(self.name, self.name)


@dataclass
class Segment:
    """The subset of a history-DB segment the signals need.

    Built by the caller from either the `segments` table or an `asr_attempts`
    row, so the same rules can score both.
    """

    audio_sec: float | None = None
    transcribe_sec: float | None = None
    raw_text: str = ""
    processed_text: str = ""
    fallback: bool = False
    rms: float | None = None
    fallback_exhausted: bool = False
    rejected: bool = False
    hotwords: str = ""
    initial_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Prefer processed text; fall back to raw so a post-processing bug
        cannot make a segment invisible to the scan."""
        return self.processed_text or self.raw_text or ""


def _is_cjk(ch: str) -> bool:
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        return False
    return "CJK" in name


def detect_simplified(text: str) -> list[str]:
    try:
        import opencc
    except ImportError:  # pragma: no cover - opencc is a hard dependency in practice
        return []
    converter = opencc.OpenCC("s2t")
    found = []
    for ch in text:
        if _is_cjk(ch) and ch not in SIMPLIFIED_WHITELIST:
            converted = converter.convert(ch)
            if converted != ch:
                found.append(f"{ch}→{converted}")
    return found


def detect_phrase_repeat(
    text: str, min_len: int = 2, max_len: int = 6, times: int = 3
) -> list[str]:
    """Consecutive repetition of a short phrase, e.g. 「量量量」的多字版本。

    `char-repeat` already covers single characters; this catches the 2–6 char
    loops that beam search also falls into.
    """
    hits: list[str] = []
    covered: list[tuple[int, int]] = []
    for size in range(min_len, max_len + 1):
        pattern = re.compile(rf"(.{{{size}}}?)\1{{{times - 1},}}", re.DOTALL)
        for m in pattern.finditer(text):
            unit = m.group(1)
            if not unit.strip():
                continue
            # A run of 「量」×300 also matches at sizes 2..6; report only the
            # shortest unit for each span so one loop yields one hit.
            if any(start <= m.start() and m.end() <= end for start, end in covered):
                continue
            covered.append((m.start(), m.end()))
            hits.append(f"'{unit}' ×{len(m.group()) // size}")
    return hits


def dominant_char_ratio(text: str) -> tuple[str, float]:
    cjk = [ch for ch in text if _is_cjk(ch)]
    if len(cjk) < 12:
        return ("", 0.0)
    counts: dict[str, int] = {}
    for ch in cjk:
        counts[ch] = counts.get(ch, 0) + 1
    ch, n = max(counts.items(), key=lambda kv: kv[1])
    return (ch, n / len(cjk))


def _prompt_terms(segment: Segment) -> list[str]:
    """Split hotwords / initial_prompt into terms long enough to be evidence.

    Single characters are excluded — they appear in ordinary speech and would
    make this signal fire on almost everything.
    """
    terms = set()
    for source in (segment.hotwords, segment.initial_prompt):
        for token in re.split(r"[\s，、。,.]+", source or ""):
            token = token.strip()
            if len(token) >= 2:
                terms.add(token)
    return sorted(terms)


# --- individual rules -------------------------------------------------------
# Each takes a Segment and returns a SignalHit or None. Registered in
# SEGMENT_SIGNALS so the scan order (and thus report order) is explicit.


def _slow(seg: Segment) -> SignalHit | None:
    dt = seg.transcribe_sec
    if dt is not None and dt > SLOW_TRANSCRIBE_SEC:
        return SignalHit("slow-transcription", f"{dt:.2f}s")
    return None


def _short_audio_slow(seg: Segment) -> SignalHit | None:
    dt, audio = seg.transcribe_sec, seg.audio_sec
    if dt is None or audio is None:
        return None
    if audio < SHORT_AUDIO_SEC and dt > SLOW_TRANSCRIBE_SEC:
        return SignalHit("short-audio-slow", f"{audio:.2f}s 音訊 / {dt:.2f}s 轉錄")
    return None


def _fallback(seg: Segment) -> SignalHit | None:
    return SignalHit("fallback-triggered") if seg.fallback else None


def _fallback_exhausted(seg: Segment) -> SignalHit | None:
    return SignalHit("fallback-exhausted") if seg.fallback_exhausted else None


def _rejected(seg: Segment) -> SignalHit | None:
    return SignalHit("rejected-segment") if seg.rejected else None


def _char_repeat(seg: Segment) -> SignalHit | None:
    hits = [m.group() for m in re.finditer(r"(.)\1{4,}", seg.text)]
    return SignalHit("char-repeat", "; ".join(h[:12] for h in hits)) if hits else None


def _phrase_repeat(seg: Segment) -> SignalHit | None:
    hits = detect_phrase_repeat(seg.text)
    return SignalHit("phrase-repeat", "; ".join(hits[:3])) if hits else None


def _dominant_char(seg: Segment) -> SignalHit | None:
    ch, ratio = dominant_char_ratio(seg.text)
    if ratio > 0.25:
        return SignalHit("dominant-char", f"'{ch}' 占 {ratio:.0%}")
    return None


def _cps(seg: Segment) -> SignalHit | None:
    audio = seg.audio_sec
    text = seg.text.strip()
    if not audio or audio < CPS_MIN_AUDIO_SEC or not text:
        return None
    cps = len(text) / audio
    if cps < CPS_LOW:
        return SignalHit("cps-low", f"{cps:.2f} 字/秒")
    if cps > CPS_HIGH:
        return SignalHit("cps-high", f"{cps:.2f} 字/秒")
    return None


def _empty(seg: Segment) -> SignalHit | None:
    if not seg.text.strip() and (seg.audio_sec or 0) >= 1.0:
        return SignalHit("empty-output", f"{seg.audio_sec:.2f}s 音訊")
    return None


def _simplified(seg: Segment) -> SignalHit | None:
    if not seg.processed_text:
        return None
    hits = detect_simplified(seg.processed_text)
    return SignalHit("simplified-residual", "; ".join(hits[:5])) if hits else None


def _halfwidth_punct(seg: Segment) -> SignalHit | None:
    hits = re.findall(r"(?<=[^\x00-\x7f])\s*([,?!;:])\s*(?=[^\x00-\x7f])", seg.text)
    return SignalHit("halfwidth-punct", "".join(hits[:5])) if hits else None


def _cjk_spacing(seg: Segment) -> SignalHit | None:
    hits = [m.group() for m in re.finditer(r"[一-鿿]\s+[一-鿿]", seg.text)]
    return SignalHit("cjk-spacing", "; ".join(hits[:3])) if hits else None


def _no_comma(seg: Segment) -> SignalHit | None:
    text = seg.text
    if len(text) >= NO_COMMA_MIN_CHARS and not re.search(r"[，,、；]", text):
        return SignalHit("no-comma-long", f"{len(text)} 字")
    return None


def _truncated(seg: Segment) -> SignalHit | None:
    text = seg.text.strip()
    if 0 < len(text) <= 5 and (seg.audio_sec or 0) >= 2.0:
        return SignalHit("truncated", f"{seg.audio_sec:.2f}s → 「{text}」")
    return None


def _prompt_echo(seg: Segment) -> SignalHit | None:
    text = seg.text
    if not text.strip():
        return None
    echoed = [t for t in _prompt_terms(seg) if t in text]
    # A single matching term is ordinary — hotwords exist because the user says
    # those words. Two or more in one segment is the echo failure mode.
    if len(echoed) >= 2:
        return SignalHit("prompt-echo", "、".join(echoed))
    return None


def _processed_missing(seg: Segment) -> SignalHit | None:
    """raw 有內容但 processed 空白。

    這代表文字在後處理或紀錄階段掉了，而不是模型沒聽到——與 `empty-output`
    （模型本身沒輸出）是不同的故障，分開標才查得動。
    """
    if seg.raw_text.strip() and not seg.processed_text.strip():
        return SignalHit("processed-missing", f"raw {len(seg.raw_text)} 字 → processed 空")
    return None


def _near_silent(seg: Segment) -> SignalHit | None:
    if seg.rms is not None and seg.rms < NEAR_SILENT_RMS:
        return SignalHit("near-silent", f"rms={seg.rms:.5f}")
    return None


# The registry fixes the order rules run in (and therefore the order signals
# appear in a report). A rule may emit more than one signal name — `_cps` emits
# either cps-low or cps-high — so this is a list of rules, not of names.
SEGMENT_SIGNALS: tuple[Callable[[Segment], SignalHit | None], ...] = (
    _slow,
    _short_audio_slow,
    _fallback,
    _fallback_exhausted,
    _rejected,
    _char_repeat,
    _phrase_repeat,
    _dominant_char,
    _cps,
    _empty,
    _simplified,
    _halfwidth_punct,
    _cjk_spacing,
    _no_comma,
    _truncated,
    _prompt_echo,
    _processed_missing,
    _near_silent,
)


def scan_segment(segment: Segment) -> list[SignalHit]:
    """Return every signal that fires on this segment, in registration order."""
    hits = []
    for rule in SEGMENT_SIGNALS:
        hit = rule(segment)
        if hit is not None:
            hits.append(hit)
    return hits
