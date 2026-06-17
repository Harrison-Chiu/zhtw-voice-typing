"""Simplified → Traditional Chinese (Taiwan) conversion via OpenCC."""

import opencc

from asr_input.processing.pipeline import TextProcessor

_s2t_detector = opencc.OpenCC("s2t")

# Minimum ratio of simplified characters (among all CJK) to trigger conversion.
# Whisper with traditional prompt outputs ~0% simplified; occasional false
# positives like 台→臺 sit around 0.5–3%.  A 5% threshold avoids those while
# still catching genuinely simplified text.
_SIMPLIFIED_THRESHOLD = 0.05


def _simplified_ratio(text: str) -> float:
    converted = _s2t_detector.convert(text)
    cjk_total = 0
    simp_count = 0
    for orig, conv in zip(text, converted, strict=False):
        if "一" <= orig <= "鿿":
            cjk_total += 1
            if orig != conv:
                simp_count += 1
    if cjk_total == 0:
        return 0.0
    return simp_count / cjk_total


class OpenCCConverter(TextProcessor):
    def __init__(self, config: str = "s2twp", threshold: float = _SIMPLIFIED_THRESHOLD) -> None:
        self._converter = opencc.OpenCC(config)
        self._threshold = threshold

    def process(self, text: str) -> str:
        if _simplified_ratio(text) < self._threshold:
            return text
        return self._converter.convert(text)
