"""Simplified → Traditional Chinese (Taiwan) conversion via OpenCC."""

import opencc

from asr_input.processing.pipeline import TextProcessor

_s2t_detector = opencc.OpenCC("s2t")

# 台 is commonly used in Taiwan (台灣, 台北) even though 臺 exists.
_TRAD_ALSO_VALID: set[str] = {"台"}


def _has_simplified(text: str) -> bool:
    converted = _s2t_detector.convert(text)
    if converted == text:
        return False
    return any(
        orig != conv and orig not in _TRAD_ALSO_VALID
        for orig, conv in zip(text, converted, strict=False)
    )


class OpenCCConverter(TextProcessor):
    def __init__(self, config: str = "s2twp") -> None:
        self._converter = opencc.OpenCC(config)

    def process(self, text: str) -> str:
        if not _has_simplified(text):
            return text
        return self._converter.convert(text)
