"""Simplified → Traditional Chinese (Taiwan) conversion via OpenCC."""

import opencc

from asr_input.processing.pipeline import TextProcessor

_s2t_detector = opencc.OpenCC("s2t")

# Characters that s2t flags as simplified but are standard traditional in Taiwan.
_TRAD_ALLOWLIST: set[str] = {"台"}


def _has_simplified(text: str) -> bool:
    converted = _s2t_detector.convert(text)
    if converted == text:
        return False
    return any(
        orig != conv and orig not in _TRAD_ALLOWLIST
        for orig, conv in zip(text, converted, strict=False)
    )


class OpenCCConverter(TextProcessor):
    def __init__(self, config: str = "s2twp") -> None:
        self._converter = opencc.OpenCC(config)

    def process(self, text: str) -> str:
        if not _has_simplified(text):
            return text
        return self._converter.convert(text)
