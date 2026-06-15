"""Simplified → Traditional Chinese (Taiwan) conversion via OpenCC."""

import opencc

from asr_input.processing.pipeline import TextProcessor


class OpenCCConverter(TextProcessor):
    def __init__(self, config: str = "s2twp") -> None:
        self._converter = opencc.OpenCC(config)

    def process(self, text: str) -> str:
        return self._converter.convert(text)
