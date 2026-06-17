"""Context-aware punctuation normalization.

Converts half-width punctuation to full-width when adjacent to CJK characters,
leaving punctuation in pure ASCII contexts (English, numbers) untouched.
"""

import re

from asr_input.processing.pipeline import TextProcessor

_CJK_RANGE = re.compile(
    r"[一-鿿㐀-䶿豈-﫿"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f"
    r"　-〿＀-￯]"
)

_HALF_TO_FULL = {
    ",": "，",
    ":": "：",
    ";": "；",
    "!": "！",
    "?": "？",
}


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RANGE.match(ch))


class PunctuationNormalizer(TextProcessor):
    def process(self, text: str) -> str:
        if not text:
            return text

        chars = list(text)
        result = []

        for i, ch in enumerate(chars):
            if ch not in _HALF_TO_FULL:
                result.append(ch)
                continue

            prev = chars[i - 1] if i > 0 else ""
            nxt = chars[i + 1] if i + 1 < len(chars) else ""

            if _is_cjk(prev) or _is_cjk(nxt):
                result.append(_HALF_TO_FULL[ch])
            else:
                result.append(ch)

        return "".join(result)
