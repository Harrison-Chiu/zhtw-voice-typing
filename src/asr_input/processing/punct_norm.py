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

# 連續 3 個以上半形句號 → 全形刪節號。多字元標點塞不進下方逐字元迴圈，
# 故另開一道 re 前處理。半形句號 "." 不在 _HALF_TO_FULL（避免誤傷小數點），不衝突。
_ELLIPSIS_RUN = re.compile(r"\.{3,}")


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RANGE.match(ch))


def _convert_ellipsis(text: str) -> str:
    """Run of 3+ half-width dots → … when adjacent to CJK; English "wait..." stays."""

    def repl(m: re.Match[str]) -> str:
        prev = text[m.start() - 1] if m.start() > 0 else ""
        nxt = text[m.end()] if m.end() < len(text) else ""
        return "…" if _is_cjk(prev) or _is_cjk(nxt) else m.group()

    return _ELLIPSIS_RUN.sub(repl, text)


class PunctuationNormalizer(TextProcessor):
    def process(self, text: str) -> str:
        if not text:
            return text

        text = _convert_ellipsis(text)
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
