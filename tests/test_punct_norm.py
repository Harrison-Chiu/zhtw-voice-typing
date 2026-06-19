"""Tests for context-aware punctuation normalization."""

import pytest

from asr_input.processing.punct_norm import PunctuationNormalizer


@pytest.fixture
def normalizer() -> PunctuationNormalizer:
    return PunctuationNormalizer()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # CJK on the left → full-width
        ("你好,世界", "你好，世界"),
        ("結束了!", "結束了！"),
        ("真的嗎?", "真的嗎？"),
        ("重點:這個", "重點：這個"),
        ("第一;第二", "第一；第二"),
        # CJK only on the right (punctuation at start of CJK run) → full-width
        (",你好", "，你好"),
        # multiple punctuation marks in one CJK sentence
        ("他說,你好,再見!", "他說，你好，再見！"),
    ],
)
def test_cjk_adjacent_becomes_fullwidth(normalizer, text, expected):
    assert normalizer.process(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # pure ASCII context → untouched
        "hello, world",
        "a:b;c",
        "Really?",
        "Stop!",
        "key: value, next",
        # numbers around punctuation are not CJK
        "1,000",
        "12:30",
    ],
)
def test_ascii_context_untouched(normalizer, text):
    assert normalizer.process(text) == text


def test_period_and_other_chars_not_in_map_pass_through(normalizer):
    # full-stop "." is intentionally NOT in the conversion map
    assert normalizer.process("好.的") == "好.的"


def test_mixed_cjk_and_ascii_boundary(normalizer):
    # comma sits between English and CJK — CJK on one side is enough
    assert normalizer.process("OK好,的") == "OK好，的"
    # comma between two ASCII tokens even if CJK elsewhere stays half-width
    assert normalizer.process("中文 a,b 結尾") == "中文 a,b 結尾"


@pytest.mark.parametrize("text", ["", "你好", "hello", ","])
def test_edge_cases_no_crash(normalizer, text):
    # lone punctuation with no neighbours stays half-width; empty/plain text unchanged
    result = normalizer.process(text)
    if text == ",":
        assert result == ","
    else:
        assert result == text
