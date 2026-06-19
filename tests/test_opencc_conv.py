"""Tests for smart OpenCC simplified→Traditional conversion."""

import pytest

from asr_input.processing.opencc_conv import OpenCCConverter


@pytest.fixture
def converter() -> OpenCCConverter:
    return OpenCCConverter()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # simplified detected → converted to Taiwan traditional + idioms
        ("视频", "影片"),
        ("软件", "軟體"),
        ("这个软件", "這個軟體"),
    ],
)
def test_simplified_gets_converted(converter, text, expected):
    assert converter.process(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "這個專案",  # pure traditional
        "你好世界",
        "hello world",  # ascii
        "",  # empty
    ],
)
def test_pure_traditional_or_ascii_untouched(converter, text):
    assert converter.process(text) == text


def test_allowlist_char_does_not_trigger_conversion(converter):
    # 台 is flagged by s2t but is standard in Taiwan; presence alone must not
    # trigger conversion (avoids 台→臺 over-conversion).
    assert converter.process("台北") == "台北"


def test_allowlist_only_gates_triggering_not_conversion(converter):
    # Documents a real edge: the allowlist stops 台 from *triggering* conversion,
    # but once other simplified chars (软件) trigger it, s2twp still maps 台→臺.
    # So 台 survives only when it's the lone flagged char (see test above).
    result = converter.process("台湾软件")
    assert "軟體" in result
    assert "臺灣" in result  # 台→臺 once conversion is running
