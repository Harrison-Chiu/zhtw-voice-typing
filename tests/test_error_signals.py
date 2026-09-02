"""Unit tests for the offline error-candidate signals.

All inputs are synthetic — no DB, no audio, no model. The point is that each
rule fires on the failure it is meant to describe and stays quiet on ordinary
speech, so that a change to one rule cannot silently widen or narrow the others.
"""

from __future__ import annotations

import pytest

from asr_input.eval.signals import (
    NO_COMMA_MIN_CHARS,
    SEGMENT_SIGNALS,
    SIGNAL_LABELS,
    Segment,
    detect_phrase_repeat,
    dominant_char_ratio,
    scan_segment,
)

NORMAL_TEXT = "今天天氣不錯，我們來測試一下語音輸入的效果好不好。"


def names(segment: Segment) -> set[str]:
    return {hit.name for hit in scan_segment(segment)}


def normal_segment(**overrides) -> Segment:
    base = {
        "audio_sec": 10.0,
        "transcribe_sec": 0.6,
        "raw_text": NORMAL_TEXT,
        "processed_text": NORMAL_TEXT,
    }
    base.update(overrides)
    return Segment(**base)


def test_ordinary_segment_fires_nothing():
    assert names(normal_segment()) == set()


def test_every_registered_rule_has_a_label():
    # A rule whose name is missing from SIGNAL_LABELS would show up in a report
    # as a bare identifier with no explanation of what it means.
    fired = set()
    for seg in (
        Segment(audio_sec=1.0, transcribe_sec=3.0, processed_text="啊" * 30),
        Segment(audio_sec=30.0, transcribe_sec=0.5, processed_text=""),
        Segment(audio_sec=5.0, transcribe_sec=0.5, raw_text="有字", processed_text=""),
        Segment(audio_sec=5.0, transcribe_sec=0.5, rms=0.0001, processed_text=NORMAL_TEXT),
        normal_segment(fallback=True, fallback_exhausted=True, rejected=True),
    ):
        fired |= names(seg)
    assert fired
    assert fired <= set(SIGNAL_LABELS)


def test_registry_is_not_empty_and_holds_callables():
    assert len(SEGMENT_SIGNALS) >= 15
    assert all(callable(rule) for rule in SEGMENT_SIGNALS)


# --- latency ---------------------------------------------------------------


def test_slow_transcription_uses_absolute_time_not_a_ratio():
    # 25s of audio taking 3.5s is a known hallucination case. Its
    # transcribe/audio ratio is 0.14, i.e. *better* than a healthy short
    # segment — a ratio rule would miss it. This test exists to make that
    # regression loud if anyone reintroduces a ratio.
    long_hallucination = Segment(audio_sec=25.2, transcribe_sec=3.5, processed_text=NORMAL_TEXT)
    assert "slow-transcription" in names(long_hallucination)

    # 0.3s of audio taking 0.4s is normal; ratio 1.3 would falsely kill it.
    short_normal = Segment(audio_sec=0.3, transcribe_sec=0.4, processed_text="好的。")
    assert "slow-transcription" not in names(short_normal)


def test_short_audio_slow_is_reported_separately():
    seg = Segment(audio_sec=0.8, transcribe_sec=3.3, processed_text="好的。")
    fired = names(seg)
    assert {"slow-transcription", "short-audio-slow"} <= fired


def test_long_audio_slow_is_not_labelled_short():
    seg = Segment(audio_sec=21.8, transcribe_sec=2.5, processed_text=NORMAL_TEXT)
    assert "short-audio-slow" not in names(seg)


# --- repetition ------------------------------------------------------------


def test_char_repeat_fires_on_single_char_loop():
    assert "char-repeat" in names(Segment(audio_sec=5.0, processed_text="量量量量量量。"))


def test_phrase_repeat_reports_one_hit_per_run():
    # 「量」×300 also matches at unit sizes 2..6; only the shortest unit is kept
    # so a single loop does not produce five near-identical findings.
    hits = detect_phrase_repeat("量" * 300)
    assert len(hits) == 1


def test_phrase_repeat_ignores_ordinary_text():
    assert detect_phrase_repeat(NORMAL_TEXT) == []


def test_dominant_char_needs_enough_characters():
    # Under 12 CJK chars the ratio is too noisy to mean anything.
    assert dominant_char_ratio("好好好") == ("", 0.0)
    ch, ratio = dominant_char_ratio("好" * 20 + "今天天氣")
    assert ch == "好"
    assert ratio > 0.7


# --- text quality ----------------------------------------------------------


def test_simplified_residual_flags_processed_only():
    seg = Segment(audio_sec=5.0, raw_text="这是简体", processed_text="這是簡體")
    assert "simplified-residual" not in names(seg)
    seg_unfixed = Segment(audio_sec=5.0, raw_text="这是简体", processed_text="这是简体")
    assert "simplified-residual" in names(seg_unfixed)


def test_taiwan_variants_are_not_treated_as_simplified():
    # 台 would be rewritten by OpenCC s2t but is standard in Taiwan usage.
    seg = Segment(audio_sec=5.0, processed_text="我在台北的裡面等你。")
    assert "simplified-residual" not in names(seg)


def test_halfwidth_punct_between_cjk():
    assert "halfwidth-punct" in names(Segment(audio_sec=5.0, processed_text="今天,天氣不錯"))
    assert "halfwidth-punct" not in names(Segment(audio_sec=5.0, processed_text="今天，天氣不錯"))


def test_english_punctuation_is_not_flagged():
    seg = Segment(audio_sec=5.0, processed_text="we use PWM, then ArduCopter")
    assert "halfwidth-punct" not in names(seg)


def test_no_comma_threshold_ignores_median_length_utterances():
    short = "啊" * (NO_COMMA_MIN_CHARS - 1)
    long = "啊" * NO_COMMA_MIN_CHARS
    assert "no-comma-long" not in names(Segment(audio_sec=20.0, processed_text=short))
    assert "no-comma-long" in names(Segment(audio_sec=20.0, processed_text=long))


# --- output/audio mismatch -------------------------------------------------


@pytest.mark.parametrize(
    ("audio_sec", "text", "expected"),
    [
        (30.0, "好的。", "cps-low"),
        (3.0, "字" * 40, "cps-high"),
    ],
)
def test_chars_per_second_outliers(audio_sec, text, expected):
    assert expected in names(Segment(audio_sec=audio_sec, processed_text=text))


def test_chars_per_second_skips_very_short_audio():
    # Short clips have too much fixed overhead in the ratio to judge.
    assert names(Segment(audio_sec=1.0, transcribe_sec=0.4, processed_text="好")) == set()


def test_empty_output_and_processed_missing_are_distinct():
    model_said_nothing = Segment(audio_sec=5.0, raw_text="", processed_text="")
    assert "empty-output" in names(model_said_nothing)
    assert "processed-missing" not in names(model_said_nothing)

    lost_in_post = Segment(audio_sec=5.0, raw_text="模型有輸出", processed_text="")
    assert "processed-missing" in names(lost_in_post)


def test_truncated_needs_long_enough_audio():
    assert "truncated" in names(Segment(audio_sec=8.0, processed_text="好的"))
    assert "truncated" not in names(Segment(audio_sec=0.5, processed_text="好的"))


# --- context-derived -------------------------------------------------------


def test_prompt_echo_needs_more_than_one_term():
    hotwords = "待辦 清單 詞表"
    one_term = Segment(audio_sec=5.0, processed_text="幫我把待辦事項記下來。", hotwords=hotwords)
    assert "prompt-echo" not in names(one_term)

    echoed = Segment(audio_sec=5.0, processed_text="希望等待辦 清單 詞表", hotwords=hotwords)
    assert "prompt-echo" in names(echoed)


def test_state_flags_are_passed_through():
    seg = normal_segment(fallback=True, fallback_exhausted=True, rejected=True)
    assert {"fallback-triggered", "fallback-exhausted", "rejected-segment"} <= names(seg)


def test_near_silent_uses_rms():
    assert "near-silent" in names(normal_segment(rms=0.0001))
    assert "near-silent" not in names(normal_segment(rms=0.02))


def test_missing_fields_do_not_crash():
    # Legacy rows can have NULL timings; a scan must degrade, not raise.
    assert isinstance(scan_segment(Segment()), list)
