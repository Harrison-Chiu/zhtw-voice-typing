"""Unit tests for the benchmark metrics.

Every expected number here is worked out by hand from a deliberately small pair,
because a metric that is quietly wrong invalidates every benchmark run made with
it — and unlike a wrong transcript, nothing downstream will look obviously off.
"""

from __future__ import annotations

import pytest

from asr_input.eval.metrics import (
    align,
    determinism,
    empty_rate,
    error_rate,
    evaluate_pair,
    insertion_runs,
    mer,
    normalize_text,
    normalized_cer,
    punctuation_f1,
    repetition_stats,
    strict_cer,
    tokenize_mixed,
    traditional_consistency,
)


def fake_s2t(text: str) -> str:
    """A stand-in converter: same length, so ratios stay comparable."""
    table = {"这": "這", "简": "簡", "体": "體", "语": "語", "输": "輸", "测": "測", "试": "試"}
    return "".join(table.get(ch, ch) for ch in text)


# --- alignment -------------------------------------------------------------


def test_identical_strings_have_no_errors():
    rate = error_rate("今天天氣很好", "今天天氣很好")
    assert (rate.substitutions, rate.deletions, rate.insertions) == (0, 0, 0)
    assert rate.rate == 0.0


def test_substitution_and_deletion_together():
    # gold 「今天天氣很好」(6) vs 「今日天氣好」(5): 天→日 substitution and 很 deleted.
    # The length difference forces exactly one deletion, so the op counts are not
    # a tie-break artefact.
    rate = error_rate("今天天氣很好", "今日天氣好")
    assert (rate.substitutions, rate.deletions, rate.insertions) == (1, 1, 0)
    assert rate.rate == pytest.approx(2 / 6)


def test_insertion_is_counted_against_the_reference_length():
    rate = error_rate("今天天氣好", "今天天氣很好")
    assert (rate.substitutions, rate.deletions, rate.insertions) == (0, 0, 1)
    assert rate.rate == pytest.approx(1 / 5)


def test_substitution_is_preferred_when_costs_tie():
    # 「很好」→「好耶」 can be read as two substitutions or as delete+insert;
    # both cost 2. The tie-break is fixed at substitution so counts are stable.
    rate = error_rate("今天天氣很好", "今天天氣好耶")
    assert (rate.substitutions, rate.deletions, rate.insertions) == (2, 0, 0)


def test_pure_deletion_and_pure_insertion():
    assert error_rate("abcde", "abc").deletions == 2
    assert error_rate("abc", "abcde").insertions == 2


def test_empty_reference_scores_1_when_anything_is_emitted():
    assert error_rate("", "").rate == 0.0
    assert error_rate("", "幻覺").rate == 1.0


def test_alignment_covers_both_sequences():
    ops = align("abc", "axc")
    assert [o.op for o in ops] == ["equal", "sub", "equal"]
    assert [o.ref for o in ops] == ["a", "b", "c"]
    assert [o.hyp for o in ops] == ["a", "x", "c"]


# --- normalisation ---------------------------------------------------------


def test_normalize_removes_whitespace_and_folds_halfwidth_punct():
    assert normalize_text("今天 ,天氣好") == "今天，天氣好"


def test_normalized_cer_ignores_punctuation_and_script_style():
    gold = "這是簡體測試"
    hyp = "这是简体测试"
    assert strict_cer(gold, hyp).errors == 5  # 是 is the only shared character
    assert normalized_cer(gold, hyp, to_traditional=fake_s2t).errors == 0


def test_normalized_cer_still_counts_real_content_errors():
    assert normalized_cer("今天天氣很好。", "今天天氣很壞。").errors == 1


# --- MER -------------------------------------------------------------------


def test_tokenizer_splits_cjk_by_char_and_latin_by_word():
    assert tokenize_mixed("我們用 ArduCopter 跑 PWM 3.3") == [
        "我",
        "們",
        "用",
        "ArduCopter",
        "跑",
        "PWM",
        "3.3",
    ]


def test_mer_charges_one_error_per_wrong_english_word():
    gold = "我們用 ArduCopter"
    hyp = "我們用 ArduPilot"
    # Character CER would charge 5 edits for the one misheard word.
    assert strict_cer(gold, hyp).errors > 1
    assert mer(gold, hyp).errors == 1
    assert mer(gold, hyp).ref_length == 4


# --- punctuation -----------------------------------------------------------


def test_punctuation_f1_is_perfect_when_marks_match():
    score = punctuation_f1("今天天氣很好，我們出門。", "今天天氣很好，我們出門。")
    assert score["micro"]["f1"] == 1.0


def test_missing_comma_lowers_recall_only():
    score = punctuation_f1("今天天氣很好，我們出門。", "今天天氣很好我們出門。")
    comma = score["per_mark"]["，"]
    assert comma["fn"] == 1
    assert comma["tp"] == 0
    assert score["per_mark"]["。"]["tp"] == 1
    assert score["micro"]["precision"] == 1.0
    assert score["micro"]["recall"] == pytest.approx(0.5)


def test_extra_comma_lowers_precision_only():
    score = punctuation_f1("今天天氣很好我們出門。", "今天天氣很好，我們出門。")
    assert score["per_mark"]["，"]["fp"] == 1
    assert score["micro"]["recall"] == 1.0
    assert score["micro"]["precision"] < 1.0


def test_halfwidth_comma_counts_as_the_fullwidth_one():
    # The normaliser's job is to fix the style; the metric should not also
    # punish it, or punctuation F1 becomes a second copy of Strict CER.
    score = punctuation_f1("今天很好，我們出門。", "今天很好,我們出門.")
    assert score["micro"]["f1"] == 1.0


def test_marks_with_no_occurrences_do_not_drag_the_score_down():
    score = punctuation_f1("今天很好。", "今天很好。")
    assert score["per_mark"]["？"]["f1"] == 1.0  # vacuously perfect, 0 tp/fp/fn


# --- hallucination / repetition -------------------------------------------


def test_insertion_runs_finds_a_fabricated_clause():
    gold = "今天天氣很好。"
    hyp = "今天天氣很好。字幕由志願者提供感謝觀看。"
    runs = insertion_runs(gold, hyp, min_chars=8)
    assert len(runs) == 1
    assert "字幕由志願者提供" in runs[0]["text"]


def test_short_insertions_are_not_called_hallucinations():
    assert insertion_runs("今天天氣很好。", "今天天氣真的很好。", min_chars=8) == []


def test_repetition_stats_catches_a_degenerate_loop():
    stats = repetition_stats("量" * 300)
    assert stats["longest_char_run"] == 300
    assert stats["dominant_char"] == "量"
    assert stats["dominant_char_ratio"] == pytest.approx(1.0)


def test_repetition_stats_on_ordinary_text():
    stats = repetition_stats("今天天氣不錯，我們來測試一下語音輸入。")
    assert stats["longest_char_run"] <= 2
    assert stats["dominant_char_ratio"] < 0.3


# --- script consistency ----------------------------------------------------


def test_traditional_consistency_counts_rewritten_chars():
    stats = traditional_consistency("这是简体", fake_s2t)
    assert stats["converted_chars"] == 3  # 是 is already shared between scripts
    assert stats["converted_ratio"] == pytest.approx(0.75)
    assert stats["changed"] is True


def test_traditional_consistency_on_already_traditional_text():
    stats = traditional_consistency("這是繁體", fake_s2t)
    assert stats["converted_chars"] == 0
    assert stats["changed"] is False


def test_traditional_consistency_reports_length_change_instead_of_a_ratio():
    stats = traditional_consistency("接口", lambda t: t.replace("接口", "介面裝置"))
    assert stats["length_changed"] is True
    assert stats["converted_ratio"] is None


# --- run-level -------------------------------------------------------------


def test_determinism_of_identical_runs():
    result = determinism(["一樣的輸出"] * 5)
    assert result.unique_outputs == 1
    assert result.exact_match_ratio == 1.0


def test_determinism_with_a_minority_disagreement():
    result = determinism(["A", "A", "A", "B"])
    assert result.unique_outputs == 2
    assert result.modal_output == "A"
    assert result.exact_match_ratio == pytest.approx(0.75)


def test_determinism_of_no_runs_is_not_a_crash():
    result = determinism([])
    assert result.runs == 0
    assert result.exact_match_ratio == 0.0


def test_empty_rate_counts_whitespace_only_as_empty():
    assert empty_rate(["有字", "", "   ", "也有字"])["empty"] == 2


def test_evaluate_pair_returns_every_section():
    result = evaluate_pair(
        "今天天氣很好，我們出門。", "今天天氣很好我們出門。", to_traditional=fake_s2t
    )
    assert set(result) == {
        "strict_cer",
        "normalized_cer",
        "mer",
        "punctuation",
        "insertion_runs",
        "repetition",
        "traditional",
    }
    assert result["strict_cer"]["errors"] == 1
    assert result["normalized_cer"]["errors"] == 0  # only punctuation differed
