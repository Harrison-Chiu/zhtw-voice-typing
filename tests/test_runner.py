"""Tests for the benchmark runner.

No ASR engine is involved: the runner takes a `transcribe` callable, so a fake
one is enough to test the parts that can silently go wrong — micro vs macro
averaging, failure isolation, and whether transcript text leaks into the result.
"""

from __future__ import annotations

import pytest

from asr_input.eval.manifest import Manifest, RunResult, Sample
from asr_input.eval.runner import (
    aggregate,
    format_markdown,
    run_benchmark,
    run_samples,
    score_run,
)


@pytest.fixture
def corpus(tmp_path):
    """Two samples with gold text, backed by real (empty) files on disk."""
    samples, private = [], {}
    golds = {"s1": "今天天氣很好，我們出門。", "s2": "語音輸入測試。"}
    for i, (sid, gold) in enumerate(golds.items(), start=1):
        audio = tmp_path / f"{sid}.wav"
        audio.write_bytes(b"")
        samples.append(
            Sample(
                sample_id=sid,
                split="smoke",
                duration_sec=float(i),
                audio_sha256="0" * 64,
                labels=["short"],
            )
        )
        private[sid] = {"audio_path": str(audio), "gold_verbatim": gold}
    return Manifest(corpus="fixture", samples=samples), private


def perfect(private):
    """A transcriber that returns the gold text for whichever file it is given."""
    by_path = {entry["audio_path"]: entry["gold_verbatim"] for entry in private.values()}
    return lambda path: by_path[str(path)]


def test_perfect_transcriber_scores_zero_error(corpus):
    manifest, private = corpus
    result = run_benchmark(manifest, private, perfect(private))

    assert result.aggregate["samples_scored"] == 2
    assert result.aggregate["strict_cer"] == 0.0
    assert result.aggregate["punctuation"]["f1"] == 1.0
    assert result.aggregate["samples_failed"] == 0


def test_a_failing_sample_does_not_abort_the_run(corpus):
    manifest, private = corpus
    gold = perfect(private)

    def flaky(path):
        if path.name == "s1.wav":
            raise RuntimeError("decode blew up")
        return gold(path)

    result = run_benchmark(manifest, private, flaky)

    assert result.aggregate["samples_failed"] == 1
    assert result.aggregate["samples_scored"] == 1  # the other sample still counted
    failed = [r for r in result.samples if r["error"]]
    assert "decode blew up" in failed[0]["error"]


def test_missing_audio_is_reported_rather_than_raised(corpus):
    manifest, private = corpus
    private["s1"]["audio_path"] = str(manifest.samples[0].sample_id) + "-nope.wav"

    runs = run_samples(manifest.samples, private, lambda p: "")

    assert "missing audio" in runs[0].error
    assert runs[0].outputs == []


def test_sample_without_a_private_entry_is_reported(corpus):
    manifest, private = corpus
    del private["s2"]

    runs = run_samples(manifest.samples, private, perfect(private))

    assert runs[1].error == "no audio_path in private map"


def test_aggregate_is_micro_not_macro():
    # A 1-char sample with 1 error and a 9-char sample with 0 errors.
    # Macro would report 50%; micro reports 1/10.
    records = [
        {"duration_sec": 1.0, "metrics": _metrics(errors=1, ref_length=1)},
        {"duration_sec": 1.0, "metrics": _metrics(errors=0, ref_length=9)},
    ]
    assert aggregate(records)["strict_cer"] == pytest.approx(0.1)


def _metrics(*, errors: int, ref_length: int) -> dict:
    rate = {"errors": errors, "ref_length": ref_length}
    return {
        "strict_cer": rate,
        "normalized_cer": rate,
        "mer": rate,
        "punctuation": {"micro": {"tp": 1, "fp": 0, "fn": 0}},
        "insertion_runs": [],
        "repetition": {},
    }


def test_aggregate_of_nothing_is_not_a_crash():
    out = aggregate([])
    assert out["samples_scored"] == 0
    assert "strict_cer" not in out


def test_repeats_are_measured_for_determinism(corpus):
    manifest, private = corpus
    outputs = iter(["A", "A", "B"] * 2)
    result = run_benchmark(manifest, private, lambda p: next(outputs), repeats=3)

    first = result.samples[0]
    assert first["determinism"]["runs"] == 3
    assert first["determinism"]["unique_outputs"] == 2


def test_unknown_tier_is_rejected(corpus):
    manifest, private = corpus
    with pytest.raises(ValueError, match="unknown tier"):
        run_benchmark(manifest, private, perfect(private), tier="everything")


def test_tier_selects_splits(corpus):
    manifest, private = corpus
    manifest.samples[1].split = "test"

    smoke = run_benchmark(manifest, private, perfect(private), tier="smoke")
    full = run_benchmark(manifest, private, perfect(private), tier="full")

    assert smoke.aggregate["samples_total"] == 1
    assert full.aggregate["samples_total"] == 2


def test_result_records_carry_no_transcript_text(corpus):
    manifest, private = corpus
    result = run_benchmark(manifest, private, lambda p: "一段不該外流的逐字稿內容")

    blob = str(result.as_dict())
    assert "不該外流" not in blob
    assert "gold_verbatim" not in blob


def test_empty_output_is_counted(corpus):
    manifest, private = corpus
    result = run_benchmark(manifest, private, lambda p: "   ")
    assert result.aggregate["empty_outputs"] == 2


def test_score_run_without_output_still_returns_a_record():
    sample = Sample(sample_id="s1", split="smoke", duration_sec=1.0, audio_sha256="x")
    from asr_input.eval.runner import SampleRun

    record = score_run(SampleRun(sample, [], [], error="boom"), "gold")

    assert record["sample_id"] == "s1"
    assert record["error"] == "boom"
    assert "metrics" not in record


def test_markdown_summary_reports_the_headline_numbers(corpus):
    manifest, private = corpus
    result = run_benchmark(manifest, private, perfect(private), engine={"name": "fake"})

    text = format_markdown(result)

    assert "# ASR benchmark — fixture" in text
    assert "Strict CER | 0.00%" in text
    assert "僅為吞吐量報告" in text


def test_markdown_lists_failed_samples():
    result = RunResult(
        run_id="r",
        manifest_corpus="c",
        engine={},
        environment={"gpu": {"available": False, "error": "x"}, "os": {}, "python": {}},
        samples=[{"sample_id": "s9", "error": "missing audio: x.wav"}],
        aggregate={"samples_total": 1, "samples_scored": 0, "samples_failed": 1},
    )
    text = format_markdown(result)
    assert "## 失敗的樣本" in text
    assert "`s9`: missing audio: x.wav" in text
