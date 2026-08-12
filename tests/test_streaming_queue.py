"""Offline queued-segment transcription tests."""

import threading

import numpy as np
import pytest

from asr_input.audio.streaming_vad import StreamingVAD
from asr_input.streaming import StreamingSession


class FakeVADModel:
    def reset_states(self):
        pass


class FakeEngine:
    def __init__(self):
        self.inputs = []

    def transcribe(self, audio, sample_rate=16000):
        self.inputs.append(audio.copy())
        return str(int(audio[0]))


class FakePipeline:
    def run(self, text):
        return f"<{text}>"


class FailingEngine(FakeEngine):
    def transcribe(self, audio, sample_rate=16000):
        raise RuntimeError("CUDA execution failed")


def make_session(engine, **kwargs):
    return StreamingSession(
        engine=engine,
        pipeline=FakePipeline(),
        vad_model=FakeVADModel(),
        sample_rate=4,
        min_hallucination_audio_sec=99,
        **kwargs,
    )


def test_presegmented_job_preserves_fifo_order():
    engine = FakeEngine()
    session = make_session(engine)
    segments = [
        (np.ones(4, dtype=np.float32), [0.9]),
        (np.ones(4, dtype=np.float32) * 2, [0.8]),
    ]

    session.start_from_segments(segments)

    assert session.stop() == "<1><2>"
    assert len(engine.inputs) == 2
    assert session.segment_stats[0]["attempts"][0]["kind"] == "original"
    assert session.segment_stats[0]["attempts"][0]["adopted"] is True


def test_open_segment_stream_transcribes_before_capture_is_closed():
    engine = FakeEngine()
    partial_ready = threading.Event()
    session = make_session(engine, on_partial=lambda latest, full: partial_ready.set())

    session.start_segment_stream()
    session.feed_segment(np.ones(4, dtype=np.float32), [0.9])

    assert partial_ready.wait(timeout=1)
    assert session.partial_text == "<1>"
    session.finish_segment_stream()
    assert session.stop() == "<1>"


def test_open_segment_stream_accepts_catch_up_then_live_segments_once():
    engine = FakeEngine()
    session = make_session(engine)
    session.start_segment_stream()

    session.feed_segment(np.ones(4, dtype=np.float32), [0.9])
    session.feed_segment(np.ones(4, dtype=np.float32) * 2, [0.8])
    session.finish_segment_stream()

    assert session.stop() == "<1><2>"
    assert [int(audio[0]) for audio in engine.inputs] == [1, 2]


def test_ui_observer_exception_does_not_fail_asr_worker():
    def fail_observer(*args):
        raise OSError("invalid tray icon handle")

    session = make_session(
        FakeEngine(),
        on_transcribing=fail_observer,
        on_partial=fail_observer,
        on_segment_done=fail_observer,
    )
    session.start_from_segments([(np.ones(4, dtype=np.float32), [0.9])])

    assert session.stop() == "<1>"


def test_worker_exception_is_raised_instead_of_returning_partial_text():
    session = make_session(FailingEngine())
    session.start_from_segments([(np.ones(4, dtype=np.float32), [0.9])])

    with pytest.raises(RuntimeError, match="ASR worker failed") as exc_info:
        session.stop()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "CUDA execution failed" in str(exc_info.value.__cause__)


def test_rms_fallback_does_not_retranscribe_unchanged_audio():
    engine = FakeEngine()
    session = make_session(engine, fallback_rms_target=0.05)

    result = session._try_rms_normalize(np.ones(4, dtype=np.float32) * 0.1)

    assert result is None
    assert engine.inputs == []


def test_rms_fallback_retries_when_gain_changes_samples():
    engine = FakeEngine()
    session = make_session(engine, fallback_rms_target=0.05)

    normalized, text, _ = session._try_rms_normalize(np.ones(4, dtype=np.float32) * 0.01)

    assert text == "0"
    assert len(engine.inputs) == 1
    np.testing.assert_allclose(normalized, 0.05)


@pytest.mark.parametrize(
    ("times", "succeeded"),
    [([0.2, 0.3], True), ([2.0, 0.3], False)],
)
def test_fallback_reports_success_separately_from_exhaustion(monkeypatch, times, succeeded):
    session = make_session(FakeEngine(), fallback_silence_ms=[500, 500])
    monkeypatch.setattr(
        StreamingVAD,
        "resegment",
        staticmethod(
            lambda audio, probs, **kwargs: [
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
            ]
        ),
    )
    results = iter((f"part-{index}", value) for index, value in enumerate(times))
    monkeypatch.setattr(session, "_transcribe_one", lambda audio: next(results))

    fallback_results, actual_succeeded = session._fallback_transcribe(
        np.ones(4, dtype=np.float32), [0.9]
    )

    assert len(fallback_results) == 2
    assert actual_succeeded is succeeded
