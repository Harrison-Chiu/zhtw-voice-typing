"""Recording/VAD collection tests without microphone or model dependencies."""

import numpy as np
import pytest

from asr_input.audio.events import VADDecisionEvent
from asr_input.recording import RecordingSession


class FakeSegmenter:
    def __init__(self, callback, event_callback):
        self.callback = callback
        self.event_callback = event_callback
        self.chunks = []
        self.reset_count = 0
        self.flush_count = 0

    def reset(self):
        self.reset_count += 1
        self.chunks.clear()

    def feed(self, chunk):
        self.chunks.append(chunk.copy())

    def flush(self):
        self.flush_count += 1
        if self.chunks:
            audio = np.concatenate(self.chunks)
            self.event_callback(
                VADDecisionEvent(
                    kind="emit",
                    start_sample=0,
                    end_sample=len(audio),
                    reason="flush",
                    rms=0.25,
                    probability_count=2,
                    probability_min=0.25,
                    probability_max=0.75,
                    probability_mean=0.5,
                )
            )
            self.callback(audio, [0.25, 0.75])


def make_session(**kwargs):
    holder = {}

    def factory(callback, event_callback):
        holder["segmenter"] = FakeSegmenter(callback, event_callback)
        return holder["segmenter"]

    return RecordingSession(factory, **kwargs), holder


def test_complete_capture_is_preserved_independently_from_vad_segments():
    session, holder = make_session(sample_rate=16000)
    first = np.array([0.1, 0.2], dtype=np.float32)
    second = np.array([0.3, 0.4, 0.5], dtype=np.float32)

    session.begin()
    session.feed(first)
    session.feed(second)
    result = session.stop()

    np.testing.assert_array_equal(result.audio, np.concatenate([first, second]))
    assert result.received_frames == 5
    assert result.duration_sec == pytest.approx(5 / 16000)
    assert result.callback_count == 2
    assert len(result.segments) == 1
    np.testing.assert_array_equal(result.segments[0].audio, result.audio)
    assert result.segments[0].probabilities == (0.25, 0.75)
    assert result.segments[0].start_sample == 0
    assert result.segments[0].end_sample == 5
    assert result.vad_events[0].reason == "flush"
    assert holder["segmenter"].flush_count == 1


def test_emitted_segment_is_forwarded_with_stable_primary_index():
    emitted = []
    session, _ = make_session(on_segment=lambda segment, index: emitted.append((segment, index)))

    session.begin()
    session.feed(np.ones(4, dtype=np.float32))
    result = session.stop()

    assert len(emitted) == 1
    assert emitted[0][0] == result.segments[0]
    assert emitted[0][1] == 0


def test_callback_status_is_recorded_at_the_current_sample_offset():
    reported = []
    session, _ = make_session(on_status=reported.append)
    session.begin()
    session.feed(np.zeros(4, dtype=np.float32))
    session.feed(np.zeros(3, dtype=np.float32), status="input overflow")
    result = session.stop()

    assert len(result.status_events) == 1
    assert result.status_events[0].sample_offset == 4
    assert result.status_events[0].message == "input overflow"
    assert reported == [result.status_events[0]]


def test_frame_mismatch_and_adc_gap_are_recorded_without_dropping_audio():
    session, _ = make_session(sample_rate=10)
    session.begin()
    session.feed(np.ones(2, dtype=np.float32), expected_frames=3, adc_time=10.0)
    session.feed(np.ones(2, dtype=np.float32), expected_frames=2, adc_time=11.0)
    result = session.stop()

    assert "frame count mismatch" in result.status_events[0].message
    assert result.gap_events[0].sample_offset == 2
    assert result.gap_events[0].expected_adc_time == pytest.approx(10.2)
    assert result.gap_events[0].actual_adc_time == pytest.approx(11.0)
    assert result.gap_events[0].gap_sec == pytest.approx(0.8)
    assert result.received_frames == 4


def test_zero_adc_timestamp_is_treated_as_unavailable_not_as_a_gap():
    session, _ = make_session(sample_rate=10)
    session.begin()
    session.feed(np.ones(2, dtype=np.float32), adc_time=0.0)
    session.feed(np.ones(2, dtype=np.float32), adc_time=0.0)
    result = session.stop()

    assert result.gap_events == ()


def test_level_updates_are_rate_limited_and_report_rms():
    levels = []
    session, _ = make_session(sample_rate=8, level_updates_hz=2, on_level=levels.append)
    session.begin()
    session.feed(np.ones(2, dtype=np.float32))
    session.feed(np.ones(2, dtype=np.float32) * 0.5)
    session.feed(np.ones(4, dtype=np.float32) * 0.25)
    session.stop()

    assert levels == pytest.approx([0.5, 0.25])


def test_level_observer_exception_is_recorded_without_escaping_capture_callback():
    def fail_level(rms):
        raise OSError("invalid tray icon handle")

    session, _ = make_session(sample_rate=8, level_updates_hz=2, on_level=fail_level)
    session.begin()
    session.feed(np.ones(4, dtype=np.float32))
    result = session.stop()

    assert result.received_frames == 4
    assert "level observer failed: OSError" in result.status_events[0].message


def test_feed_and_stop_require_an_active_recording():
    session, _ = make_session()

    with pytest.raises(RuntimeError, match="not active"):
        session.feed(np.zeros(1, dtype=np.float32))
    with pytest.raises(RuntimeError, match="not active"):
        session.stop()


def test_begin_cannot_overlap_an_existing_recording():
    session, _ = make_session()
    session.begin()

    with pytest.raises(RuntimeError, match="already active"):
        session.begin()
