import numpy as np
import pytest

from asr_input.audio.streaming_vad import StreamingVAD


class ProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def __call__(self, tensor, sample_rate):
        return np.array(next(self.probabilities), dtype=np.float32)

    def reset_states(self):
        pass


def make_vad(probabilities, *, min_speech_ms=1, min_energy=0.005):
    segments = []
    events = []
    vad = StreamingVAD(
        on_speech_segment=lambda audio, probs: segments.append((audio, probs)),
        on_decision_event=events.append,
        sample_rate=16000,
        min_speech_ms=min_speech_ms,
        silence_trigger_ms=32,
        silence_min_ms=32,
        speech_pad_ms=0,
        min_energy=min_energy,
    )
    vad.load(ProbabilityModel(probabilities))
    return vad, segments, events


def test_emit_event_has_exact_offsets_and_probability_summary():
    vad, segments, events = make_vad([0.9, 0.1])

    vad.feed(np.concatenate([np.ones(512, dtype=np.float32), np.zeros(512)]))

    assert len(segments) == 1
    event = events[0]
    assert (event.kind, event.start_sample, event.end_sample) == ("emit", 0, 512)
    assert event.reason == "靜音"
    assert event.probability_count == 1
    assert event.probability_mean == pytest.approx(0.9)


def test_low_energy_and_short_segments_are_visible_discards():
    low_vad, _, low_events = make_vad([0.9, 0.1])
    low_vad.feed(np.zeros(1024, dtype=np.float32))

    short_vad, _, short_events = make_vad([0.9, 0.1], min_speech_ms=100)
    short_vad.feed(
        np.concatenate([np.ones(512, dtype=np.float32), np.zeros(512, dtype=np.float32)])
    )

    assert low_events[0].kind == "discard"
    assert low_events[0].reason == "low_energy"
    assert short_events[0].kind == "discard"
    assert short_events[0].reason == "too_short"


def test_subwindow_tail_is_logged_instead_of_disappearing():
    vad, _, events = make_vad([])

    vad.feed(np.ones(100, dtype=np.float32))
    vad.flush()

    assert len(events) == 1
    assert events[0].reason == "unprocessed_tail_below_silero_window"
    assert (events[0].start_sample, events[0].end_sample) == (0, 100)
