"""Offline evaluation helpers: error-candidate signals and (later) metrics."""

from asr_input.eval.signals import (
    SEGMENT_SIGNALS,
    SIGNAL_LABELS,
    Segment,
    SignalHit,
    scan_segment,
)

__all__ = ["SEGMENT_SIGNALS", "SIGNAL_LABELS", "Segment", "SignalHit", "scan_segment"]
