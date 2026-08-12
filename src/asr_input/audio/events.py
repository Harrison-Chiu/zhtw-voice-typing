"""Lightweight audio decision events shared by capture and VAD."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VADDecisionEvent:
    kind: str
    start_sample: int
    end_sample: int
    reason: str
    rms: float
    probability_count: int
    probability_min: float | None
    probability_max: float | None
    probability_mean: float | None
