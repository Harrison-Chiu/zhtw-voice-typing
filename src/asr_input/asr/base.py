"""Abstract ASR engine interface."""

from abc import ABC, abstractmethod

import numpy as np


class ASREngine(ABC):
    """Base class for ASR engines. Implement this to add a new engine."""

    # Local decoder latency can expose repetition degeneration. Network-backed
    # engines must override this because transport latency is not a quality signal.
    transcription_latency_is_quality_signal = True

    @abstractmethod
    def load(self) -> None:
        """Load model into memory / GPU."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio array to text."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
