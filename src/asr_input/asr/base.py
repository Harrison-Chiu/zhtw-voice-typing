"""Abstract ASR engine interface."""

from abc import ABC, abstractmethod

import numpy as np


class ASREngine(ABC):
    """Base class for ASR engines. Implement this to add a new engine."""

    @abstractmethod
    def load(self) -> None:
        """Load model into memory / GPU."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio array to text."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
