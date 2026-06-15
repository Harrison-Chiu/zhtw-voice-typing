"""Audio capture from microphone."""

from abc import ABC, abstractmethod

import numpy as np


class AudioSource(ABC):
    """Abstract audio source — swap mic, file, or stream."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> np.ndarray: ...


class MicrophoneCapture(AudioSource):
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._frames, axis=0).flatten()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._frames.append(indata.copy())
