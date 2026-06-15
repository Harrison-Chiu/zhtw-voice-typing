"""Audio capture from microphone."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AudioSource(ABC):
    """Abstract audio source — swap mic, file, or stream."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> np.ndarray: ...


class FileAudioSource(AudioSource):
    """Fake microphone backed by an audio file.

    Implements the same start()/stop() contract as MicrophoneCapture, so the full
    mic pipeline can be exercised without real hardware (handy for VMs / CI / tests).
    `stop()` returns the whole file as one buffer, exactly like recording then stopping.
    """

    def __init__(self, path: str | Path, sample_rate: int = 16000) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._audio: np.ndarray | None = None

    def start(self) -> None:
        import librosa

        audio, _ = librosa.load(str(self.path), sr=self.sample_rate, mono=True)
        self._audio = audio.astype(np.float32)

    def stop(self) -> np.ndarray:
        if self._audio is None:
            return np.array([], dtype=np.float32)
        audio, self._audio = self._audio, None
        return audio


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
