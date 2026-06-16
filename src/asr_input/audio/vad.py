"""Silero VAD — detect speech segments in audio."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpeechSegment:
    start_sample: int
    end_sample: int


class SileroVAD:
    """Thin wrapper around Silero VAD v5."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._model: torch.jit.ScriptModule | None = None

    def load(self) -> None:
        model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self._model = model

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> list[SpeechSegment]:
        if self._model is None:
            raise RuntimeError("VAD not loaded. Call load() first.")

        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        from silero_vad import get_speech_timestamps

        raw = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )
        return [SpeechSegment(start_sample=s["start"], end_sample=s["end"]) for s in raw]

    def unload(self) -> None:
        self._model = None
