"""Qwen3-ASR engine implementation."""

import numpy as np

from asr_input.asr.base import ASREngine


class QwenASREngine(ASREngine):
    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-1.7B", device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        # TODO: implement model loading with transformers
        raise NotImplementedError("Qwen3-ASR loading — will implement in next phase")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        # TODO: implement transcription
        raise NotImplementedError("Qwen3-ASR transcription — will implement in next phase")

    def unload(self) -> None:
        self._model = None
        self._processor = None
