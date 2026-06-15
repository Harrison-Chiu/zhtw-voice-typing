"""Qwen3-ASR engine implementation."""

import numpy as np
import torch

from asr_input.asr.base import ASREngine


class QwenASREngine(ASREngine):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        language: str | None = "Chinese",
        context: str = "",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.language = language
        self.context = context
        self._model = None

    def load(self) -> None:
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
            max_new_tokens=512,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self._model.transcribe(
            audio=(audio, sample_rate),
            context=self.context,
            language=self.language,
        )
        return results[0].text if results else ""

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
