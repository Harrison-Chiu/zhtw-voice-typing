"""faster-whisper engine implementation.

Unlike Qwen3-ASR (whose `context` is inert for script), Whisper's `initial_prompt`
biases output script AND punctuation: a short Traditional prompt with full-width
punctuation yields native Traditional output with punctuation. See
`experiments/experiment_whisper_prompt.py`.
"""

import numpy as np

from asr_input.asr.base import ASREngine


class WhisperFWEngine(ASREngine):
    def __init__(
        self,
        model_id: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = "zh",
        initial_prompt: str = "繁體中文，台灣用語。",
        hotwords: str | None = None,
        beam_size: int = 5,
        temperature: float | list[float] | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.beam_size = beam_size
        self.temperature = temperature
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        segments, _ = self._model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=self.beam_size,
            temperature=self.temperature,
            initial_prompt=self.initial_prompt or None,
            hotwords=self.hotwords or None,
        )
        return "".join(seg.text for seg in segments).strip()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
