"""Real-time streaming VAD using Silero VAD v5.

Processes audio chunk-by-chunk and fires a callback when a complete speech
segment is detected (i.e., speech followed by sufficient silence).
"""

from collections.abc import Callable

import numpy as np
import torch


class StreamingVAD:
    """Feed audio chunks in; get speech segments out via callback.

    Silero VAD v5 expects 512-sample chunks at 16 kHz (32 ms each).
    We accept arbitrary chunk sizes from the microphone and internally
    buffer/split them into 512-sample windows for the model.
    """

    SILERO_CHUNK_SAMPLES = 512  # 32 ms at 16 kHz

    def __init__(
        self,
        on_speech_segment: Callable[[np.ndarray], None],
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        silence_trigger_ms: int = 1000,
        speech_pad_ms: int = 100,
        max_segment_sec: float = 30.0,
    ) -> None:
        self._on_speech_segment = on_speech_segment
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self._silence_trigger_samples = int(silence_trigger_ms * sample_rate / 1000)
        self._speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self._max_segment_samples = int(max_segment_sec * sample_rate)

        self._model: torch.jit.ScriptModule | None = None

        # Buffering state
        self._pending: np.ndarray = np.array([], dtype=np.float32)
        self._speech_buf: list[np.ndarray] = []
        self._in_speech = False
        self._silence_samples = 0
        self._speech_samples = 0
        # Pre-speech ring buffer for padding
        self._pre_buf: list[np.ndarray] = []
        self._pre_buf_samples = 0

    def load(self, model: torch.jit.ScriptModule) -> None:
        self._model = model

    def reset(self) -> None:
        """Reset state for a new streaming session."""
        self._pending = np.array([], dtype=np.float32)
        self._speech_buf.clear()
        self._in_speech = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._pre_buf.clear()
        self._pre_buf_samples = 0
        if self._model is not None:
            self._model.reset_states()

    def feed(self, chunk: np.ndarray) -> None:
        """Feed an audio chunk from the microphone. May trigger callback."""
        if self._model is None:
            raise RuntimeError("StreamingVAD not loaded.")

        self._pending = np.concatenate([self._pending, chunk.flatten().astype(np.float32)])

        while len(self._pending) >= self.SILERO_CHUNK_SAMPLES:
            window = self._pending[: self.SILERO_CHUNK_SAMPLES]
            self._pending = self._pending[self.SILERO_CHUNK_SAMPLES :]
            self._process_window(window)

    def flush(self) -> None:
        """Flush any remaining speech at end of session."""
        if self._in_speech and self._speech_buf:
            self._emit_segment()

    def _process_window(self, window: np.ndarray) -> None:
        tensor = torch.from_numpy(window)
        prob = self._model(tensor, self._sample_rate).item()
        is_speech = prob >= self._threshold

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._silence_samples = 0
                self._speech_samples = 0
                # Prepend the pre-speech padding buffer
                if self._pre_buf:
                    self._speech_buf.extend(self._pre_buf)
                    self._pre_buf.clear()
                    self._pre_buf_samples = 0
            self._speech_buf.append(window)
            self._speech_samples += len(window)
            self._silence_samples = 0
            if self._speech_samples >= self._max_segment_samples:
                self._emit_segment()
        else:
            if self._in_speech:
                self._speech_buf.append(window)
                self._silence_samples += len(window)
                if self._silence_samples >= self._silence_trigger_samples:
                    self._emit_segment()
            else:
                # Not in speech — maintain a rolling pre-buffer for padding
                self._pre_buf.append(window)
                self._pre_buf_samples += len(window)
                while self._pre_buf_samples > self._speech_pad_samples and len(self._pre_buf) > 1:
                    removed = self._pre_buf.pop(0)
                    self._pre_buf_samples -= len(removed)

    def _emit_segment(self) -> None:
        if not self._speech_buf:
            self._in_speech = False
            return

        audio = np.concatenate(self._speech_buf)

        # Trim trailing silence (keep a small pad)
        if self._silence_samples > self._speech_pad_samples:
            trim = self._silence_samples - self._speech_pad_samples
            audio = audio[: len(audio) - trim]

        if len(audio) >= self._min_speech_samples:
            self._on_speech_segment(audio)

        self._speech_buf.clear()
        self._in_speech = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._pre_buf.clear()
        self._pre_buf_samples = 0
