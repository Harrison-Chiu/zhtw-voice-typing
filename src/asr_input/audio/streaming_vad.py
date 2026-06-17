"""Real-time streaming VAD using Silero VAD v5.

Processes audio chunk-by-chunk and fires a callback when a complete speech
segment is detected (i.e., speech followed by sufficient silence).
"""

import sys
from collections.abc import Callable

import numpy as np
import torch


class StreamingVAD:
    """Feed audio chunks in; get speech segments out via callback.

    Silero VAD v5 expects 512-sample chunks at 16 kHz (32 ms each).
    We accept arbitrary chunk sizes from the microphone and internally
    buffer/split them into 512-sample windows for the model.

    Silence trigger ramps linearly from silence_trigger_ms down to
    silence_min_ms between ramp_start_sec and ramp_end_sec of accumulated
    speech, in 100ms steps.
    """

    SILERO_CHUNK_SAMPLES = 512  # 32 ms at 16 kHz

    def __init__(
        self,
        on_speech_segment: Callable[[np.ndarray], None],
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        silence_trigger_ms: int = 1000,
        silence_min_ms: int = 300,
        ramp_start_sec: float = 10.0,
        ramp_end_sec: float = 25.0,
        speech_pad_ms: int = 100,
        max_segment_sec: float = 30.0,
        min_energy: float = 0.005,
        verbose: bool = False,
    ) -> None:
        self._on_speech_segment = on_speech_segment
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self._silence_trigger_ms = silence_trigger_ms
        self._silence_min_ms = silence_min_ms
        self._ramp_start_samples = int(ramp_start_sec * sample_rate)
        self._ramp_end_samples = int(ramp_end_sec * sample_rate)
        self._speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self._max_segment_samples = int(max_segment_sec * sample_rate)
        self._min_energy = min_energy
        self._verbose = verbose
        self._total_fed_samples = 0

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
        self._total_fed_samples = 0
        self._segment_count = 0
        if self._model is not None:
            self._model.reset_states()

    def feed(self, chunk: np.ndarray) -> None:
        """Feed an audio chunk from the microphone. May trigger callback."""
        if self._model is None:
            raise RuntimeError("StreamingVAD not loaded.")

        flat = chunk.flatten().astype(np.float32)
        self._total_fed_samples += len(flat)
        self._pending = np.concatenate([self._pending, flat])

        while len(self._pending) >= self.SILERO_CHUNK_SAMPLES:
            window = self._pending[: self.SILERO_CHUNK_SAMPLES]
            self._pending = self._pending[self.SILERO_CHUNK_SAMPLES :]
            self._process_window(window)

    def flush(self) -> None:
        """Flush any remaining speech at end of session."""
        if self._in_speech and self._speech_buf:
            self._emit_segment()

    def _current_silence_trigger(self) -> int:
        """Return the effective silence trigger based on how long speech has accumulated.

        Linear ramp from silence_trigger_ms to silence_min_ms between
        ramp_start and ramp_end, quantized to 100ms steps.
        """
        if self._speech_samples <= self._ramp_start_samples:
            ms = self._silence_trigger_ms
        elif self._speech_samples >= self._ramp_end_samples:
            ms = self._silence_min_ms
        else:
            ramp_range = self._ramp_end_samples - self._ramp_start_samples
            progress = (self._speech_samples - self._ramp_start_samples) / ramp_range
            delta = self._silence_min_ms - self._silence_trigger_ms
            ms = self._silence_trigger_ms + progress * delta
        ms = round(ms / 100) * 100
        return int(ms * self._sample_rate / 1000)

    def _status_line(self, text: str) -> None:
        """Overwrite the current line with a status message (no newline)."""
        sys.stdout.write(f"\r  {text:<72}")
        sys.stdout.flush()

    def _clear_status(self) -> None:
        """Clear the status line before printing a result."""
        sys.stdout.write("\r" + " " * 76 + "\r")
        sys.stdout.flush()

    def _process_window(self, window: np.ndarray) -> None:
        tensor = torch.from_numpy(window)
        prob = self._model(tensor, self._sample_rate).item()
        is_speech = prob >= self._threshold

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._silence_samples = 0
                self._speech_samples = 0
                if self._pre_buf:
                    self._speech_buf.extend(self._pre_buf)
                    self._pre_buf.clear()
                    self._pre_buf_samples = 0
            self._speech_buf.append(window)
            self._speech_samples += len(window)
            self._silence_samples = 0

            if self._verbose and self._speech_samples % (self._sample_rate // 2) < len(window):
                speech_sec = self._speech_samples / self._sample_rate
                trigger_ms = self._current_silence_trigger() * 1000 // self._sample_rate
                self._status_line(
                    f"[語音] {speech_sec:.1f}s | 門檻 {trigger_ms}ms"
                )

            if self._speech_samples >= self._max_segment_samples:
                self._emit_segment("上限")
        else:
            if self._in_speech:
                self._speech_buf.append(window)
                self._silence_samples += len(window)
                trigger = self._current_silence_trigger()

                if self._verbose and self._silence_samples % (self._sample_rate // 4) < len(window):
                    speech_sec = self._speech_samples / self._sample_rate
                    silence_ms = self._silence_samples * 1000 // self._sample_rate
                    trigger_ms = trigger * 1000 // self._sample_rate
                    self._status_line(
                        f"[語音] {speech_sec:.1f}s | [靜音] {silence_ms}/{trigger_ms}ms"
                    )

                if self._silence_samples >= trigger:
                    self._emit_segment("靜音")
            else:
                self._pre_buf.append(window)
                self._pre_buf_samples += len(window)
                while self._pre_buf_samples > self._speech_pad_samples and len(self._pre_buf) > 1:
                    removed = self._pre_buf.pop(0)
                    self._pre_buf_samples -= len(removed)

    def _emit_segment(self, reason: str = "") -> None:
        if not self._speech_buf:
            self._in_speech = False
            return

        audio = np.concatenate(self._speech_buf)

        if self._silence_samples > self._speech_pad_samples:
            trim = self._silence_samples - self._speech_pad_samples
            audio = audio[: len(audio) - trim]

        if len(audio) >= self._min_speech_samples:
            rms = np.sqrt(np.mean(audio**2))
            if rms >= self._min_energy:
                self._segment_count = getattr(self, "_segment_count", 0) + 1
                if self._verbose:
                    seg_sec = len(audio) / self._sample_rate
                    self._clear_status()
                    print(
                        f"  [切句] #{self._segment_count} "
                        f"{seg_sec:.1f}s ({reason}) → 辨識中...",
                        flush=True,
                    )
                self._on_speech_segment(audio)
            elif self._verbose:
                self._clear_status()
                print(
                    f"  [捨棄] 低能量 RMS={rms:.4f}",
                    flush=True,
                )

        self._speech_buf.clear()
        self._in_speech = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._pre_buf.clear()
        self._pre_buf_samples = 0
