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
        min_silence_duration_ms: int = 800,
        speech_pad_ms: int = 100,
        max_segment_sec: float = 60.0,
        fallback_silence_ms: tuple[int, ...] = (500, 300),
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_segment_sec = max_segment_sec
        self.fallback_silence_ms = fallback_silence_ms
        self._model: torch.jit.ScriptModule | None = None

    def load(self) -> None:
        model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self._model = model

    def _detect_raw(
        self, audio: np.ndarray, sample_rate: int, silence_ms: int
    ) -> list[SpeechSegment]:
        from silero_vad import get_speech_timestamps

        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        raw = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )
        return [SpeechSegment(start_sample=s["start"], end_sample=s["end"]) for s in raw]

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> list[SpeechSegment]:
        """Detect speech segments with adaptive re-splitting for long segments."""
        if self._model is None:
            raise RuntimeError("VAD not loaded. Call load() first.")

        segments = self._detect_raw(audio, sample_rate, self.min_silence_duration_ms)

        max_samples = int(self.max_segment_sec * sample_rate)
        for silence_ms in self.fallback_silence_ms:
            refined: list[SpeechSegment] = []
            for seg in segments:
                seg_len = seg.end_sample - seg.start_sample
                if seg_len <= max_samples:
                    refined.append(seg)
                    continue
                chunk = audio[seg.start_sample : seg.end_sample]
                sub_segs = self._detect_raw(chunk, sample_rate, silence_ms)
                if len(sub_segs) <= 1:
                    refined.append(seg)
                else:
                    for sub in sub_segs:
                        refined.append(
                            SpeechSegment(
                                start_sample=seg.start_sample + sub.start_sample,
                                end_sample=seg.start_sample + sub.end_sample,
                            )
                        )
            segments = refined

        return segments

    def unload(self) -> None:
        self._model = None
