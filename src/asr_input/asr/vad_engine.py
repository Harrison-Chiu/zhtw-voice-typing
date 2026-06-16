"""VAD-segmented ASR engine wrapper.

Wraps any ASREngine: long audio is split into speech segments via Silero VAD,
each segment is transcribed independently, and results are concatenated.
Short audio or single-segment audio bypasses the overhead.
"""

import time

import numpy as np

from asr_input.asr.base import ASREngine
from asr_input.audio.vad import SileroVAD


class VadSegmentedEngine(ASREngine):
    """Decorator that adds VAD segmentation to any ASREngine."""

    def __init__(
        self,
        inner: ASREngine,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
        min_audio_len_sec: float = 30.0,
    ) -> None:
        self._inner = inner
        self._min_audio_len_sec = min_audio_len_sec
        self._vad = SileroVAD(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def load(self) -> None:
        self._inner.load()
        print("載入 VAD 模型...", flush=True)
        self._vad.load()
        print("VAD 模型載入完成。", flush=True)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        audio_len_sec = len(audio) / sample_rate

        if audio_len_sec < self._min_audio_len_sec:
            return self._inner.transcribe(audio, sample_rate)

        segments = self._vad.detect(audio, sample_rate)
        if not segments:
            return ""
        if len(segments) == 1:
            chunk = audio[segments[0].start_sample : segments[0].end_sample]
            return self._inner.transcribe(chunk, sample_rate)

        print(f"VAD: {len(segments)} 段語音（音訊 {audio_len_sec:.1f}s）", flush=True)
        results: list[str] = []
        for i, seg in enumerate(segments):
            chunk = audio[seg.start_sample : seg.end_sample]
            chunk_sec = len(chunk) / sample_rate
            t0 = time.time()
            text = self._inner.transcribe(chunk, sample_rate)
            dt = time.time() - t0
            print(f"  段 {i + 1}/{len(segments)}: {chunk_sec:.1f}s → {dt:.1f}s", flush=True)
            if text.strip():
                results.append(text.strip())

        return "".join(results)

    def unload(self) -> None:
        self._inner.unload()
        self._vad.unload()
