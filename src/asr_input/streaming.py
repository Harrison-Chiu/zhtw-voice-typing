"""Streaming ASR session — real-time VAD + per-sentence transcription.

Coordinates: microphone → streaming VAD → ASR engine → post-processing pipeline.
Accumulates all transcribed text; copies to clipboard on stop.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import numpy as np
import torch

from asr_input.asr.base import ASREngine
from asr_input.audio.streaming_vad import StreamingVAD
from asr_input.processing.pipeline import ProcessingPipeline


class StreamingSession:
    """One streaming recording session (start → speak → stop → get result)."""

    def __init__(
        self,
        engine: ASREngine,
        pipeline: ProcessingPipeline,
        vad_model: torch.jit.ScriptModule,
        sample_rate: int = 16000,
        silence_trigger_ms: int = 1000,
        vad_threshold: float = 0.5,
        min_speech_ms: int = 250,
        speech_pad_ms: int = 100,
        max_segment_sec: float = 30.0,
        adaptive_thresholds: list[tuple[float, int]] | None = None,
        min_energy: float = 0.005,
        on_partial: Callable[[str, str], None] | None = None,
    ) -> None:
        self._engine = engine
        self._pipeline = pipeline
        self._sample_rate = sample_rate
        self._on_partial = on_partial

        self._segments: list[str] = []
        self._segment_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stream = None

        self._vad = StreamingVAD(
            on_speech_segment=self._on_speech_segment,
            sample_rate=sample_rate,
            threshold=vad_threshold,
            min_speech_ms=min_speech_ms,
            silence_trigger_ms=silence_trigger_ms,
            speech_pad_ms=speech_pad_ms,
            max_segment_sec=max_segment_sec,
            adaptive_thresholds=adaptive_thresholds,
            min_energy=min_energy,
        )
        self._vad.load(vad_model)

    def start(self) -> None:
        """Start recording and processing."""
        self._segments.clear()
        self._vad.reset()
        self._stop_event.clear()

        self._worker = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._worker.start()

        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> str:
        """Stop recording, flush remaining audio, return full text."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._vad.flush()

        # Signal worker to finish
        self._segment_queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=30)
            self._worker = None

        return "".join(self._segments)

    @property
    def partial_text(self) -> str:
        """Current accumulated text (may still be incomplete)."""
        return "".join(self._segments)

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        self._vad.feed(indata)

    def _on_speech_segment(self, audio: np.ndarray) -> None:
        """Called by StreamingVAD when a complete speech segment is detected."""
        self._segment_queue.put(audio)

    def _transcribe_loop(self) -> None:
        """Worker thread: pull segments from queue, transcribe, accumulate."""
        while True:
            segment_audio = self._segment_queue.get()
            if segment_audio is None:
                break

            t0 = time.time()
            raw_text = self._engine.transcribe(segment_audio, self._sample_rate)
            dt = time.time() - t0

            if not raw_text.strip():
                continue

            processed = self._pipeline.run(raw_text)
            self._segments.append(processed)

            audio_sec = len(segment_audio) / self._sample_rate
            print(
                f"  [串流] {audio_sec:.1f}s 音訊 → {dt:.1f}s 辨識: {processed}",
                flush=True,
            )

            if self._on_partial:
                self._on_partial(processed, "".join(self._segments))
