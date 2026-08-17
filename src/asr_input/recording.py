"""Microphone recording independent from model loading and transcription."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from asr_input.audio.events import VADDecisionEvent


@dataclass(frozen=True)
class CapturedSegment:
    audio: np.ndarray
    probabilities: tuple[float, ...]
    start_sample: int | None
    end_sample: int | None


@dataclass(frozen=True)
class CaptureStatusEvent:
    sample_offset: int
    message: str


@dataclass(frozen=True)
class CaptureGapEvent:
    sample_offset: int
    expected_adc_time: float
    actual_adc_time: float
    gap_sec: float


@dataclass(frozen=True)
class CaptureResult:
    audio: np.ndarray
    segments: tuple[CapturedSegment, ...]
    status_events: tuple[CaptureStatusEvent, ...]
    gap_events: tuple[CaptureGapEvent, ...]
    vad_events: tuple[VADDecisionEvent, ...]
    sample_rate: int
    received_frames: int
    callback_count: int

    @property
    def duration_sec(self) -> float:
        return self.received_frames / self.sample_rate


class Segmenter(Protocol):
    def reset(self) -> None: ...

    def feed(self, chunk: np.ndarray) -> None: ...

    def flush(self) -> None: ...


SegmenterFactory = Callable[
    [
        Callable[[np.ndarray, list[float]], None],
        Callable[[VADDecisionEvent], None],
    ],
    Segmenter,
]


class RecordingSession:
    """Collect one start→stop recording while VAD emits derived segments.

    No ASR engine is referenced here.  A recording can therefore finish while
    Whisper is unloaded or loading, and its immutable result can be queued for a
    single transcription worker later.
    """

    def __init__(
        self,
        segmenter_factory: SegmenterFactory,
        *,
        sample_rate: int = 16000,
        blocksize: int = 1024,
        level_updates_hz: float = 4.0,
        on_level: Callable[[float], None] | None = None,
        on_status: Callable[[CaptureStatusEvent], None] | None = None,
        on_segment: Callable[[CapturedSegment, int], None] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._on_level = on_level
        self._on_status = on_status
        self._on_segment_emitted = on_segment
        self._level_interval_samples = max(1, int(sample_rate / level_updates_hz))
        self._segmenter = segmenter_factory(self._on_segment, self._on_vad_event)
        self._lock = threading.Lock()

        self._stream = None
        self._recording = False
        self._chunks: list[np.ndarray] = []
        self._segments: list[CapturedSegment] = []
        self._status_events: list[CaptureStatusEvent] = []
        self._gap_events: list[CaptureGapEvent] = []
        self._vad_events: list[VADDecisionEvent] = []
        self._received_frames = 0
        self._callback_count = 0
        self._samples_since_level = 0
        self._last_adc_end: float | None = None

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def callback_count(self) -> int:
        with self._lock:
            return self._callback_count

    @property
    def segments(self) -> tuple[CapturedSegment, ...]:
        """Stable snapshot of segments emitted so far in the active capture."""
        with self._lock:
            return tuple(self._segments)

    def start(self) -> None:
        """Open the configured microphone stream."""
        import sounddevice as sd

        self.begin()
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=self._blocksize,
            )
            self._stream.start()
        except Exception:
            self._recording = False
            self._stream = None
            raise

    def begin(self) -> None:
        """Begin an in-memory capture; useful for file/fake-microphone tests."""
        if self._recording:
            raise RuntimeError("Recording session is already active")
        self._chunks.clear()
        self._segments.clear()
        self._status_events.clear()
        self._gap_events.clear()
        self._vad_events.clear()
        self._received_frames = 0
        self._callback_count = 0
        self._samples_since_level = 0
        self._last_adc_end = None
        self._segmenter.reset()
        self._recording = True

    def feed(
        self,
        chunk: np.ndarray,
        *,
        status: object | None = None,
        expected_frames: int | None = None,
        adc_time: float | None = None,
    ) -> None:
        """Feed one callback chunk while preserving the unmodified capture."""
        if not self._recording:
            raise RuntimeError("Recording session is not active")

        flat = np.asarray(chunk, dtype=np.float32).reshape(-1).copy()
        if expected_frames is not None and expected_frames != len(flat):
            status = (
                f"{status}; " if status else ""
            ) + f"frame count mismatch: callback={expected_frames}, samples={len(flat)}"
        status_event = None
        with self._lock:
            if status:
                status_event = CaptureStatusEvent(
                    sample_offset=self._received_frames,
                    message=str(status),
                )
                self._status_events.append(status_event)
            valid_adc_time = adc_time is not None and adc_time > 0
            if valid_adc_time and self._last_adc_end is not None:
                gap = adc_time - self._last_adc_end
                block_duration = len(flat) / self._sample_rate
                if abs(gap) > max(0.05, block_duration * 0.5):
                    self._gap_events.append(
                        CaptureGapEvent(
                            sample_offset=self._received_frames,
                            expected_adc_time=self._last_adc_end,
                            actual_adc_time=adc_time,
                            gap_sec=gap,
                        )
                    )
            if valid_adc_time:
                self._last_adc_end = adc_time + len(flat) / self._sample_rate
            else:
                self._last_adc_end = None
            self._chunks.append(flat)
            self._received_frames += len(flat)
            self._callback_count += 1
            self._samples_since_level += len(flat)

        if status_event is not None and self._on_status is not None:
            try:
                self._on_status(status_event)
            except Exception as exc:
                self._record_observer_error("status", exc)

        self._segmenter.feed(flat)

        if self._on_level and self._samples_since_level >= self._level_interval_samples:
            self._samples_since_level %= self._level_interval_samples
            rms = float(np.sqrt(np.mean(flat**2))) if len(flat) else 0.0
            try:
                self._on_level(rms)
            except Exception as exc:
                self._record_observer_error("level", exc)

    def stop(self) -> CaptureResult:
        """Stop capture, flush VAD, and return immutable audio/job data."""
        if not self._recording:
            raise RuntimeError("Recording session is not active")

        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
            self._segmenter.flush()
        finally:
            # A PortAudio teardown or VAD flush error must not leave the
            # session permanently marked active and make every later start fail.
            self._recording = False

        with self._lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.array([], dtype=np.float32)
            return CaptureResult(
                audio=audio,
                segments=tuple(self._segments),
                status_events=tuple(self._status_events),
                gap_events=tuple(self._gap_events),
                vad_events=tuple(self._vad_events),
                sample_rate=self._sample_rate,
                received_frames=self._received_frames,
                callback_count=self._callback_count,
            )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        adc_time = (
            time_info.get("inputBufferAdcTime")
            if isinstance(time_info, dict)
            else getattr(time_info, "inputBufferAdcTime", None)
        )
        self.feed(
            indata,
            status=status,
            expected_frames=frames,
            adc_time=adc_time,
        )

    def _on_segment(self, audio: np.ndarray, probabilities: list[float]) -> None:
        with self._lock:
            emitted = next(
                (event for event in reversed(self._vad_events) if event.kind == "emit"),
                None,
            )
        segment = CapturedSegment(
            audio=np.asarray(audio, dtype=np.float32).reshape(-1).copy(),
            probabilities=tuple(float(value) for value in probabilities),
            start_sample=emitted.start_sample if emitted else None,
            end_sample=emitted.end_sample if emitted else None,
        )
        with self._lock:
            self._segments.append(segment)
            segment_index = len(self._segments) - 1
        if self._on_segment_emitted is not None:
            try:
                self._on_segment_emitted(segment, segment_index)
            except Exception as exc:
                self._record_observer_error("segment", exc)

    def _on_vad_event(self, event: VADDecisionEvent) -> None:
        with self._lock:
            self._vad_events.append(event)

    def _record_observer_error(self, observer: str, exc: Exception) -> None:
        with self._lock:
            self._status_events.append(
                CaptureStatusEvent(
                    sample_offset=self._received_frames,
                    message=f"{observer} observer failed: {type(exc).__name__}: {exc}",
                )
            )
