"""Streaming ASR session — real-time VAD + per-sentence transcription.

Coordinates: microphone → streaming VAD → ASR engine → post-processing pipeline.
Accumulates all transcribed text; copies to clipboard on stop.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass

import numpy as np
import torch

from asr_input.asr.base import ASREngine
from asr_input.audio.streaming_vad import StreamingVAD
from asr_input.output.session_log import SessionLogger
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
        silence_min_ms: int = 300,
        ramp_start_sec: float = 10.0,
        ramp_end_sec: float = 25.0,
        vad_threshold: float = 0.5,
        min_speech_ms: int = 250,
        speech_pad_ms: int = 100,
        max_segment_sec: float = 30.0,
        min_energy: float = 0.005,
        hallucination_threshold_sec: float = 1.5,
        min_hallucination_audio_sec: float = 3.0,
        fallback_silence_ms: list[int] | None = None,
        fallback_rms_target: float = 0.05,
        on_partial: Callable[[str, str], None] | None = None,
        on_transcribing: Callable[[float], None] | None = None,
        on_segment_done: Callable[[], None] | None = None,
        verbose: bool = False,
        session_logger: SessionLogger | None = None,
    ) -> None:
        self._engine = engine
        self._pipeline = pipeline
        self._sample_rate = sample_rate
        self._vad_threshold = vad_threshold
        self._min_speech_ms = min_speech_ms
        self._speech_pad_ms = speech_pad_ms
        self._min_energy = min_energy
        self._hallucination_threshold = hallucination_threshold_sec
        self._min_hallucination_audio_sec = min_hallucination_audio_sec
        self._fallback_silence_ms = fallback_silence_ms or [500, 300]
        self._fallback_rms_target = fallback_rms_target
        self._on_partial = on_partial
        self._on_transcribing = on_transcribing
        self._on_segment_done = on_segment_done
        self._verbose = verbose
        self._logger = session_logger
        self._session_start: float = 0

        self._segments: list[str] = []
        self._segment_stats: list[dict] = []
        self._segment_queue: queue.Queue[tuple[np.ndarray, list[float]] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._stop_event = threading.Event()
        self._stream = None

        self._vad = StreamingVAD(
            on_speech_segment=self._on_speech_segment,
            sample_rate=sample_rate,
            threshold=vad_threshold,
            min_speech_ms=min_speech_ms,
            silence_trigger_ms=silence_trigger_ms,
            silence_min_ms=silence_min_ms,
            ramp_start_sec=ramp_start_sec,
            ramp_end_sec=ramp_end_sec,
            speech_pad_ms=speech_pad_ms,
            max_segment_sec=max_segment_sec,
            min_energy=min_energy,
            verbose=verbose,
        )
        self._vad.load(vad_model)

    def start(self) -> None:
        """Start recording from microphone and processing."""
        self._begin()

        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def start_from_file(self, audio: np.ndarray, chunk_size: int = 1024) -> None:
        """Feed a pre-loaded audio array as if it came from a microphone."""
        self._begin()
        for i in range(0, len(audio), chunk_size):
            self._vad.feed(audio[i : i + chunk_size])
        self._vad.flush()
        self._segment_queue.put(None)

    def start_from_segments(self, segments: list[tuple[np.ndarray, list[float]]]) -> None:
        """Transcribe VAD segments captured before the ASR model was ready."""
        self.start_segment_stream()
        for audio, probabilities in segments:
            self.feed_segment(audio, probabilities)
        self.finish_segment_stream()

    def start_segment_stream(self) -> None:
        """Start an externally-fed segment stream.

        Recording/VAD may continue producing segments while this worker drains
        them.  This restores live transcription without coupling microphone
        ownership back into the ASR session.
        """
        self._begin()

    def feed_segment(self, audio: np.ndarray, probabilities: list[float]) -> None:
        """Queue one already-segmented utterance for transcription."""
        if self._worker is None:
            raise RuntimeError("Segment stream is not active")
        self._segment_queue.put((audio, probabilities))

    def finish_segment_stream(self) -> None:
        """Signal that no more externally-fed segments will arrive."""
        if self._worker is None:
            raise RuntimeError("Segment stream is not active")
        self._segment_queue.put(None)

    def stop(self) -> str:
        """Stop recording, flush remaining audio, return full text."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._vad.flush()
            self._segment_queue.put(None)

        if self._worker is not None:
            self._worker.join(timeout=60)
            if self._worker.is_alive():
                raise TimeoutError("ASR worker did not finish within 60 seconds")
            self._worker = None

        if self._worker_error is not None:
            error, self._worker_error = self._worker_error, None
            raise RuntimeError("ASR worker failed") from error

        return "".join(self._segments)

    @property
    def segment_stats(self) -> list[dict]:
        """Per-segment statistics: audio_sec, transcribe_sec, ratio, raw, processed."""
        return list(self._segment_stats)

    def _begin(self) -> None:
        self._segments.clear()
        self._segment_stats.clear()
        self._segment_queue = queue.Queue()
        self._vad.reset()
        self._stop_event.clear()
        self._worker_error = None
        self._session_start = time.time()
        self._worker = threading.Thread(target=self._run_transcribe_worker, daemon=True)
        self._worker.start()

    def _run_transcribe_worker(self) -> None:
        try:
            self._transcribe_loop()
        except BaseException as exc:
            self._worker_error = exc

    @property
    def partial_text(self) -> str:
        """Current accumulated text (may still be incomplete)."""
        return "".join(self._segments)

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._vad.feed(indata)

    def _on_speech_segment(self, audio: np.ndarray, probs: list[float]) -> None:
        """Called by StreamingVAD when a complete speech segment is detected."""
        self._segment_queue.put((audio, probs))

    def _transcribe_one(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe a single audio segment. Returns (raw_text, elapsed_sec)."""
        t0 = time.time()
        raw_text = self._engine.transcribe(audio, self._sample_rate)
        dt = time.time() - t0
        return raw_text, dt

    def _engine_usage(self) -> dict | None:
        usage = getattr(self._engine, "last_usage", None)
        if usage is None:
            return None
        if is_dataclass(usage):
            return asdict(usage)
        return dict(usage) if isinstance(usage, dict) else None

    def _fallback_transcribe(
        self, audio: np.ndarray, probs: list[float]
    ) -> tuple[list[tuple[str, float, float]], bool]:
        """Try re-segmenting with gradually shorter silence thresholds.

        Steps down from fallback_silence_ms[0] to fallback_silence_ms[-1] in
        100ms increments. At each step, resegment (free — no model needed) and
        check if it produces a different segmentation. On the first threshold
        that yields a new split, transcribe the sub-segments. If any sub-segment
        still hallucinates, continue stepping down and try the next new split.

        Returns ``(results, succeeded)``. ``succeeded`` is false when every
        available split was exhausted with at least one still-slow subsegment.
        """
        start_ms = self._fallback_silence_ms[0]
        end_ms = self._fallback_silence_ms[-1]
        step_ms = 100

        last_results: list[tuple[str, float, float]] = []
        prev_seg_lens: tuple[int, ...] = (len(audio),)

        silence_ms = start_ms
        while silence_ms >= end_ms:
            sub_segments = StreamingVAD.resegment(
                audio,
                probs,
                silence_trigger_ms=silence_ms,
                sample_rate=self._sample_rate,
                threshold=self._vad_threshold,
                min_speech_ms=self._min_speech_ms,
                speech_pad_ms=self._speech_pad_ms,
                min_energy=self._min_energy,
            )

            if not sub_segments:
                silence_ms -= step_ms
                continue

            cur_seg_lens = tuple(len(s) for s in sub_segments)
            is_new_split = cur_seg_lens != prev_seg_lens
            prev_seg_lens = cur_seg_lens

            if not is_new_split:
                silence_ms -= step_ms
                continue

            if self._verbose:
                print(
                    f"  [fallback] {silence_ms}ms → {len(sub_segments)} 子段",
                    flush=True,
                )

            results: list[tuple[str, float, float]] = []
            any_hallucination = False

            for sub_audio in sub_segments:
                sub_sec = len(sub_audio) / self._sample_rate
                raw, dt = self._transcribe_one(sub_audio)
                if self._verbose:
                    warn = " ⚠仍幻覺" if dt > self._hallucination_threshold else ""
                    print(
                        f"    子段 {sub_sec:.1f}s→{dt:.1f}s{warn} | {raw[:50]}",
                        flush=True,
                    )
                results.append((raw, sub_sec, dt))
                if dt > self._hallucination_threshold:
                    any_hallucination = True

            last_results = results
            if not any_hallucination:
                return results, True

            silence_ms -= step_ms

        return last_results, False

    def _try_rms_normalize(self, audio: np.ndarray) -> tuple[np.ndarray, str, float] | None:
        """RMS-normalize and retry only when the samples actually change."""
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms == 0 or rms >= self._fallback_rms_target:
            return None
        gain = self._fallback_rms_target / rms
        normalized = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        text, dt = self._transcribe_one(normalized)
        return normalized, text, dt

    def _transcribe_loop(self) -> None:
        """Worker thread: pull segments from queue, transcribe, accumulate."""
        while True:
            item = self._segment_queue.get()
            if item is None:
                break

            segment_audio, segment_probs = item
            audio_sec = len(segment_audio) / self._sample_rate
            rms = float(np.sqrt(np.mean(segment_audio**2)))
            fallback_exhausted = False

            self._notify_observer(self._on_transcribing, audio_sec)

            raw_text, dt = self._transcribe_one(segment_audio)
            usage = self._engine_usage()
            attempts = [
                {
                    "kind": "original",
                    "audio_sec": round(audio_sec, 2),
                    "transcribe_sec": round(dt, 2),
                    "raw": raw_text,
                    "adopted": False,
                    "settings": {"sample_rate": self._sample_rate},
                    **({"usage": usage} if usage else {}),
                }
            ]
            adopted_attempt = 0

            if (
                getattr(self._engine, "transcription_latency_is_quality_signal", True)
                and dt > self._hallucination_threshold
                and audio_sec >= self._min_hallucination_audio_sec
            ):
                if self._verbose:
                    print(
                        f"  [⚠幻覺偵測] {audio_sec:.1f}s→{dt:.1f}s "
                        f"(>{self._hallucination_threshold}s)",
                        flush=True,
                    )

                # Step 1: try RMS normalize on the full segment
                normalized_attempt = self._try_rms_normalize(segment_audio)
                if normalized_attempt is not None:
                    normalized, raw_norm, dt_norm = normalized_attempt
                    attempts.append(
                        {
                            "kind": "rms_normalize",
                            "audio_sec": round(audio_sec, 2),
                            "transcribe_sec": round(dt_norm, 2),
                            "raw": raw_norm,
                            "adopted": False,
                            "settings": {
                                "target_rms": self._fallback_rms_target,
                                "samples_changed": not np.array_equal(normalized, segment_audio),
                            },
                        }
                    )
                    if dt_norm <= self._hallucination_threshold:
                        raw_text = raw_norm
                        dt = dt_norm
                        adopted_attempt = len(attempts) - 1
                        if self._verbose:
                            print(
                                f"  [✓RMS normalize] {dt_norm:.2f}s — 幻覺消除",
                                flush=True,
                            )
                    elif self._verbose:
                        print(
                            f"  [RMS normalize] {dt_norm:.2f}s — 仍幻覺 → fallback 重切",
                            flush=True,
                        )
                elif self._verbose:
                    print(
                        "  [RMS normalize] 音量已達目標，跳過相同音訊的單純重試",
                        flush=True,
                    )

                # Step 2: if still hallucinating, try resegment
                if dt > self._hallucination_threshold:
                    fallback_results, fallback_succeeded = self._fallback_transcribe(
                        segment_audio, segment_probs
                    )

                    if fallback_results:
                        sub_stats = []
                        for sub_raw, sub_sec, sub_dt in fallback_results:
                            if sub_raw.strip():
                                processed = self._pipeline.run(sub_raw)
                                self._segments.append(processed)
                            else:
                                processed = ""
                            sub_stats.append(
                                {
                                    "audio_sec": round(sub_sec, 2),
                                    "transcribe_sec": round(sub_dt, 2),
                                    "raw": sub_raw,
                                    "processed": processed,
                                }
                            )
                            attempts.append(
                                {
                                    "kind": "resegment",
                                    "audio_sec": round(sub_sec, 2),
                                    "transcribe_sec": round(sub_dt, 2),
                                    "raw": sub_raw,
                                    "processed": processed,
                                    "adopted": True,
                                    "settings": {"fallback_silence_ms": self._fallback_silence_ms},
                                }
                            )

                        self._segment_stats.append(
                            {
                                "audio_sec": round(audio_sec, 2),
                                "transcribe_sec": round(dt, 2),
                                "rms": round(rms, 5),
                                "fallback": True,
                                "fallback_succeeded": fallback_succeeded,
                                "fallback_exhausted": not fallback_succeeded,
                                "original_raw": raw_text,
                                "original_dt": round(dt, 2),
                                "sub_segments": sub_stats,
                                "attempts": attempts,
                            }
                        )

                        if self._logger:
                            self._logger.log_fallback_segment(
                                original_audio=segment_audio,
                                original_raw=raw_text,
                                original_dt=dt,
                                audio_sec=audio_sec,
                                rms=rms,
                                sub_segments=sub_stats,
                            )

                        seg_num = len(self._segments)
                        combined = "".join(s["processed"] for s in sub_stats if s["processed"])
                        print(
                            f"  [#{seg_num}] {audio_sec:.1f}s→fallback "
                            f"({len(sub_stats)}子段"
                            f"{'，已用盡仍偏慢' if not fallback_succeeded else ''}) | {combined}",
                            flush=True,
                        )

                        self._notify_observer(self._on_partial, combined, "".join(self._segments))
                        self._notify_observer(self._on_segment_done)
                        continue
                    fallback_exhausted = True

            attempts[adopted_attempt]["adopted"] = True

            if not raw_text.strip():
                self._segment_stats.append(
                    {
                        "audio_sec": round(audio_sec, 2),
                        "transcribe_sec": round(dt, 2),
                        "ratio": round(dt / audio_sec, 3) if audio_sec > 0 else 0,
                        "rms": round(rms, 5),
                        "raw": "",
                        "processed": "",
                        "empty": True,
                        "fallback_exhausted": fallback_exhausted,
                        "attempts": attempts,
                    }
                )
                if self._logger:
                    self._logger.log_segment(
                        audio=segment_audio,
                        raw_text="",
                        processed_text="",
                        audio_sec=audio_sec,
                        transcribe_sec=dt,
                        rms=rms,
                        extra={"empty": True, **({"usage": usage} if usage else {})},
                    )
                self._notify_observer(self._on_segment_done)
                continue

            processed = self._pipeline.run(raw_text)
            self._segments.append(processed)

            self._segment_stats.append(
                {
                    "audio_sec": round(audio_sec, 2),
                    "transcribe_sec": round(dt, 2),
                    "ratio": round(dt / audio_sec, 3) if audio_sec > 0 else 0,
                    "rms": round(rms, 5),
                    "raw": raw_text,
                    "processed": processed,
                    "empty": False,
                    "fallback_exhausted": fallback_exhausted,
                    "attempts": attempts,
                }
            )

            if self._logger:
                self._logger.log_segment(
                    audio=segment_audio,
                    raw_text=raw_text,
                    processed_text=processed,
                    audio_sec=audio_sec,
                    transcribe_sec=dt,
                    rms=rms,
                    extra={"usage": usage} if usage else None,
                )

            seg_num = len(self._segments)
            print(
                f"  [#{seg_num}] {audio_sec:.1f}s→{dt:.1f}s | {processed}",
                flush=True,
            )

            self._notify_observer(self._on_partial, processed, "".join(self._segments))
            self._notify_observer(self._on_segment_done)

    @staticmethod
    def _notify_observer(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            print(
                f"ASR observer failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
