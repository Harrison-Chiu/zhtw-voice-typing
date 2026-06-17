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
        verbose: bool = False,
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
        self._verbose = verbose
        self._session_start: float = 0

        self._segments: list[str] = []
        self._segment_stats: list[dict] = []
        self._segment_queue: queue.Queue[tuple[np.ndarray, list[float]] | None] = (
            queue.Queue()
        )
        self._worker: threading.Thread | None = None
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
            self._worker = None

        return "".join(self._segments)

    @property
    def segment_stats(self) -> list[dict]:
        """Per-segment statistics: audio_sec, transcribe_sec, ratio, raw, processed."""
        return list(self._segment_stats)

    def _begin(self) -> None:
        self._segments.clear()
        self._segment_stats.clear()
        self._vad.reset()
        self._stop_event.clear()
        self._session_start = time.time()
        self._worker = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._worker.start()

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

    def _on_speech_segment(self, audio: np.ndarray, probs: list[float]) -> None:
        """Called by StreamingVAD when a complete speech segment is detected."""
        self._segment_queue.put((audio, probs))

    def _transcribe_one(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe a single audio segment. Returns (raw_text, elapsed_sec)."""
        t0 = time.time()
        raw_text = self._engine.transcribe(audio, self._sample_rate)
        dt = time.time() - t0
        return raw_text, dt

    def _fallback_transcribe(
        self, audio: np.ndarray, probs: list[float]
    ) -> list[tuple[str, float, float]]:
        """Try re-segmenting with gradually shorter silence thresholds.

        Steps down from fallback_silence_ms[0] to fallback_silence_ms[-1] in
        100ms increments. At each step, resegment (free — no model needed) and
        check if it produces a different segmentation. On the first threshold
        that yields a new split, transcribe the sub-segments. If any sub-segment
        still hallucinates, continue stepping down and try the next new split.

        Returns list of (raw_text, audio_sec, transcribe_sec) for each sub-segment.
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
                return results

            silence_ms -= step_ms

        return last_results

    def _try_rms_normalize(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, str, float]:
        """RMS-normalize audio and re-transcribe. Returns (normalized_audio, text, dt)."""
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms == 0 or rms >= self._fallback_rms_target:
            text, dt = self._transcribe_one(audio)
            return audio, text, dt
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

            if self._on_transcribing:
                self._on_transcribing(audio_sec)

            raw_text, dt = self._transcribe_one(segment_audio)

            if (
                dt > self._hallucination_threshold
                and audio_sec >= self._min_hallucination_audio_sec
            ):
                if self._verbose:
                    print(
                        f"  [⚠幻覺偵測] {audio_sec:.1f}s→{dt:.1f}s "
                        f"(>{self._hallucination_threshold}s)",
                        flush=True,
                    )

                # Step 1: try RMS normalize on the full segment
                normalized, raw_norm, dt_norm = self._try_rms_normalize(
                    segment_audio
                )
                if dt_norm <= self._hallucination_threshold:
                    raw_text = raw_norm
                    dt = dt_norm
                    if self._verbose:
                        print(
                            f"  [✓RMS normalize] {dt_norm:.2f}s — 幻覺消除",
                            flush=True,
                        )
                else:
                    if self._verbose:
                        print(
                            f"  [RMS normalize] {dt_norm:.2f}s — 仍幻覺 → fallback 重切",
                            flush=True,
                        )

                # Step 2: if still hallucinating, try resegment
                if dt > self._hallucination_threshold:
                    fallback_results = self._fallback_transcribe(
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
                            sub_stats.append({
                                "audio_sec": round(sub_sec, 2),
                                "transcribe_sec": round(sub_dt, 2),
                                "raw": sub_raw,
                                "processed": processed,
                            })

                        self._segment_stats.append({
                            "audio_sec": round(audio_sec, 2),
                            "transcribe_sec": round(dt, 2),
                            "rms": round(rms, 5),
                            "fallback": True,
                            "original_raw": raw_text,
                            "original_dt": round(dt, 2),
                            "sub_segments": sub_stats,
                        })

                        seg_num = len(self._segments)
                        combined = "".join(
                            s["processed"] for s in sub_stats if s["processed"]
                        )
                        print(
                            f"  [#{seg_num}] {audio_sec:.1f}s→fallback "
                            f"({len(sub_stats)}子段) | {combined}",
                            flush=True,
                        )

                        if self._on_partial:
                            self._on_partial(combined, "".join(self._segments))
                        continue

            if not raw_text.strip():
                self._segment_stats.append({
                    "audio_sec": round(audio_sec, 2),
                    "transcribe_sec": round(dt, 2),
                    "ratio": round(dt / audio_sec, 3) if audio_sec > 0 else 0,
                    "rms": round(rms, 5),
                    "raw": "",
                    "processed": "",
                    "empty": True,
                })
                continue

            processed = self._pipeline.run(raw_text)
            self._segments.append(processed)

            self._segment_stats.append({
                "audio_sec": round(audio_sec, 2),
                "transcribe_sec": round(dt, 2),
                "ratio": round(dt / audio_sec, 3) if audio_sec > 0 else 0,
                "rms": round(rms, 5),
                "raw": raw_text,
                "processed": processed,
                "empty": False,
            })

            seg_num = len(self._segments)
            print(
                f"  [#{seg_num}] {audio_sec:.1f}s→{dt:.1f}s | {processed}",
                flush=True,
            )

            if self._on_partial:
                self._on_partial(processed, "".join(self._segments))
