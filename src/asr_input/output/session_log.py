"""Per-session logging with audio segments — for offline experimentation.

Each session creates a timestamped directory under data/logs/sessions/ containing:
- seg_NNN.wav  — raw audio for each segment (before any normalization)
- session.jsonl — one JSON line per segment with metadata
- session_meta.json — session-level config snapshot
"""

from __future__ import annotations

import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from asr_input.paths import logs_dir

LOG_ROOT = logs_dir() / "sessions"


class SessionLogger:
    """Logs audio + metadata for one streaming session."""

    def __init__(
        self,
        sample_rate: int = 16000,
        enabled: bool = True,
        config_snapshot: dict | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._enabled = enabled
        self._seg_index = 0
        self._session_dir: Path | None = None

        if not enabled:
            return

        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self._session_dir = LOG_ROOT / ts
        self._session_dir.mkdir(parents=True, exist_ok=True)

        if config_snapshot:
            meta_path = self._session_dir / "session_meta.json"
            meta_path.write_text(
                json.dumps(config_snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def log_segment(
        self,
        audio: np.ndarray,
        raw_text: str,
        processed_text: str,
        audio_sec: float,
        transcribe_sec: float,
        rms: float,
        extra: dict | None = None,
    ) -> None:
        if not self._enabled or self._session_dir is None:
            return

        seg_name = f"seg_{self._seg_index:03d}"
        self._seg_index += 1

        wav_path = self._session_dir / f"{seg_name}.wav"
        _write_wav(wav_path, audio, self._sample_rate)

        entry: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "seg": seg_name,
            "audio_file": wav_path.name,
            "audio_sec": round(audio_sec, 2),
            "transcribe_sec": round(transcribe_sec, 2),
            "rms": round(rms, 5),
            "raw": raw_text,
            "processed": processed_text,
        }
        if extra:
            entry.update(extra)

        jsonl_path = self._session_dir / "session.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_fallback_segment(
        self,
        original_audio: np.ndarray,
        original_raw: str,
        original_dt: float,
        audio_sec: float,
        rms: float,
        sub_segments: list[dict],
    ) -> None:
        """Log a segment that went through hallucination fallback."""
        if not self._enabled or self._session_dir is None:
            return

        seg_name = f"seg_{self._seg_index:03d}"
        self._seg_index += 1

        wav_path = self._session_dir / f"{seg_name}.wav"
        _write_wav(wav_path, original_audio, self._sample_rate)

        entry: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "seg": seg_name,
            "audio_file": wav_path.name,
            "audio_sec": round(audio_sec, 2),
            "transcribe_sec": round(original_dt, 2),
            "rms": round(rms, 5),
            "raw": original_raw,
            "fallback": True,
            "sub_segments": sub_segments,
        }

        jsonl_path = self._session_dir / "session.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
