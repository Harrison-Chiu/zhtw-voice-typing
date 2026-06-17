"""Append-only transcript log — one JSONL entry per transcription."""

import json
from datetime import UTC, datetime
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "transcripts.jsonl"


def log_transcript(
    raw_text: str,
    processed_text: str,
    audio_duration_sec: float | None = None,
    source: str = "mic",
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "source": source,
        "raw": raw_text,
        "processed": processed_text,
    }
    if audio_duration_sec is not None:
        entry["audio_sec"] = round(audio_duration_sec, 1)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
