"""Non-destructively index legacy JSONL/WAV logs in the SQLite history store."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.output.history_store import DEFAULT_DB_PATH, HistoryStore


def _entries(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def import_logs(store: HistoryStore, sessions: Path, transcripts: Path) -> tuple[int, int]:
    imported = skipped = 0
    for jsonl in sorted(sessions.glob("*/session.jsonl")):
        for line_number, entry in _entries(jsonl):
            audio_file = entry.get("audio_file")
            metadata = {"legacy_entry": entry}
            if audio_file:
                metadata["legacy_audio_path"] = str(jsonl.parent / audio_file)
            audio_sec = float(entry.get("audio_sec") or 0)
            added = store.import_legacy_entry(
                f"{jsonl.resolve()}:{line_number}",
                created_at=entry.get("ts") or datetime.now(UTC).isoformat(),
                text=entry.get("processed") or entry.get("raw") or "",
                sample_rate=16000,
                received_frames=round(audio_sec * 16000),
                metadata=metadata,
            )
            imported += int(added)
            skipped += int(not added)

    for line_number, entry in _entries(transcripts):
        added = store.import_legacy_entry(
            f"{transcripts.resolve()}:{line_number}",
            created_at=entry.get("ts") or datetime.now(UTC).isoformat(),
            text=entry.get("processed") or entry.get("raw") or "",
            sample_rate=16000,
            received_frames=round(float(entry.get("audio_sec") or 0) * 16000),
            metadata={"legacy_entry": entry},
        )
        imported += int(added)
        skipped += int(not added)
    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--sessions", type=Path, default=Path("data/logs/sessions"))
    parser.add_argument("--transcripts", type=Path, default=Path("data/logs/transcripts.jsonl"))
    args = parser.parse_args()

    imported, skipped = import_logs(HistoryStore(args.db), args.sessions, args.transcripts)
    print(f"Imported {imported} legacy entries; skipped {skipped} already indexed entries.")


if __name__ == "__main__":
    main()
