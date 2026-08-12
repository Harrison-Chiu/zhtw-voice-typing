"""SQLite metadata store and lossless audio files for jobs and recent results."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
DEFAULT_DB_PATH = Path("data/logs/history.sqlite3")


@dataclass(frozen=True)
class StoredJob:
    id: str
    created_at: str
    state: str
    sample_rate: int
    received_frames: int
    audio_path: Path | None
    text: str | None
    error: str | None
    metadata: dict[str, Any]


class HistoryStore:
    """Small local database; SQLite is embedded and requires no server."""

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = path
        self._write_lock = threading.RLock()
        self.audio_dir = path.parent / "audio"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_pending(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        received_frames: int,
        metadata: dict[str, Any] | None = None,
        persist_audio: bool = True,
    ) -> str:
        job_id = uuid.uuid4().hex
        audio_path = self.audio_dir / f"{job_id}.wav" if persist_audio else None
        if audio_path is not None:
            self._write_wav_atomic(audio_path, audio, sample_rate)
        now = datetime.now(UTC).isoformat()
        try:
            with self._write_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        id, created_at, state, sample_rate, received_frames,
                        audio_path, metadata_json
                    ) VALUES (?, ?, 'pending', ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        now,
                        sample_rate,
                        received_frames,
                        str(audio_path) if audio_path is not None else None,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
        except Exception:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
            raise
        return job_id

    def complete(self, job_id: str, text: str, *, retain_audio: bool) -> None:
        audio_path = self._audio_path_for(job_id)
        with self._write_connection() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET state = 'completed', completed_at = ?, text = ?, error = NULL,
                    audio_path = CASE WHEN ? THEN audio_path ELSE NULL END
                WHERE id = ?
                """,
                (datetime.now(UTC).isoformat(), text, int(retain_audio), job_id),
            )
        if not retain_audio and audio_path is not None:
            audio_path.unlink(missing_ok=True)

    def fail(self, job_id: str, error: str) -> None:
        with self._write_connection() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            metadata = json.loads(row[0]) if row and row[0] else {}
            error_history = metadata.setdefault("error_history", [])
            error_history.append({"at": datetime.now(UTC).isoformat(), "error": error})
            connection.execute(
                "UPDATE jobs SET state = 'failed', error = ?, metadata_json = ? WHERE id = ?",
                (error, json.dumps(metadata, ensure_ascii=False), job_id),
            )

    def mark_pending(self, job_id: str) -> None:
        with self._write_connection() as connection:
            connection.execute(
                "UPDATE jobs SET state = 'pending', error = NULL WHERE id = ?",
                (job_id,),
            )

    def record_segments(self, job_id: str, segments: list[dict[str, Any]]) -> None:
        """Store derived ASR observations; the full capture remains the audio SSOT."""
        with self._write_connection() as connection:
            for index, segment in enumerate(segments):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO segments (
                        job_id, segment_index, start_sample, end_sample,
                        audio_sec, transcribe_sec,
                        raw_text, processed_text, fallback, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        index,
                        segment.get("start_sample"),
                        segment.get("end_sample"),
                        segment.get("audio_sec"),
                        segment.get("transcribe_sec"),
                        segment.get("raw", segment.get("original_raw")),
                        segment.get("processed"),
                        int(bool(segment.get("fallback"))),
                        json.dumps(segment, ensure_ascii=False),
                    ),
                )
                for attempt_index, attempt in enumerate(segment.get("attempts", [])):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO asr_attempts (
                            job_id, segment_index, attempt_index, kind,
                            audio_sec, transcribe_sec, raw_text, processed_text,
                            adopted, settings_json, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job_id,
                            index,
                            attempt_index,
                            attempt.get("kind", "unknown"),
                            attempt.get("audio_sec"),
                            attempt.get("transcribe_sec"),
                            attempt.get("raw"),
                            attempt.get("processed"),
                            int(bool(attempt.get("adopted"))),
                            json.dumps(attempt.get("settings", {}), ensure_ascii=False),
                            json.dumps(attempt, ensure_ascii=False),
                        ),
                    )

    def add_corpus_label(
        self,
        job_id: str,
        label: str,
        *,
        segment_index: int = -1,
        source: str,
        reason: str = "",
    ) -> None:
        if source not in {"observed", "derived", "hypothesis"}:
            raise ValueError(f"Invalid evidence source: {source}")
        with self._write_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO corpus_labels (
                    job_id, segment_index, label, source, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    segment_index,
                    label,
                    source,
                    reason,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def add_correction_candidate(
        self,
        original_text: str,
        replacement_text: str,
        context_text: str,
        *,
        source: str,
        job_id: str | None = None,
    ) -> int:
        if source not in {"observed", "derived", "hypothesis"}:
            raise ValueError(f"Invalid evidence source: {source}")
        with self._write_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO corrections (
                    job_id, original_text, replacement_text, context_text,
                    status, source, created_at
                ) VALUES (?, ?, ?, ?, 'candidate', ?, ?)
                """,
                (
                    job_id,
                    original_text,
                    replacement_text,
                    context_text,
                    source,
                    datetime.now(UTC).isoformat(),
                ),
            )
            return cursor.lastrowid

    def decide_correction(self, correction_id: int, status: str) -> None:
        if status not in {"confirmed", "rejected"}:
            raise ValueError("Correction status must be confirmed or rejected")
        with self._write_connection() as connection:
            connection.execute(
                "UPDATE corrections SET status = ? WHERE id = ?",
                (status, correction_id),
            )

    def import_legacy_entry(
        self,
        source_key: str,
        *,
        created_at: str,
        text: str,
        sample_rate: int,
        received_frames: int,
        metadata: dict[str, Any],
    ) -> bool:
        """Import metadata once without moving, copying, or deleting legacy files."""
        job_id = uuid.uuid5(uuid.NAMESPACE_URL, source_key).hex
        with self._write_connection() as connection:
            if connection.execute(
                "SELECT 1 FROM legacy_imports WHERE source_key = ?", (source_key,)
            ).fetchone():
                return False
            connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    id, created_at, completed_at, state, sample_rate,
                    received_frames, text, metadata_json
                ) VALUES (?, ?, ?, 'completed', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    created_at,
                    created_at,
                    sample_rate,
                    received_frames,
                    text,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            connection.execute(
                "INSERT INTO legacy_imports(source_key, job_id, imported_at) VALUES (?, ?, ?)",
                (source_key, job_id, datetime.now(UTC).isoformat()),
            )
        return True

    def recoverable(self, states: tuple[str, ...] = ("pending", "failed")) -> list[StoredJob]:
        allowed = {"pending", "failed"}
        if not states or not set(states).issubset(allowed):
            raise ValueError("recoverable states must contain pending and/or failed")
        placeholders = ", ".join("?" for _ in states)
        return self._query_jobs(
            f"WHERE state IN ({placeholders}) AND audio_path IS NOT NULL ORDER BY created_at",
            states,
        )

    def recent(self, limit: int = 20) -> list[StoredJob]:
        return self._query_jobs(
            "WHERE state = 'completed' AND text IS NOT NULL AND text <> '' "
            "ORDER BY completed_at DESC LIMIT ?",
            (limit,),
        )

    def analysis_entries(self) -> list[dict[str, Any]]:
        """Expose viewer/search records without making tools query schema details."""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT j.id, j.created_at, j.audio_path, j.text, j.metadata_json,
                       s.segment_index, s.start_sample, s.end_sample, s.audio_sec,
                       s.transcribe_sec, s.raw_text, s.processed_text,
                       s.metadata_json AS segment_metadata_json,
                       l.source_key AS legacy_source_key
                FROM jobs j
                LEFT JOIN segments s ON s.job_id = j.id
                LEFT JOIN legacy_imports l ON l.job_id = j.id
                WHERE j.state = 'completed'
                ORDER BY j.created_at, s.segment_index
                """
            ).fetchall()
        entries = []
        for row in rows:
            job_metadata = json.loads(row["metadata_json"])
            segment_metadata = (
                json.loads(row["segment_metadata_json"]) if row["segment_metadata_json"] else {}
            )
            audio_path_text = row["audio_path"] or job_metadata.get("legacy_audio_path")
            audio_path = Path(audio_path_text) if audio_path_text else None
            audio_url = ""
            if audio_path:
                try:
                    audio_url = (
                        audio_path.resolve().relative_to(self.path.parent.resolve()).as_posix()
                    )
                except ValueError:
                    audio_url = audio_path.as_posix()
            entries.append(
                {
                    "ts": row["created_at"],
                    "raw": row["raw_text"] or "",
                    "processed": row["processed_text"] or row["text"] or "",
                    "audio_sec": row["audio_sec"],
                    "transcribe_sec": row["transcribe_sec"],
                    "start_sample": row["start_sample"],
                    "end_sample": row["end_sample"],
                    "fallback": bool(segment_metadata.get("fallback")),
                    "fallback_exhausted": bool(segment_metadata.get("fallback_exhausted")),
                    "audio_file": audio_path.name if audio_path else "",
                    "_audio_url": audio_url,
                    "_session_dir": str(audio_path.parent) if audio_path else "",
                    "_source": f"sqlite:{row['id'][:8]}",
                    "_job_id": row["id"],
                    "_legacy_source_key": row["legacy_source_key"],
                }
            )
        return entries

    def enforce_audio_budget(self, max_bytes: int) -> int:
        """Delete oldest unprotected completed audio, retaining all metadata."""
        rows = self._query_jobs(
            "WHERE state = 'completed' AND audio_path IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM corpus_labels c WHERE c.job_id = jobs.id) "
            "ORDER BY completed_at DESC"
        )
        sizes = [(job, self._file_size(job.audio_path)) for job in rows]
        total = sum(size for _, size in sizes)
        removed = 0
        for job, size in reversed(sizes):
            if total <= max_bytes:
                break
            if job.audio_path is not None:
                job.audio_path.unlink(missing_ok=True)
            with self._write_connection() as connection:
                connection.execute("UPDATE jobs SET audio_path = NULL WHERE id = ?", (job.id,))
            total -= size
            removed += 1
        return removed

    def expire_recovery(self, max_age_hours: float = 24) -> int:
        """Drop stale recovery audio while retaining an explicit metadata tombstone."""
        cutoff = (datetime.now(UTC) - timedelta(hours=max_age_hours)).isoformat()
        jobs = self._query_jobs(
            "WHERE state IN ('pending', 'failed') AND audio_path IS NOT NULL "
            "AND created_at < ? ORDER BY created_at",
            (cutoff,),
        )
        with self._write_connection() as connection:
            for job in jobs:
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'failed', audio_path = NULL,
                        error = 'recovery audio expired'
                    WHERE id = ?
                    """,
                    (job.id,),
                )
        for job in jobs:
            if job.audio_path is not None:
                job.audio_path.unlink(missing_ok=True)
        return len(jobs)

    def export_data(self) -> dict[str, list[dict[str, Any]]]:
        """Return a stable, JSON-serializable snapshot for tools and support."""
        tables = (
            "jobs",
            "segments",
            "asr_attempts",
            "corrections",
            "corpus_labels",
            "legacy_imports",
            "app_settings",
        )
        exported: dict[str, list[dict[str, Any]]] = {}
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            for table in tables:
                exported[table] = [
                    dict(row) for row in connection.execute(f"SELECT * FROM {table}")
                ]
        return exported

    def backup_to(self, destination: Path) -> None:
        """Create a transactionally consistent SQLite backup with atomic publish."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            with self._connect() as source:
                target = sqlite3.connect(temporary)
                try:
                    source.backup(target)
                finally:
                    target.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._write_connection() as connection:
            connection.execute(
                """
                INSERT INTO app_settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, datetime.now(UTC).isoformat()),
            )

    def _initialize(self) -> None:
        with self._write_connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    state TEXT NOT NULL CHECK(state IN ('pending', 'completed', 'failed')),
                    sample_rate INTEGER NOT NULL,
                    received_frames INTEGER NOT NULL,
                    audio_path TEXT,
                    text TEXT,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS jobs_recent
                    ON jobs(state, completed_at DESC);
                CREATE TABLE IF NOT EXISTS segments (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL,
                    start_sample INTEGER,
                    end_sample INTEGER,
                    audio_sec REAL,
                    transcribe_sec REAL,
                    raw_text TEXT,
                    processed_text TEXT,
                    fallback INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(job_id, segment_index)
                );
                CREATE TABLE IF NOT EXISTS corpus_labels (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL DEFAULT -1,
                    label TEXT NOT NULL,
                    source TEXT NOT NULL CHECK(source IN ('observed', 'derived', 'hypothesis')),
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, segment_index, label)
                );
                CREATE TABLE IF NOT EXISTS asr_attempts (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    audio_sec REAL,
                    transcribe_sec REAL,
                    raw_text TEXT,
                    processed_text TEXT,
                    adopted INTEGER NOT NULL DEFAULT 0,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(job_id, segment_index, attempt_index)
                );
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
                    original_text TEXT NOT NULL,
                    replacement_text TEXT NOT NULL,
                    context_text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate',
                    source TEXT NOT NULL CHECK(source IN ('observed', 'derived', 'hypothesis')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS legacy_imports (
                    source_key TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row[0] == 1:
                columns = {
                    column[1] for column in connection.execute("PRAGMA table_info(segments)")
                }
                if "start_sample" not in columns:
                    connection.execute("ALTER TABLE segments ADD COLUMN start_sample INTEGER")
                if "end_sample" not in columns:
                    connection.execute("ALTER TABLE segments ADD COLUMN end_sample INTEGER")
                connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
            elif row[0] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported history schema {row[0]} (expected {SCHEMA_VERSION})"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self._connect() as connection:
            yield connection

    def _query_jobs(self, suffix: str, parameters: tuple[Any, ...] = ()) -> list[StoredJob]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, state, sample_rate, received_frames,
                       audio_path, text, error, metadata_json
                FROM jobs
                """
                + suffix,
                parameters,
            ).fetchall()
        return [
            StoredJob(
                id=row[0],
                created_at=row[1],
                state=row[2],
                sample_rate=row[3],
                received_frames=row[4],
                audio_path=Path(row[5]) if row[5] else None,
                text=row[6],
                error=row[7],
                metadata=json.loads(row[8]),
            )
            for row in rows
        ]

    def _audio_path_for(self, job_id: str) -> Path | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT audio_path FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return Path(row[0]) if row and row[0] else None

    @staticmethod
    def _file_size(path: Path | None) -> int:
        if path is None:
            return 0
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    @staticmethod
    def _write_wav_atomic(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        import soundfile as sf

        temporary = path.with_suffix(".tmp.wav")
        try:
            # Keep the legacy PCM16 quantization rule until codec experiments
            # explicitly approve a different storage format.
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            sf.write(temporary, pcm, sample_rate, format="WAV", subtype="PCM_16")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
