"""SQLite history, recovery, lossless audio, and retention tests."""

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import soundfile as sf

from asr_input.output.history_store import HistoryStore


def test_pending_audio_round_trips_through_pcm16_wav(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    audio = np.array([-0.5, -0.125, 0.0, 0.125, 0.5], dtype=np.float32)

    job_id = store.create_pending(
        audio,
        sample_rate=16000,
        received_frames=len(audio),
        metadata={"callback_count": 2},
    )

    job = store.recoverable()[0]
    assert job.id == job_id
    assert job.metadata == {"callback_count": 2}
    decoded, sample_rate = sf.read(job.audio_path, dtype="float32")
    assert sample_rate == 16000
    np.testing.assert_allclose(decoded, audio, atol=1 / 32768)


def test_history_wav_uses_legacy_pcm16_quantization(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    audio = np.linspace(-0.95, 0.95, 1000, dtype=np.float32)
    expected_pcm = (audio * 32767).astype(np.int16)
    job_id = store.create_pending(audio, sample_rate=16000, received_frames=len(audio))

    stored_pcm, _ = sf.read(store.recoverable()[0].audio_path, dtype="int16")

    assert job_id
    np.testing.assert_array_equal(stored_pcm, expected_pcm)


def test_success_keeps_recent_text_but_can_remove_transient_audio(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )
    audio_path = store.recoverable()[0].audio_path

    store.complete(job_id, "辨識結果", retain_audio=False)

    assert not audio_path.exists()
    assert store.recoverable() == []
    recent = store.recent()
    assert recent[0].text == "辨識結果"
    assert recent[0].audio_path is None


def test_recent_text_is_stored_even_when_audio_persistence_is_disabled(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(10, dtype=np.float32),
        sample_rate=16000,
        received_frames=10,
        persist_audio=False,
    )

    store.complete(job_id, "text only", retain_audio=False)

    assert store.recent()[0].text == "text only"
    assert list(store.audio_dir.iterdir()) == []


def test_failure_remains_recoverable(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )

    store.fail(job_id, "CUDA unavailable")

    job = store.recoverable()[0]
    assert job.state == "failed"
    assert job.error == "CUDA unavailable"
    assert job.audio_path.exists()
    assert job.metadata["error_history"][0]["error"] == "CUDA unavailable"

    store.mark_pending(job_id)
    retried = store.recoverable()[0]
    assert retried.state == "pending"
    assert retried.error is None
    assert retried.metadata["error_history"][0]["error"] == "CUDA unavailable"


def test_successful_retry_keeps_prior_error_history(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )
    store.fail(job_id, "first failure")
    store.mark_pending(job_id)
    store.complete(job_id, "recovered", retain_audio=True)

    job = store.recent()[0]
    assert job.error is None
    assert job.metadata["error_history"][0]["error"] == "first failure"


def test_recovery_can_select_pending_without_automatically_retrying_failures(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    pending_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )
    failed_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )
    store.fail(failed_id, "CUDA unavailable")

    assert [job.id for job in store.recoverable(("pending",))] == [pending_id]
    assert [job.id for job in store.recoverable(("failed",))] == [failed_id]


def test_budget_removes_oldest_audio_and_preserves_metadata(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    ids = []
    for index in range(2):
        job_id = store.create_pending(
            np.zeros(400, dtype=np.float32), sample_rate=16000, received_frames=400
        )
        store.complete(job_id, f"result-{index}", retain_audio=True)
        ids.append(job_id)

    removed = store.enforce_audio_budget(0)

    assert removed == 2
    assert {job.id for job in store.recent()} == set(ids)
    assert all(job.audio_path is None for job in store.recent())


def test_budget_never_removes_labeled_regression_audio(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(400, dtype=np.float32), sample_rate=16000, received_frames=400
    )
    store.complete(job_id, "important", retain_audio=True)
    store.add_corpus_label(job_id, "regression", source="observed")

    assert store.enforce_audio_budget(0) == 0
    assert store.recent()[0].audio_path.exists()


def test_recovery_audio_expires_but_metadata_tombstone_remains(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (old, job_id))

    assert store.expire_recovery(max_age_hours=24) == 1

    with sqlite3.connect(store.path) as connection:
        state, audio_path, error = connection.execute(
            "SELECT state, audio_path, error FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert (state, audio_path, error) == ("failed", None, "recovery audio expired")


def test_database_enables_wal_and_tracks_schema_version(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 2


def test_segment_metadata_and_evidence_provenance_are_structured(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )
    store.record_segments(
        job_id,
        [
            {
                "audio_sec": 0.5,
                "start_sample": 100,
                "end_sample": 8100,
                "transcribe_sec": 1.8,
                "raw": "測試",
                "processed": "測試",
                "fallback": True,
                "attempts": [
                    {
                        "kind": "original",
                        "audio_sec": 0.5,
                        "transcribe_sec": 1.8,
                        "raw": "测试",
                        "adopted": False,
                        "settings": {"beam_size": 5},
                    },
                    {
                        "kind": "resegment",
                        "audio_sec": 0.4,
                        "transcribe_sec": 0.3,
                        "raw": "測試",
                        "adopted": True,
                    },
                ],
            }
        ],
    )
    store.add_corpus_label(
        job_id,
        "engine-specific",
        segment_index=0,
        source="observed",
        reason="absolute latency exceeded 1.5s",
    )

    with sqlite3.connect(store.path) as connection:
        segment = connection.execute(
            "SELECT start_sample, end_sample, transcribe_sec, fallback "
            "FROM segments WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        label = connection.execute(
            "SELECT label, source FROM corpus_labels WHERE job_id = ?", (job_id,)
        ).fetchone()
        attempts = connection.execute(
            "SELECT kind, adopted FROM asr_attempts WHERE job_id = ? ORDER BY attempt_index",
            (job_id,),
        ).fetchall()
    assert segment == (100, 8100, 1.8, 1)
    assert label == ("engine-specific", "observed")
    assert attempts == [("original", 0), ("resegment", 1)]


def test_evidence_source_must_be_explicit(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(1, dtype=np.float32), sample_rate=16000, received_frames=1
    )

    with pytest.raises(ValueError, match="evidence source"):
        store.add_corpus_label(job_id, "candidate", source="guessed")


def test_correction_remains_candidate_until_explicit_decision(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    correction_id = store.add_correction_candidate(
        "錯的完整詞組",
        "對的完整詞組",
        "這是包含錯的完整詞組的一整句。",
        source="observed",
    )

    with sqlite3.connect(store.path) as connection:
        status = connection.execute(
            "SELECT status FROM corrections WHERE id = ?", (correction_id,)
        ).fetchone()[0]
    assert status == "candidate"

    store.decide_correction(correction_id, "confirmed")
    with sqlite3.connect(store.path) as connection:
        status = connection.execute(
            "SELECT status FROM corrections WHERE id = ?", (correction_id,)
        ).fetchone()[0]
    assert status == "confirmed"


def test_analysis_entries_hide_schema_details_from_viewer(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(16000, dtype=np.float32), sample_rate=16000, received_frames=16000
    )
    store.record_segments(
        job_id,
        [
            {
                "start_sample": 100,
                "end_sample": 8100,
                "audio_sec": 0.5,
                "transcribe_sec": 1.8,
                "raw": "测试",
                "processed": "測試",
                "fallback_exhausted": True,
            }
        ],
    )
    store.complete(job_id, "測試", retain_audio=True)

    entry = store.analysis_entries()[0]

    assert entry["raw"] == "测试"
    assert entry["processed"] == "測試"
    assert (entry["start_sample"], entry["end_sample"]) == (100, 8100)
    assert entry["fallback_exhausted"] is True
    assert entry["audio_file"].endswith(".wav")
    assert entry["_audio_url"].startswith("audio/")


def test_legacy_import_is_idempotent_and_does_not_manage_old_audio(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    metadata = {"legacy_audio_path": "data/logs/sessions/old/seg_000.wav"}

    assert store.import_legacy_entry(
        "old/session.jsonl:1",
        created_at="2026-01-01T00:00:00+00:00",
        text="舊資料",
        sample_rate=16000,
        received_frames=32000,
        metadata=metadata,
    )
    assert not store.import_legacy_entry(
        "old/session.jsonl:1",
        created_at="2026-01-01T00:00:00+00:00",
        text="舊資料",
        sample_rate=16000,
        received_frames=32000,
        metadata=metadata,
    )

    job = store.recent()[0]
    assert job.audio_path is None
    assert job.metadata == metadata


def test_export_and_backup_are_readable_and_complete(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    job_id = store.create_pending(
        np.zeros(10, dtype=np.float32), sample_rate=16000, received_frames=10
    )
    store.complete(job_id, "backup me", retain_audio=False)

    exported = store.export_data()
    backup = tmp_path / "backup" / "history.sqlite3"
    store.backup_to(backup)

    assert exported["jobs"][0]["text"] == "backup me"
    with sqlite3.connect(backup) as connection:
        assert connection.execute("SELECT text FROM jobs").fetchone()[0] == "backup me"


def test_app_setting_round_trips_and_updates(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    assert store.get_setting("idle_unload_minutes") is None

    store.set_setting("idle_unload_minutes", "30")
    store.set_setting("idle_unload_minutes", "0")

    assert store.get_setting("idle_unload_minutes") == "0"


def test_unknown_schema_version_fails_visibly(tmp_path):
    path = tmp_path / "history.sqlite3"
    store = HistoryStore(path)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE schema_meta SET version = 999")

    with pytest.raises(RuntimeError, match="Unsupported history schema 999"):
        HistoryStore(path)


def test_schema_v1_migrates_offsets_without_losing_tables(tmp_path):
    path = tmp_path / "history.sqlite3"
    store = HistoryStore(path)
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE segments")
        connection.execute(
            """
            CREATE TABLE segments (
                job_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                audio_sec REAL,
                transcribe_sec REAL,
                raw_text TEXT,
                processed_text TEXT,
                fallback INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(job_id, segment_index)
            )
            """
        )
        connection.execute("UPDATE schema_meta SET version = 1")

    HistoryStore(path)

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(segments)")}
        version = connection.execute("SELECT version FROM schema_meta").fetchone()[0]
    assert {"start_sample", "end_sample"} <= columns
    assert version == 2


def test_corrupt_database_fails_visibly(tmp_path):
    path = tmp_path / "history.sqlite3"
    path.write_bytes(b"not a sqlite database")

    with pytest.raises(sqlite3.DatabaseError):
        HistoryStore(path)
