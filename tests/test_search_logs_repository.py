import json

from scripts import search_logs

from asr_input.output.history_store import HistoryStore


def test_search_loader_reads_sqlite_and_deduplicates_imported_jsonl(tmp_path, monkeypatch):
    log_root = tmp_path / "logs"
    sessions = log_root / "sessions"
    session = sessions / "old"
    session.mkdir(parents=True)
    jsonl = session / "session.jsonl"
    entry = {
        "ts": "2026-01-01T00:00:00+00:00",
        "audio_file": "seg_000.wav",
        "processed": "同一筆",
    }
    jsonl.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    database = log_root / "history.sqlite3"
    store = HistoryStore(database)
    store.import_legacy_entry(
        f"{jsonl.resolve()}:1",
        created_at=entry["ts"],
        text=entry["processed"],
        sample_rate=16000,
        received_frames=0,
        metadata={
            "legacy_entry": entry,
            "legacy_audio_path": str(session / entry["audio_file"]),
        },
    )
    monkeypatch.setattr(search_logs, "DEFAULT_DB_PATH", database)
    monkeypatch.setattr(search_logs, "SESSION_DIR", sessions)

    loaded = search_logs.load_session_entries()

    assert len(loaded) == 1
    assert loaded[0]["processed"] == "同一筆"
    assert loaded[0]["_source"].startswith("sqlite:")
