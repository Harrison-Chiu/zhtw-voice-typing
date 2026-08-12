import json

from scripts.import_legacy_logs import import_logs

from asr_input.output.history_store import HistoryStore


def test_importer_indexes_session_and_transcript_logs_without_modifying_them(tmp_path):
    sessions = tmp_path / "sessions"
    session = sessions / "old"
    session.mkdir(parents=True)
    session_jsonl = session / "session.jsonl"
    session_jsonl.write_text(
        json.dumps(
            {
                "ts": "2026-01-01T00:00:00+00:00",
                "audio_file": "seg_000.wav",
                "audio_sec": 1.25,
                "raw": "测试",
                "processed": "測試",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    transcripts = tmp_path / "transcripts.jsonl"
    transcripts.write_text(
        json.dumps({"processed": "只有文字"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    before = session_jsonl.read_bytes(), transcripts.read_bytes()
    store = HistoryStore(tmp_path / "history.sqlite3")

    assert import_logs(store, sessions, transcripts) == (2, 0)
    assert import_logs(store, sessions, transcripts) == (0, 2)
    assert before == (session_jsonl.read_bytes(), transcripts.read_bytes())
    assert {job.text for job in store.recent()} == {"測試", "只有文字"}
