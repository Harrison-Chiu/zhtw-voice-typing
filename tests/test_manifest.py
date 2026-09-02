"""Tests for the benchmark manifest schema.

The property that matters most here is the public/private split: the repo is
public, so a manifest that quietly carries an audio path or a gold transcript is
a leak, not a cosmetic problem.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from asr_input.eval.manifest import (
    Manifest,
    ManifestError,
    RunResult,
    Sample,
    check_private_map,
    load_manifest,
    save_manifest,
    sha256_file,
    validate_manifest,
)


def make_sample(sample_id="s0001", split="dev", **kw) -> Sample:
    defaults = {"duration_sec": 4.2, "audio_sha256": "0" * 64, "labels": ["mixed_lang"]}
    defaults.update(kw)
    return Sample(sample_id=sample_id, split=split, **defaults)


def test_roundtrip_through_json(tmp_path):
    manifest = Manifest(
        corpus="daily-2026-09", samples=[make_sample(), make_sample("s0002", "test")]
    )
    path = tmp_path / "manifest.json"

    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert [s.sample_id for s in loaded.samples] == ["s0001", "s0002"]
    assert loaded.samples[0].labels == ["mixed_lang"]
    assert loaded.corpus == "daily-2026-09"


def test_by_split_selects_only_that_split():
    manifest = Manifest(
        corpus="c",
        samples=[make_sample("a", "dev"), make_sample("b", "test"), make_sample("c", "dev")],
    )
    assert [s.sample_id for s in manifest.by_split("dev")] == ["a", "c"]


@pytest.mark.parametrize("leak_key", ["audio_path", "transcript", "text", "gold", "filename"])
def test_private_fields_are_rejected(leak_key):
    data = {
        "version": 1,
        "corpus": "c",
        "samples": [
            {
                "sample_id": "s1",
                "split": "dev",
                "duration_sec": 1.0,
                "audio_sha256": "0" * 64,
                leak_key: "data/logs/audio/abc.wav",
            }
        ],
    }
    with pytest.raises(ManifestError, match="private fields"):
        validate_manifest(data)


def test_saving_refuses_what_loading_would_reject(tmp_path, monkeypatch):
    # A Sample cannot carry a path, so the mistake is simulated at the dict level:
    # save_manifest must validate its own output rather than trust the caller.
    manifest = Manifest(corpus="c", samples=[make_sample()])
    monkeypatch.setattr(
        Manifest,
        "as_dict",
        lambda self: {
            "version": 1,
            "corpus": "c",
            "samples": [
                {
                    "sample_id": "s1",
                    "split": "dev",
                    "duration_sec": 1.0,
                    "audio_sha256": "0" * 64,
                    "audio_path": "leak.wav",
                }
            ],
        },
    )
    path = tmp_path / "m.json"
    with pytest.raises(ManifestError):
        save_manifest(manifest, path)
    assert not path.exists()


def test_unknown_split_is_rejected():
    data = {
        "version": 1,
        "corpus": "c",
        "samples": [
            {"sample_id": "s1", "split": "train", "duration_sec": 1.0, "audio_sha256": "x"}
        ],
    }
    with pytest.raises(ManifestError, match="unknown split"):
        validate_manifest(data)


def test_duplicate_sample_ids_are_rejected():
    entry = {"sample_id": "s1", "split": "dev", "duration_sec": 1.0, "audio_sha256": "x"}
    with pytest.raises(ManifestError, match="duplicate"):
        validate_manifest({"version": 1, "corpus": "c", "samples": [entry, dict(entry)]})


def test_missing_required_field_is_rejected():
    with pytest.raises(ManifestError, match="missing"):
        validate_manifest(
            {"version": 1, "corpus": "c", "samples": [{"sample_id": "s1", "split": "dev"}]}
        )


def test_version_mismatch_is_rejected():
    with pytest.raises(ManifestError, match="version"):
        validate_manifest({"version": 99, "corpus": "c", "samples": []})


def test_corpus_name_is_required():
    with pytest.raises(ManifestError, match="corpus"):
        validate_manifest({"version": 1, "samples": []})


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "a.bin"
    payload = b"some audio bytes" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_private_map_check_reports_both_directions():
    manifest = Manifest(corpus="c", samples=[make_sample("s1"), make_sample("s2")])
    private = {"s1": {"audio_path": "x.wav"}, "s3": {"audio_path": "y.wav"}}

    problems = check_private_map(manifest, private)

    assert "s2: missing from private map" in problems
    assert "s3: in private map but not in manifest" in problems


def test_private_map_check_is_silent_when_consistent():
    manifest = Manifest(corpus="c", samples=[make_sample("s1")])
    assert check_private_map(manifest, {"s1": {"audio_path": "x.wav"}}) == []


def test_private_entry_without_audio_path_is_a_problem():
    manifest = Manifest(corpus="c", samples=[make_sample("s1")])
    problems = check_private_map(manifest, {"s1": {"gold_verbatim": "..."}})
    assert problems == ["s1: private entry has no audio_path"]


def test_run_result_saves_readable_json(tmp_path):
    result = RunResult(
        run_id="r1",
        manifest_corpus="c",
        engine={"name": "faster-whisper"},
        environment={"gpu": {"available": False}},
        samples=[{"sample_id": "s1", "strict_cer": {"rate": 0.1}}],
        aggregate={"strict_cer": 0.1},
    )
    path = tmp_path / "runs" / "r1.json"
    result.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "r1"
    assert data["samples"][0]["sample_id"] == "s1"
    assert data["aggregate"]["strict_cer"] == 0.1
