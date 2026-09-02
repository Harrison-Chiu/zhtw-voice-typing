"""Tests for the safe wav → FLAC conversion helper.

The property under test is not "FLAC compresses" but "a failure never costs us the
original recording". Every failure path is therefore checked by comparing the
source file's bytes before and after the call.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from asr_input.storage.flac_archive import convert_to_flac

SAMPLE_RATE = 16000


def write_wav(path, frames: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.integers(-8000, 8000, size=(frames, 1), dtype=np.int16)
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")
    return data


def test_converted_flac_is_sample_identical(tmp_path):
    src = tmp_path / "a.wav"
    data = write_wav(src)

    result = convert_to_flac(src)

    assert result.ok
    assert result.target.exists()
    back, rate = sf.read(str(result.target), dtype="int16", always_2d=True)
    assert rate == SAMPLE_RATE
    assert np.array_equal(back, data)


def test_source_is_kept_by_default(tmp_path):
    src = tmp_path / "a.wav"
    before = write_wav(src)

    result = convert_to_flac(src)

    assert src.exists()
    assert result.source_removed is False
    assert np.array_equal(sf.read(str(src), dtype="int16", always_2d=True)[0], before)


def test_source_removed_only_when_asked(tmp_path):
    src = tmp_path / "a.wav"
    write_wav(src)

    result = convert_to_flac(src, remove_source=True)

    assert result.ok
    assert result.source_removed is True
    assert not src.exists()
    assert result.target.exists()


def test_empty_wav_is_skipped_not_failed(tmp_path):
    # 3 of 542 real recordings on 2026-09-03 were 44-byte header-only files.
    src = tmp_path / "empty.wav"
    sf.write(str(src), np.zeros((0, 1), dtype=np.int16), SAMPLE_RATE, subtype="PCM_16")
    before = src.read_bytes()

    result = convert_to_flac(src, remove_source=True)

    assert result.status == "skipped-empty"
    assert result.target is None
    assert src.read_bytes() == before
    assert not (tmp_path / "empty.flac").exists()


def test_unreadable_source_fails_without_touching_anything(tmp_path):
    src = tmp_path / "broken.wav"
    src.write_bytes(b"not a wav at all")
    before = src.read_bytes()

    result = convert_to_flac(src, remove_source=True)

    assert result.status == "failed"
    assert "read failed" in result.error
    assert src.read_bytes() == before
    assert list(tmp_path.iterdir()) == [src]


@pytest.mark.parametrize("stage", ["encode", "verify_read", "mismatch", "replace"])
def test_every_failure_stage_preserves_the_source(tmp_path, monkeypatch, stage):
    src = tmp_path / "a.wav"
    before = src.parent / "a.wav"
    write_wav(src)
    original_bytes = before.read_bytes()

    real_read = sf.read
    real_write = sf.write

    if stage == "encode":
        monkeypatch.setattr(
            sf, "write", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    elif stage == "verify_read":

        def read_fails_on_flac(path, *a, **k):
            if str(path).endswith(".tmp"):
                raise RuntimeError("boom")
            return real_read(path, *a, **k)

        monkeypatch.setattr(sf, "read", read_fails_on_flac)
    elif stage == "mismatch":

        def read_returns_wrong_data(path, *a, **k):
            data, rate = real_read(path, *a, **k)
            if str(path).endswith(".tmp"):
                data = data + 1  # decoded audio differs from the source
            return data, rate

        monkeypatch.setattr(sf, "read", read_returns_wrong_data)
    elif stage == "replace":
        monkeypatch.setattr(
            "asr_input.storage.flac_archive.os.replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
        )

    result = convert_to_flac(src, remove_source=True)

    monkeypatch.setattr(sf, "read", real_read)
    monkeypatch.setattr(sf, "write", real_write)

    assert result.status == "failed", stage
    assert result.target is None
    assert src.exists() and src.read_bytes() == original_bytes
    assert not (tmp_path / "a.flac").exists()
    # No half-written temporary file is left behind either.
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_verify_can_be_disabled(tmp_path):
    src = tmp_path / "a.wav"
    write_wav(src)

    result = convert_to_flac(src, verify=False)

    assert result.ok
    assert result.target.exists()


def test_explicit_target_path_is_used(tmp_path):
    src = tmp_path / "a.wav"
    write_wav(src)
    target = tmp_path / "nested" / "out.flac"

    result = convert_to_flac(src, target=target)

    assert result.ok
    assert result.target == target
    assert target.exists()
