"""Safe wav → FLAC conversion for the audio archive (design + tests only, not wired in).

Nothing in the running app calls this yet. It exists so the safety rule that
`docs/roadmap.md` asks for — **a failed conversion must leave the original wav
untouched** — is written down as executable behaviour and covered by tests before
any migration is considered.

The rule is enforced by ordering, not by cleanup-after-the-fact:

1. Encode into a temporary file next to the target, never onto the target path.
2. Read the temporary file back and compare every sample against the source.
   FLAC is lossless in principle; this checks that it was lossless *here*.
3. Only after the comparison passes, `os.replace()` the temporary file into place
   (atomic within a filesystem).
4. Only after that, and only when explicitly asked, remove the source wav.

Any failure before step 3 leaves the archive exactly as it was; the temporary file
is removed. The caller decides whether to delete the source, so a conversion run
can be done in two passes (convert everything, verify, then reclaim space).

Empty wavs are a real case in this archive: 3 of 542 files on 2026-09-03 were
44-byte header-only recordings. FLAC cannot encode a zero-frame stream
(libsndfile 1.2.2 fails to open the file for writing), so they are reported as
`skipped-empty` rather than as failures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ConversionResult:
    """What happened to one file. `source_intact` is the safety property."""

    source: Path
    target: Path | None
    status: str  # "converted" | "skipped-empty" | "failed"
    source_bytes: int
    target_bytes: int | None = None
    source_removed: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "converted"


def convert_to_flac(
    wav_path: Path,
    *,
    target: Path | None = None,
    remove_source: bool = False,
    verify: bool = True,
) -> ConversionResult:
    """Convert one wav to FLAC, leaving the source untouched unless it fully succeeds.

    `remove_source=True` deletes the wav only after the FLAC is in place and (when
    `verify`) has been confirmed sample-identical. `verify=False` skips the
    read-back — faster, but then a silent encoder bug would be undetectable, so
    the default is on.
    """
    wav_path = Path(wav_path)
    source_bytes = wav_path.stat().st_size
    out_path = Path(target) if target else wav_path.with_suffix(".flac")
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    def failed(message: str) -> ConversionResult:
        tmp_path.unlink(missing_ok=True)
        return ConversionResult(
            source=wav_path, target=None, status="failed", source_bytes=source_bytes, error=message
        )

    try:
        data, samplerate = sf.read(str(wav_path), dtype="int16", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - 讀不起來也算失敗，原檔不動
        return failed(f"read failed: {exc!r}")

    if data.shape[0] == 0:
        return ConversionResult(
            source=wav_path,
            target=None,
            status="skipped-empty",
            source_bytes=source_bytes,
            error="0 frames; FLAC cannot encode an empty stream",
        )

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(tmp_path), data, samplerate, format="FLAC", subtype="PCM_16")
    except Exception as exc:  # noqa: BLE001
        return failed(f"encode failed: {exc!r}")

    if verify:
        try:
            back, back_rate = sf.read(str(tmp_path), dtype="int16", always_2d=True)
        except Exception as exc:  # noqa: BLE001
            return failed(f"verify read failed: {exc!r}")
        if back_rate != samplerate or back.shape != data.shape or not np.array_equal(back, data):
            return failed("verify mismatch: decoded FLAC differs from source PCM")

    try:
        os.replace(tmp_path, out_path)
    except OSError as exc:
        return failed(f"replace failed: {exc!r}")

    removed = False
    if remove_source and wav_path.resolve() != out_path.resolve():
        try:
            wav_path.unlink()
            removed = True
        except OSError as exc:
            # The FLAC is already in place and verified, so the conversion itself
            # succeeded; a leftover wav costs space, not data.
            return ConversionResult(
                source=wav_path,
                target=out_path,
                status="converted",
                source_bytes=source_bytes,
                target_bytes=out_path.stat().st_size,
                source_removed=False,
                error=f"source not removed: {exc!r}",
            )

    return ConversionResult(
        source=wav_path,
        target=out_path,
        status="converted",
        source_bytes=source_bytes,
        target_bytes=out_path.stat().st_size,
        source_removed=removed,
    )
