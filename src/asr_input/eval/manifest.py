"""Benchmark manifest and result schemas (`docs/asr-benchmark-proposal.md` §3, §5).

The repo is public, so the split matters: a **manifest** carries only anonymous
sample IDs, labels, durations and audio hashes and is safe to commit; the
**private map** carries the actual audio paths and gold transcripts and must stay
in a gitignored directory. `validate_manifest()` enforces that split by rejecting
any manifest entry that carries a path or transcript text — it is easier to fail
loudly here than to notice a leaked transcript after it is in the history.

The audio hash is what ties the two together. It also detects the quieter
failure: a benchmark re-run whose numbers moved because the audio was
re-encoded, not because the model changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_VERSION = 1
SPLITS = ("smoke", "dev", "test")

# Keys that must never appear in a committed manifest.
FORBIDDEN_SAMPLE_KEYS = (
    "audio_path",
    "path",
    "file",
    "filename",
    "text",
    "transcript",
    "gold",
    "verbatim",
    "input_ready",
)


class ManifestError(ValueError):
    """Raised when a manifest would leak private data or is internally invalid."""


@dataclass
class Sample:
    """One benchmark item. Anonymous by construction — no path, no transcript."""

    sample_id: str
    split: str
    duration_sec: float
    audio_sha256: str
    sample_rate: int = 16000
    labels: list[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Manifest:
    corpus: str
    samples: list[Sample] = field(default_factory=list)
    version: int = MANIFEST_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "corpus": self.corpus,
            "created_at": self.created_at,
            "description": self.description,
            "samples": [s.as_dict() for s in self.samples],
        }

    def by_split(self, split: str) -> list[Sample]:
        return [s for s in self.samples if s.split == split]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(data: dict) -> Manifest:
    """Parse and check a manifest dict, raising `ManifestError` on any problem."""
    if data.get("version") != MANIFEST_VERSION:
        raise ManifestError(f"unsupported manifest version: {data.get('version')!r}")
    if not data.get("corpus"):
        raise ManifestError("manifest needs a corpus name")

    samples: list[Sample] = []
    seen: set[str] = set()
    for raw in data.get("samples", []):
        leaked = sorted(k for k in raw if k.lower() in FORBIDDEN_SAMPLE_KEYS)
        if leaked:
            raise ManifestError(
                f"sample {raw.get('sample_id')!r} carries private fields {leaked}; "
                "paths and transcripts belong in the private map, not the manifest"
            )
        missing = [
            k for k in ("sample_id", "split", "duration_sec", "audio_sha256") if k not in raw
        ]
        if missing:
            raise ManifestError(f"sample {raw.get('sample_id')!r} missing {missing}")
        if raw["split"] not in SPLITS:
            raise ManifestError(f"sample {raw['sample_id']!r} has unknown split {raw['split']!r}")
        if raw["sample_id"] in seen:
            raise ManifestError(f"duplicate sample_id {raw['sample_id']!r}")
        seen.add(raw["sample_id"])
        samples.append(
            Sample(
                sample_id=raw["sample_id"],
                split=raw["split"],
                duration_sec=float(raw["duration_sec"]),
                audio_sha256=raw["audio_sha256"],
                sample_rate=int(raw.get("sample_rate", 16000)),
                labels=list(raw.get("labels", [])),
                notes=raw.get("notes", ""),
            )
        )

    return Manifest(
        corpus=data["corpus"],
        samples=samples,
        version=data["version"],
        created_at=data.get("created_at", ""),
        description=data.get("description", ""),
    )


def load_manifest(path: Path) -> Manifest:
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def save_manifest(manifest: Manifest, path: Path) -> None:
    validate_manifest(manifest.as_dict())  # never write something we would reject
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- private side ----------------------------------------------------------


def load_private_map(path: Path) -> dict[str, dict]:
    """Load the sample_id → {audio_path, gold_*} mapping.

    Kept deliberately dumb: this file lives outside version control and is the
    only place the two halves are joined.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError("private map must be an object keyed by sample_id")
    return data


def check_private_map(manifest: Manifest, private: dict[str, dict]) -> list[str]:
    """Return the problems found, empty list if the two sides line up."""
    problems = []
    for sample in manifest.samples:
        entry = private.get(sample.sample_id)
        if entry is None:
            problems.append(f"{sample.sample_id}: missing from private map")
            continue
        if not entry.get("audio_path"):
            problems.append(f"{sample.sample_id}: private entry has no audio_path")
    extra = sorted(set(private) - {s.sample_id for s in manifest.samples})
    problems.extend(f"{sid}: in private map but not in manifest" for sid in extra)
    return problems


# --- results ---------------------------------------------------------------


@dataclass
class RunResult:
    """One benchmark run over one manifest with one engine configuration."""

    run_id: str
    manifest_corpus: str
    engine: dict
    environment: dict
    samples: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    version: int = MANIFEST_VERSION

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
