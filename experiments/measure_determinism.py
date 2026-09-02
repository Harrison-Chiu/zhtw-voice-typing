"""Measure whether faster-whisper output is reproducible across repeated runs.

Why this exists
---------------
`docs/roadmap.md` records a 2026-07-31 pilot: one difficult WAV produced 7
distinct texts over 10 runs under the default temperature fallback, and 10
identical texts once `temperature` was pinned to 0. That pilot took the *first
16* segments of a list whose leading entries all carried a hardcoded
`known_asr_error` label, so its sample was biased toward exactly the segments
most likely to be unstable. It could not say how common instability is.

This script re-measures on a **seeded, stratified random sample of all
segments**, drawn without looking at any label. Labels are reported afterwards
so the composition of the sample is visible, but they never influence selection.

Two arms, one variable
----------------------
Everything is held at the production config except `temperature`:

    A. production  - temperature=(0.0, 0.2, ..., 1.0), the shipped fallback ladder
    B. pinned      - temperature=0.0

Each segment is transcribed `--repeats` times per arm. The metric is the number
of distinct raw outputs; scoring is on raw engine text, because post-processing
is deterministic and would only mask differences.

Audio comes from the job's full capture WAV, sliced by the segment's sample
offsets - the same bytes the production run saw.

Output goes to `experiments/results/` (gitignored; it contains transcripts).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from asr_input.config import load_config  # noqa: E402

DB = ROOT / "data" / "logs" / "history.sqlite3"
OUT_DIR = ROOT / "experiments" / "results"

# Duration bands. Reproducibility plausibly depends on length (a long segment
# has more places to diverge; a very short one is where hallucinations live),
# so the sample is stratified rather than uniform over an unbalanced pool.
BANDS = [
    ("<1s", 0.0, 1.0),
    ("1-3s", 1.0, 3.0),
    ("3-8s", 3.0, 8.0),
    ("8-20s", 8.0, 20.0),
    (">=20s", 20.0, float("inf")),
]

ARMS = {
    "production": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "pinned": (0.0,),
}


def band_of(seconds: float) -> str:
    for name, low, high in BANDS:
        if low <= seconds < high:
            return name
    return BANDS[-1][0]


def pick_samples(conn: sqlite3.Connection, per_band: int, seed: int) -> list[dict]:
    """Seeded stratified sample over every segment with reachable audio.

    Selection looks only at duration. No label, no transcript, no timing - the
    2026-07-31 pilot's mistake was letting an error label decide the sample.
    """
    rows = conn.execute(
        """
        SELECT s.job_id, s.segment_index, s.audio_sec, s.start_sample, s.end_sample,
               s.raw_text, j.audio_path, j.sample_rate
        FROM segments s JOIN jobs j ON j.id = s.job_id
        WHERE j.audio_path IS NOT NULL AND s.audio_sec > 0
        """
    ).fetchall()

    buckets: dict[str, list[dict]] = {name: [] for name, _, _ in BANDS}
    for row in rows:
        item = dict(row)
        if not (ROOT / item["audio_path"]).exists():
            continue
        buckets[band_of(item["audio_sec"])].append(item)

    rng = random.Random(seed)
    picked: list[dict] = []
    for name, _, _ in BANDS:
        pool = sorted(buckets[name], key=lambda r: (r["job_id"], r["segment_index"]))
        picked.extend(rng.sample(pool, min(per_band, len(pool))))
    return picked


def labels_for(conn: sqlite3.Connection, job_ids: set[str]) -> dict[str, list[str]]:
    if not job_ids:
        return {}
    marks = ",".join("?" * len(job_ids))
    out: dict[str, list[str]] = {}
    for job_id, label in conn.execute(
        f"SELECT job_id, label FROM corpus_labels WHERE job_id IN ({marks})", tuple(job_ids)
    ):
        out.setdefault(job_id, []).append(label)
    return out


def load_segment_audio(item: dict) -> np.ndarray:
    audio, _ = sf.read(str(ROOT / item["audio_path"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio[item["start_sample"] : item["end_sample"]]


def build_engine_for(temperature, config):
    from asr_input.asr.whisper_fw import WhisperFWEngine

    asr = dict(config["asr"])
    asr.pop("engine", None)
    asr.pop("temperature", None)
    return WhisperFWEngine(temperature=temperature, **asr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-band", type=int, default=6, help="每個時長分帶抽幾段")
    parser.add_argument("--repeats", type=int, default=5, help="每段每組跑幾次")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    samples = pick_samples(conn, args.per_band, args.seed)
    label_map = labels_for(conn, {s["job_id"] for s in samples})
    print(f"抽樣 {len(samples)} 段（seed={args.seed}，每帶 {args.per_band}）", flush=True)

    config = load_config(args.config)
    audio_cache = {(s["job_id"], s["segment_index"]): load_segment_audio(s) for s in samples}

    records: list[dict] = []
    for arm, temperature in ARMS.items():
        print(f"\n=== arm={arm} temperature={temperature} ===", flush=True)
        engine = build_engine_for(temperature, config)
        engine.load()
        for i, item in enumerate(samples, start=1):
            audio = audio_cache[(item["job_id"], item["segment_index"])]
            outputs, seconds = [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                outputs.append(engine.transcribe(audio, item["sample_rate"] or 16000))
                seconds.append(time.perf_counter() - start)
            unique = len(set(outputs))
            records.append(
                {
                    "arm": arm,
                    "job_id": item["job_id"],
                    "segment_index": item["segment_index"],
                    "band": band_of(item["audio_sec"]),
                    "audio_sec": round(item["audio_sec"], 2),
                    "labels": label_map.get(item["job_id"], []),
                    "repeats": args.repeats,
                    "unique_outputs": unique,
                    "identical": unique == 1,
                    "median_transcribe_sec": round(sorted(seconds)[len(seconds) // 2], 3),
                    "outputs": sorted(set(outputs)),
                    "production_raw_text": item["raw_text"],
                }
            )
            print(f"  [{i}/{len(samples)}] {item['audio_sec']:.1f}s -> {unique} 種", flush=True)
        engine.unload()

    summary = {}
    for arm in ARMS:
        arm_records = [r for r in records if r["arm"] == arm]
        by_band = Counter(r["band"] for r in arm_records if not r["identical"])
        summary[arm] = {
            "segments": len(arm_records),
            "identical": sum(1 for r in arm_records if r["identical"]),
            "unstable": sum(1 for r in arm_records if not r["identical"]),
            "max_unique": max((r["unique_outputs"] for r in arm_records), default=0),
            "unstable_by_band": dict(by_band),
        }

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"determinism_{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "seed": args.seed,
                "per_band": args.per_band,
                "repeats": args.repeats,
                "arms": {k: list(v) for k, v in ARMS.items()},
                "summary": summary,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
