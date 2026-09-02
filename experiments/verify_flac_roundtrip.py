"""Verify that PCM16 wav -> FLAC -> PCM16 is bit-exact, and measure what it costs.

Why this exists: `docs/roadmap.md` names FLAC as the first candidate for shrinking
`data/logs/audio/`, gated on three things being measured rather than assumed —
lossless round-trip, actual capacity ratio on *this* corpus, and encode/decode
cost. This script measures all three. It only reads the existing wav files and
writes FLAC into a temporary directory; nothing under `data/logs/` is modified.

"Bit-exact" here means every sample of every channel compares equal as int16.
A single mismatching file fails the whole run — an audio archive that is lossless
"almost always" is not lossless.

Usage:
    python experiments/verify_flac_roundtrip.py                 # 全量
    python experiments/verify_flac_roundtrip.py --sample 60     # 隨機抽樣
    python experiments/verify_flac_roundtrip.py --out results/flac.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.paths import logs_dir  # noqa: E402

DURATION_BUCKETS = (1.0, 3.0, 6.0, 12.0, 24.0, 60.0)


def bucket_of(seconds: float) -> str:
    low = 0.0
    for edge in DURATION_BUCKETS:
        if seconds < edge:
            return f"{low:g}-{edge:g}s"
        low = edge
    return f">{DURATION_BUCKETS[-1]:g}s"


def check_one(path: Path, tmp_dir: Path) -> dict:
    """Round-trip a single file and return its measurements."""
    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype="int16", always_2d=True)

    flac_path = tmp_dir / (path.stem + ".flac")
    t0 = time.perf_counter()
    sf.write(str(flac_path), data, sr, format="FLAC", subtype="PCM_16")
    encode_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    back, sr_back = sf.read(str(flac_path), dtype="int16", always_2d=True)
    decode_sec = time.perf_counter() - t0

    identical = sr_back == sr and back.shape == data.shape and bool(np.array_equal(back, data))
    first_bad = None
    if not identical and back.shape == data.shape:
        diff = np.flatnonzero(np.any(back != data, axis=1))
        if diff.size:
            first_bad = int(diff[0])

    wav_bytes = path.stat().st_size
    flac_bytes = flac_path.stat().st_size
    flac_path.unlink()

    duration = info.frames / info.samplerate if info.samplerate else 0.0
    return {
        "file": path.name,
        "identical": identical,
        "first_mismatch_frame": first_bad,
        "duration_sec": round(duration, 3),
        "bucket": bucket_of(duration),
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "wav_bytes": wav_bytes,
        "flac_bytes": flac_bytes,
        "ratio": round(flac_bytes / wav_bytes, 4) if wav_bytes else None,
        "encode_sec": round(encode_sec, 4),
        "decode_sec": round(decode_sec, 4),
    }


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["identical"]]
    wav_total = sum(r["wav_bytes"] for r in rows)
    flac_total = sum(r["flac_bytes"] for r in rows)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    buckets = {}
    for name in sorted(by_bucket, key=lambda b: by_bucket[b][0]["duration_sec"]):
        group = by_bucket[name]
        g_wav = sum(r["wav_bytes"] for r in group)
        g_flac = sum(r["flac_bytes"] for r in group)
        buckets[name] = {
            "files": len(group),
            "wav_bytes": g_wav,
            "flac_bytes": g_flac,
            "ratio": round(g_flac / g_wav, 4) if g_wav else None,
            "median_encode_sec": round(statistics.median(r["encode_sec"] for r in group), 4),
            "median_decode_sec": round(statistics.median(r["decode_sec"] for r in group), 4),
        }

    subtypes: dict[str, int] = defaultdict(int)
    for row in rows:
        subtypes[f"{row['subtype']}@{row['samplerate']}x{row['channels']}"] += 1

    return {
        "files": len(rows),
        "bit_exact": len(ok),
        "mismatched": [r["file"] for r in rows if not r["identical"]],
        "wav_bytes": wav_total,
        "flac_bytes": flac_total,
        "overall_ratio": round(flac_total / wav_total, 4) if wav_total else None,
        "total_audio_sec": round(sum(r["duration_sec"] for r in rows), 1),
        "encode_sec_total": round(sum(r["encode_sec"] for r in rows), 2),
        "decode_sec_total": round(sum(r["decode_sec"] for r in rows), 2),
        "by_bucket": buckets,
        "source_formats": dict(subtypes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=logs_dir() / "audio")
    parser.add_argument("--sample", type=int, default=0, help="隨機抽 N 個檔（0＝全量）")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", type=Path, default=None, help="結果 JSON 路徑")
    args = parser.parse_args()

    files = sorted(args.audio_dir.glob("*.wav"))
    if not files:
        print(f"找不到 wav：{args.audio_dir}")
        return 1
    if args.sample and args.sample < len(files):
        files = random.Random(args.seed).sample(files, args.sample)
        files.sort()

    print(f"來源 {args.audio_dir}，檔案 {len(files)} 個")
    rows = []
    with tempfile.TemporaryDirectory(prefix="flac_roundtrip_") as tmp:
        tmp_dir = Path(tmp)
        for i, path in enumerate(files, 1):
            try:
                rows.append(check_one(path, tmp_dir))
            except Exception as exc:  # noqa: BLE001 - 單檔失敗不該中斷整批
                rows.append(
                    {
                        "file": path.name,
                        "identical": False,
                        "error": repr(exc),
                        "duration_sec": 0.0,
                        "bucket": "error",
                        "samplerate": None,
                        "channels": None,
                        "subtype": None,
                        "wav_bytes": path.stat().st_size,
                        "flac_bytes": 0,
                        "ratio": None,
                        "encode_sec": 0.0,
                        "decode_sec": 0.0,
                        "first_mismatch_frame": None,
                    }
                )
                print(f"  [{i}] {path.name} 失敗：{exc!r}")
            if i % 50 == 0:
                print(f"  ...{i}/{len(files)}")

    stats = summarise(rows)
    print(f"\nbit-exact {stats['bit_exact']}/{stats['files']}")
    if stats["mismatched"]:
        print("不符的檔案：" + ", ".join(stats["mismatched"][:10]))
    print(
        f"wav {stats['wav_bytes'] / 2**20:.1f} MiB → flac {stats['flac_bytes'] / 2**20:.1f} MiB"
        f"（{stats['overall_ratio']:.3f}，省 {1 - stats['overall_ratio']:.1%}）"
    )
    print(f"音訊總長 {stats['total_audio_sec'] / 60:.1f} 分鐘")
    print(f"編碼總耗時 {stats['encode_sec_total']}s，解碼 {stats['decode_sec_total']}s")
    print("\n各時長區間：")
    for name, b in stats["by_bucket"].items():
        print(
            f"  {name:>10}  n={b['files']:<4} ratio={b['ratio']}  "
            f"enc(med)={b['median_encode_sec']}s dec(med)={b['median_decode_sec']}s"
        )
    print(f"\n來源格式分布：{stats['source_formats']}")

    out = args.out or (
        Path(__file__).resolve().parent
        / "results"
        / f"flac_roundtrip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "audio_dir": str(args.audio_dir),
                "soundfile": sf.__version__,
                "libsndfile": sf.__libsndfile_version__,
                "sample": args.sample,
                "seed": args.seed,
                "stats": stats,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已寫出 {out}")
    return 0 if stats["bit_exact"] == stats["files"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
