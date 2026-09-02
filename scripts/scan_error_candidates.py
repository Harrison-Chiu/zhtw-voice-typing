"""Scan the history DB for error candidates worth a human review (roadmap E line).

High recall by design: every deterministic signal in `asr_input.eval.signals` is
applied to every segment, and anything that fires becomes a candidate. False
positives are expected — the point is to not miss real errors. Each candidate
records *which* signals fired, so the signals themselves can be judged after the
review (which ones only ever produce noise, which ones actually find errors).

A control group of segments that no signal touched is sampled alongside, matched
on audio-duration buckets. Without it a review only ever sees flagged segments
and there is no way to tell how many errors the signals miss.

Output is JSON under `data/logs/review/` — gitignored, because it contains
transcript text.

Usage:
    python scripts/scan_error_candidates.py                  # 掃描並寫出候選 JSON
    python scripts/scan_error_candidates.py --summary        # 只印統計，不寫檔
    python scripts/scan_error_candidates.py --control-ratio 0.3 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.eval.signals import SIGNAL_LABELS, Segment, scan_segment  # noqa: E402
from asr_input.paths import logs_dir  # noqa: E402

DEFAULT_DB = logs_dir() / "history.sqlite3"
DEFAULT_OUT_DIR = logs_dir() / "review"

# Duration buckets used to match the control group to the candidates. Boundaries
# follow the observed audio_sec distribution (p25 4.3s, p50 9.0s, p75 18.0s).
DURATION_BUCKETS = (1.0, 3.0, 6.0, 12.0, 24.0)


def bucket_of(audio_sec: float | None) -> str:
    if audio_sec is None:
        return "unknown"
    low = 0.0
    for edge in DURATION_BUCKETS:
        if audio_sec < edge:
            return f"{low:g}-{edge:g}s"
        low = edge
    return f">{DURATION_BUCKETS[-1]:g}s"


def load_segments(db_path: Path) -> list[dict]:
    """Read every segment plus the job context the signals need."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    jobs = {}
    for row in conn.execute(
        "SELECT id, created_at, audio_path, sample_rate, metadata_json FROM jobs"
    ):
        meta = json.loads(row["metadata_json"] or "{}")
        asr_cfg = meta.get("asr_config", {}) or {}
        jobs[row["id"]] = {
            "created_at": row["created_at"],
            "audio_path": row["audio_path"],
            "sample_rate": row["sample_rate"],
            "hotwords": asr_cfg.get("hotwords") or "",
            "initial_prompt": asr_cfg.get("initial_prompt") or "",
            "model_id": asr_cfg.get("model_id"),
            "beam_size": asr_cfg.get("beam_size"),
        }

    records = []
    query = (
        "SELECT job_id, segment_index, start_sample, end_sample, audio_sec, "
        "transcribe_sec, raw_text, processed_text, fallback, metadata_json "
        "FROM segments ORDER BY job_id, segment_index"
    )
    for row in conn.execute(query):
        job = jobs.get(row["job_id"], {})
        meta = json.loads(row["metadata_json"] or "{}")
        records.append(
            {
                "job_id": row["job_id"],
                "segment_index": row["segment_index"],
                "created_at": job.get("created_at"),
                "audio_path": job.get("audio_path"),
                "sample_rate": job.get("sample_rate"),
                "start_sample": row["start_sample"],
                "end_sample": row["end_sample"],
                "audio_sec": row["audio_sec"],
                "transcribe_sec": row["transcribe_sec"],
                "raw_text": row["raw_text"] or "",
                "processed_text": row["processed_text"] or "",
                "model_id": job.get("model_id"),
                "beam_size": job.get("beam_size"),
                "_segment": Segment(
                    audio_sec=row["audio_sec"],
                    transcribe_sec=row["transcribe_sec"],
                    raw_text=row["raw_text"] or "",
                    processed_text=row["processed_text"] or "",
                    fallback=bool(row["fallback"]),
                    rms=meta.get("rms"),
                    fallback_exhausted=bool(meta.get("fallback_exhausted")),
                    rejected=bool(meta.get("rejected")),
                    hotwords=job.get("hotwords", ""),
                    initial_prompt=job.get("initial_prompt", ""),
                ),
            }
        )
    conn.close()
    return records


def split_candidates(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (candidates, clean) — clean is everything no signal touched."""
    candidates, clean = [], []
    for rec in records:
        hits = scan_segment(rec["_segment"])
        entry = {k: v for k, v in rec.items() if not k.startswith("_")}
        entry["duration_bucket"] = bucket_of(rec["audio_sec"])
        if hits:
            entry["signals"] = [{"name": h.name, "detail": h.detail} for h in hits]
            entry["group"] = "candidate"
            candidates.append(entry)
        else:
            entry["signals"] = []
            entry["group"] = "control"
            clean.append(entry)
    return candidates, clean


def sample_control(
    candidates: list[dict], clean: list[dict], ratio: float, seed: int
) -> list[dict]:
    """Sample clean segments matching the candidates' duration distribution.

    Matched sampling matters because the signals are duration-correlated (short
    audio is over-represented among the latency signals). An unmatched control
    would differ from the candidates in duration as well as in flagged-ness, so
    a reviewer could not attribute a difference in error rate to the signals.
    """
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for entry in clean:
        by_bucket[entry["duration_bucket"]].append(entry)

    wanted = Counter()
    for entry in candidates:
        wanted[entry["duration_bucket"]] += 1

    picked: list[dict] = []
    for bucket, count in wanted.items():
        pool = by_bucket.get(bucket, [])
        take = min(len(pool), max(1, round(count * ratio)) if count else 0)
        picked.extend(rng.sample(pool, take))
    return picked


def summarise(candidates: list[dict], clean: list[dict]) -> dict:
    signal_counts = Counter()
    for entry in candidates:
        for sig in entry["signals"]:
            signal_counts[sig["name"]] += 1
    only_signal = Counter()
    for entry in candidates:
        if len(entry["signals"]) == 1:
            only_signal[entry["signals"][0]["name"]] += 1
    return {
        "segments_total": len(candidates) + len(clean),
        "candidates": len(candidates),
        "clean": len(clean),
        "by_signal": dict(signal_counts.most_common()),
        "sole_signal": dict(only_signal.most_common()),
        "by_bucket": dict(Counter(e["duration_bucket"] for e in candidates)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=None, help="輸出 JSON 路徑")
    parser.add_argument(
        "--control-ratio",
        type=float,
        default=0.25,
        help="對照組相對候選數的比例（預設 0.25）",
    )
    parser.add_argument("--seed", type=int, default=20260903, help="對照組抽樣種子（可重現）")
    parser.add_argument("--summary", action="store_true", help="只印統計，不寫檔")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"找不到 DB：{args.db}")
        return 1

    records = load_segments(args.db)
    candidates, clean = split_candidates(records)
    stats = summarise(candidates, clean)

    print(f"DB: {args.db}")
    print(
        f"總段數 {stats['segments_total']}，候選 {stats['candidates']}"
        f"（{stats['candidates'] / max(1, stats['segments_total']):.1%}），"
        f"未打中 {stats['clean']}"
    )
    print("\n各訊號命中數（括號為「只有這個訊號打中」的數量）：")
    for name, count in stats["by_signal"].items():
        sole = stats["sole_signal"].get(name, 0)
        print(f"  {count:>5}  ({sole:>4})  {name:<22} {SIGNAL_LABELS.get(name, '')}")
    unfired = [n for n in SIGNAL_LABELS if n not in stats["by_signal"]]
    if unfired:
        print(f"\n未命中任何段的訊號：{', '.join(unfired)}")

    if args.summary:
        return 0

    control = sample_control(candidates, clean, args.control_ratio, args.seed)
    out_path = args.out or (
        DEFAULT_OUT_DIR / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "db_path": str(args.db),
        "control_ratio": args.control_ratio,
        "seed": args.seed,
        "signal_labels": SIGNAL_LABELS,
        "stats": stats,
        "control_count": len(control),
        "entries": candidates + control,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n對照組 {len(control)} 筆（與候選同時長分布，seed={args.seed}）")
    print(f"已寫出 {len(payload['entries'])} 筆到 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
