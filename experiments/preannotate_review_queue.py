"""Rank review-queue segments by how much independent ASR sources disagree.

Why this exists
---------------
`scripts/build_review_queue.py` produces a queue of candidate segments, but it
gives no order within the queue: a reviewer has to listen to all of them to find
the ones that are actually wrong. Human listening is the single biggest
bottleneck on both the A and E lines, so anything that puts the most doubtful
segments first converts machine time into reviewer time.

The ordering signal here is **disagreement between independent transcriptions of
the same audio**. Where several sources converge, production's output is
probably fine; where they diverge, something is worth a human ear.

This is a **ranking signal only**. It never writes a corpus label, never decides
gold, and none of the sources is treated as more correct than another - a
disagreement says "look here", not "production is wrong".

Sources
-------
    production  - the text already in the DB (what the user actually got);
                  not re-run, because that output is the thing under review
    fw_pinned   - faster-whisper at production config but temperature=(0.0,);
                  a different decoding path through the same engine
    qwen        - Qwen3-ASR-1.7B; a genuinely different backend (LLM-ASR),
                  so its errors are unlikely to correlate with Whisper's

Scoring uses `normalized_cer`, which folds whitespace, punctuation style and
script before comparing. That matters because Qwen emits simplified Chinese and
differing punctuation - without folding, every pair would look maximally
disagreeing for reasons that have nothing to do with the words.

Each engine runs in its own subprocess. faster-whisper (CTranslate2) and Qwen
(torch) both bring their own cuDNN, and loading them into one process aborts the
interpreter at `Could not load symbol cudnnGetLibConfig` - even with the first
engine unloaded first, because the DLL stays resident. Isolation is the fix;
`--worker` is that subprocess mode and is not meant to be called by hand.

Output goes to `data/logs/review/` (gitignored; it contains transcripts).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from asr_input.config import load_config  # noqa: E402
from asr_input.eval.metrics import normalized_cer  # noqa: E402

REVIEW_DIR = ROOT / "data" / "logs" / "review"


def latest_candidates() -> Path:
    files = sorted(REVIEW_DIR.glob("candidates_*.json"))
    if not files:
        raise SystemExit("找不到 candidates_*.json，請先跑 scripts/scan_error_candidates.py")
    return files[-1]


def to_traditional():
    """OpenCC s2t converter, or None if OpenCC is unavailable.

    Script folding is not essential to the ranking - it only stops Qwen's
    simplified output from inflating every pair - so a missing OpenCC degrades
    the signal rather than failing the run.
    """
    try:
        import opencc

        conv = opencc.OpenCC("s2t")
        return conv.convert
    except Exception as exc:  # noqa: BLE001 - optional dependency, degrade quietly
        print(f"警告：OpenCC 不可用（{exc}），簡繁差異會被算進 disagreement", flush=True)
        return None


def load_audio(entry: dict) -> np.ndarray:
    audio, _ = sf.read(str(ROOT / entry["audio_path"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio[entry["start_sample"] : entry["end_sample"]]


def build_fw_pinned(config):
    from asr_input.asr.whisper_fw import WhisperFWEngine

    asr = dict(config["asr"])
    asr.pop("engine", None)
    asr.pop("temperature", None)
    return WhisperFWEngine(temperature=(0.0,), **asr)


def build_qwen(model_id: str):
    from asr_input.asr.qwen import QwenASREngine

    return QwenASREngine(model_id=model_id, device="cuda", language="Chinese")


def transcribe_all(entries, engine, audio_cache, label) -> dict[tuple[str, int], str]:
    """Run one engine over every entry, isolating per-segment failures."""
    out: dict[tuple[str, int], str] = {}
    engine.load()
    try:
        for i, entry in enumerate(entries, start=1):
            key = (entry["job_id"], entry["segment_index"])
            start = time.perf_counter()
            try:
                out[key] = engine.transcribe(audio_cache[key], entry["sample_rate"] or 16000)
            except Exception as exc:  # noqa: BLE001 - one bad segment must not kill the run
                out[key] = ""
                print(f"  [{i}] 失敗：{type(exc).__name__}: {exc}", flush=True)
                continue
            if i % 10 == 0 or i == len(entries):
                print(
                    f"  [{i}/{len(entries)}] {label} {time.perf_counter() - start:.2f}s",
                    flush=True,
                )
    finally:
        engine.unload()
    return out


def run_worker(source: str, entries, args) -> dict[tuple[str, int], str]:
    """Transcribe every entry with one engine. Runs in its own process."""
    audio_cache = {(e["job_id"], e["segment_index"]): load_audio(e) for e in entries}
    if source == "fw_pinned":
        engine = build_fw_pinned(load_config(args.config))
    elif source == "qwen":
        engine = build_qwen(args.qwen_model)
    else:
        raise SystemExit(f"未知的 source: {source}")
    return transcribe_all(entries, engine, audio_cache, source)


def spawn_worker(source: str, argv_extra: list[str]) -> dict[tuple[str, int], str]:
    """Run one engine in a subprocess and read back its transcriptions."""
    out_path = REVIEW_DIR / f"_pre_{source}.json"
    out_path.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", source, *argv_extra]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 or not out_path.exists():
        raise SystemExit(f"{source} 子行程失敗（returncode={proc.returncode}）")
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    out_path.unlink(missing_ok=True)
    return {(r["job_id"], r["segment_index"]): r["text"] for r in raw}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 段（0 = 全部）")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--worker", default=None, help="內部用：在子行程跑單一引擎")
    args = parser.parse_args()

    src = args.candidates or latest_candidates()
    data = json.loads(src.read_text(encoding="utf-8"))
    entries = data["entries"]
    if args.limit:
        entries = entries[: args.limit]
    entries = [e for e in entries if (ROOT / e["audio_path"]).exists()]

    if args.worker:
        texts = run_worker(args.worker, entries, args)
        (REVIEW_DIR / f"_pre_{args.worker}.json").write_text(
            json.dumps(
                [
                    {"job_id": job_id, "segment_index": index, "text": text}
                    for (job_id, index), text in texts.items()
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return 0

    print(f"讀入 {src.name}，可用 {len(entries)} 段", flush=True)
    passthrough = ["--candidates", str(src), "--qwen-model", args.qwen_model]
    if args.limit:
        passthrough += ["--limit", str(args.limit)]

    print("\n=== fw_pinned ===", flush=True)
    sources = {"fw_pinned": spawn_worker("fw_pinned", passthrough)}
    if not args.skip_qwen:
        print("\n=== qwen ===", flush=True)
        sources["qwen"] = spawn_worker("qwen", passthrough)

    convert = to_traditional()
    names = ["production", *sources]
    records = []
    for entry in entries:
        key = (entry["job_id"], entry["segment_index"])
        texts = {"production": entry["raw_text"] or ""}
        texts.update({name: sources[name][key] for name in sources})

        pairs = {}
        for a, b in combinations(names, 2):
            rate = normalized_cer(texts[a], texts[b], to_traditional=convert)
            pairs[f"{a}|{b}"] = round(rate.rate, 4)
        values = list(pairs.values())

        records.append(
            {
                "job_id": entry["job_id"],
                "segment_index": entry["segment_index"],
                "audio_sec": entry["audio_sec"],
                "group": entry["group"],
                "signals": [s["name"] for s in entry.get("signals", [])],
                "mean_disagreement": round(sum(values) / len(values), 4),
                "max_disagreement": max(values),
                "pairwise_normalized_cer": pairs,
                "empty_sources": [n for n in names if not texts[n].strip()],
                "texts": texts,
            }
        )

    records.sort(key=lambda r: r["mean_disagreement"], reverse=True)
    for rank, rec in enumerate(records, start=1):
        rec["rank"] = rank

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = REVIEW_DIR / f"preannotation_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "candidates_file": src.name,
                "sources": names,
                "note": "排序訊號，不是 gold；任何來源都不視為正確答案",
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    top = [r for r in records if r["group"] == "candidate"][:10]
    print("\n分歧最高的 10 段（candidate）：", flush=True)
    for rec in top:
        print(
            f"  #{rec['rank']:3d} {rec['audio_sec']:6.1f}s "
            f"mean={rec['mean_disagreement']:.3f} signals={rec['signals']}"
        )
    controls = [r["mean_disagreement"] for r in records if r["group"] == "control"]
    cands = [r["mean_disagreement"] for r in records if r["group"] == "candidate"]
    if controls and cands:
        print(
            f"\ncandidate 平均分歧 {sum(cands) / len(cands):.4f}"
            f"（n={len(cands)}） vs control {sum(controls) / len(controls):.4f}"
            f"（n={len(controls)}）"
        )
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
