"""Strict ASR regression test for lossless/lossy audio log formats.

The purpose of saved log audio is to reproduce Whisper recognition mistakes.
Therefore a codec passes only when its raw Faster Whisper output is exactly
equal to the WAV baseline for every selected clip, including punctuation,
spaces, letter case, and Unicode characters.

Usage:
    .venv\\Scripts\\python.exe experiments/experiment_audio_codec.py --mode determinism
    .venv\\Scripts\\python.exe experiments/experiment_audio_codec.py --mode codec
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from asr_input.asr import build_engine

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data" / "logs" / "sessions"
RESULT_ROOT = ROOT / "experiments" / "results"

ERROR_TERMS = ("外餐", "表點", "標頂", "辦形", "轉入", "以有")
CODECS = {
    "flac": ("flac", ["-c:a", "flac"]),
    "opus_16k": (
        "ogg",
        ["-c:a", "libopus", "-b:a", "16k", "-vbr", "on", "-application", "voip"],
    ),
    "opus_24k": (
        "ogg",
        ["-c:a", "libopus", "-b:a", "24k", "-vbr", "on", "-application", "voip"],
    ),
    "opus_32k": (
        "ogg",
        ["-c:a", "libopus", "-b:a", "32k", "-vbr", "on", "-application", "voip"],
    ),
    "opus_48k": (
        "ogg",
        ["-c:a", "libopus", "-b:a", "48k", "-vbr", "on", "-application", "voip"],
    ),
    "opus_64k": (
        "ogg",
        ["-c:a", "libopus", "-b:a", "64k", "-vbr", "on", "-application", "voip"],
    ),
    "mp3_32k": ("mp3", ["-c:a", "libmp3lame", "-b:a", "32k"]),
    "mp3_64k": ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    audio_path: str
    audio_sec: float
    rms: float
    transcribe_sec: float
    raw_log_text: str
    fallback: bool
    tags: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("determinism", "codec", "all"), default="all")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--repeat-samples", type=int, default=16)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--temperature", type=float)
    return parser.parse_args()


def load_samples(limit: int) -> list[Sample]:
    candidates: list[Sample] = []
    for jsonl_path in sorted(LOG_ROOT.glob("*/session.jsonl"), reverse=True):
        session = jsonl_path.parent.name
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                audio_name = entry.get("audio_file")
                if not audio_name:
                    continue
                audio_path = jsonl_path.parent / audio_name
                if not audio_path.exists():
                    continue

                duration = float(entry.get("audio_sec", 0))
                rms = float(entry.get("rms", 0))
                transcribe_sec = float(entry.get("transcribe_sec", 0))
                raw = entry.get("raw", "")
                tags: list[str] = []
                if any(term in raw for term in ERROR_TERMS):
                    tags.append("known_asr_error")
                if entry.get("fallback"):
                    tags.append("fallback")
                if duration <= 5:
                    tags.append("short")
                if duration >= 25:
                    tags.append("long")
                if rms <= 0.0075:
                    tags.append("low_rms")
                if duration and transcribe_sec / duration >= 0.08:
                    tags.append("slow_decode")
                if any(ord(char) < 128 and char.isalpha() for char in raw):
                    tags.append("mixed_language")

                candidates.append(
                    Sample(
                        sample_id=f"{session}/{entry.get('seg', audio_path.stem)}",
                        audio_path=str(audio_path.relative_to(ROOT)),
                        audio_sec=duration,
                        rms=rms,
                        transcribe_sec=transcribe_sec,
                        raw_log_text=raw,
                        fallback=bool(entry.get("fallback")),
                        tags=tuple(tags),
                    )
                )

    # First preserve rare/difficult cases, then fill the remainder evenly across
    # recent history instead of taking one contiguous time period.
    chosen: list[Sample] = []
    seen: set[str] = set()
    quotas = {
        "known_asr_error": 24,
        "fallback": 20,
        "short": 18,
        "long": 18,
        "low_rms": 18,
        "slow_decode": 12,
        "mixed_language": 12,
    }
    for tag, quota in quotas.items():
        for sample in candidates:
            if tag in sample.tags and sample.sample_id not in seen:
                chosen.append(sample)
                seen.add(sample.sample_id)
                quota -= 1
                if quota == 0 or len(chosen) == limit:
                    break
        if len(chosen) == limit:
            return chosen

    remaining = [sample for sample in candidates if sample.sample_id not in seen]
    slots = limit - len(chosen)
    if slots > 0 and remaining:
        indices = np.linspace(0, len(remaining) - 1, min(slots, len(remaining)), dtype=int)
        chosen.extend(remaining[index] for index in indices)
    return chosen[:limit]


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"{path}: expected 16000 Hz, got {sample_rate}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32)


def first_difference(expected: str, actual: str) -> dict:
    matcher = difflib.SequenceMatcher(a=expected, b=actual)
    operations = [
        {
            "operation": operation,
            "expected": expected[a0:a1],
            "actual": actual[b0:b1],
            "expected_span": [a0, a1],
            "actual_span": [b0, b1],
        }
        for operation, a0, a1, b0, b1 in matcher.get_opcodes()
        if operation != "equal"
    ]
    return {"operations": operations}


def encode_decode(source: Path, codec: str, temp_dir: Path) -> tuple[np.ndarray, int]:
    extension, codec_args = CODECS[codec]
    encoded = temp_dir / f"{source.stem}_{codec}.{extension}"
    decoded = temp_dir / f"{source.stem}_{codec}.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            *codec_args,
            str(encoded),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(encoded),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(decoded),
        ],
        check=True,
    )
    return load_audio(decoded), encoded.stat().st_size


def transcribe(engine, audio: np.ndarray) -> tuple[str, float]:
    started = time.perf_counter()
    text = engine.transcribe(audio, 16000)
    return text, time.perf_counter() - started


def determinism_test(engine, samples: list[Sample], runs: int) -> dict:
    rows = []
    for index, sample in enumerate(samples, start=1):
        audio = load_audio(ROOT / sample.audio_path)
        outputs = []
        timings = []
        for _ in range(runs):
            text, elapsed = transcribe(engine, audio)
            outputs.append(text)
            timings.append(elapsed)
        passed = len(set(outputs)) == 1
        rows.append(
            {
                "sample_id": sample.sample_id,
                "passed": passed,
                "outputs": outputs,
                "timings_sec": timings,
            }
        )
        print(f"[determinism {index:>2}/{len(samples)}] {'PASS' if passed else 'FAIL'}")
    return {
        "runs_per_sample": runs,
        "sample_count": len(samples),
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }


def codec_test(engine, samples: list[Sample], runs: int) -> dict:
    baseline: dict[str, dict] = {}
    for index, sample in enumerate(samples, start=1):
        audio = load_audio(ROOT / sample.audio_path)
        outputs = [transcribe(engine, audio)[0] for _ in range(runs)]
        baseline[sample.sample_id] = {
            "stable": len(set(outputs)) == 1,
            "outputs": outputs,
        }
        print(
            f"[baseline {index:>3}/{len(samples)}] "
            f"{'STABLE' if baseline[sample.sample_id]['stable'] else 'UNSTABLE'}"
        )

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="asr_codec_") as directory:
        temp_dir = Path(directory)
        for codec in CODECS:
            rows = []
            source_bytes = 0
            encoded_bytes = 0
            for index, sample in enumerate(samples, start=1):
                source = ROOT / sample.audio_path
                audio, size = encode_decode(source, codec, temp_dir)
                actual_runs = [transcribe(engine, audio) for _ in range(runs)]
                actual_outputs = [text for text, _ in actual_runs]
                baseline_result = baseline[sample.sample_id]
                expected = baseline_result["outputs"][0]
                conclusive = baseline_result["stable"]
                passed = conclusive and all(text == expected for text in actual_outputs)
                source_bytes += source.stat().st_size
                encoded_bytes += size
                row = {
                    "sample_id": sample.sample_id,
                    "passed": passed,
                    "conclusive": conclusive,
                    "transcribe_sec": [elapsed for _, elapsed in actual_runs],
                    "encoded_bytes": size,
                }
                if not conclusive:
                    row["reason"] = "WAV baseline is not deterministic"
                    row["baseline_outputs"] = baseline_result["outputs"]
                    row["actual_outputs"] = actual_outputs
                elif not passed:
                    row.update(
                        {
                            "expected": expected,
                            "actual_outputs": actual_outputs,
                            "differences": [
                                first_difference(expected, actual)
                                for actual in actual_outputs
                                if actual != expected
                            ],
                        }
                    )
                rows.append(row)
                print(
                    f"[{codec} {index:>3}/{len(samples)}] {'PASS' if passed else 'FAIL'}",
                    flush=True,
                )
            results[codec] = {
                "passed": all(row["passed"] for row in rows),
                "exact_matches": sum(row["passed"] for row in rows),
                "inconclusive": sum(not row["conclusive"] for row in rows),
                "sample_count": len(rows),
                "size_ratio": encoded_bytes / source_bytes,
                "source_bytes": source_bytes,
                "encoded_bytes": encoded_bytes,
                "rows": rows,
            }
    return {"baseline": baseline, "codecs": results}


def write_results(payload: dict) -> tuple[Path, Path]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    json_path = RESULT_ROOT / f"audio_codec_{stamp}.json"
    md_path = RESULT_ROOT / f"audio_codec_{stamp}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 音訊編碼對 Faster Whisper 的嚴格回歸測試",
        "",
        "通過條件：每一段的原始辨識文字逐字相同，包含標點、空格、大小寫與 Unicode。",
        "",
        f"- 樣本數：{len(payload['samples'])}",
    ]
    determinism = payload.get("determinism")
    if determinism:
        lines.append(
            f"- 同一 WAV 重跑：{'通過' if determinism['passed'] else '失敗'}"
            f"（{determinism['sample_count']} 段 × {determinism['runs_per_sample']} 次）"
        )
    codec = payload.get("codec")
    if codec:
        lines.extend(
            [
                "",
                "| 編碼 | 完全相同 | WAV 不穩定 | 容量比例 | 結論 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for name, result in codec["codecs"].items():
            lines.append(
                f"| {name} | {result['exact_matches']}/{result['sample_count']} | "
                f"{result['inconclusive']} | {result['size_ratio']:.1%} | "
                f"{'通過' if result['passed'] else '不通過'} |"
            )
        lines.extend(
            [
                "",
                "任何一段不同就判定該編碼不適合作為可重現 ASR 錯誤的開發原始紀錄。",
                "完整逐段差異與參數請見同名 JSON。",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)
    if args.sample_id:
        requested = set(args.sample_id)
        samples = [sample for sample in samples if sample.sample_id in requested]
        missing = requested - {sample.sample_id for sample in samples}
        if missing:
            raise ValueError(f"sample IDs not found in selected set: {sorted(missing)}")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if args.beam_size is not None:
        config["asr"]["beam_size"] = args.beam_size
    if args.temperature is not None:
        config["asr"]["temperature"] = args.temperature
    engine = build_engine(config["asr"], vad_cfg=None)

    print(f"選取 {len(samples)} 段；載入 Faster Whisper...", flush=True)
    started = time.perf_counter()
    engine.load()
    print(f"模型載入完成：{time.perf_counter() - started:.1f}s", flush=True)

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "criteria": "exact raw Faster Whisper text equality",
        "config": config["asr"],
        "samples": [asdict(sample) for sample in samples],
    }
    if args.mode in ("determinism", "all"):
        payload["determinism"] = determinism_test(
            engine,
            samples[: args.repeat_samples],
            args.repeat_runs,
        )
    if args.mode in ("codec", "all"):
        payload["codec"] = codec_test(engine, samples, args.repeat_runs)

    json_path, md_path = write_results(payload)
    print(f"JSON: {json_path}")
    print(f"摘要: {md_path}")


if __name__ == "__main__":
    main()
