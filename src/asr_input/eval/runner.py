"""Benchmark runner skeleton (`docs/asr-benchmark-proposal.md` §6).

The runner is engine-agnostic on purpose: it takes a `transcribe(audio_path)`
callable, so the same code can measure faster-whisper, a candidate engine, or a
replay of stored transcripts. Loading a model is the caller's job — this module
never imports an ASR engine, which keeps it importable (and testable) in a
process with no GPU.

Shape of a run:

    manifest + private map → for each sample, transcribe `repeats` times →
    metrics against the gold text → per-sample records → aggregate → Markdown

Aggregation is **micro**: total errors divided by total reference length, not the
mean of per-sample rates. A macro mean lets one 2-second sample weigh as much as
one 60-second sample, which is the wrong summary for "how good is this engine on
my speech".

Note on `speed`: the runner reports absolute transcription seconds alongside a
real-time factor. The factor is a throughput report only. It is deliberately not
plumbed into any quality or hallucination decision — that judgement uses absolute
transcription time (see CLAUDE.md, 短段幻覺分級門檻).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from asr_input.eval import environment
from asr_input.eval.manifest import Manifest, RunResult, Sample
from asr_input.eval.metrics import determinism, evaluate_pair

Transcriber = Callable[[Path], str]

TIERS = {
    "smoke": ("smoke",),
    "dev": ("smoke", "dev"),
    "full": ("smoke", "dev", "test"),
}


@dataclass
class SampleRun:
    sample: Sample
    outputs: list[str]
    seconds: list[float]
    error: str | None = None


def _resolve(private: dict[str, dict], sample: Sample) -> tuple[Path | None, str]:
    entry = private.get(sample.sample_id, {})
    path = entry.get("audio_path")
    return (Path(path) if path else None), entry.get("gold_verbatim", "")


def run_samples(
    samples: Iterable[Sample],
    private: dict[str, dict],
    transcribe: Transcriber,
    *,
    repeats: int = 1,
) -> list[SampleRun]:
    """Transcribe each sample `repeats` times, recording failures rather than raising.

    One unreadable file should not throw away the rest of a run that may have
    taken an hour of GPU time.
    """
    runs: list[SampleRun] = []
    for sample in samples:
        audio_path, _ = _resolve(private, sample)
        if audio_path is None:
            runs.append(SampleRun(sample, [], [], error="no audio_path in private map"))
            continue
        if not audio_path.exists():
            runs.append(SampleRun(sample, [], [], error=f"missing audio: {audio_path}"))
            continue

        outputs: list[str] = []
        seconds: list[float] = []
        error: str | None = None
        for _ in range(repeats):
            start = time.perf_counter()
            try:
                outputs.append(transcribe(audio_path))
            except Exception as exc:  # noqa: BLE001 - a bad sample must not kill the run
                error = f"{type(exc).__name__}: {exc}"
                break
            seconds.append(time.perf_counter() - start)
        runs.append(SampleRun(sample, outputs, seconds, error=error))
    return runs


def score_run(run: SampleRun, gold: str, *, to_traditional=None) -> dict:
    """Metrics for one sample, using the first output as the scored transcript."""
    record: dict = {
        "sample_id": run.sample.sample_id,
        "split": run.sample.split,
        "labels": list(run.sample.labels),
        "duration_sec": run.sample.duration_sec,
        "repeats": len(run.outputs),
        "error": run.error,
    }
    if run.seconds:
        record["transcribe_sec"] = round(median(run.seconds), 4)
    if not run.outputs:
        return record

    record["metrics"] = evaluate_pair(gold, run.outputs[0], to_traditional=to_traditional)
    # Only the emptiness is kept, never the text: a result JSON must stay shareable.
    record["empty_output"] = not run.outputs[0].strip()
    if len(run.outputs) > 1:
        det = determinism(run.outputs)
        record["determinism"] = {
            "runs": det.runs,
            "unique_outputs": det.unique_outputs,
            "exact_match_ratio": det.exact_match_ratio,
        }
    return record


def aggregate(records: list[dict]) -> dict:
    """Micro-average the per-sample records into one summary."""
    scored = [r for r in records if "metrics" in r]
    out: dict = {
        "samples_total": len(records),
        "samples_scored": len(scored),
        "samples_failed": sum(1 for r in records if r.get("error")),
    }
    if not scored:
        return out

    for key in ("strict_cer", "normalized_cer", "mer"):
        errors = sum(r["metrics"][key]["errors"] for r in scored)
        length = sum(r["metrics"][key]["ref_length"] for r in scored)
        out[key] = round(errors / length, 4) if length else 0.0

    tp = sum(r["metrics"]["punctuation"]["micro"]["tp"] for r in scored)
    fp = sum(r["metrics"]["punctuation"]["micro"]["fp"] for r in scored)
    fn = sum(r["metrics"]["punctuation"]["micro"]["fn"] for r in scored)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    out["punctuation"] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall
        else 0.0,
    }

    out["samples_with_insertion_runs"] = sum(1 for r in scored if r["metrics"]["insertion_runs"])
    out["empty_outputs"] = sum(1 for r in scored if r.get("empty_output"))

    audio_sec = sum(r["duration_sec"] for r in scored if "transcribe_sec" in r)
    transcribe_sec = sum(r["transcribe_sec"] for r in scored if "transcribe_sec" in r)
    out["speed"] = {
        "audio_sec": round(audio_sec, 2),
        "transcribe_sec": round(transcribe_sec, 2),
        # Throughput report only — never a quality or hallucination signal.
        "realtime_factor": round(transcribe_sec / audio_sec, 4) if audio_sec else None,
    }
    return out


def run_benchmark(
    manifest: Manifest,
    private: dict[str, dict],
    transcribe: Transcriber,
    *,
    tier: str = "smoke",
    repeats: int = 1,
    engine: dict | None = None,
    to_traditional=None,
) -> RunResult:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}, expected one of {sorted(TIERS)}")
    splits = TIERS[tier]
    samples = [s for s in manifest.samples if s.split in splits]

    runs = run_samples(samples, private, transcribe, repeats=repeats)
    records = [
        score_run(r, _resolve(private, r.sample)[1], to_traditional=to_traditional) for r in runs
    ]

    return RunResult(
        run_id=f"{tier}-{uuid.uuid4().hex[:8]}",
        manifest_corpus=manifest.corpus,
        engine=dict(engine or {}, tier=tier, repeats=repeats),
        environment=environment.collect(model=engine),
        samples=records,
        aggregate=aggregate(records),
    )


# --- reporting -------------------------------------------------------------


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def format_markdown(result: RunResult) -> str:
    """Human-readable summary. Contains no transcript text, so it is safe to share."""
    agg = result.aggregate
    lines = [
        f"# ASR benchmark — {result.manifest_corpus}",
        "",
        f"- run: `{result.run_id}` ({result.created_at})",
        f"- engine: {result.engine}",
        f"- env: {environment.format_fingerprint(result.environment)}",
        f"- samples: {agg.get('samples_scored', 0)} scored / "
        f"{agg.get('samples_total', 0)} total, {agg.get('samples_failed', 0)} failed",
        "",
        "## 指標",
        "",
        "| 指標 | 值 |",
        "|------|-----|",
        f"| Strict CER | {_pct(agg.get('strict_cer'))} |",
        f"| Normalized CER | {_pct(agg.get('normalized_cer'))} |",
        f"| MER | {_pct(agg.get('mer'))} |",
    ]
    punct = agg.get("punctuation")
    if punct:
        lines.append(
            f"| Punctuation F1 | {_pct(punct['f1'])} "
            f"(P {_pct(punct['precision'])} / R {_pct(punct['recall'])}) |"
        )
    lines.append(f"| 有插入串的段數 | {agg.get('samples_with_insertion_runs', 0)} |")
    lines.append(f"| 空輸出 | {agg.get('empty_outputs', 0)} |")

    speed = agg.get("speed")
    if speed:
        lines += [
            "",
            f"速度：{speed['transcribe_sec']} s 轉錄 / {speed['audio_sec']} s 音訊"
            f"（RTF {speed['realtime_factor']}，僅為吞吐量報告）。",
        ]

    failed = [r for r in result.samples if r.get("error")]
    if failed:
        lines += ["", "## 失敗的樣本", ""]
        lines += [f"- `{r['sample_id']}`: {r['error']}" for r in failed]

    lines += ["", "> 逐樣本數據見同名 JSON。音訊與逐字稿不在本報告內。", ""]
    return "\n".join(lines)
