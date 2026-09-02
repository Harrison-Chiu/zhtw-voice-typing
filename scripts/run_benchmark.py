"""Run the ASR benchmark over a manifest and write JSON + Markdown (roadmap E line).

The manifest is public (anonymous IDs, hashes, labels); the private map holds the
audio paths and gold transcripts and lives under `data/logs/` where git cannot
reach it. Output goes to the same gitignored directory, because per-sample
records are derived from personal recordings.

The Smoke tier is the default: it is meant to be runnable in a couple of minutes
so a change can be checked before committing to a full run.

Usage:
    python scripts/run_benchmark.py --manifest data/benchmark/manifest.json
    python scripts/run_benchmark.py --tier dev --repeats 3
    python scripts/run_benchmark.py --stage raw        # 不套後處理，比較引擎本身
    python scripts/run_benchmark.py --dry-run          # 只檢查 manifest 與私密對應

Requires a manifest to exist. There is no corpus in the repo — building one from
reviewed segments is the next step of the E line.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.eval.manifest import (  # noqa: E402
    ManifestError,
    check_private_map,
    load_manifest,
    load_private_map,
)
from asr_input.eval.runner import TIERS, format_markdown, run_benchmark  # noqa: E402
from asr_input.paths import logs_dir  # noqa: E402

DEFAULT_MANIFEST = logs_dir() / "benchmark" / "manifest.json"
DEFAULT_PRIVATE = logs_dir() / "benchmark" / "private_map.json"
DEFAULT_OUT_DIR = logs_dir() / "benchmark" / "runs"


def build_transcriber(config_path: Path, *, stage: str = "processed"):
    """Load the configured ASR engine and return a `transcribe(path) -> str`.

    `stage` decides what is scored. `processed` runs the same post-processing the
    user's clipboard gets (punctuation normalisation, OpenCC, optionally the
    Taiwan term list) and is the default, because that is the text the tool
    actually produces. `raw` scores the engine output alone, which is what you
    want when comparing engines without the pipeline in the way. Mixing them —
    raw output against processed gold — silently inflates CER and destroys
    punctuation recall, so the choice is recorded in the result.

    Imported lazily so `--dry-run` (and the tests) never pay for a model load.
    """
    import soundfile as sf

    from asr_input.asr import build_engine
    from asr_input.config import load_config

    config = load_config(config_path)
    asr_cfg = config["asr"]
    # VAD segmentation is left off: a benchmark sample is one utterance, and an
    # extra segmentation layer would measure something the streaming path does
    # differently anyway.
    engine = build_engine(asr_cfg, vad_cfg=None)
    engine.load()

    pipeline = None
    if stage == "processed":
        from asr_input.main import build_pipeline

        pipeline = build_pipeline(config, include_output=False)

    def transcribe(path: Path) -> str:
        audio, sample_rate = sf.read(str(path), dtype="float32")
        text = engine.transcribe(audio, sample_rate)
        return pipeline.run(text) if pipeline is not None else text

    engine_info = {
        "engine": asr_cfg.get("engine"),
        "model_id": asr_cfg.get("model_id"),
        "beam_size": asr_cfg.get("beam_size"),
        "initial_prompt": asr_cfg.get("initial_prompt"),
        "stage": stage,
    }
    return transcribe, engine_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--tier", choices=sorted(TIERS), default="smoke")
    parser.add_argument("--repeats", type=int, default=1, help="每段跑幾次（>1 才量得到決定性）")
    parser.add_argument(
        "--stage",
        choices=("processed", "raw"),
        default="processed",
        help="評分對象：processed（含後處理，預設）或 raw（引擎原始輸出）",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="只驗證 manifest 與私密對應")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"找不到 manifest：{args.manifest}", file=sys.stderr)
        print("目前 repo 內沒有語料，manifest 需由審核過的段落產生。", file=sys.stderr)
        return 2

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"manifest 無效：{exc}", file=sys.stderr)
        return 2

    private = load_private_map(args.private) if args.private.exists() else {}
    problems = check_private_map(manifest, private)
    if problems:
        print(f"manifest 與私密對應不一致（{len(problems)} 項）：", file=sys.stderr)
        for line in problems[:20]:
            print(f"  - {line}", file=sys.stderr)
        if not args.dry_run:
            return 2

    print(f"corpus {manifest.corpus}：{len(manifest.samples)} 段，tier={args.tier}")
    if args.dry_run:
        for split in sorted({s.split for s in manifest.samples}):
            print(f"  {split}: {len(manifest.by_split(split))} 段")
        return 0

    transcribe, engine_info = build_transcriber(args.config, stage=args.stage)
    result = run_benchmark(
        manifest,
        private,
        transcribe,
        tier=args.tier,
        repeats=args.repeats,
        engine=engine_info,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{stamp}_{result.run_id}.json"
    md_path = json_path.with_suffix(".md")
    result.save(json_path)
    md_path.write_text(format_markdown(result), encoding="utf-8")

    print(json.dumps(result.aggregate, ensure_ascii=False, indent=2))
    print(f"\n→ {json_path}\n→ {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
