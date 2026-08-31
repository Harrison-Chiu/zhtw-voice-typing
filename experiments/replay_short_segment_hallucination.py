"""重現「極短子段被當成正常輸出」的實際案例，用真實模型跑存檔音訊。

案例：job 364bdc50（2026-08-12），seg3 觸發幻覺 fallback，重切出 0.29s／0.52s 兩個
子段；切分門檻用盡後兩段仍偏慢，卻被原樣採用，使用者剪貼簿因此出現
「作詞・作曲・編曲，男高等部分方向政府，強行固定中文名字。」。

用法：
    python experiments/replay_short_segment_hallucination.py [--job 364bdc50] [--segment 3]

注意：幻覺文字跨 session 不具決定性（同音訊不同 session 可能吐出不同垃圾），
但「極短音訊 → 轉錄時間數秒」的延遲訊號在本例中可穩定重現。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import wave
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.asr import build_engine  # noqa: E402


def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        sample_rate = handle.getframerate()
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="364bdc50", help="job id 前綴")
    parser.add_argument("--segment", type=int, default=3)
    parser.add_argument(
        "--split-sec",
        type=float,
        nargs="*",
        default=[0.29, 0.81],
        help="在段內的切點（秒），對應當時 fallback 重切出來的子段邊界",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    connection = sqlite3.connect("data/logs/history.sqlite3")
    connection.row_factory = sqlite3.Row
    job = connection.execute("SELECT * FROM jobs WHERE id LIKE ?", (args.job + "%",)).fetchone()
    segment = connection.execute(
        "SELECT * FROM segments WHERE job_id = ? AND segment_index = ?",
        (job["id"], args.segment),
    ).fetchone()
    if not job["audio_path"] or not Path(job["audio_path"]).exists():
        raise SystemExit(f"音訊已被容量護欄淘汰：{job['audio_path']}")

    audio, sample_rate = load_wav(job["audio_path"])
    clip = audio[segment["start_sample"] : segment["end_sample"]]
    print(f"job {job['id'][:8]} seg{args.segment}: {len(clip) / sample_rate:.2f}s")

    engine = build_engine(config["asr"])
    started = time.time()
    engine.load()
    print(f"模型載入 {time.time() - started:.1f}s")

    bounds = [0.0, *args.split_sec]
    pieces = [("整段", clip)]
    for start, end in zip(bounds, bounds[1:], strict=False):
        pieces.append(
            (
                f"子段 {start:.2f}–{end:.2f}s",
                clip[int(start * sample_rate) : int(end * sample_rate)],
            )
        )

    for label, piece in pieces:
        started = time.time()
        text = engine.transcribe(piece, sample_rate)
        elapsed = time.time() - started
        flag = " ⚠" if elapsed > config["streaming"]["hallucination_threshold_sec"] else ""
        print(f"{label}: {len(piece) / sample_rate:.2f}s → {elapsed:.2f}s{flag} | {text!r}")


if __name__ == "__main__":
    main()
