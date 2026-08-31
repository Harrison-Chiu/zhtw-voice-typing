"""從真實 history log 抽出短音訊（< 3 秒）的辨識嘗試，供短段幻覺規則回歸。

背景：`streaming.min_hallucination_audio_sec = 3.0` 讓 3 秒以下的段完全跳過幻覺
偵測，實際 log 中因此出現 0.26 秒音訊輸出 60 字亂碼並直接進剪貼簿的案例。
`docs/roadmap.md` 要求「先從現有短段 log 建立測試，再決定附加訊號或分級門檻」，
本腳本就是那份測試資料的產生器。

用法：
    python experiments/extract_short_segment_cases.py            # 印出候選，不寫檔
    python experiments/extract_short_segment_cases.py --write    # 覆寫 fixture 的 cases

輸出檔 `tests/data/short_segment_cases.json` 的 `label` 欄是人工判讀結果，
`--write` 會沿用既有檔案中相同 source 的 label／note，不會把人工標註洗掉。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/logs/history.sqlite3")
FIXTURE_PATH = Path("tests/data/short_segment_cases.json")
MAX_AUDIO_SEC = 3.0


def collect(db_path: Path, max_audio_sec: float) -> list[dict]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT job_id, segment_index, attempt_index, kind,
               audio_sec, transcribe_sec, raw_text, adopted
        FROM asr_attempts
        WHERE audio_sec < ?
        ORDER BY transcribe_sec DESC, audio_sec
        """,
        (max_audio_sec,),
    ).fetchall()
    connection.close()
    return [
        {
            "source": f"{row['job_id'][:8]}#seg{row['segment_index']}#att{row['attempt_index']}",
            "kind": row["kind"],
            "audio_sec": round(row["audio_sec"], 2),
            "transcribe_sec": round(row["transcribe_sec"], 2),
            "adopted": bool(row["adopted"]),
            "raw": row["raw_text"],
            "label": "normal",
        }
        for row in rows
    ]


def merge_labels(cases: list[dict], fixture_path: Path) -> list[dict]:
    if not fixture_path.exists():
        return cases
    previous = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    known = {case["source"]: case for case in previous}
    for case in cases:
        old = known.get(case["source"])
        if old is None:
            continue
        case["label"] = old["label"]
        if "note" in old:
            case["note"] = old["note"]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--max-audio-sec", type=float, default=MAX_AUDIO_SEC)
    parser.add_argument("--write", action="store_true", help="寫回 fixture 的 cases 欄")
    args = parser.parse_args()

    cases = merge_labels(collect(args.db, args.max_audio_sec), FIXTURE_PATH)
    for case in cases[:20]:
        print(
            f"{case['audio_sec']:5.2f}s → {case['transcribe_sec']:5.2f}s "
            f"[{case['label']:>13}] {case['raw'][:60]}"
        )
    print(f"... 共 {len(cases)} 筆")

    if args.write:
        document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        document["cases"] = cases
        FIXTURE_PATH.write_text(
            json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"已寫入 {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
