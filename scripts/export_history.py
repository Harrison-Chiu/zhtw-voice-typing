"""Export or consistently back up the SQLite ASR history store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.output.history_store import DEFAULT_DB_PATH, HistoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "jsonl", "sqlite"), default="json")
    args = parser.parse_args()

    store = HistoryStore(args.db)
    if args.format == "sqlite":
        store.backup_to(args.output)
        return

    data = store.export_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        if args.format == "json":
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        else:
            for table, rows in data.items():
                for row in rows:
                    handle.write(
                        json.dumps({"table": table, "row": row}, ensure_ascii=False) + "\n"
                    )


if __name__ == "__main__":
    main()
