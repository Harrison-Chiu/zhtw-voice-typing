"""Search session logs for quality issues — tag each segment, filter by tags.

Defaults to scanning new session logs only (data/logs/sessions/).
Use --legacy to also include old transcript logs (data/logs/transcripts.jsonl).

Usage:
    python scripts/search_logs.py                # 掃新版 log，印摘要 + 有問題的條目
    python scripts/search_logs.py --summary      # 只印統計摘要
    python scripts/search_logs.py --tag punct    # 只看特定標籤
    python scripts/search_logs.py --tag model    # 所有模型層問題
    python scripts/search_logs.py --tag postproc # 所有後處理問題
    python scripts/search_logs.py --text "某詞"  # 搜尋包含特定文字的條目
    python scripts/search_logs.py --slow 2.0     # 轉錄超過 N 秒的段
    python scripts/search_logs.py --legacy       # 也掃舊版 transcripts.jsonl

Tags (each is a deterministic rule, no guessing):
  Model-level (checked on raw model output):
    punct      — 半形標點出現在 CJK 字旁（regex: CJK + [,?!;:] + CJK）
    simplified — 含簡體字（逐字 OpenCC s2t 比對有差異）
    spacing    — CJK 字間有空格（regex: [CJK] + whitespace + [CJK]）
    repeat     — 同字連續 ≥5 次（regex: (.)\1{4,}）
    no_comma   — ≥30 字且無任何 ，,、；
    truncated  — 全文 ≤5 字

  Post-processing (checked on processed output, only when raw available):
    overconv       — 含 OpenCC 過度轉換詞（臺灣、臺北…）
    punct_unfixed  — raw 有半形標點且 processed 仍有
    simplified_unfixed — processed 仍含簡體字

  Shortcuts:
    model    — all model-level tags
    postproc — all post-processing tags
    all      — everything (default)
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

LOG_FILE = Path("data/logs/transcripts.jsonl")
SESSION_DIR = Path("data/logs/sessions")

OVERCONV_WORDS = {"臺灣", "臺北", "臺中", "臺南", "臺東", "臺大", "臺詞", "臺階"}

MODEL_TAGS = {"punct", "simplified", "spacing", "repeat", "no_comma", "truncated"}
POSTPROC_TAGS = {"overconv", "punct_unfixed", "simplified_unfixed"}
ALL_TAGS = MODEL_TAGS | POSTPROC_TAGS

TAG_LABELS = {
    "punct": "模型：半形標點",
    "simplified": "模型：簡體字",
    "spacing": "模型：字間空格（逗號缺失）",
    "repeat": "模型：重複字退化",
    "no_comma": "模型：長段無逗號",
    "truncated": "按錯/截斷（≤5字）",
    "overconv": "後處理：OpenCC 過度轉換",
    "punct_unfixed": "後處理：半形標點未修正",
    "simplified_unfixed": "後處理：簡體字未修正",
}

# Reusable white-list for characters that look simplified but are standard in Taiwan
SIMPLIFIED_WHITELIST = {"台", "里", "面", "干", "后", "才"}


def is_cjk(ch: str) -> bool:
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        return False
    return "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name


def detect_simplified(text: str) -> list[str]:
    try:
        import opencc

        converter = opencc.OpenCC("s2t")
    except ImportError:
        return []
    found = []
    for ch in text:
        if is_cjk(ch):
            converted = converter.convert(ch)
            if converted != ch and ch not in SIMPLIFIED_WHITELIST:
                found.append(f"{ch}→{converted}")
    return found


def detect_halfwidth_punct(text: str) -> list[str]:
    issues = []
    for m in re.finditer(r"(?<=[^\x00-\x7f])\s*([,?!;:])\s*(?=[^\x00-\x7f])", text):
        start = max(0, m.start() - 5)
        end = min(len(text), m.end() + 5)
        issues.append(f"'{m.group(1)}' …{text[start:end]}…")
    return issues


def detect_cjk_spacing(text: str) -> list[str]:
    return [f"…{m.group()}…" for m in re.finditer(r"[一-鿿]\s+[一-鿿]", text)]


def detect_overconv(text: str) -> list[str]:
    return [f"{w}→應為{w.replace('臺', '台')}" for w in OVERCONV_WORDS if w in text]


def detect_repeat(text: str) -> list[str]:
    return [f"'{m.group()[:10]}…' ({len(m.group())}次)" for m in re.finditer(r"(.)\1{4,}", text)]


def has_no_comma(text: str) -> bool:
    return len(text) >= 30 and not re.search(r"[，,、；]", text)


def is_truncated(text: str) -> bool:
    return 0 < len(text.strip()) <= 5


# ---------------------------------------------------------------------------


def tag_entry(entry: dict) -> dict[str, list[str] | bool]:
    raw = entry.get("raw", "")
    processed = entry.get("processed", "")
    has_raw = bool(raw) and raw != "(streaming)"

    model_text = raw if has_raw else processed
    if not model_text:
        return {}

    tags: dict[str, list[str] | bool] = {}

    # Model-level tags (on raw)
    if p := detect_halfwidth_punct(model_text):
        tags["punct"] = p
    if s := detect_simplified(model_text):
        tags["simplified"] = s
    if sp := detect_cjk_spacing(model_text):
        tags["spacing"] = sp
    if r := detect_repeat(model_text):
        tags["repeat"] = r
    if has_no_comma(model_text):
        tags["no_comma"] = True
    if is_truncated(model_text):
        tags["truncated"] = True

    # Post-processing tags (on processed, only when raw is available)
    if has_raw and processed:
        if oc := detect_overconv(processed):
            tags["overconv"] = oc
        if "punct" in tags and (pu := detect_halfwidth_punct(processed)):
            tags["punct_unfixed"] = pu
        if su := detect_simplified(processed):
            tags["simplified_unfixed"] = su

    return tags


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_session_entries() -> list[dict]:
    entries = []
    if not SESSION_DIR.exists():
        return entries
    for session_dir in sorted(SESSION_DIR.iterdir()):
        jsonl = session_dir / "session.jsonl"
        if jsonl.exists():
            for e in _load_jsonl(jsonl):
                e["_source"] = session_dir.name
                e["_session_dir"] = str(session_dir)
                entries.append(e)
    return entries


def load_legacy_entries() -> list[dict]:
    entries = []
    for e in _load_jsonl(LOG_FILE):
        e["_source"] = "legacy"
        entries.append(e)
    return entries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_entry(idx: int, entry: dict, tags: dict) -> None:
    raw = entry.get("raw", "")
    processed = entry.get("processed", "")
    has_raw = bool(raw) and raw != "(streaming)"
    display = raw if has_raw else processed
    if not display:
        return

    ts = entry.get("ts", "")[:19]
    src = entry.get("_source", "")
    audio_file = entry.get("audio_file", "")

    header_parts = [f"[{idx:>4}]", ts]
    if src and src != "legacy":
        header_parts.append(f"[{src}]")
    elif src == "legacy":
        header_parts.append("[舊版]")
    header_parts.append("(raw)" if has_raw else "(processed-only)")
    if audio_file:
        header_parts.append(f"🔊{audio_file}")

    tag_str = " ".join(f"[{t}]" for t in tags)
    print(f"  {' '.join(header_parts)} {tag_str}")
    print(f"         {display[:100]}")
    if has_raw and processed and raw != processed:
        print(f"     pp: {processed[:100]}")
    for key, detail in tags.items():
        if isinstance(detail, list):
            print(f"         → {key}: {', '.join(detail[:5])}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="掃描 session log 品質問題",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default="all",
        help="篩選標籤：punct/simplified/spacing/repeat/no_comma/truncated/"
        "overconv/punct_unfixed/simplified_unfixed/model/postproc/all",
    )
    parser.add_argument("--text", help="搜尋包含特定文字的條目")
    parser.add_argument("--slow", type=float, help="轉錄超過 N 秒的段")
    parser.add_argument("--summary", action="store_true", help="只印統計摘要")
    parser.add_argument("--legacy", action="store_true", help="也掃舊版 transcripts.jsonl")
    parser.add_argument("--limit", type=int, default=50, help="最多顯示幾筆（預設 50）")
    args = parser.parse_args()

    # Load entries
    entries = load_session_entries()
    if args.legacy:
        entries = load_legacy_entries() + entries

    if not entries:
        if args.legacy:
            print("找不到任何 log 檔案")
        else:
            print("新版 session log 尚無資料（data/logs/sessions/ 為空）")
            print("提示：開始使用 tray 錄音後會自動累積")
            print("      加 --legacy 可掃舊版 transcripts.jsonl")
        return

    source_desc = "新版 session log"
    if args.legacy:
        source_desc += " + 舊版 transcripts.jsonl"
    print(f"掃描 {source_desc}：共 {len(entries)} 筆\n")

    # --- Text search mode ---
    if args.text:
        matches = [
            (i, e)
            for i, e in enumerate(entries)
            if args.text in (e.get("raw", "") + e.get("processed", ""))
        ]
        print(f"包含「{args.text}」的條目：{len(matches)} 筆")
        for i, e in matches[: args.limit]:
            raw = e.get("raw", "")
            processed = e.get("processed", "")
            text = raw if (raw and raw != "(streaming)") else processed
            ts = e.get("ts", "")[:19]
            audio = e.get("audio_file", "")
            audio_str = f" 🔊{audio}" if audio else ""
            print(f"  [{i:>4}] {ts}{audio_str} | {text[:100]}")
        return

    # --- Slow transcription mode ---
    if args.slow is not None:
        matches = [(i, e) for i, e in enumerate(entries) if e.get("transcribe_sec", 0) > args.slow]
        print(f"轉錄超過 {args.slow}s 的段：{len(matches)} 筆")
        for i, e in matches[: args.limit]:
            text = e.get("raw", "") or e.get("processed", "")
            dt = e.get("transcribe_sec", "?")
            audio_sec = e.get("audio_sec", "?")
            src = e.get("_source", "")
            audio_file = e.get("audio_file", "")
            audio_str = f" 🔊{audio_file}" if audio_file else ""
            print(f"  [{i:>4}] {audio_sec}s→{dt}s [{src}]{audio_str} | {text[:80]}")
        return

    # --- Tag scan mode ---
    tag_filter = args.tag.lower()
    if tag_filter == "model":
        filter_set = MODEL_TAGS
    elif tag_filter == "postproc":
        filter_set = POSTPROC_TAGS
    elif tag_filter == "all":
        filter_set = ALL_TAGS
    elif tag_filter in ALL_TAGS:
        filter_set = {tag_filter}
    else:
        print(f"未知標籤：{tag_filter}")
        print(f"可用：{', '.join(sorted(ALL_TAGS))} / model / postproc / all")
        return

    counts: dict[str, int] = {t: 0 for t in ALL_TAGS}
    flagged: list[tuple[int, dict, dict]] = []

    for i, entry in enumerate(entries):
        tags = tag_entry(entry)
        if not tags:
            continue
        for t in tags:
            if t in counts:
                counts[t] += 1
        if filter_set & set(tags.keys()):
            flagged.append((i, entry, tags))

    # Summary
    print("=== 品質問題統計 ===")
    print("  [模型層]")
    for t in ["punct", "simplified", "spacing", "repeat", "no_comma", "truncated"]:
        c = counts[t]
        bar = "█" * min(c, 40)
        print(f"    {TAG_LABELS[t]:<24s} {c:>4d}  {bar}")
    print("  [後處理]")
    for t in ["overconv", "punct_unfixed", "simplified_unfixed"]:
        c = counts[t]
        bar = "█" * min(c, 40)
        print(f"    {TAG_LABELS[t]:<24s} {c:>4d}  {bar}")

    if args.summary:
        return

    if not flagged:
        print("\n（無符合條件的條目）")
        return

    print(f"\n=== 符合 [{tag_filter}] 的條目：{len(flagged)} 筆 ===")
    for i, entry, tags in flagged[: args.limit]:
        print_entry(i, entry, tags)


if __name__ == "__main__":
    main()
