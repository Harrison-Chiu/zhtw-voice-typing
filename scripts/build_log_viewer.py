"""Generate an HTML viewer for session logs with tag filtering and audio playback.

Usage:
    python scripts/build_log_viewer.py              # 產生 data/logs/viewer.html
    python scripts/build_log_viewer.py --legacy      # 也包含舊版 transcripts.jsonl

Output: data/logs/viewer.html (open in browser, audio playback via relative paths)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_logs import (
    TAG_LABELS,
    load_legacy_entries,
    load_session_entries,
    tag_entry,
)

OUTPUT = Path("data/logs/viewer.html")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>ASR Session Log Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", "Microsoft JhengFei", sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { color: #e94560; margin-bottom: 10px; font-size: 1.4em; }
.stats { color: #888; margin-bottom: 16px; font-size: 0.9em; }

.controls { background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.control-row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.control-row:last-child { margin-bottom: 0; }
.control-label { color: #e94560; font-weight: 600; font-size: 0.85em; min-width: 60px; }

.tag-btn { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em;
  cursor: pointer; border: 1px solid #444; background: #0f3460; color: #ccc; margin: 2px; transition: all .15s; user-select: none; }
.tag-btn.active { background: #e94560; color: #fff; border-color: #e94560; }
.tag-btn:hover { border-color: #e94560; }
.tag-btn.model { border-left: 3px solid #53d8fb; }
.tag-btn.postproc { border-left: 3px solid #f0a500; }

input[type=text] { background: #0f3460; border: 1px solid #444; color: #e0e0e0; padding: 6px 12px;
  border-radius: 6px; font-size: 0.9em; width: 300px; }
input[type=text]::placeholder { color: #666; }

.summary-bar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.summary-item { background: #16213e; border-radius: 6px; padding: 6px 12px; font-size: 0.8em; }
.summary-item .count { color: #e94560; font-weight: 700; font-size: 1.1em; }

.entries { display: flex; flex-direction: column; gap: 8px; }
.entry { background: #16213e; border-radius: 8px; padding: 12px 16px; border-left: 3px solid #333; }
.entry.has-issue { border-left-color: #e94560; }
.entry-header { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 0.8em; color: #888; margin-bottom: 6px; }
.entry-header .session { color: #53d8fb; }
.entry-header .tag { display: inline-block; background: #e94560; color: #fff; padding: 1px 7px; border-radius: 8px; font-size: 0.75em; }
.entry-header .tag.postproc { background: #f0a500; color: #000; }

.text-raw { color: #e0e0e0; font-size: 0.95em; line-height: 1.6; margin-bottom: 4px; }
.text-processed { color: #888; font-size: 0.85em; line-height: 1.5; }
.text-processed::before { content: "pp: "; color: #666; }
.diff-char { background: #e9456033; color: #ff6b6b; padding: 0 1px; border-radius: 2px; }

audio { height: 28px; margin-top: 4px; }

.empty-state { text-align: center; color: #666; padding: 40px; font-size: 1.1em; }
.showing-count { color: #888; font-size: 0.85em; margin-bottom: 8px; }
</style>
</head>
<body>

<h1>ASR Session Log Viewer</h1>
<div class="stats" id="stats"></div>

<div class="controls">
  <div class="control-row">
    <span class="control-label">模型層</span>
    <div id="model-tags"></div>
  </div>
  <div class="control-row">
    <span class="control-label">後處理</span>
    <div id="postproc-tags"></div>
  </div>
  <div class="control-row">
    <span class="control-label">搜尋</span>
    <input type="text" id="search" placeholder="搜尋文字內容...">
    <label style="font-size:.85em;color:#888;cursor:pointer">
      <input type="checkbox" id="show-clean"> 顯示無標籤的段
    </label>
  </div>
</div>

<div class="summary-bar" id="summary"></div>
<div class="showing-count" id="showing-count"></div>
<div class="entries" id="entries"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;
const TAG_LABELS = __TAG_LABELS_PLACEHOLDER__;
const MODEL_TAGS = ["punct","simplified","spacing","repeat","no_comma","truncated"];
const POSTPROC_TAGS = ["overconv","punct_unfixed","simplified_unfixed"];

let activeTags = new Set();
let searchText = "";
let showClean = false;

function init() {
  document.getElementById("stats").textContent =
    `${DATA.length} 筆記錄（${DATA.filter(d=>d._source!=="legacy").length} 新版 / ${DATA.filter(d=>d._source==="legacy").length} 舊版）`;

  renderTagButtons("model-tags", MODEL_TAGS, "model");
  renderTagButtons("postproc-tags", POSTPROC_TAGS, "postproc");
  renderSummary();

  document.getElementById("search").addEventListener("input", e => { searchText = e.target.value; render(); });
  document.getElementById("show-clean").addEventListener("change", e => { showClean = e.target.checked; render(); });

  render();
}

function renderTagButtons(containerId, tags, cls) {
  const container = document.getElementById(containerId);
  tags.forEach(tag => {
    const count = DATA.filter(d => d._tags && d._tags[tag]).length;
    if (count === 0 && cls === "postproc") return;
    const btn = document.createElement("span");
    btn.className = `tag-btn ${cls}`;
    btn.textContent = `${TAG_LABELS[tag] || tag} (${count})`;
    btn.dataset.tag = tag;
    btn.addEventListener("click", () => {
      if (activeTags.has(tag)) { activeTags.delete(tag); btn.classList.remove("active"); }
      else { activeTags.add(tag); btn.classList.add("active"); }
      render();
    });
    container.appendChild(btn);
  });
}

function renderSummary() {
  const bar = document.getElementById("summary");
  const counts = {};
  [...MODEL_TAGS, ...POSTPROC_TAGS].forEach(t => counts[t] = 0);
  DATA.forEach(d => { if (d._tags) Object.keys(d._tags).forEach(t => { if (t in counts) counts[t]++; }); });
  bar.innerHTML = "";
  for (const [t, c] of Object.entries(counts)) {
    if (c === 0) continue;
    const item = document.createElement("span");
    item.className = "summary-item";
    item.innerHTML = `<span class="count">${c}</span> ${TAG_LABELS[t] || t}`;
    bar.appendChild(item);
  }
}

function highlightDiff(raw, processed) {
  if (!raw || !processed || raw === processed) return "";
  let html = "";
  let pi = 0;
  for (let i = 0; i < processed.length; i++) {
    if (pi < raw.length && processed[i] === raw[pi]) {
      html += escHtml(processed[i]);
      pi++;
    } else {
      html += `<span class="diff-char">${escHtml(processed[i])}</span>`;
      if (pi < raw.length && raw[pi] !== processed[i]) pi++;
    }
  }
  return html;
}

function escHtml(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function render() {
  const container = document.getElementById("entries");
  container.innerHTML = "";
  let shown = 0;
  const maxShow = 200;

  const filtered = DATA.filter(d => {
    const raw = d.raw || "";
    const processed = d.processed || "";
    const text = raw + processed;
    if (searchText && !text.includes(searchText)) return false;

    const tags = d._tags || {};
    const hasTags = Object.keys(tags).length > 0;

    if (activeTags.size > 0) {
      return [...activeTags].some(t => t in tags);
    }
    return showClean || hasTags;
  });

  document.getElementById("showing-count").textContent =
    `顯示 ${Math.min(filtered.length, maxShow)} / ${filtered.length} 筆` +
    (filtered.length > maxShow ? `（上限 ${maxShow}，請用標籤或搜尋縮小範圍）` : "");

  filtered.slice(0, maxShow).forEach((d, idx) => {
    const raw = d.raw || "";
    const processed = d.processed || "";
    const hasRaw = raw && raw !== "(streaming)";
    const tags = d._tags || {};
    const hasTags = Object.keys(tags).length > 0;

    const entry = document.createElement("div");
    entry.className = `entry ${hasTags ? "has-issue" : ""}`;

    // Header
    const header = document.createElement("div");
    header.className = "entry-header";

    const ts = (d.ts || "").substring(0, 19);
    header.innerHTML = `<span>${ts}</span>`;
    if (d._source && d._source !== "legacy") header.innerHTML += `<span class="session">[${d._source}]</span>`;
    if (d._source === "legacy") header.innerHTML += `<span style="color:#666">[舊版]</span>`;
    if (hasRaw) header.innerHTML += `<span style="color:#53d8fb">(raw)</span>`;
    else header.innerHTML += `<span style="color:#666">(processed-only)</span>`;
    if (d.audio_sec) header.innerHTML += `<span>${d.audio_sec}s</span>`;
    if (d.transcribe_sec) header.innerHTML += `<span>→${d.transcribe_sec}s</span>`;

    Object.keys(tags).forEach(t => {
      const cls = POSTPROC_TAGS.includes(t) ? "tag postproc" : "tag";
      header.innerHTML += `<span class="${cls}">${t}</span>`;
    });
    entry.appendChild(header);

    // Text
    const displayText = hasRaw ? raw : processed;
    const textDiv = document.createElement("div");
    textDiv.className = "text-raw";
    textDiv.textContent = displayText;
    entry.appendChild(textDiv);

    if (hasRaw && processed && raw !== processed) {
      const ppDiv = document.createElement("div");
      ppDiv.className = "text-processed";
      const diffHtml = highlightDiff(raw, processed);
      if (diffHtml) ppDiv.innerHTML = diffHtml;
      else ppDiv.textContent = processed;
      entry.appendChild(ppDiv);
    }

    // Audio
    if (d.audio_file && d._session_dir) {
      const audioPath = `sessions/${d._source}/${d.audio_file}`;
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = audioPath;
      entry.appendChild(audio);
    }

    container.appendChild(entry);
  });

  if (filtered.length === 0) {
    container.innerHTML = '<div class="empty-state">沒有符合條件的記錄</div>';
  }
}

init();
</script>
</body>
</html>"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="產生 session log HTML 檢視器")
    parser.add_argument("--legacy", action="store_true", help="也包含舊版 transcripts.jsonl")
    args = parser.parse_args()

    entries = load_session_entries()
    if args.legacy:
        entries = load_legacy_entries() + entries

    if not entries:
        print("沒有找到任何 log 資料")
        return

    for entry in entries:
        entry["_tags"] = tag_entry(entry)

    data_json = json.dumps(entries, ensure_ascii=False)
    labels_json = json.dumps(TAG_LABELS, ensure_ascii=False)

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    html = html.replace("__TAG_LABELS_PLACEHOLDER__", labels_json)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"已產生：{OUTPUT}（{len(entries)} 筆記錄）")
    print("用瀏覽器開啟即可使用（音訊播放需從 data/logs/ 目錄結構存取）")


if __name__ == "__main__":
    main()
