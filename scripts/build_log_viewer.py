"""Generate an HTML viewer for session logs with tag filtering and audio playback.

Usage:
    python scripts/build_log_viewer.py              # 產生 data/logs/viewer.html
    python scripts/build_log_viewer.py --legacy      # 也包含舊版 transcripts.jsonl
    python scripts/build_log_viewer.py --serve       # 產生後啟動 HTTP server（支援音訊播放）
    python scripts/build_log_viewer.py --serve 9000  # 指定 port

Output: data/logs/viewer.html (open in browser, audio playback via relative paths)
Note: Audio playback requires HTTP server (not file://) due to browser security.
      Use --serve or manually run a server from data/logs/.
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

.server-hint { background: #2a1a00; border: 1px solid #f0a500; border-radius: 6px; padding: 10px 14px;
  margin-bottom: 16px; font-size: 0.85em; color: #f0a500; display: none; }
.server-hint code { background: #0f3460; padding: 2px 6px; border-radius: 3px; color: #53d8fb; cursor: pointer; }
.server-hint code:hover { background: #1a4a7a; }

.controls { background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.control-row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.control-row:last-child { margin-bottom: 0; }
.control-label { color: #e94560; font-weight: 600; font-size: 0.85em; min-width: 70px; }
.control-hint { color: #666; font-size: 0.75em; margin-left: 8px; }

.tag-btn { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 12px; font-size: 0.8em;
  cursor: pointer; border: 1px solid #444; background: #0f3460; color: #ccc; margin: 2px; transition: all .15s; user-select: none; }
.tag-btn:hover { border-color: #888; }
.tag-btn.include { background: #1a6b3a; color: #8eff8e; border-color: #2a8a4a; }
.tag-btn.exclude { background: #6b1a1a; color: #ff8e8e; border-color: #8a2a2a; }
.tag-btn .tag-icon { font-size: 0.75em; }
.tag-btn.model { border-left: 3px solid #53d8fb; }
.tag-btn.include.model { border-left-color: #53d8fb; }
.tag-btn.exclude.model { border-left-color: #53d8fb; }
.tag-btn.postproc { border-left: 3px solid #f0a500; }
.tag-btn.include.postproc { border-left-color: #f0a500; }
.tag-btn.exclude.postproc { border-left-color: #f0a500; }

.filter-group { display: inline-flex; gap: 2px; background: #0a1a30; border-radius: 6px; padding: 2px; }
.filter-opt { padding: 3px 10px; border-radius: 4px; font-size: 0.8em; cursor: pointer; color: #888;
  border: none; background: transparent; transition: all .15s; }
.filter-opt.active { background: #0f3460; color: #e0e0e0; }
.filter-opt:hover { color: #ccc; }

input[type=text] { background: #0f3460; border: 1px solid #444; color: #e0e0e0; padding: 6px 12px;
  border-radius: 6px; font-size: 0.9em; width: 300px; }
input[type=text]::placeholder { color: #666; }

.active-filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
.active-filters:empty { display: none; }
.active-filter-chip { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 10px;
  font-size: 0.75em; }
.active-filter-chip.include { background: #1a6b3a; color: #8eff8e; }
.active-filter-chip.exclude { background: #6b1a1a; color: #ff8e8e; }
.clear-filters { color: #e94560; cursor: pointer; font-size: 0.8em; padding: 2px 8px; }
.clear-filters:hover { text-decoration: underline; }

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
<div class="server-hint" id="server-hint">
  音訊無法播放？瀏覽器安全限制阻擋了本地檔案存取。請改用：<br>
  <code id="server-cmd" title="點擊複製">python scripts/build_log_viewer.py --serve</code>
  &nbsp;（會自動啟動 HTTP server 並開啟瀏覽器）
</div>

<div class="controls">
  <div class="control-row">
    <span class="control-label">模型層</span>
    <div id="model-tags"></div>
    <span class="control-hint" id="tag-hint">點擊切換：不篩選 → +包含 → -排除</span>
  </div>
  <div class="control-row">
    <span class="control-label">後處理</span>
    <div id="postproc-tags"></div>
  </div>
  <div class="control-row">
    <span class="control-label">搜尋</span>
    <input type="text" id="search" placeholder="搜尋文字內容...">
    <div class="filter-group" id="search-target">
      <button class="filter-opt active" data-target="both">全部</button>
      <button class="filter-opt" data-target="raw">模型層</button>
      <button class="filter-opt" data-target="processed">後處理</button>
    </div>
  </div>
</div>

<div class="active-filters" id="active-filters"></div>
<div class="summary-bar" id="summary"></div>
<div class="showing-count" id="showing-count"></div>
<div class="entries" id="entries"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;
const TAG_LABELS = __TAG_LABELS_PLACEHOLDER__;
const MODEL_TAGS = ["punct","simplified","spacing","repeat","no_comma","truncated"];
const POSTPROC_TAGS = ["overconv","punct_unfixed","simplified_unfixed"];

// tag state: "neutral" | "include" | "exclude"
const tagState = {};
[...MODEL_TAGS, ...POSTPROC_TAGS].forEach(t => tagState[t] = "neutral");

let searchText = "";
let searchTarget = "both"; // "both" | "raw" | "processed"

function init() {
  document.getElementById("stats").textContent =
    `${DATA.length} 筆記錄（${DATA.filter(d=>d._source!=="legacy").length} 新版 / ${DATA.filter(d=>d._source==="legacy").length} 舊版）`;

  renderTagButtons("model-tags", MODEL_TAGS, "model");
  renderTagButtons("postproc-tags", POSTPROC_TAGS, "postproc");
  renderSummary();

  document.getElementById("search").addEventListener("input", e => { searchText = e.target.value; render(); });

  document.getElementById("search-target").addEventListener("click", e => {
    const btn = e.target.closest(".filter-opt");
    if (!btn) return;
    document.querySelectorAll("#search-target .filter-opt").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    searchTarget = btn.dataset.target;
    render();
  });

  document.getElementById("server-cmd").addEventListener("click", () => {
    navigator.clipboard.writeText("cd data/logs && python -m http.server 8000").catch(() => {});
    document.getElementById("server-cmd").textContent = "已複製！";
    setTimeout(() => { document.getElementById("server-cmd").textContent = "cd data/logs && python -m http.server 8000"; }, 1500);
  });

  // detect file:// protocol — show server hint if audio won't work
  if (location.protocol === "file:") {
    document.getElementById("server-hint").style.display = "block";
  }

  render();
}

function renderTagButtons(containerId, tags, cls) {
  const container = document.getElementById(containerId);
  tags.forEach(tag => {
    const count = DATA.filter(d => d._tags && d._tags[tag]).length;
    if (count === 0 && cls === "postproc") return;
    const btn = document.createElement("span");
    btn.className = `tag-btn ${cls}`;
    btn.innerHTML = `<span class="tag-icon"></span>${TAG_LABELS[tag] || tag} (${count})`;
    btn.dataset.tag = tag;
    btn.addEventListener("click", () => {
      const states = ["neutral", "include", "exclude"];
      const cur = states.indexOf(tagState[tag]);
      tagState[tag] = states[(cur + 1) % 3];
      updateTagBtnStyle(btn, tag);
      render();
    });
    container.appendChild(btn);
  });
}

function updateTagBtnStyle(btn, tag) {
  btn.classList.remove("include", "exclude");
  const icon = btn.querySelector(".tag-icon");
  if (tagState[tag] === "include") {
    btn.classList.add("include");
    icon.textContent = "+";
  } else if (tagState[tag] === "exclude") {
    btn.classList.add("exclude");
    icon.textContent = "−";
  } else {
    icon.textContent = "";
  }
}

function renderActiveFilters() {
  const container = document.getElementById("active-filters");
  container.innerHTML = "";
  let hasFilter = false;
  for (const [tag, state] of Object.entries(tagState)) {
    if (state === "neutral") continue;
    hasFilter = true;
    const chip = document.createElement("span");
    chip.className = `active-filter-chip ${state}`;
    chip.textContent = `${state === "include" ? "+" : "−"} ${TAG_LABELS[tag] || tag}`;
    container.appendChild(chip);
  }
  if (hasFilter) {
    const clear = document.createElement("span");
    clear.className = "clear-filters";
    clear.textContent = "清除全部";
    clear.addEventListener("click", () => {
      Object.keys(tagState).forEach(t => tagState[t] = "neutral");
      document.querySelectorAll(".tag-btn").forEach(btn => {
        updateTagBtnStyle(btn, btn.dataset.tag);
      });
      render();
    });
    container.appendChild(clear);
  }
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
  renderActiveFilters();

  const container = document.getElementById("entries");
  container.innerHTML = "";
  const maxShow = 200;

  const includeTags = Object.entries(tagState).filter(([,s]) => s === "include").map(([t]) => t);
  const excludeTags = Object.entries(tagState).filter(([,s]) => s === "exclude").map(([t]) => t);

  const filtered = DATA.filter(d => {
    const raw = d.raw || "";
    const processed = d.processed || "";

    // search filter
    if (searchText) {
      let haystack;
      if (searchTarget === "raw") haystack = raw;
      else if (searchTarget === "processed") haystack = processed;
      else haystack = raw + processed;
      if (!haystack.includes(searchText)) return false;
    }

    const tags = d._tags || {};
    const entryTags = Object.keys(tags);

    // exclude filter: reject if entry has any excluded tag
    if (excludeTags.length > 0 && excludeTags.some(t => entryTags.includes(t))) return false;

    // include filter: require at least one included tag
    if (includeTags.length > 0) return includeTags.some(t => entryTags.includes(t));

    // no include tags active: show all (that passed exclude)
    return true;
  });

  document.getElementById("showing-count").textContent =
    `顯示 ${Math.min(filtered.length, maxShow)} / ${filtered.length} 筆（共 ${DATA.length} 筆）` +
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
      header.innerHTML += `<span class="${cls}">${TAG_LABELS[t] || t}</span>`;
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

    // Audio — load via fetch+blob to avoid Range request issues
    if (d.audio_file && d._session_dir) {
      const audioPath = `sessions/${d._source}/${d.audio_file}`;
      const audioWrap = document.createElement("div");
      const playBtn = document.createElement("button");
      playBtn.textContent = "▶ 播放";
      playBtn.style.cssText = "background:#0f3460;color:#53d8fb;border:1px solid #444;border-radius:4px;padding:3px 10px;font-size:0.8em;cursor:pointer;margin-top:4px;";
      playBtn.addEventListener("click", async () => {
        playBtn.textContent = "…載入中";
        playBtn.disabled = true;
        try {
          const resp = await fetch(audioPath);
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          const audio = document.createElement("audio");
          audio.controls = true;
          audio.src = url;
          audioWrap.replaceChild(audio, playBtn);
          audio.play();
        } catch(e) {
          playBtn.textContent = "✖ 無法載入（需透過 HTTP server 開啟）";
          playBtn.style.color = "#e94560";
        }
      }, {once: true});
      audioWrap.appendChild(playBtn);
      entry.appendChild(audioWrap);
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


def _start_server(directory: Path, port: int) -> None:
    """Start a threaded HTTP server for viewer + audio playback."""
    import http.server
    import os
    import webbrowser

    os.chdir(directory)

    url = f"http://localhost:{port}/viewer.html"
    print(f"啟動 HTTP server：{url}")
    print("Ctrl+C 停止")
    webbrowser.open(url)

    server = http.server.ThreadingHTTPServer(("", port), http.server.SimpleHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="產生 session log HTML 檢視器")
    parser.add_argument("--legacy", action="store_true", help="也包含舊版 transcripts.jsonl")
    parser.add_argument(
        "--serve",
        nargs="?",
        const=8000,
        type=int,
        metavar="PORT",
        help="產生後啟動 HTTP server（預設 port 8000）",
    )
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

    if args.serve is not None:
        _start_server(OUTPUT.parent, args.serve)
    else:
        print("提示：加 --serve 可啟動 HTTP server 並自動開啟瀏覽器（含音訊播放）")


if __name__ == "__main__":
    main()
