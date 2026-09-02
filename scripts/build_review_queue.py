"""Build the human review queue viewer for error candidates (roadmap E line).

Takes the JSON written by `scan_error_candidates.py` and produces an HTML page
that plays each segment's audio, shows raw vs processed text and which signals
fired, and lets a reviewer mark every entry 正確／錯誤／不確定 plus a free-text note.

Candidates and the matched control group are shuffled together and the queue
hides which group an entry belongs to until it has been marked. Otherwise a
reviewer who can see the labels judges flagged segments more harshly, and the
control group stops measuring what it was sampled to measure.

Marks are written back to `data/logs/review/marks.json` by the bundled server
(POST /api/marks), and mirrored to localStorage so nothing is lost if the server
is not running. Everything under `data/logs/` is gitignored — this page and the
marks contain transcript text.

Usage:
    python scripts/build_review_queue.py                 # 用最新一份候選 JSON 產生佇列
    python scripts/build_review_queue.py --serve         # 產生後啟動 server 並開瀏覽器
    python scripts/build_review_queue.py --input <path> --serve 9100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.paths import logs_dir, project_root  # noqa: E402

REVIEW_DIR = logs_dir() / "review"
OUTPUT = REVIEW_DIR / "queue.html"
MARKS = REVIEW_DIR / "marks.json"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>ASR 錯誤候選審核佇列</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", "Microsoft JhengHei", sans-serif; background: #14141f; color: #e0e0e0;
  padding: 20px; max-width: 1100px; margin: 0 auto; }
h1 { color: #e94560; font-size: 1.3em; margin-bottom: 4px; }
.sub { color: #777; font-size: 0.85em; margin-bottom: 14px; }
.bar { background: #1b1b2c; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;
  display: flex; gap: 18px; align-items: center; flex-wrap: wrap; font-size: 0.85em; }
.bar b { color: #53d8fb; }
.bar .save-state { margin-left: auto; color: #666; }
.bar .save-state.ok { color: #8eff8e; }
.bar .save-state.warn { color: #f0a500; }
button { background: #0f3460; color: #cfe6ff; border: 1px solid #2a4a70; border-radius: 6px;
  padding: 5px 12px; font-size: 0.85em; cursor: pointer; font-family: inherit; }
button:hover { border-color: #53d8fb; }
button:disabled { opacity: .45; cursor: default; }
select { background: #0f3460; color: #cfe6ff; border: 1px solid #2a4a70; border-radius: 6px; padding: 4px 8px; }

.entry { background: #1b1b2c; border: 1px solid #262640; border-left: 4px solid #333; border-radius: 8px;
  padding: 14px 16px; margin-bottom: 12px; }
.entry.current { border-color: #53d8fb; border-left-color: #53d8fb; }
.entry.done-ok { border-left-color: #2a8a4a; }
.entry.done-error { border-left-color: #e94560; }
.entry.done-unsure { border-left-color: #f0a500; }
.head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 0.78em; color: #888;
  margin-bottom: 8px; }
.idx { color: #53d8fb; font-weight: 600; }
.sig { background: #3a1020; color: #ff9db0; border-radius: 10px; padding: 2px 9px; font-size: 0.95em; }
.sig .detail { color: #c98; }
.group { border-radius: 10px; padding: 2px 9px; font-size: 0.95em; }
.group.candidate { background: #3a1020; color: #ff9db0; }
.group.control { background: #10303a; color: #8ed8e8; }
.text { font-size: 1.02em; line-height: 1.7; margin: 6px 0; white-space: pre-wrap; word-break: break-word; }
.text.raw { color: #cfd8e8; }
.label { color: #666; font-size: 0.75em; margin-right: 6px; }
.diff-char { background: #2a3f1a; color: #b8ff8e; border-radius: 2px; }
.controls { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
.verdict { border-radius: 6px; padding: 5px 14px; }
.verdict.sel-ok { background: #1a6b3a; color: #d6ffd6; border-color: #2a8a4a; }
.verdict.sel-error { background: #6b1a1a; color: #ffd6d6; border-color: #8a2a2a; }
.verdict.sel-unsure { background: #6b5000; color: #ffe9a8; border-color: #8a6a00; }
input[type=text] { background: #0f1424; border: 1px solid #2a2a44; color: #e0e0e0; padding: 6px 10px;
  border-radius: 6px; font-size: 0.85em; flex: 1; min-width: 220px; font-family: inherit; }
.hint { color: #555; font-size: 0.78em; margin: 14px 0 24px; line-height: 1.8; }
kbd { background: #0f3460; border: 1px solid #2a4a70; border-radius: 4px; padding: 1px 6px; font-size: 0.9em; }
.empty { color: #666; text-align: center; padding: 40px; }
</style>
</head>
<body>
<h1>ASR 錯誤候選審核佇列</h1>
<div class="sub" id="sub"></div>

<div class="bar">
  <span>已標記 <b id="done">0</b> / <span id="total">0</span></span>
  <span>正確 <b id="c-ok">0</b>・錯誤 <b id="c-error">0</b>・不確定 <b id="c-unsure">0</b></span>
  <label>顯示
    <select id="filter">
      <option value="todo">未標記</option>
      <option value="all">全部</option>
      <option value="done">已標記</option>
    </select>
  </label>
  <button id="export">匯出 marks.json</button>
  <span class="save-state" id="save-state">尚未儲存</span>
</div>

<div id="entries"></div>

<div class="hint">
  快捷鍵：<kbd>j</kbd>/<kbd>k</kbd> 下一筆／上一筆・<kbd>空白</kbd> 播放目前這筆・
  <kbd>1</kbd> 正確・<kbd>2</kbd> 錯誤・<kbd>3</kbd> 不確定。<br>
  音訊是從整段 job 錄音裁出這一段來播，需要透過 HTTP server 開啟（<code>--serve</code>）才讀得到檔案。<br>
  分組（候選／對照）在標記前不顯示，標記後才揭露，避免先看到標籤影響判斷。
</div>

<script>
const DATA = __DATA__;
const LABELS = __LABELS__;
const INITIAL_MARKS = __MARKS__;
const AUDIO_BASE = "__AUDIO_BASE__";
const SUBTITLE = __SUBTITLE__;

let marks = Object.assign({}, INITIAL_MARKS);
let current = 0;
let filterMode = "todo";
const audioCache = new Map();
let playing = null;
let ctx = null;

function key(d) { return d.job_id + ":" + d.segment_index; }
function escHtml(s) { return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function highlightDiff(raw, processed) {
  if (!raw || !processed || raw === processed) return "";
  let html = "", pi = 0;
  for (let i = 0; i < processed.length; i++) {
    if (pi < raw.length && processed[i] === raw[pi]) { html += escHtml(processed[i]); pi++; }
    else { html += `<span class="diff-char">${escHtml(processed[i])}</span>`;
           if (pi < raw.length && raw[pi] !== processed[i]) pi++; }
  }
  return html;
}

async function loadBuffer(path) {
  if (!audioCache.has(path)) {
    audioCache.set(path, (async () => {
      const resp = await fetch(path);
      if (!resp.ok) throw new Error(resp.status);
      ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
      return await ctx.decodeAudioData(await resp.arrayBuffer());
    })());
  }
  return audioCache.get(path);
}

async function play(d, btn) {
  if (playing) { try { playing.stop(); } catch(e){} playing = null; }
  if (!d.audio_path) { btn.textContent = "無音訊"; return; }
  const path = AUDIO_BASE + d.audio_path.replace(/\\/g, "/");
  btn.textContent = "…載入中";
  try {
    const buf = await loadBuffer(path);
    const sr = d.sample_rate || buf.sampleRate;
    const offset = (d.start_sample || 0) / sr;
    const dur = d.audio_sec || ((d.end_sample - d.start_sample) / sr);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    // Rewire back to play: btn.onclick is swapped to "stop" below, and without
    // this a finished segment could not be replayed until the next render().
    src.onended = () => {
      btn.textContent = "▶ 播放";
      btn.onclick = () => play(d, btn);
      playing = null;
    };
    src.start(0, offset, dur);
    playing = src;
    btn.textContent = "■ 停止";
    btn.onclick = () => { try { src.stop(); } catch(e){} };
  } catch (e) {
    btn.textContent = "✖ 讀不到音訊（需 --serve）";
  }
}

function setVerdict(d, verdict) {
  const k = key(d);
  if (marks[k] && marks[k].verdict === verdict) delete marks[k];
  else marks[k] = Object.assign({}, marks[k], {
    verdict, at: new Date().toISOString(),
    job_id: d.job_id, segment_index: d.segment_index,
    signals: (d.signals||[]).map(s => s.name), group: d.group,
  });
  save();
  render();
}

let saveTimer = null;
function save() {
  try { localStorage.setItem("asr-review-marks", JSON.stringify(marks)); } catch(e) {}
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const el = document.getElementById("save-state");
    try {
      const resp = await fetch("api/marks", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({updated_at: new Date().toISOString(), marks}),
      });
      if (!resp.ok) throw new Error(resp.status);
      el.textContent = "已寫入 marks.json " + new Date().toLocaleTimeString();
      el.className = "save-state ok";
    } catch (e) {
      el.textContent = "只存在瀏覽器（server 未啟動）— 請按「匯出 marks.json」";
      el.className = "save-state warn";
    }
  }, 400);
}

function visible() {
  return DATA.filter(d => {
    const marked = !!marks[key(d)];
    if (filterMode === "todo") return !marked;
    if (filterMode === "done") return marked;
    return true;
  });
}

function renderStats() {
  const counts = {ok: 0, error: 0, unsure: 0};
  Object.values(marks).forEach(m => { if (m.verdict in counts) counts[m.verdict]++; });
  document.getElementById("done").textContent = Object.keys(marks).length;
  document.getElementById("total").textContent = DATA.length;
  document.getElementById("c-ok").textContent = counts.ok;
  document.getElementById("c-error").textContent = counts.error;
  document.getElementById("c-unsure").textContent = counts.unsure;
}

function render() {
  renderStats();
  const list = visible();
  if (current >= list.length) current = Math.max(0, list.length - 1);
  const container = document.getElementById("entries");
  container.innerHTML = "";
  if (!list.length) {
    container.innerHTML = '<div class="empty">這個篩選條件下沒有項目。</div>';
    return;
  }
  list.forEach((d, i) => {
    const mark = marks[key(d)];
    const el = document.createElement("div");
    el.className = "entry" + (i === current ? " current" : "") + (mark ? " done-" + mark.verdict : "");

    const head = document.createElement("div");
    head.className = "head";
    let h = `<span class="idx">#${i + 1}</span>`;
    h += `<span>${escHtml((d.created_at||"").substring(0,19).replace("T"," "))}</span>`;
    h += `<span>${d.audio_sec ?? "?"}s → ${d.transcribe_sec ?? "?"}s</span>`;
    if (mark) h += `<span class="group ${d.group}">${d.group === "candidate" ? "候選" : "對照"}</span>`;
    (d.signals || []).forEach(s => {
      h += `<span class="sig">${escHtml(LABELS[s.name] || s.name)}` +
           (s.detail ? ` <span class="detail">${escHtml(s.detail)}</span>` : "") + `</span>`;
    });
    head.innerHTML = h;
    el.appendChild(head);

    const raw = d.raw_text || "", proc = d.processed_text || "";
    const rawDiv = document.createElement("div");
    rawDiv.className = "text raw";
    rawDiv.innerHTML = `<span class="label">raw</span>` + escHtml(raw || "（空）");
    el.appendChild(rawDiv);
    if (proc !== raw) {
      const pDiv = document.createElement("div");
      pDiv.className = "text";
      const diff = highlightDiff(raw, proc);
      pDiv.innerHTML = `<span class="label">processed</span>` + (diff || escHtml(proc || "（空）"));
      el.appendChild(pDiv);
    }

    const controls = document.createElement("div");
    controls.className = "controls";
    const playBtn = document.createElement("button");
    playBtn.textContent = "▶ 播放";
    playBtn.onclick = () => play(d, playBtn);
    controls.appendChild(playBtn);
    [["ok","1 正確"],["error","2 錯誤"],["unsure","3 不確定"]].forEach(([v, text]) => {
      const b = document.createElement("button");
      b.className = "verdict" + (mark && mark.verdict === v ? " sel-" + v : "");
      b.textContent = text;
      b.onclick = () => setVerdict(d, v);
      controls.appendChild(b);
    });
    const note = document.createElement("input");
    note.type = "text";
    note.placeholder = "備註：正確答案應該是什麼？（選填）";
    note.value = (mark && mark.note) || "";
    note.oninput = () => {
      const k = key(d);
      marks[k] = Object.assign({job_id: d.job_id, segment_index: d.segment_index,
        group: d.group, verdict: (marks[k]||{}).verdict || "unsure"}, marks[k], {note: note.value});
      save();
      renderStats();
    };
    controls.appendChild(note);
    el.appendChild(controls);
    container.appendChild(el);
  });
  const cur = container.children[current];
  if (cur && cur.scrollIntoView) cur.scrollIntoView({block: "nearest"});
}

document.getElementById("filter").onchange = (e) => { filterMode = e.target.value; current = 0; render(); };
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify({updated_at: new Date().toISOString(), marks}, null, 2)],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "marks.json";
  a.click();
};

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const list = visible();
  const d = list[current];
  if (e.key === "j") { current = Math.min(current + 1, list.length - 1); render(); }
  else if (e.key === "k") { current = Math.max(current - 1, 0); render(); }
  else if (e.key === " " && d) {
    e.preventDefault();
    const btn = document.getElementById("entries").children[current].querySelector("button");
    play(d, btn);
  }
  else if (d && (e.key === "1" || e.key === "2" || e.key === "3")) {
    setVerdict(d, {"1": "ok", "2": "error", "3": "unsure"}[e.key]);
  }
});

// marks.json is embedded at build time; localStorage may hold marks made after
// that build (or made with the server down). Keep whichever has more entries,
// with the embedded file winning on conflicting keys since it is the shared copy.
try {
  const local = JSON.parse(localStorage.getItem("asr-review-marks") || "{}");
  if (Object.keys(local).length > Object.keys(marks).length) marks = Object.assign(local, marks);
} catch (e) {}

document.getElementById("sub").textContent = SUBTITLE;
render();
</script>
</body>
</html>"""


def build(entries: list[dict], labels: dict, marks: dict, subtitle: str, audio_base: str) -> str:
    def dump(value) -> str:
        return json.dumps(value, ensure_ascii=False)

    html = HTML_TEMPLATE.replace("__DATA__", dump(entries))
    html = html.replace("__LABELS__", dump(labels))
    html = html.replace("__MARKS__", dump(marks))
    html = html.replace("__SUBTITLE__", dump(subtitle))
    return html.replace("__AUDIO_BASE__", audio_base)


def latest_candidates() -> Path | None:
    files = sorted(REVIEW_DIR.glob("candidates_*.json"))
    return files[-1] if files else None


def _serve(port: int, open_browser: bool = True) -> None:
    """Serve the repo root so the page can read both itself and the job wavs.

    POST /api/marks writes marks.json; without it the page can still mark
    entries but only into localStorage.
    """
    import http.server
    import webbrowser
    from functools import partial

    root = project_root()
    rel = OUTPUT.relative_to(root).as_posix()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib naming
            if self.path.rstrip("/").endswith("api/marks"):
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length))
                except json.JSONDecodeError:
                    self.send_error(400, "invalid json")
                    return
                MARKS.parent.mkdir(parents=True, exist_ok=True)
                MARKS.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            else:
                self.send_error(404)

        def log_message(self, *args):  # keep the console readable
            pass

    url = f"http://localhost:{port}/{rel}"
    print(f"啟動 HTTP server：{url}")
    print(f"標記會寫回 {MARKS}")
    print("Ctrl+C 停止")
    if open_browser:
        webbrowser.open(url)
    server = http.server.ThreadingHTTPServer(("", port), partial(Handler, directory=str(root)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


def main() -> int:
    parser = argparse.ArgumentParser(description="產生錯誤候選審核佇列 HTML")
    parser.add_argument("--input", type=Path, default=None, help="候選 JSON（預設取最新一份）")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20260903, help="候選／對照混合順序的種子")
    parser.add_argument(
        "--serve", nargs="?", const=8100, type=int, metavar="PORT", help="產生後啟動 server"
    )
    parser.add_argument("--no-browser", action="store_true", help="啟動 server 但不自動開瀏覽器")
    args = parser.parse_args()

    src = args.input or latest_candidates()
    if src is None or not src.exists():
        print(f"找不到候選 JSON（{src or REVIEW_DIR}）。先跑 scripts/scan_error_candidates.py")
        return 1

    payload = json.loads(src.read_text(encoding="utf-8"))
    entries = payload["entries"]
    random.Random(args.seed).shuffle(entries)

    marks = {}
    if MARKS.exists():
        marks = json.loads(MARKS.read_text(encoding="utf-8")).get("marks", {})

    stats = payload.get("stats", {})
    subtitle = (
        f"來源 {src.name}・候選 {stats.get('candidates', '?')} + 對照 "
        f"{payload.get('control_count', '?')} = {len(entries)} 筆"
        f"（母體 {stats.get('segments_total', '?')} 段，產生於 "
        f"{payload.get('generated_at', '')[:19]}）"
    )
    # The page lives under data/logs/review/; audio_path in the DB is relative to
    # the repo root, which the server serves, so the URL has to climb back out.
    audio_base = "../../../"

    html = build(entries, payload.get("signal_labels", {}), marks, subtitle, audio_base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(
        f"已產生：{args.out}（{len(entries)} 筆，已標記 {len(marks)} 筆，seed={args.seed}，"
        f"更新於 {datetime.now(UTC).isoformat(timespec='seconds')}）"
    )

    if args.serve is not None:
        _serve(args.serve, open_browser=not args.no_browser)
    else:
        print("提示：加 --serve 才播得出音訊（瀏覽器不允許 file:// 讀取本機音檔）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
