"""實驗：Qwen3-ASR 的 context(system prompt) 能否引導輸出繁體中文?（2026-06-29）

設計（嚴謹邊界）：
- 受測：Qwen3-ASR-1.7B，唯一自由文字引導槽 = `context`（進對話模板的 system 訊息）。
  `language` 受限固定清單只有 "Chinese"，無繁簡選項，故不測 language。
- 指標：簡體殘留率 = OpenCC s2t 會改動的 CJK 字數 / 總 CJK 字數。0=全繁、越高越簡。
- 交叉驗證：每個 context probe 同時跑兩個短檔；要判定「有效」須兩檔方向一致。
- 結論只限：此模型 / 這兩個短檔 / 這些 probe / 此指標。不外推到其他音訊或長度。

用法：.venv\\Scripts\\python.exe experiments/steer_qwen_traditional.py
"""

import gc
import io
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from opencc import OpenCC  # noqa: E402

from asr_input.asr.qwen import QwenASREngine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "data/test_audio"
OUT = PROJECT_ROOT / "experiments/results/steer_qwen_2026-06-29.json"
SHORT_FILES = ["中英錄音測試.m4a", "機器人展示，語音轉錄測試.m4a"]

_s2t = OpenCC("s2t")


def simplified_ratio(text: str) -> dict:
    """以 OpenCC s2t 為準：被改動的 CJK 字 = 簡體字。回傳殘留率與被改字清單。"""
    cjk = [c for c in text if "一" <= c <= "鿿"]
    if not cjk:
        return {"ratio": 0.0, "n_cjk": 0, "changed": []}
    changed = [c for c in cjk if _s2t.convert(c) != c]
    uniq = sorted(set(changed))
    return {"ratio": round(len(changed) / len(cjk), 3), "n_cjk": len(cjk), "changed": uniq}


def decode(path: Path, sr: int) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# context probes：baseline + 多種繁體引導（中文指令 / 強指令 / 繁體前文 / 英文指令）
PROBES = {
    "baseline_empty": "",
    "zh_simple": "請以繁體中文輸出。",
    "zh_strong": "使用台灣正體中文，禁止輸出任何簡體字。",
    "zh_prefix": "以下是一段繁體中文的逐字稿：",
    "en_instruct": "Output in Traditional Chinese (Taiwan).",
    "zh_terms": "繁體中文，台灣用語。視訊請寫成影片，軟體不要寫軟件。",
}


def main():
    sr = 16000
    audios = {fn: decode(AUDIO_DIR / fn, sr) for fn in SHORT_FILES}
    for fn, a in audios.items():
        print(f"音檔 {fn} {round(len(a) / sr, 1)}s")

    engine = QwenASREngine(
        model_id="Qwen/Qwen3-ASR-1.7B", device="cuda", language="Chinese", context=""
    )
    print("載入 Qwen3-ASR ...")
    engine.load()

    results = {
        "metric": "simplified_ratio via OpenCC s2t (0=全繁,高=簡)",
        "files": SHORT_FILES,
        "probes": {},
    }
    for pname, ctx in PROBES.items():
        engine.context = ctx
        per_file = {}
        for fn, a in audios.items():
            t0 = time.time()
            raw = engine.transcribe(a, sr)
            sr_stat = simplified_ratio(raw)
            per_file[fn] = {
                "raw": raw,
                "simplified_ratio": sr_stat["ratio"],
                "n_cjk": sr_stat["n_cjk"],
                "changed": sr_stat["changed"],
                "sec": round(time.time() - t0, 2),
            }
        results["probes"][pname] = {"context": ctx, "per_file": per_file}
        ratios = [per_file[fn]["simplified_ratio"] for fn in SHORT_FILES]
        print(f"\n=== probe '{pname}'  context={ctx!r} ===")
        for fn in SHORT_FILES:
            r = per_file[fn]
            print(f"  [{fn}] 簡體率={r['simplified_ratio']} (CJK {r['n_cjk']})")
            print(f"      {r['raw']}")
        print(f"  → 兩檔簡體率 {ratios}")

    engine.unload()
    gc.collect()
    torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 交叉驗證彙整：相對 baseline 兩檔是否一致下降
    base = [
        results["probes"]["baseline_empty"]["per_file"][fn]["simplified_ratio"]
        for fn in SHORT_FILES
    ]
    print(f"\n{'=' * 60}\n交叉驗證（相對 baseline {base}）：")
    for pname in PROBES:
        if pname == "baseline_empty":
            continue
        r = [results["probes"][pname]["per_file"][fn]["simplified_ratio"] for fn in SHORT_FILES]
        d = [round(r[i] - base[i], 3) for i in range(len(base))]
        both_down = all(x < -0.01 for x in d)
        verdict = (
            "兩檔一致下降✓"
            if both_down
            else ("無一致效果" if not all(x < 0 for x in d) else "微幅")
        )
        print(f"  {pname:14} 簡體率={r} Δ={d} {verdict}")
    print(f"\n結論邊界：僅此模型/這兩短檔/這些 probe/此指標。存檔 → {OUT.name}")


if __name__ == "__main__":
    main()
