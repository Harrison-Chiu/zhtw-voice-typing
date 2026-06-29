"""Qwen context 繁體引導：措辭變體邊界測試（2026-06-29 follow-up）。

主實驗發現 context='請以繁體中文輸出。' 在兩短檔一致引導成繁體，其餘 5 probe 零效果。
本 follow-up 圍繞該句做措辭變體，測效果對「確切措辭」的敏感度（n=2 檔交叉）。
結論邊界同主實驗：僅此模型/這兩短檔/這些 probe。
"""

import io, json, subprocess, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, torch
from opencc import OpenCC
from asr_input.asr.qwen import QwenASREngine

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "data/test_audio"
OUT = ROOT / "experiments/results/steer_qwen_variants_2026-06-29.json"
FILES = ["中英錄音測試.m4a", "機器人展示，語音轉錄測試.m4a"]
_s2t = OpenCC("s2t")


def simp(t):
    cjk = [c for c in t if "一" <= c <= "鿿"]
    if not cjk:
        return 0.0, 0
    ch = [c for c in cjk if _s2t.convert(c) != c]
    return round(len(ch) / len(cjk), 3), len(cjk)


def dec(p, sr=16000):
    r = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(p),
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
    return np.frombuffer(r, dtype=np.float32).copy()


PROBES = {
    "known_good": "請以繁體中文輸出。",
    "no_period": "請以繁體中文輸出",
    "qing_out": "請輸出繁體中文。",
    "yi_out": "以繁體中文輸出。",
    "qing_use": "請用繁體中文。",
    "zhengti": "請以正體中文輸出。",
    "bare_kw": "繁體中文",
    "repeat_good": "請以繁體中文輸出。",
}


def main():
    sr = 16000
    auds = {f: dec(AUD / f) for f in FILES}
    eng = QwenASREngine(
        model_id="Qwen/Qwen3-ASR-1.7B", device="cuda", language="Chinese", context=""
    )
    eng.load()
    res = {"files": FILES, "probes": {}}
    for name, ctx in PROBES.items():
        eng.context = ctx
        pf = {}
        for f, a in auds.items():
            raw = eng.transcribe(a, sr)
            r, n = simp(raw)
            pf[f] = {"raw": raw, "simplified_ratio": r, "n_cjk": n}
        res["probes"][name] = {"context": ctx, "per_file": pf}
        rr = [pf[f]["simplified_ratio"] for f in FILES]
        print(f"{name:12} ctx={ctx!r:30} 簡體率={rr}")
    eng.unload()
    torch.cuda.empty_cache()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n存檔 {OUT.name}")


main()
