"""實驗：Qwen `context` 槽的「用途維度」探測 + baseline faster-whisper 對照（2026-06-29）。

不再測「輸出繁體」的措辭強弱（已知 n 夠）。改用質性不同的 context，分離它控制的維度：
  - script   : 字體（繁/簡）
  - terms    : 台灣用語（用詞在地化）
  - prefix   : 前文延續（塞半句看模型接不接 = context 是不是被當 prefix）
  - domain   : 主題/領域熱詞偏置（官方文件聲稱的用途）
指標：簡體率(OpenCC s2t) / 全形+半形標點數 / 前文是否被覆誦(prefix echo) / 原始輸出留存。
對照組：baseline faster-whisper（現行 config 的 initial_prompt，無 VAD，逐段）。
樣本：3 段不同內容（01 純中文 / 03 中英混合 / 05 數字規格）。交叉驗證跨內容。
結論邊界：僅此模型 / 這幾段 / 這些 probe / 這些指標。
"""

import io, json, subprocess, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, torch
from opencc import OpenCC
from asr_input.asr.qwen import QwenASREngine
from asr_input.asr import build_engine
from asr_input.config import load_config

ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "data/test_audio/segments"
OUT = ROOT / "experiments/results/probe_context_purpose_2026-06-29.json"
FILES = ["01_純中文長句_自我介紹.wav", "03_中英混合_Arduino.wav", "05_數字混合_電池規格.wav"]
_s2t = OpenCC("s2t")


def simp(t):
    cjk = [c for c in t if "一" <= c <= "鿿"]
    if not cjk:
        return 0.0
    return round(sum(1 for c in cjk if _s2t.convert(c) != c) / len(cjk), 3)


def punct(t):
    full = sum(t.count(c) for c in "，。、！？；：")
    half = sum(t.count(c) for c in ",.!?;:")
    return full, half


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


# 每個 probe 測「不同維度」，不是同方向的強弱變體
PREFIX_TEXT = "大家好我叫陳小明今天"  # prefix echo 探針：看輸出會不會帶出這串（音訊裡沒有）
PROBES = {
    "C0_baseline": "",
    "C_script": "請以繁體中文輸出",
    "C_terms": "請使用台灣在地用語",
    "C_prefix": PREFIX_TEXT,
    "C_domain": "主題：中山大學機器人團隊、Arduino 開發板、鋰電池規格",
}


def main():
    sr = 16000
    auds = {f: dec(SEG / f) for f in FILES}
    for f, a in auds.items():
        print(f"{f} {round(len(a) / sr, 1)}s")
    res = {"prefix_probe_text": PREFIX_TEXT, "files": FILES, "baseline_fw": {}, "qwen_probes": {}}

    # --- baseline faster-whisper（現行 config，逐段無 VAD）---
    cfg = load_config()
    acfg = dict(cfg["asr"])
    print("\n載入 baseline faster-whisper ...")
    fw = build_engine(acfg, vad_cfg=None)
    fw.load()
    for f, a in auds.items():
        t0 = time.time()
        raw = fw.transcribe(a, sr)
        res["baseline_fw"][f] = {
            "raw": raw,
            "simp": simp(raw),
            "punct": punct(raw),
            "sec": round(time.time() - t0, 2),
        }
        print(f"  [FW {f}] simp={simp(raw)} punct={punct(raw)}\n     {raw}")
    fw.unload()
    torch.cuda.empty_cache()

    # --- Qwen context 用途探測 ---
    print("\n載入 Qwen3-ASR ...")
    eng = QwenASREngine(
        model_id="Qwen/Qwen3-ASR-1.7B", device="cuda", language="Chinese", context=""
    )
    eng.load()
    for name, ctx in PROBES.items():
        eng.context = ctx
        pf = {}
        for f, a in auds.items():
            raw = eng.transcribe(a, sr)
            echo = PREFIX_TEXT[:6] in raw  # 前文是否被覆誦
            pf[f] = {"raw": raw, "simp": simp(raw), "punct": punct(raw), "prefix_echo": echo}
        res["qwen_probes"][name] = {"context": ctx, "per_file": pf}
        print(f"\n=== Qwen {name}  context={ctx!r} ===")
        for f in FILES:
            r = pf[f]
            print(
                f"  [{f}] simp={r['simp']} punct={r['punct']} echo={r['prefix_echo']}\n     {r['raw']}"
            )
    eng.unload()
    torch.cuda.empty_cache()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n存檔 {OUT.name}")


main()
