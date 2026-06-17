"""Experiment v2: isolate what makes a prompt produce full-width punctuation.

Outputs full text for every segment x prompt combination, plus detailed
analysis (punctuation, simplified Chinese, character count differences).

Usage:
    .venv\\Scripts\\python.exe experiment_punctuation_v2.py
"""

import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa
import numpy as np
from faster_whisper import WhisperModel

SEGMENT_DIR = Path("data/test_audio/segments")
SR = 16000
OUT_FILE = Path("data/experiment_punct_v2.json")

PROMPTS = {
    "A_baseline": {
        "val": "繁體中文，台灣用語。",
        "note": "Current production prompt (short, 1 full-width comma)",
    },
    "B_original_long": {
        "val": "繁體中文，台灣用語。這是一段語音辨識的測試，請使用全形標點符號！好的，沒問題。",
        "note": "Previous best (80% full-width) — has ！ which may cause garbled output",
    },
    "G_long_no_exclaim": {
        "val": "繁體中文，台灣用語。這是一段語音辨識的測試，我們來看看結果。",
        "note": "Same length as B, but no ！ — test if exclamation causes garbled chars",
    },
    "H_many_commas": {
        "val": "今天天氣很好，我們一起去公園，看看風景，聊聊天，吃個午餐。",
        "note": "Many full-width commas, natural sentence, no explicit instruction",
    },
    "I_multi_sentence": {
        "val": "繁體中文，台灣用語。今天的會議很順利，大家都有提出建議。接下來，我們需要確認時間表。",
        "note": "3 sentences with full-width commas + periods, natural style",
    },
    "J_comma_heavy": {
        "val": "首先，我要說明的是，這個計畫的目標，是提升品質，改善效率，降低成本。",
        "note": "Comma-heavy single sentence, tests if high comma density helps",
    },
    "K_mixed_punct": {
        "val": "繁體中文，台灣用語。這樣可以嗎？當然沒問題，我們繼續。太好了，開始吧。",
        "note": "Mix of ，。？ — test if varied full-width punctuation helps",
    },
}

HALF_PUNCT = {",": "半形逗號", ".": "半形句號", "?": "半形問號", "!": "半形驚嘆", ":": "半形冒號", ";": "半形分號"}
FULL_PUNCT = {"，": "全形逗號", "。": "全形句號", "？": "全形問號", "！": "全形驚嘆", "：": "全形冒號", "；": "全形分號", "、": "頓號"}

SIMPLIFIED_CHARS = re.compile(
    r"[这这说为与关于来对时从会让进这种没还这个这些这样这里这么"
    r"这就这些这可这应这将这很发现开关问题运动经过图国际际际"
    r"设计们个么来对时会还过进让说这将发开关图际运经问题]"
)


def count_punct(text):
    result = {}
    for ch, name in {**HALF_PUNCT, **FULL_PUNCT}.items():
        cnt = text.count(ch)
        if cnt > 0:
            result[ch] = {"count": cnt, "name": name, "width": "half" if ch in HALF_PUNCT else "full"}
    return result


def detect_simplified(text):
    simp_patterns = [
        ("们", "們"), ("这", "這"), ("来", "來"), ("对", "對"),
        ("时", "時"), ("会", "會"), ("还", "還"), ("过", "過"),
        ("进", "進"), ("让", "讓"), ("说", "說"), ("发", "發"),
        ("开", "開"), ("关", "關"), ("图", "圖"), ("运", "運"),
        ("经", "經"), ("设", "設"), ("为", "為"), ("与", "與"),
    ]
    found = []
    for simp, trad in simp_patterns:
        positions = [i for i, c in enumerate(text) if c == simp]
        if positions:
            found.append({"char": simp, "traditional": trad, "count": len(positions)})
    return found


def main():
    wav_files = sorted(SEGMENT_DIR.glob("*.wav"))
    if not wav_files:
        print("找不到測試片段，請先執行 extract_test_segments.py")
        return

    print("載入模型...", flush=True)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("模型載入完成。\n", flush=True)

    segments = {}
    for f in wav_files:
        audio, _ = librosa.load(str(f), sr=SR, mono=True)
        segments[f.stem] = audio.astype(np.float32)
        print(f"  載入: {f.stem} ({len(audio)/SR:.1f}s)")
    print()

    all_results = {}

    for prompt_name, prompt_info in PROMPTS.items():
        prompt_val = prompt_info["val"]
        print(f"{'='*80}")
        print(f"  {prompt_name}")
        print(f"  Prompt: {prompt_val}")
        print(f"  Note: {prompt_info['note']}")
        print(f"{'='*80}")

        prompt_results = {}
        total_half = 0
        total_full = 0

        for seg_name, audio in segments.items():
            segs, _ = model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                initial_prompt=prompt_val,
            )
            text = "".join(s.text for s in segs).strip()

            punct = count_punct(text)
            simplified = detect_simplified(text)
            half_cnt = sum(v["count"] for v in punct.values() if v["width"] == "half")
            full_cnt = sum(v["count"] for v in punct.values() if v["width"] == "full")
            total_half += half_cnt
            total_full += full_cnt

            prompt_results[seg_name] = {
                "text": text,
                "char_count": len(text),
                "punct": punct,
                "half_total": half_cnt,
                "full_total": full_cnt,
                "simplified": simplified,
            }

            print(f"\n  [{seg_name}]")
            print(f"  字數: {len(text)} | 半形: {half_cnt} | 全形: {full_cnt}")
            if simplified:
                simp_str = ", ".join(f"{s['char']}→{s['traditional']} x{s['count']}" for s in simplified)
                print(f"  ⚠ 簡體字: {simp_str}")
            print(f"  全文: {text}")

        total = total_half + total_full
        pct = total_full / total * 100 if total else 0
        print(f"\n  >>> 全形率: {pct:.1f}% (半形:{total_half} 全形:{total_full})\n")

        all_results[prompt_name] = {
            "prompt": prompt_val,
            "note": prompt_info["note"],
            "segments": prompt_results,
            "total_half": total_half,
            "total_full": total_full,
            "fullwidth_pct": round(pct, 1),
        }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n完整結果已存: {OUT_FILE}")

    print(f"\n{'='*80}")
    print("=== 總結 ===")
    print(f"{'='*80}\n")
    for name, r in all_results.items():
        simp_total = sum(
            sum(s["count"] for s in seg["simplified"])
            for seg in r["segments"].values()
        )
        print(
            f"  {name:<22} 全形率:{r['fullwidth_pct']:>5.1f}%  "
            f"半:{r['total_half']:>2}  全:{r['total_full']:>2}  "
            f"簡體字:{simp_total}"
        )


if __name__ == "__main__":
    main()
