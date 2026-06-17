"""Experiment v3: targeted punctuation control — prompts + hotwords + suppress_tokens.

Two independent experiments in one script:
  Exp 1 — Three new prompt designs (long natural text, embedded instruction, extra-long)
  Exp 2 — hotwords / suppress_tokens for half-width comma (token 11)

All tested on the same 5 standard segments.

Usage:
    .venv\\Scripts\\python.exe experiment_punctuation_v3.py
"""

import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

import librosa
import numpy as np
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENT_DIR = PROJECT_ROOT / "data/test_audio/segments"
SR = 16000
OUT_FILE = Path(__file__).resolve().parent / "results/experiment_punct_v3.json"

# ── Experiment 1: Prompt designs ──────────────────────────────────────────────
# Based on v2 findings:
#   - Full-width comma density in prompt → output follows (statistical continuation)
#   - ！ causes garbled output → avoid
#   - Longer prompt may sustain effect into longer segments
#   - Instruction-as-text is a new variable to isolate

PROMPT_L = (
    "各位同學，大家好。今天我們要來討論的主題是人工智慧在日常生活中的應用。"
    "首先，我想跟大家分享幾個有趣的案例。第一個例子是語音辨識技術，"
    "它可以把我們說的話轉換成文字，這在會議記錄、字幕生成等方面都非常實用。"
    "第二個例子是推薦系統，像是影片平台會根據你的觀看紀錄，推薦你可能感興趣的內容。"
)

PROMPT_M = (
    "以下是一段繁體中文的語音辨識結果，文字使用全形標點符號，"
    "包括逗號、句號、問號等。這是正式的中文書寫格式，所有標點都採用全形。"
    "好的，那我們接下來看看今天要討論的內容。首先，關於這次專案的進度，"
    "目前已經完成了大部分的開發工作，接下來需要進行測試和優化。"
)

PROMPT_N = (
    "各位好，歡迎來到今天的技術分享會。我是這次的主講人，今天要跟大家介紹的主題是"
    "如何利用深度學習技術來改善語音辨識的準確度。在開始之前，我想先簡單回顧一下"
    "語音辨識技術的發展歷程。早期的語音辨識系統主要依賴統計模型，像是隱馬可夫模型，"
    "搭配高斯混合模型來進行聲學建模。雖然這些傳統方法在特定場景下表現不錯，"
    "但在面對複雜的口語環境時，辨識率往往會大幅下降。後來，隨著深度學習的興起，"
    "研究人員開始嘗試用神經網路來取代傳統的聲學模型，結果發現效果有了顯著的提升。"
)

PROMPTS_EXP1 = {
    "L_natural_long": {
        "val": PROMPT_L,
        "note": f"自然長文 ({len(PROMPT_L)} chars)，適中逗號密度，無指令",
    },
    "M_embedded_instruction": {
        "val": PROMPT_M,
        "note": f"嵌入式指令 ({len(PROMPT_M)} chars)，自然地提到「全形標點符號」",
    },
    "N_extra_long": {
        "val": PROMPT_N,
        "note": f"超長前文 ({len(PROMPT_N)} chars)，測試長前文是否更穩定",
    },
}

# ── Experiment 2: hotwords & suppress_tokens ──────────────────────────────────
# Token 11 = ',' (half-width comma) — single token, easy to suppress
# hotwords="，" biases decoder toward full-width comma tokens
# Use baseline short prompt (A) to isolate the effect of these parameters

BASELINE_PROMPT = "繁體中文，台灣用語。"
BEST_V2_PROMPT = "繁體中文，台灣用語。這樣可以嗎？當然沒問題，我們繼續。太好了，開始吧。"  # K

CONFIGS_EXP2 = {
    "A_baseline": {
        "prompt": BASELINE_PROMPT,
        "hotwords": None,
        "suppress": None,
        "note": "對照組：短 prompt，無 hotwords/suppress",
    },
    "A+hotwords_comma": {
        "prompt": BASELINE_PROMPT,
        "hotwords": "，",
        "suppress": None,
        "note": "短 prompt + hotwords 偏置全形逗號",
    },
    "A+suppress_half_comma": {
        "prompt": BASELINE_PROMPT,
        "hotwords": None,
        "suppress": [11],
        "note": "短 prompt + suppress 半形逗號 (token 11)",
    },
    "A+hotwords+suppress": {
        "prompt": BASELINE_PROMPT,
        "hotwords": "，",
        "suppress": [11],
        "note": "短 prompt + hotwords + suppress 雙管齊下",
    },
    "K+suppress_half_comma": {
        "prompt": BEST_V2_PROMPT,
        "hotwords": None,
        "suppress": [11],
        "note": "v2 最佳 prompt K + suppress 半形逗號",
    },
    "K+hotwords+suppress": {
        "prompt": BEST_V2_PROMPT,
        "hotwords": "，",
        "suppress": [11],
        "note": "v2 最佳 prompt K + hotwords + suppress",
    },
}

# ── Shared analysis ───────────────────────────────────────────────────────────

HALF_PUNCT = {",": "半形逗號", ".": "半形句號", "?": "半形問號", "!": "半形驚嘆", ":": "半形冒號", ";": "半形分號"}
FULL_PUNCT = {"，": "全形逗號", "。": "全形句號", "？": "全形問號", "！": "全形驚嘆", "：": "全形冒號", "；": "全形分號", "、": "頓號"}


def count_punct(text):
    result = {}
    for ch, name in {**HALF_PUNCT, **FULL_PUNCT}.items():
        cnt = text.count(ch)
        if cnt > 0:
            result[ch] = {"count": cnt, "name": name, "width": "half" if ch in HALF_PUNCT else "full"}
    return result


def detect_simplified(text):
    pairs = [
        ("们", "們"), ("这", "這"), ("来", "來"), ("对", "對"),
        ("时", "時"), ("会", "會"), ("还", "還"), ("过", "過"),
        ("进", "進"), ("让", "讓"), ("说", "說"), ("发", "發"),
        ("开", "開"), ("关", "關"), ("图", "圖"), ("运", "運"),
        ("经", "經"), ("设", "設"), ("为", "為"), ("与", "與"),
    ]
    found = []
    for simp, trad in pairs:
        cnt = text.count(simp)
        if cnt:
            found.append({"char": simp, "traditional": trad, "count": cnt})
    return found


def analyze(text):
    punct = count_punct(text)
    simplified = detect_simplified(text)
    half_cnt = sum(v["count"] for v in punct.values() if v["width"] == "half")
    full_cnt = sum(v["count"] for v in punct.values() if v["width"] == "full")
    return {
        "text": text,
        "char_count": len(text),
        "punct": punct,
        "half_total": half_cnt,
        "full_total": full_cnt,
        "simplified": simplified,
    }


def transcribe_segment(model, audio, prompt, hotwords=None, suppress=None):
    kwargs = dict(language="zh", beam_size=5, initial_prompt=prompt)
    if hotwords:
        kwargs["hotwords"] = hotwords
    if suppress:
        kwargs["suppress_tokens"] = suppress
    segs, _ = model.transcribe(audio.astype(np.float32), **kwargs)
    return "".join(s.text for s in segs).strip()


def run_experiment(model, segments, configs, exp_name):
    print(f"\n{'#'*80}")
    print(f"#  {exp_name}")
    print(f"{'#'*80}")

    all_results = {}

    for config_name, cfg in configs.items():
        prompt = cfg.get("val") or cfg.get("prompt")
        hotwords = cfg.get("hotwords")
        suppress = cfg.get("suppress")
        note = cfg["note"]

        print(f"\n{'='*80}")
        print(f"  {config_name}")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        if hotwords:
            print(f"  Hotwords: {hotwords}")
        if suppress:
            print(f"  Suppress tokens: {suppress}")
        print(f"  Note: {note}")
        print(f"{'='*80}")

        config_results = {}
        total_half = 0
        total_full = 0

        for seg_name, audio in segments.items():
            text = transcribe_segment(model, audio, prompt, hotwords, suppress)
            result = analyze(text)
            config_results[seg_name] = result
            total_half += result["half_total"]
            total_full += result["full_total"]

            simp_str = ""
            if result["simplified"]:
                simp_str = " | 簡體: " + ", ".join(
                    f"{s['char']}→{s['traditional']}x{s['count']}" for s in result["simplified"]
                )

            print(f"\n  [{seg_name}]")
            print(f"  字數:{result['char_count']} 半形:{result['half_total']} 全形:{result['full_total']}{simp_str}")
            print(f"  {text[:150]}{'...' if len(text) > 150 else ''}")

        total = total_half + total_full
        pct = total_full / total * 100 if total else 0
        print(f"\n  >>> 全形率: {pct:.1f}% (半形:{total_half} 全形:{total_full})")

        all_results[config_name] = {
            "prompt": prompt,
            "hotwords": hotwords,
            "suppress_tokens": suppress,
            "note": note,
            "segments": config_results,
            "total_half": total_half,
            "total_full": total_full,
            "fullwidth_pct": round(pct, 1),
        }

    return all_results


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

    # Exp 1: prompt designs
    exp1_configs = {k: {"val": v["val"], "note": v["note"]} for k, v in PROMPTS_EXP1.items()}
    results_exp1 = run_experiment(model, segments, exp1_configs, "Experiment 1: Prompt Designs")

    # Exp 2: hotwords & suppress_tokens
    results_exp2 = run_experiment(model, segments, CONFIGS_EXP2, "Experiment 2: Hotwords & Suppress Tokens")

    # Save all
    combined = {"exp1_prompts": results_exp1, "exp2_params": results_exp2}
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n完整結果已存: {OUT_FILE}")

    # Summary
    print(f"\n{'='*80}")
    print("=== 總結 ===")
    print(f"{'='*80}")
    print("\nExp 1 — Prompt Designs:")
    for name, r in results_exp1.items():
        print(f"  {name:<30} 全形率:{r['fullwidth_pct']:>5.1f}%  半:{r['total_half']:>2}  全:{r['total_full']:>2}")
    print("\nExp 2 — Hotwords & Suppress:")
    for name, r in results_exp2.items():
        print(f"  {name:<30} 全形率:{r['fullwidth_pct']:>5.1f}%  半:{r['total_half']:>2}  全:{r['total_full']:>2}")


if __name__ == "__main__":
    main()
