"""實驗：解碼參數與 prompt 對逗號輸出的影響。

測試段落來自 session log（已知有逗號問題的 + 正常的做對照）。
每組參數跑全部段落，比較逗號密度和文字穩定度。
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")

# --- 測試段落 ---
BASE = Path("data/logs/sessions")
TEST_SEGMENTS = [
    # 問題段：完全無逗號
    ("no_comma", BASE / "20260629_093837/seg_000.wav", "73字完全無逗號"),
    ("no_comma", BASE / "20260629_094339/seg_000.wav", "67字無逗號"),
    # 問題段：空格代逗號
    ("space", BASE / "20260630_164252/seg_000.wav", "多空格長句"),
    ("space", BASE / "20260630_151848/seg_003.wav", "短句多空格"),
    # 正常段（對照）
    ("normal", BASE / "20260629_092345/seg_000.wav", "正常有逗號20s"),
    ("normal", BASE / "20260629_092600/seg_000.wav", "正常有逗號15s"),
    ("normal", BASE / "20260629_092623/seg_000.wav", "正常有逗號17s"),
]

# --- 參數組合 ---
CONFIGS = [
    {
        "name": "baseline",
        "initial_prompt": "使用繁體中文，台灣用語。",
        "condition_on_previous_text": True,
    },
    {
        "name": "prompt_more_commas",
        "initial_prompt": "使用繁體中文，正確標點，語句間以逗號分隔。",
        "condition_on_previous_text": True,
    },
    {
        "name": "prompt_demo_commas",
        "initial_prompt": "繁體中文，正確使用逗號，台灣用語。",
        "condition_on_previous_text": True,
    },
    {
        "name": "cond_off",
        "initial_prompt": "使用繁體中文，台灣用語。",
        "condition_on_previous_text": False,
    },
    {
        "name": "cond_off+more_commas",
        "initial_prompt": "使用繁體中文，正確標點，語句間以逗號分隔。",
        "condition_on_previous_text": False,
    },
]

FIXED_PARAMS = {
    "model_id": "large-v3-turbo",
    "device": "cuda",
    "compute_type": "float16",
    "language": "zh",
    "beam_size": 5,
    "hotwords": "詞表 待辦 清單 聲學 標點 主分支 逗號",
}


def count_cjk(text: str) -> int:
    return len(re.findall(r"[一-鿿]", text))


def count_commas(text: str) -> int:
    return text.count(",") + text.count("，")


def comma_density(text: str) -> float:
    cjk = count_cjk(text)
    if cjk == 0:
        return 0.0
    return count_commas(text) / cjk * 100


def text_similarity(a: str, b: str) -> float:
    """粗略的文字相似度：共同字元比例。"""
    a_chars = set(a)
    b_chars = set(b)
    if not a_chars and not b_chars:
        return 1.0
    return len(a_chars & b_chars) / len(a_chars | b_chars)


def main():
    from faster_whisper import WhisperModel

    # 載入模型（只載一次）
    print("載入模型...", flush=True)
    model = WhisperModel(
        FIXED_PARAMS["model_id"],
        device=FIXED_PARAMS["device"],
        compute_type=FIXED_PARAMS["compute_type"],
    )
    print("模型載入完成", flush=True)

    # 預載音檔
    audios = {}
    for cat, wav_path, desc in TEST_SEGMENTS:
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            raise ValueError(f"{wav_path} sample rate {sr} != 16000")
        audios[str(wav_path)] = audio
        print(f"  載入 {wav_path.name} ({len(audio)/sr:.1f}s) — {desc}")

    print(f"\n{'='*80}")
    print(f"開始實驗：{len(CONFIGS)} 組參數 × {len(TEST_SEGMENTS)} 段音檔")
    print(f"{'='*80}\n")

    all_results = []

    # 先跑 baseline 拿參考文字
    baseline_texts = {}

    for config in CONFIGS:
        print(f"\n--- 參數組: {config['name']} ---")
        print(f"  prompt: {config['initial_prompt']}")
        print(f"  condition_on_previous_text: {config['condition_on_previous_text']}")

        for cat, wav_path, desc in TEST_SEGMENTS:
            audio = audios[str(wav_path)]
            t0 = time.time()
            segments, info = model.transcribe(
                audio.astype(np.float32),
                language=FIXED_PARAMS["language"],
                beam_size=FIXED_PARAMS["beam_size"],
                initial_prompt=config["initial_prompt"],
                hotwords=FIXED_PARAMS["hotwords"],
                condition_on_previous_text=config["condition_on_previous_text"],
            )
            raw = "".join(seg.text for seg in segments).strip()
            dt = time.time() - t0

            cjk = count_cjk(raw)
            commas = count_commas(raw)
            density = comma_density(raw)
            spaces = len(re.findall(r"[一-鿿]\s[一-鿿]", raw))

            # 文字穩定度（跟 baseline 比）
            key = str(wav_path)
            if config["name"] == "baseline":
                baseline_texts[key] = raw
                sim = 1.0
            else:
                sim = text_similarity(raw, baseline_texts.get(key, ""))

            result = {
                "config": config["name"],
                "category": cat,
                "segment": wav_path.name,
                "desc": desc,
                "audio_sec": round(len(audio) / 16000, 1),
                "transcribe_sec": round(dt, 2),
                "cjk_chars": cjk,
                "commas": commas,
                "comma_density": round(density, 1),
                "spaces_in_cjk": spaces,
                "similarity_to_baseline": round(sim, 3),
                "raw": raw,
            }
            all_results.append(result)

            tag = "✓" if commas > 0 else "✗"
            print(
                f"  {tag} [{cat}] {wav_path.name} | "
                f"逗號={commas} 密度={density:.1f}% 空格={spaces} "
                f"相似={sim:.2f} {dt:.2f}s"
            )
            if len(raw) <= 120:
                print(f"    {raw}")
            else:
                print(f"    {raw[:120]}...")

    # 儲存結果
    out_path = Path("experiments/results/experiment_comma_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存: {out_path}")

    # 摘要表
    print(f"\n{'='*80}")
    print("摘要：各參數組在問題段的逗號密度")
    print(f"{'='*80}")
    print(f"{'config':<25} {'問題段平均密度':>12} {'正常段平均密度':>12} {'文字穩定度':>10}")
    print("-" * 65)
    for config in CONFIGS:
        cfg_results = [r for r in all_results if r["config"] == config["name"]]
        problem = [r for r in cfg_results if r["category"] in ("no_comma", "space")]
        normal = [r for r in cfg_results if r["category"] == "normal"]
        prob_density = sum(r["comma_density"] for r in problem) / len(problem) if problem else 0
        norm_density = sum(r["comma_density"] for r in normal) / len(normal) if normal else 0
        sim = sum(r["similarity_to_baseline"] for r in cfg_results) / len(cfg_results)
        print(f"{config['name']:<25} {prob_density:>10.1f}% {norm_density:>10.1f}% {sim:>10.2f}")


if __name__ == "__main__":
    main()
