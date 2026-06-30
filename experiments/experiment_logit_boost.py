"""Logit 調節實驗：測試不同參數組合對逗號輸出的影響。

參數網格：
- threshold: 距上次標點多少字後開始 boost（8, 10, 12, 15）
- alpha: 每多一字增加多少 logit boost（0.1, 0.15, 0.2, 0.3, 0.5）
- max_boost: boost 上限（2.0, 3.0, 5.0）

對全部 session log 段落跑每一組參數，比較：
- 問題段（原本無逗號 >20字）的逗號改善率
- 正常段的誤插率（文字變化率）
- 整體品質指標

輸出：experiments/results/experiment_logit_boost.json
"""

import json
import re
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.stdout.reconfigure(encoding="utf-8")

SEGMENTS_DIR = Path("data/logs/sessions")
OUTPUT_PATH = Path("experiments/results/experiment_logit_boost.json")

TID_COMMA_HW = 11
PUNCT_TIDS = {11, 1543, 1231}  # , 。 、
# CJK Unicode ranges for detection
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


class CommaBoostProcessor(LogitsProcessor):
    """根據距上次標點的 CJK 字數，對半形逗號 token 加 logit boost。"""

    def __init__(self, tokenizer, threshold: int, alpha: float, max_boost: float):
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.alpha = alpha
        self.max_boost = max_boost
        self.chars_since_punct = 0
        self.prev_was_cjk = False

    def reset(self):
        self.chars_since_punct = 0
        self.prev_was_cjk = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        last_tid = input_ids[0, -1].item()

        if last_tid >= 50000:
            return scores

        last_token = self.tokenizer.decode([last_tid])
        is_punct = last_tid in PUNCT_TIDS
        is_cjk = bool(CJK_RE.search(last_token))

        if is_punct:
            self.chars_since_punct = 0
            self.prev_was_cjk = False
        elif is_cjk:
            cjk_count = len(CJK_RE.findall(last_token))
            self.chars_since_punct += cjk_count
            self.prev_was_cjk = True
        else:
            self.prev_was_cjk = False

        if self.prev_was_cjk and self.chars_since_punct > self.threshold:
            excess = self.chars_since_punct - self.threshold
            boost = min(self.alpha * excess, self.max_boost)
            scores[0, TID_COMMA_HW] += boost

        return scores


def find_all_segments():
    segs = []
    for session_dir in sorted(SEGMENTS_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        for wav in sorted(session_dir.glob("seg_*.wav")):
            segs.append(wav)
    return segs


def count_cjk(text: str) -> int:
    return len(CJK_RE.findall(text))


def strip_punct(text: str) -> str:
    return re.sub(r"[，,。、！？!?\s]", "", text)


def run_experiment():
    print("載入模型...", flush=True)
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo", dtype=torch.float16
    ).to("cuda")
    model.eval()
    tokenizer = processor.tokenizer
    prompt_ids = processor.get_prompt_ids("繁體中文，台灣用語。", return_tensors="pt").to("cuda")
    print(f"模型載入: {time.time() - t0:.1f}s", flush=True)

    segments = find_all_segments()
    print(f"找到 {len(segments)} 段", flush=True)

    # 預載音檔
    audios = {}
    for wav in segments:
        audio, sr = sf.read(wav, dtype="float32")
        if sr == 16000 and len(audio) / 16000 >= 1.0:
            audios[str(wav)] = audio

    valid_segments = [s for s in segments if str(s) in audios]
    print(f"有效段落: {len(valid_segments)}", flush=True)

    # --- Baseline (no boost) ---
    print("\n--- Baseline (no boost) ---", flush=True)
    baseline_texts = {}
    for i, wav in enumerate(valid_segments):
        audio = audios[str(wav)]
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.float16)

        with torch.no_grad():
            out = model.generate(
                input_features,
                language="zh",
                prompt_ids=prompt_ids,
                max_new_tokens=420,
                num_beams=1,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if "繁體中文，台灣用語。" in text:
            text = text.split("繁體中文，台灣用語。")[-1].strip()
        baseline_texts[str(wav)] = text

        if (i + 1) % 20 == 0:
            print(f"  baseline [{i+1}/{len(valid_segments)}]", flush=True)

    print(f"Baseline 完成: {len(baseline_texts)} 段", flush=True)

    # 分類 baseline
    problem_segs = []
    normal_segs = []
    for wav_str, text in baseline_texts.items():
        cjk = count_cjk(text)
        commas = text.count(",") + text.count("，")
        if cjk > 20 and commas == 0:
            problem_segs.append(wav_str)
        elif commas > 0:
            normal_segs.append(wav_str)

    print(f"問題段(>20字無逗號): {len(problem_segs)}, 正常段: {len(normal_segs)}", flush=True)

    # --- 參數網格 ---
    thresholds = [8, 10, 12, 15]
    alphas = [0.1, 0.15, 0.2, 0.3, 0.5]
    max_boosts = [2.0, 3.0, 5.0]

    # 減少組合：先固定 max_boost=3.0 跑 threshold × alpha，
    # 找到最佳 threshold+alpha 後再測不同 max_boost
    phase1_configs = [(t, a, 3.0) for t, a in product(thresholds, alphas)]
    print(f"\nPhase 1: {len(phase1_configs)} 組參數", flush=True)

    all_results = []
    boost_processor = CommaBoostProcessor(tokenizer, 12, 0.2, 3.0)

    for cfg_idx, (threshold, alpha, max_boost) in enumerate(phase1_configs):
        boost_processor.threshold = threshold
        boost_processor.alpha = alpha
        boost_processor.max_boost = max_boost

        cfg_name = f"t{threshold}_a{alpha}_m{max_boost}"
        cfg_results = {
            "config": cfg_name,
            "threshold": threshold,
            "alpha": alpha,
            "max_boost": max_boost,
            "segments": [],
        }

        # 統計
        problem_gained_comma = 0
        normal_text_changed = 0
        normal_comma_added = 0
        total_problem = 0
        total_normal = 0

        t0 = time.time()
        for wav in valid_segments:
            wav_str = str(wav)
            audio = audios[wav_str]
            base_text = baseline_texts[wav_str]

            input_features = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.float16)

            boost_processor.reset()
            with torch.no_grad():
                out = model.generate(
                    input_features,
                    language="zh",
                    prompt_ids=prompt_ids,
                    max_new_tokens=420,
                    num_beams=1,
                    logits_processor=LogitsProcessorList([boost_processor]),
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            if "繁體中文，台灣用語。" in text:
                text = text.split("繁體中文，台灣用語。")[-1].strip()

            base_commas = base_text.count(",") + base_text.count("，")
            new_commas = text.count(",") + text.count("，")
            base_content = strip_punct(base_text)
            new_content = strip_punct(text)
            content_match = base_content == new_content

            is_problem = wav_str in problem_segs
            is_normal = wav_str in normal_segs

            if is_problem:
                total_problem += 1
                if new_commas > 0:
                    problem_gained_comma += 1
            elif is_normal:
                total_normal += 1
                if not content_match:
                    normal_text_changed += 1
                if new_commas > base_commas:
                    normal_comma_added += 1

            seg_result = {
                "wav": wav_str,
                "category": "problem" if is_problem else ("normal" if is_normal else "other"),
                "baseline_commas": base_commas,
                "new_commas": new_commas,
                "comma_delta": new_commas - base_commas,
                "content_match": content_match,
                "baseline_text": base_text,
                "new_text": text,
            }
            cfg_results["segments"].append(seg_result)

        dt = time.time() - t0

        cfg_results["summary"] = {
            "problem_total": total_problem,
            "problem_gained_comma": problem_gained_comma,
            "problem_success_rate": round(problem_gained_comma / max(total_problem, 1) * 100, 1),
            "normal_total": total_normal,
            "normal_text_changed": normal_text_changed,
            "normal_text_change_rate": round(normal_text_changed / max(total_normal, 1) * 100, 1),
            "normal_comma_added": normal_comma_added,
            "normal_comma_add_rate": round(normal_comma_added / max(total_normal, 1) * 100, 1),
            "time_sec": round(dt, 1),
        }

        s = cfg_results["summary"]
        print(
            f"  [{cfg_idx+1}/{len(phase1_configs)}] {cfg_name}: "
            f"問題段改善={s['problem_gained_comma']}/{s['problem_total']} "
            f"({s['problem_success_rate']}%) | "
            f"正常段文字變={s['normal_text_changed']}/{s['normal_total']} "
            f"({s['normal_text_change_rate']}%) | "
            f"正常段多逗號={s['normal_comma_added']} "
            f"({s['normal_comma_add_rate']}%) | "
            f"{dt:.0f}s",
            flush=True,
        )

        all_results.append(cfg_results)

    # --- Phase 2: 最佳 threshold+alpha 下測不同 max_boost ---
    # 找 Phase 1 最佳組合（問題改善率高 + 正常變化率低）
    best_cfg = max(
        all_results,
        key=lambda r: (
            r["summary"]["problem_success_rate"]
            - r["summary"]["normal_text_change_rate"] * 2
            - r["summary"]["normal_comma_add_rate"] * 0.5
        ),
    )
    best_t = best_cfg["threshold"]
    best_a = best_cfg["alpha"]
    print(f"\nPhase 1 最佳: threshold={best_t}, alpha={best_a}", flush=True)

    phase2_configs = [(best_t, best_a, mb) for mb in max_boosts if mb != 3.0]
    print(f"Phase 2: 測 max_boost {[mb for mb in max_boosts if mb != 3.0]}", flush=True)

    for cfg_idx, (threshold, alpha, max_boost) in enumerate(phase2_configs):
        boost_processor.threshold = threshold
        boost_processor.alpha = alpha
        boost_processor.max_boost = max_boost
        cfg_name = f"t{threshold}_a{alpha}_m{max_boost}"

        cfg_results = {
            "config": cfg_name,
            "threshold": threshold,
            "alpha": alpha,
            "max_boost": max_boost,
            "segments": [],
        }

        problem_gained_comma = 0
        normal_text_changed = 0
        normal_comma_added = 0
        total_problem = 0
        total_normal = 0

        t0 = time.time()
        for wav in valid_segments:
            wav_str = str(wav)
            audio = audios[wav_str]
            base_text = baseline_texts[wav_str]

            input_features = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.float16)

            boost_processor.reset()
            with torch.no_grad():
                out = model.generate(
                    input_features,
                    language="zh",
                    prompt_ids=prompt_ids,
                    max_new_tokens=420,
                    num_beams=1,
                    logits_processor=LogitsProcessorList([boost_processor]),
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            if "繁體中文，台灣用語。" in text:
                text = text.split("繁體中文，台灣用語。")[-1].strip()

            base_commas = base_text.count(",") + base_text.count("，")
            new_commas = text.count(",") + text.count("，")
            base_content = strip_punct(base_text)
            new_content = strip_punct(text)
            content_match = base_content == new_content

            is_problem = wav_str in problem_segs
            is_normal = wav_str in normal_segs

            if is_problem:
                total_problem += 1
                if new_commas > 0:
                    problem_gained_comma += 1
            elif is_normal:
                total_normal += 1
                if not content_match:
                    normal_text_changed += 1
                if new_commas > base_commas:
                    normal_comma_added += 1

            cfg_results["segments"].append({
                "wav": wav_str,
                "category": "problem" if is_problem else ("normal" if is_normal else "other"),
                "baseline_commas": base_commas,
                "new_commas": new_commas,
                "comma_delta": new_commas - base_commas,
                "content_match": content_match,
                "baseline_text": base_text,
                "new_text": text,
            })

        dt = time.time() - t0
        cfg_results["summary"] = {
            "problem_total": total_problem,
            "problem_gained_comma": problem_gained_comma,
            "problem_success_rate": round(problem_gained_comma / max(total_problem, 1) * 100, 1),
            "normal_total": total_normal,
            "normal_text_changed": normal_text_changed,
            "normal_text_change_rate": round(normal_text_changed / max(total_normal, 1) * 100, 1),
            "normal_comma_added": normal_comma_added,
            "normal_comma_add_rate": round(normal_comma_added / max(total_normal, 1) * 100, 1),
            "time_sec": round(dt, 1),
        }

        s = cfg_results["summary"]
        print(
            f"  [{cfg_idx+1}/{len(phase2_configs)}] {cfg_name}: "
            f"問題段改善={s['problem_gained_comma']}/{s['problem_total']} "
            f"({s['problem_success_rate']}%) | "
            f"正常段文字變={s['normal_text_changed']}/{s['normal_total']} "
            f"({s['normal_text_change_rate']}%) | "
            f"正常段多逗號={s['normal_comma_added']} "
            f"({s['normal_comma_add_rate']}%) | "
            f"{dt:.0f}s",
            flush=True,
        )

        all_results.append(cfg_results)

    # --- 儲存 ---
    output = {
        "meta": {
            "model": "openai/whisper-large-v3-turbo",
            "date": "2026-07-01",
            "total_segments": len(valid_segments),
            "problem_segments": len(problem_segs),
            "normal_segments": len(normal_segs),
            "phase1_configs": len(phase1_configs),
            "phase2_configs": len(phase2_configs),
        },
        "baseline_summary": {
            "problem_segs": problem_segs,
            "normal_segs": normal_segs,
        },
        "results": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存: {OUTPUT_PATH}", flush=True)

    # --- 最終摘要 ---
    print("\n" + "=" * 80)
    print("最終摘要")
    print("=" * 80)
    print(f"{'config':<25} {'問題改善%':>8} {'正常文字變%':>10} {'正常多逗號%':>10}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x["summary"]["problem_success_rate"], reverse=True):
        s = r["summary"]
        print(
            f"{r['config']:<25} {s['problem_success_rate']:>7.1f}% "
            f"{s['normal_text_change_rate']:>9.1f}% "
            f"{s['normal_comma_add_rate']:>9.1f}%"
        )


if __name__ == "__main__":
    run_experiment()
