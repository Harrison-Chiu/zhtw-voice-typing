"""Phase 3b: Shallow Fusion — Bigram Prior LogitsProcessor

Phase 3a 發現純語言特徵（bigram 先驗）AUC=0.824，能獨立預測逗號位置。
本實驗驗證：將 bigram 先驗作為外部語言信號注入 LogitsProcessor。

公式（shallow fusion）：
  boosted_logit[comma] = original_logit[comma] + alpha * log(P_bigram(comma | prev_token))

假說：
- bigram 先驗提供精準的「何時 boost」信號（只在語言上適合的位置 boost）
- 預期比 Phase 1 的固定規則更精準：相同改善率下副作用更低
- 看到（改善>50%, 副作用<10%）→ shallow fusion 有效，可整合到產品
- 沒看到 → bigram 先驗不夠強或 alpha 需更精細調校

參數：
- alpha: boost 強度（1.0, 2.0, 3.0, 5.0）— 控制先驗的影響力
- min_prior: 先驗閾值（0.1, 0.2, 0.3）— 低於此不 boost（避免噪音）
- chars_gate: 最少字元數（0, 5, 8）— 剛開始不 boost（避免句首誤觸發）
"""

import json
import math
import re
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.stdout.reconfigure(encoding="utf-8")

SEGMENTS_DIR = Path("data/logs/sessions")
PROBE_PATH = Path("experiments/results/comma_logit_probe.json")
OUTPUT_PATH = Path("experiments/results/experiment_shallow_fusion.json")

TID_COMMA_HW = 11
PUNCT_TIDS = {11, 1543, 1231}
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def build_bigram_priors():
    """Build P(comma | prev_token) from logit probe data (normal segments only)."""
    with open(PROBE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["logit_probe"]
    bigram_before_comma = Counter()
    bigram_total = Counter()

    for seg in segments:
        if seg["commas"] == 0 and seg["cjk_chars"] > 20:
            continue
        steps = seg["steps"]
        for i in range(1, len(steps)):
            prev_tok = steps[i - 1]["actual_token"]
            bigram_total[prev_tok] += 1
            if steps[i]["actual_tid"] == TID_COMMA_HW:
                bigram_before_comma[prev_tok] += 1

    priors = {}
    for tok, count in bigram_total.items():
        if count >= 3:
            priors[tok] = bigram_before_comma.get(tok, 0) / count

    return priors


class ShallowFusionCommaProcessor(LogitsProcessor):
    """Bigram-prior-guided comma boost."""

    def __init__(self, tokenizer, priors: dict, alpha: float,
                 min_prior: float = 0.1, chars_gate: int = 5):
        self.tokenizer = tokenizer
        self.priors = priors
        self.alpha = alpha
        self.min_prior = min_prior
        self.chars_gate = chars_gate
        self.prev_token_str = ""
        self.chars_since_punct = 0

    def reset(self):
        self.prev_token_str = ""
        self.chars_since_punct = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        last_tid = input_ids[0, -1].item()
        if last_tid >= 50000:
            return scores

        last_token = self.tokenizer.decode([last_tid])
        is_punct = last_tid in PUNCT_TIDS
        is_cjk = bool(CJK_RE.search(last_token))

        if is_punct:
            self.chars_since_punct = 0
        elif is_cjk:
            self.chars_since_punct += len(CJK_RE.findall(last_token))

        # Lookup bigram prior for previous token
        prior = self.priors.get(self.prev_token_str, 0.0)

        if prior >= self.min_prior and self.chars_since_punct >= self.chars_gate:
            boost = self.alpha * math.log(prior + 1e-10) - self.alpha * math.log(1 - prior + 1e-10)
            # Clamp to avoid extreme values
            boost = max(0, min(boost, 8.0))
            scores[0, TID_COMMA_HW] += boost

        self.prev_token_str = last_token
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


def run_experiment():
    # Build priors
    print("建立 bigram 先驗...", flush=True)
    priors = build_bigram_priors()
    top_priors = sorted(priors.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"共 {len(priors)} 個 token 有先驗")
    print("Top tokens:")
    for tok, p in top_priors:
        print(f"  '{tok}' → {p:.3f}")

    # Load model
    print("\n載入模型...", flush=True)
    t0 = time.time()
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo", dtype=torch.float16
    ).to("cuda")
    model.eval()
    tokenizer = proc.tokenizer
    prompt_ids = proc.get_prompt_ids("繁體中文，台灣用語。", return_tensors="pt").to("cuda")
    print(f"模型載入: {time.time() - t0:.1f}s", flush=True)

    # Load audio
    segments = find_all_segments()
    audios = {}
    for wav in segments:
        audio, sr = sf.read(wav, dtype="float32")
        if sr == 16000 and len(audio) / 16000 >= 1.0:
            audios[str(wav)] = audio
    valid_segments = [s for s in segments if str(s) in audios]
    print(f"有效段落: {len(valid_segments)}", flush=True)

    # Baseline
    print("\n--- Baseline ---", flush=True)
    baseline_texts = {}
    for i, wav in enumerate(valid_segments):
        audio = audios[str(wav)]
        input_features = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to(
            "cuda", dtype=torch.float16
        )
        with torch.no_grad():
            out = model.generate(
                input_features, language="zh", prompt_ids=prompt_ids,
                max_new_tokens=420, num_beams=1,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if "繁體中文，台灣用語。" in text:
            text = text.split("繁體中文，台灣用語。")[-1].strip()
        baseline_texts[str(wav)] = text
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(valid_segments)}]", flush=True)

    # Classify segments
    problem_segs = []
    normal_segs = []
    for wav_str, text in baseline_texts.items():
        cjk = count_cjk(text)
        commas = text.count(",") + text.count("，")
        if cjk > 20 and commas == 0:
            problem_segs.append(wav_str)
        elif commas > 0:
            normal_segs.append(wav_str)
    print(f"問題段: {len(problem_segs)}, 正常段: {len(normal_segs)}", flush=True)

    # Parameter grid — focused on alpha and min_prior
    configs = []
    for alpha in [1.0, 2.0, 3.0, 5.0]:
        for min_prior in [0.1, 0.2, 0.3]:
            for chars_gate in [0, 5]:
                configs.append((alpha, min_prior, chars_gate))

    print(f"\n--- {len(configs)} 組參數 ---", flush=True)

    fusion_proc = ShallowFusionCommaProcessor(tokenizer, priors, 1.0)
    all_results = []

    for cfg_idx, (alpha, min_prior, chars_gate) in enumerate(configs):
        fusion_proc.alpha = alpha
        fusion_proc.min_prior = min_prior
        fusion_proc.chars_gate = chars_gate

        cfg_name = f"a{alpha}_mp{min_prior}_cg{chars_gate}"
        t_start = time.time()

        problem_improved = 0
        normal_text_changed = 0
        normal_comma_added = 0
        seg_details = []

        for wav in valid_segments:
            wav_str = str(wav)
            if wav_str not in baseline_texts:
                continue

            audio = audios[wav_str]
            input_features = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to(
                "cuda", dtype=torch.float16
            )

            fusion_proc.reset()
            with torch.no_grad():
                out = model.generate(
                    input_features, language="zh", prompt_ids=prompt_ids,
                    max_new_tokens=420, num_beams=1,
                    logits_processor=LogitsProcessorList([fusion_proc]),
                )
            new_text = tokenizer.decode(out[0], skip_special_tokens=True)
            if "繁體中文，台灣用語。" in new_text:
                new_text = new_text.split("繁體中文，台灣用語。")[-1].strip()

            base_text = baseline_texts[wav_str]
            base_commas = base_text.count(",") + base_text.count("，")
            new_commas = new_text.count(",") + new_text.count("，")
            is_problem = wav_str in problem_segs
            is_normal = wav_str in normal_segs

            if is_problem and new_commas > 0:
                problem_improved += 1
            if is_normal and new_text != base_text:
                normal_text_changed += 1
            if is_normal and new_commas > base_commas:
                normal_comma_added += 1

            seg_details.append({
                "wav": wav_str,
                "category": "problem" if is_problem else ("normal" if is_normal else "other"),
                "baseline_commas": base_commas,
                "new_commas": new_commas,
                "comma_delta": new_commas - base_commas,
                "content_match": new_text == base_text,
                "baseline_text": base_text,
                "new_text": new_text,
            })

        elapsed = time.time() - t_start
        result = {
            "config": cfg_name,
            "alpha": alpha,
            "min_prior": min_prior,
            "chars_gate": chars_gate,
            "summary": {
                "problem_improved": problem_improved,
                "problem_total": len(problem_segs),
                "problem_rate": problem_improved / max(len(problem_segs), 1),
                "normal_text_changed": normal_text_changed,
                "normal_total": len(normal_segs),
                "normal_text_rate": normal_text_changed / max(len(normal_segs), 1),
                "normal_comma_added": normal_comma_added,
                "normal_comma_rate": normal_comma_added / max(len(normal_segs), 1),
            },
            "segments": seg_details,
        }
        all_results.append(result)

        imp_pct = problem_improved / max(len(problem_segs), 1) * 100
        se_pct = normal_comma_added / max(len(normal_segs), 1) * 100
        txt_pct = normal_text_changed / max(len(normal_segs), 1) * 100
        print(f"  [{cfg_idx+1}/{len(configs)}] {cfg_name}: "
              f"問題改善={problem_improved}/{len(problem_segs)} ({imp_pct:.1f}%) | "
              f"正常文字變={normal_text_changed}/{len(normal_segs)} ({txt_pct:.1f}%) | "
              f"正常多逗號={normal_comma_added} ({se_pct:.1f}%) | {elapsed:.0f}s",
              flush=True)

    # Summary
    print("\n" + "=" * 70)
    print("摘要（按改善率排序）")
    print("=" * 70)
    print(f"{'config':30s}  {'問題改善%':>10}  {'正常文字變%':>12}  {'正常多逗號%':>12}")
    print("-" * 70)
    sorted_results = sorted(all_results, key=lambda r: r["summary"]["problem_rate"], reverse=True)
    for r in sorted_results:
        s = r["summary"]
        print(f"{r['config']:30s}  {s['problem_rate']*100:8.1f}%  {s['normal_text_rate']*100:10.1f}%  "
              f"{s['normal_comma_rate']*100:10.1f}%")

    # Find best tradeoff
    print("\n--- 最佳 tradeoff ---")
    best = None
    best_score = -1
    for r in all_results:
        s = r["summary"]
        score = s["problem_rate"] - 2.0 * s["normal_comma_rate"]
        if score > best_score:
            best_score = score
            best = r
    if best:
        s = best["summary"]
        print(f"Config: {best['config']}")
        print(f"問題改善: {s['problem_rate']*100:.1f}%")
        print(f"正常多逗號: {s['normal_comma_rate']*100:.1f}%")

    # Compare with Phase 1 best
    print("\n--- vs Phase 1 (t15_a0.2) ---")
    print(f"Phase 1: 改善 51.4%, 副作用 21.7%")
    if best:
        s = best["summary"]
        print(f"Phase 3b: 改善 {s['problem_rate']*100:.1f}%, 副作用 {s['normal_comma_rate']*100:.1f}%")

    # Save
    output = {
        "meta": {
            "method": "shallow_fusion_bigram_prior",
            "date": "2026-07-01",
            "total_segments": len(valid_segments),
            "problem_segments": len(problem_segs),
            "normal_segments": len(normal_segs),
            "num_configs": len(configs),
            "num_priors": len(priors),
        },
        "priors_top50": dict(top_priors[:50]) if len(top_priors) >= 50 else dict(top_priors),
        "results": [
            {k: v for k, v in r.items() if k != "segments"}
            for r in sorted_results
        ],
        "results_with_text": sorted_results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_experiment()
