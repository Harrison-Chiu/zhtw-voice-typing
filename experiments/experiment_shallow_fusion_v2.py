"""Phase 3b-v2: Shallow Fusion — Fixed boost formula + Combined model

v1 bug: log-odds formula makes boost=0 for prior < 0.5, filtering out most tokens.
v2 fix: boost = alpha * prior (direct scaling), and test combined model:
  boost = alpha * prior + beta * overdue_factor

Combined model 假說：
- chars_since_punct 提供「已經太久沒逗號」的信號（Phase 1 的有效部分）
- bigram prior 提供「這裡語言上適合逗號」的信號（Phase 3a 發現）
- 兩者相乘/相加比單獨使用更精準
- 預期：相同改善率下副作用更低（因為兩個條件都要滿足才 boost）
"""

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.stdout.reconfigure(encoding="utf-8")

SEGMENTS_DIR = Path("data/logs/sessions")
PROBE_PATH = Path("experiments/results/comma_logit_probe.json")
OUTPUT_PATH = Path("experiments/results/experiment_shallow_fusion_v2.json")

TID_COMMA_HW = 11
PUNCT_TIDS = {11, 1543, 1231}
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def build_bigram_priors():
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


class CombinedFusionProcessor(LogitsProcessor):
    """Combined: bigram prior * overdue factor."""

    def __init__(self, tokenizer, priors, alpha, beta, min_prior, chars_gate):
        self.tokenizer = tokenizer
        self.priors = priors
        self.alpha = alpha  # prior weight
        self.beta = beta    # overdue weight
        self.min_prior = min_prior
        self.chars_gate = chars_gate
        self.prev_token_str = ""
        self.chars_since_punct = 0

    def reset(self):
        self.prev_token_str = ""
        self.chars_since_punct = 0

    def __call__(self, input_ids, scores):
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

        prior = self.priors.get(self.prev_token_str, 0.0)
        overdue = max(0, self.chars_since_punct - self.chars_gate)

        # Three modes based on alpha/beta:
        # 1. alpha>0, beta=0: pure prior (boost = alpha * prior)
        # 2. alpha=0, beta>0: pure overdue (boost = beta * overdue) — same as Phase 1
        # 3. alpha>0, beta>0: combined (boost = alpha * prior * (1 + beta * overdue))
        #    Prior gates the boost, overdue amplifies it

        if self.alpha > 0 and self.beta > 0:
            # Combined: prior gates, overdue amplifies
            if prior >= self.min_prior and overdue > 0:
                boost = self.alpha * prior * (1.0 + self.beta * overdue)
                boost = min(boost, 8.0)
                scores[0, TID_COMMA_HW] += boost
        elif self.alpha > 0:
            # Pure prior
            if prior >= self.min_prior:
                boost = self.alpha * prior
                boost = min(boost, 8.0)
                scores[0, TID_COMMA_HW] += boost
        elif self.beta > 0:
            # Pure overdue
            if overdue > 0:
                boost = self.beta * overdue
                boost = min(boost, 8.0)
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


def count_cjk(text):
    return len(CJK_RE.findall(text))


def run_experiment():
    print("建立 bigram 先驗...", flush=True)
    priors = build_bigram_priors()
    high_prior = {k: v for k, v in priors.items() if v >= 0.1}
    print(f"共 {len(priors)} tokens, {len(high_prior)} with prior >= 0.1")

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
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
        with torch.no_grad():
            out = model.generate(feats, language="zh", prompt_ids=prompt_ids, max_new_tokens=420, num_beams=1)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if "繁體中文，台灣用語。" in text:
            text = text.split("繁體中文，台灣用語。")[-1].strip()
        baseline_texts[str(wav)] = text
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(valid_segments)}]", flush=True)

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

    # Configs — three groups:
    # A. Pure prior (fixed formula, higher alpha)
    # B. Combined (prior gates + overdue amplifies)
    # C. Multiplicative (prior * overdue as score)
    configs = []
    # A: Pure prior with direct scaling
    for alpha in [3.0, 5.0, 8.0, 12.0]:
        configs.append(("prior", alpha, 0.0, 0.1, 0))
    # B: Combined — prior gates, overdue amplifies
    for alpha in [3.0, 5.0, 8.0]:
        for beta in [0.1, 0.2, 0.3]:
            configs.append(("combined", alpha, beta, 0.1, 5))
    # C: Combined with higher chars_gate
    for alpha in [5.0, 8.0]:
        for beta in [0.1, 0.2]:
            configs.append(("combined_g8", alpha, beta, 0.1, 8))

    print(f"\n--- {len(configs)} 組參數 ---", flush=True)

    fusion_proc = CombinedFusionProcessor(tokenizer, priors, 1.0, 0.0, 0.1, 0)
    all_results = []

    for cfg_idx, (mode, alpha, beta, min_prior, chars_gate) in enumerate(configs):
        fusion_proc.alpha = alpha
        fusion_proc.beta = beta
        fusion_proc.min_prior = min_prior
        fusion_proc.chars_gate = chars_gate

        cfg_name = f"{mode}_a{alpha}_b{beta}_cg{chars_gate}"
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
            feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)

            fusion_proc.reset()
            with torch.no_grad():
                out = model.generate(
                    feats, language="zh", prompt_ids=prompt_ids,
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
                "baseline_text": base_text,
                "new_text": new_text,
            })

        elapsed = time.time() - t_start
        result = {
            "config": cfg_name,
            "mode": mode,
            "alpha": alpha,
            "beta": beta,
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

        imp = problem_improved / max(len(problem_segs), 1) * 100
        se = normal_comma_added / max(len(normal_segs), 1) * 100
        tc = normal_text_changed / max(len(normal_segs), 1) * 100
        print(f"  [{cfg_idx+1}/{len(configs)}] {cfg_name}: "
              f"改善={problem_improved}/{len(problem_segs)} ({imp:.1f}%) | "
              f"文字變={normal_text_changed} ({tc:.1f}%) | "
              f"多逗號={normal_comma_added} ({se:.1f}%) | {elapsed:.0f}s",
              flush=True)

    # Summary
    print("\n" + "=" * 80)
    print("摘要")
    print("=" * 80)
    sorted_r = sorted(all_results, key=lambda r: r["summary"]["problem_rate"], reverse=True)
    print(f"{'config':40s}  {'改善%':>8}  {'文字變%':>8}  {'多逗號%':>8}")
    print("-" * 70)
    for r in sorted_r:
        s = r["summary"]
        print(f"{r['config']:40s}  {s['problem_rate']*100:6.1f}%  {s['normal_text_rate']*100:6.1f}%  "
              f"{s['normal_comma_rate']*100:6.1f}%")

    # Best tradeoff
    print("\n--- 最佳 tradeoff (score = 改善 - 2*副作用) ---")
    best = max(all_results, key=lambda r: r["summary"]["problem_rate"] - 2 * r["summary"]["normal_comma_rate"])
    s = best["summary"]
    print(f"Config: {best['config']}")
    print(f"改善: {s['problem_rate']*100:.1f}%, 副作用: {s['normal_comma_rate']*100:.1f}%")
    print(f"\nvs Phase 1 (t15_a0.2): 改善 51.4%, 副作用 21.7%")

    # Sample improved problem segments from best config
    best_segs = best["segments"]
    improved = [s for s in best_segs if s["category"] == "problem" and s["comma_delta"] > 0]
    if improved:
        print(f"\n--- 改善的問題段原文（{best['config']}）---")
        for s in improved[:5]:
            seg_id = '/'.join(s['wav'].replace('\\', '/').split('/')[-2:])
            print(f"\n[{seg_id}]")
            print(f"  原: {s['baseline_text']}")
            print(f"  改: {s['new_text']}")

    # Save (without full text for size)
    output = {
        "meta": {
            "method": "combined_fusion_v2",
            "date": "2026-07-01",
            "problem_segments": len(problem_segs),
            "normal_segments": len(normal_segs),
            "num_configs": len(configs),
            "v1_bug": "log-odds formula zeroed boost for prior<0.5",
        },
        "results": [{k: v for k, v in r.items() if k != "segments"} for r in sorted_r],
        "best_config": {
            "config": best["config"],
            "improved_segments": [
                {"wav": s["wav"], "baseline": s["baseline_text"], "new": s["new_text"]}
                for s in improved
            ] if improved else [],
        },
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_experiment()
