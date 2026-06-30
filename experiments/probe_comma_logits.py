"""Phase 1+2：逗號 logit 探針 + 聲學間隔分析。

對所有 session log 段落，用 transformers 後端取得每步解碼的 logits，
分析逗號 token 的機率分佈和排名。同時用 faster-whisper 取 word timestamps
分析聲學間隔與標點的關聯。

輸出：experiments/results/comma_logit_probe.json
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.stdout.reconfigure(encoding="utf-8")

# --- Token IDs ---
TID_COMMA_HW = 11        # ,
TID_FULLWIDTH_B0 = 171    # first byte of ，and other fullwidth chars
TID_FULLWIDTH_B1 = 120    # second byte of fullwidth punctuation block
TID_FULLWIDTH_COMMA_B2 = 234  # third byte → ，
TID_PERIOD = 1543         # 。
TID_ENUM_COMMA = 1231     # 、
TID_SPACE = 220           # ' '
TID_EOT = 50257           # <|endoftext|>

PUNCT_TOKENS = {TID_COMMA_HW, TID_PERIOD, TID_ENUM_COMMA}

SEGMENTS_DIR = Path("data/logs/sessions")
OUTPUT_PATH = Path("experiments/results/comma_logit_probe.json")


def find_all_segments():
    """找出所有 wav 段落，按 session/seg 排序。"""
    segs = []
    for session_dir in sorted(SEGMENTS_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        for wav in sorted(session_dir.glob("seg_*.wav")):
            segs.append(wav)
    return segs


def count_cjk(text: str) -> int:
    return len(re.findall(r"[一-鿿㐀-䶿]", text))


def run_logit_probe(segments: list[Path]):
    """Phase 1: transformers 後端取 logits。"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("=== Phase 1: Logit Probe ===")
    print("載入 transformers 模型...", flush=True)
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo",
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    print(f"模型載入: {time.time() - t0:.1f}s", flush=True)

    tokenizer = processor.tokenizer

    # prompt_ids: 跟 faster-whisper 設定一致
    prompt_text = "繁體中文，台灣用語。"
    prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt").to("cuda")

    results = []

    for i, wav_path in enumerate(segments):
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            print(f"  SKIP {wav_path} (sr={sr})")
            continue
        audio_sec = len(audio) / 16000
        if audio_sec < 1.0:
            continue

        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.float16)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_features,
                language="zh",
                task="transcribe",
                prompt_ids=prompt_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=420,
                num_beams=1,  # greedy for clean logit analysis
            )
        dt = time.time() - t0

        # sequences 包含 decoder_start + prompt + content tokens
        token_ids = out.sequences[0].cpu().tolist()
        scores = out.scores  # tuple of (vocab_size,) tensors, one per generated step

        # 解碼完整文字
        full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        # 去掉 prompt echo
        if prompt_text in full_text:
            full_text = full_text.split(prompt_text, 1)[-1].strip()

        # 逐步分析 scores
        step_data = []
        for step_idx, score_tensor in enumerate(scores):
            logits = score_tensor[0].float()  # (vocab_size,)
            probs = torch.softmax(logits, dim=-1)

            # 這一步實際選了什麼 token
            # token_ids 的結構: [decoder_start_tokens...] + [generated tokens]
            # scores[i] 對應 generated token i
            # 但 token_ids 包含了 prefix，需要算 offset
            gen_offset = len(token_ids) - len(scores)
            actual_tid = token_ids[gen_offset + step_idx]
            actual_token = tokenizer.decode([actual_tid])

            # 各關注 token 的機率和排名
            def get_prob_rank(tid):
                p = probs[tid].item()
                rank = (probs > probs[tid]).sum().item() + 1
                return p, int(rank)

            hw_prob, hw_rank = get_prob_rank(TID_COMMA_HW)
            fw_b0_prob, fw_b0_rank = get_prob_rank(TID_FULLWIDTH_B0)
            space_prob, space_rank = get_prob_rank(TID_SPACE)
            period_prob, period_rank = get_prob_rank(TID_PERIOD)
            enum_prob, enum_rank = get_prob_rank(TID_ENUM_COMMA)

            # top-5
            top5_probs, top5_ids = torch.topk(probs, 5)
            top5 = [
                {"tid": tid.item(), "token": tokenizer.decode([tid.item()]), "prob": p.item()}
                for tid, p in zip(top5_ids, top5_probs)
            ]

            step_data.append({
                "step": step_idx,
                "actual_tid": actual_tid,
                "actual_token": actual_token,
                "comma_hw": {"prob": hw_prob, "rank": hw_rank},
                "fullwidth_b0": {"prob": fw_b0_prob, "rank": fw_b0_rank},
                "space": {"prob": space_prob, "rank": space_rank},
                "period": {"prob": period_prob, "rank": period_rank},
                "enum_comma": {"prob": enum_prob, "rank": enum_rank},
                "top5": top5,
            })

        cjk_count = count_cjk(full_text)
        comma_count = full_text.count(",") + full_text.count("，")
        density = (comma_count / cjk_count * 100) if cjk_count > 0 else 0

        result = {
            "wav": str(wav_path),
            "session": wav_path.parent.name,
            "seg": wav_path.stem,
            "audio_sec": round(audio_sec, 2),
            "transcribe_sec": round(dt, 2),
            "text": full_text,
            "cjk_chars": cjk_count,
            "commas": comma_count,
            "comma_density_pct": round(density, 1),
            "total_steps": len(scores),
            "steps": step_data,
        }
        results.append(result)

        tag = "✓" if comma_count > 0 else "✗"
        print(
            f"  [{i+1}/{len(segments)}] {tag} {wav_path.parent.name}/{wav_path.name} "
            f"| {audio_sec:.1f}s | 逗號={comma_count} 密度={density:.1f}% "
            f"| {dt:.1f}s | {full_text[:60]}"
        )

    # 釋放 VRAM
    del model
    torch.cuda.empty_cache()

    return results


def run_word_timestamps(segments: list[Path]):
    """Phase 2: faster-whisper word timestamps 取聲學間隔。"""
    from faster_whisper import WhisperModel

    print("\n=== Phase 2: Word Timestamps ===")
    print("載入 faster-whisper...", flush=True)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    results = []
    for i, wav_path in enumerate(segments):
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000 or len(audio) / 16000 < 1.0:
            continue
        audio_sec = len(audio) / 16000

        segs_iter, info = model.transcribe(
            audio.astype(np.float32),
            language="zh",
            beam_size=5,
            initial_prompt="繁體中文，台灣用語。",
            word_timestamps=True,
        )

        words = []
        full_text = ""
        for seg in segs_iter:
            full_text += seg.text
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    })

        # 計算相鄰 word 之間的 gap
        gaps = []
        for j in range(1, len(words)):
            gap_ms = round((words[j]["start"] - words[j - 1]["end"]) * 1000)
            gaps.append({
                "before": words[j - 1]["word"],
                "after": words[j]["word"],
                "gap_ms": gap_ms,
                "position": j,
            })

        cjk_count = count_cjk(full_text)
        comma_count = full_text.count(",") + full_text.count("，")
        density = (comma_count / cjk_count * 100) if cjk_count > 0 else 0

        results.append({
            "wav": str(wav_path),
            "session": wav_path.parent.name,
            "seg": wav_path.stem,
            "audio_sec": round(audio_sec, 2),
            "text": full_text.strip(),
            "cjk_chars": cjk_count,
            "commas": comma_count,
            "comma_density_pct": round(density, 1),
            "words": words,
            "gaps": gaps,
        })

        print(
            f"  [{i+1}/{len(segments)}] {wav_path.parent.name}/{wav_path.name} "
            f"| words={len(words)} gaps={len(gaps)} | {full_text.strip()[:60]}"
        )

    del model
    torch.cuda.empty_cache()
    return results


def analyze_and_summarize(logit_results, timestamp_results):
    """統計分析並印摘要。"""
    print("\n" + "=" * 80)
    print("分析摘要")
    print("=" * 80)

    # --- Logit 分析 ---
    # 把每步分成：實際輸出了逗號 vs 沒輸出逗號
    comma_steps = []      # 模型選了逗號的步驟
    non_comma_steps = []   # 模型沒選逗號的步驟（排除特殊 token、句首等）

    for seg in logit_results:
        for s in seg["steps"]:
            tid = s["actual_tid"]
            if tid >= 50000:  # special tokens
                continue
            is_comma = (tid == TID_COMMA_HW or
                        (tid == TID_FULLWIDTH_B0))  # fullwidth comma starts with 171
            if is_comma:
                comma_steps.append(s)
            else:
                non_comma_steps.append(s)

    print(f"\n逗號步驟: {len(comma_steps)}, 非逗號步驟: {len(non_comma_steps)}")

    if comma_steps:
        avg_comma_prob = np.mean([s["comma_hw"]["prob"] for s in comma_steps])
        avg_comma_rank = np.mean([s["comma_hw"]["rank"] for s in comma_steps])
        print(f"逗號步驟的半形逗號 avg prob: {avg_comma_prob:.4f}, avg rank: {avg_comma_rank:.1f}")

    # 在非逗號步驟中，逗號 token 的分佈
    if non_comma_steps:
        hw_probs = [s["comma_hw"]["prob"] for s in non_comma_steps]
        hw_ranks = [s["comma_hw"]["rank"] for s in non_comma_steps]
        print(f"\n非逗號步驟的半形逗號統計:")
        print(f"  prob: mean={np.mean(hw_probs):.6f}, median={np.median(hw_probs):.6f}, "
              f"max={np.max(hw_probs):.4f}, p95={np.percentile(hw_probs, 95):.6f}")
        print(f"  rank: mean={np.mean(hw_ranks):.0f}, median={np.median(hw_ranks):.0f}, "
              f"min={np.min(hw_ranks)}, p5={np.percentile(hw_ranks, 5):.0f}")

        # 有多少步驟逗號排在前 10?
        top10 = sum(1 for r in hw_ranks if r <= 10)
        top50 = sum(1 for r in hw_ranks if r <= 50)
        print(f"  rank<=10: {top10} ({top10/len(hw_ranks)*100:.1f}%), "
              f"rank<=50: {top50} ({top50/len(hw_ranks)*100:.1f}%)")

    # 區分有逗號段 vs 無逗號段
    has_comma_segs = [s for s in logit_results if s["commas"] > 0]
    no_comma_segs = [s for s in logit_results if s["commas"] == 0 and s["cjk_chars"] > 20]

    print(f"\n段落統計: 有逗號={len(has_comma_segs)}, 無逗號(>20字)={len(no_comma_segs)}")

    # 無逗號段中，逗號最高機率出現在哪裡？
    if no_comma_segs:
        print(f"\n--- 無逗號段的逗號機率 top moments ---")
        for seg in no_comma_segs[:5]:  # show up to 5
            best_steps = sorted(seg["steps"], key=lambda s: s["comma_hw"]["prob"], reverse=True)[:3]
            print(f"  {seg['session']}/{seg['seg']} ({seg['cjk_chars']}字):")
            for bs in best_steps:
                ctx_start = max(0, bs["step"] - 2)
                context_tokens = [seg["steps"][j]["actual_token"] for j in range(ctx_start, bs["step"])]
                ctx = "".join(context_tokens)
                print(f"    step {bs['step']}: comma_prob={bs['comma_hw']['prob']:.4f} "
                      f"rank={bs['comma_hw']['rank']} | context='...{ctx}' → '{bs['actual_token']}'")

    # --- Word timestamp 分析 ---
    if timestamp_results:
        print(f"\n--- 聲學間隔分析 ---")
        # 有逗號的 gap vs 沒逗號的 gap
        comma_gaps = []
        no_comma_gaps = []
        for seg in timestamp_results:
            for g in seg["gaps"]:
                has_punct = any(c in g["before"] for c in "，,。、！？")
                if has_punct:
                    comma_gaps.append(g["gap_ms"])
                else:
                    no_comma_gaps.append(g["gap_ms"])

        if comma_gaps:
            print(f"  標點後的 gap: mean={np.mean(comma_gaps):.0f}ms, "
                  f"median={np.median(comma_gaps):.0f}ms (n={len(comma_gaps)})")
        if no_comma_gaps:
            print(f"  無標點的 gap: mean={np.mean(no_comma_gaps):.0f}ms, "
                  f"median={np.median(no_comma_gaps):.0f}ms (n={len(no_comma_gaps)})")

        # 無逗號段的 gap 分佈
        no_comma_seg_gaps = []
        for seg in timestamp_results:
            if seg["commas"] == 0 and seg["cjk_chars"] > 20:
                for g in seg["gaps"]:
                    no_comma_seg_gaps.append(g["gap_ms"])
        if no_comma_seg_gaps:
            print(f"  無逗號段的所有 gap: mean={np.mean(no_comma_seg_gaps):.0f}ms, "
                  f"median={np.median(no_comma_seg_gaps):.0f}ms, "
                  f"max={np.max(no_comma_seg_gaps)}ms (n={len(no_comma_seg_gaps)})")
            over_200 = sum(1 for g in no_comma_seg_gaps if g > 200)
            over_300 = sum(1 for g in no_comma_seg_gaps if g > 300)
            print(f"  其中 >200ms: {over_200}, >300ms: {over_300}")


def main():
    segments = find_all_segments()
    print(f"找到 {len(segments)} 段 wav")

    # Phase 1: Logit probe
    logit_results = run_logit_probe(segments)

    # Phase 2: Word timestamps
    timestamp_results = run_word_timestamps(segments)

    # 合併儲存
    combined = {
        "meta": {
            "total_segments": len(segments),
            "logit_segments": len(logit_results),
            "timestamp_segments": len(timestamp_results),
            "model": "openai/whisper-large-v3-turbo",
            "date": "2026-07-01",
        },
        "logit_probe": logit_results,
        "word_timestamps": timestamp_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存: {OUTPUT_PATH}")

    # 分析
    analyze_and_summarize(logit_results, timestamp_results)


if __name__ == "__main__":
    main()
