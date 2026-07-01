"""MCTS-lite 實驗：在逗號 near-miss 位置用樹搜索 + 標點獎勵，驗證能否改善無逗號長句。

核心假說：
  beam=10 全部零逗號，是因為 beam search 只用 model log-prob 評分。
  如果引入外部標點獎勵（懲罰長段無逗號），MCTS 的 lookahead 能否找到
  model 自己走不到但「加了逗號反而更好」的路徑？

  看到（問題段改善 + 正常段不變）→ 模型內部有逗號信號，只是 greedy 挖不出來
  沒看到 → 模型真的認為不加逗號更好，問題在模型層，採樣改不了

方法：
  Phase 0: 診斷 — 在 near-miss 位置比較「選 winner」vs「強制逗號」的續寫 log-prob
  Phase 1: MCTS-lite — 在 near-miss 位置展開 K 條分支，短 rollout 評估，
           用 model_log_prob + λ * punct_reward 選路

段落選擇：
  14 個真正問題段（>40字無標點）+ 10 個正常段（有逗號）
"""

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PATH = Path("experiments/results/experiment_mcts_comma.json")
PROBE_PATH = Path("experiments/results/comma_logit_probe.json")

TID_COMMA = 11  # half-width comma ","
TID_PERIOD = 1231  # "。"
TID_ENUM = 1543  # "、"
PUNCT_TIDS = {TID_COMMA, TID_PERIOD, TID_ENUM}
TID_EOS = 50257
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def count_cjk(text):
    return len(CJK_RE.findall(text))


def has_any_punct(text):
    return bool(re.search(r"[，,。、！？!?；;]", text))


def select_segments():
    """Pick 14 real problem segs + 10 normal segs from probe data."""
    with open(PROBE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    problem = []
    normal = []
    for seg in data["logit_probe"]:
        cjk = seg["cjk_chars"]
        commas = seg["commas"]
        text = seg["text"]
        wav = seg["wav"]
        if commas == 0 and cjk > 40 and not has_any_punct(text):
            problem.append(wav)
        elif commas > 0:
            normal.append(wav)

    normal_sample = normal[:10]
    print(f"選定段落: {len(problem)} 問題段 + {len(normal_sample)} 正常段")
    return problem, normal_sample


def load_audio(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}")
    return audio


def prepare_audio(processor_obj, audio):
    """Prepare input_features tensor from raw audio."""
    return processor_obj(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to("cuda", dtype=torch.float16)


def baseline_generate(model, input_features, prompt_ids, tokenizer):
    """Use model.generate for reliable baseline decode."""
    prompt_ids_pt = prompt_ids.unsqueeze(0).to("cuda") if prompt_ids.dim() == 1 else prompt_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_features,
            language="zh",
            prompt_ids=prompt_ids_pt.squeeze(0),
            max_new_tokens=420,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
    token_ids = out.sequences[0].cpu().tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    prompt_text = "繁體中文，台灣用語。"
    if prompt_text in text:
        text = text.split(prompt_text, 1)[-1].strip()
    return text, out.scores


def build_decoder_prefix(model, input_features, prompt_ids):
    """Build the initial decoder_input_ids and run encoder.
    Returns (encoder_outputs, decoder_start_ids) where decoder_start_ids
    includes SOT + language + task + prompt tokens."""
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_features)

    # Build prefix: <|startoftranscript|> <|zh|> <|transcribe|> <|notimestamps|> + prompt
    sot = torch.tensor([[50258]], device="cuda")  # <|startoftranscript|>
    lang = torch.tensor([[50260]], device="cuda")  # <|zh|>
    task = torch.tensor([[50359]], device="cuda")  # <|transcribe|>
    notimestamps = torch.tensor([[50364]], device="cuda")  # <|notimestamps|>

    prompt = prompt_ids.to("cuda")
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)

    # Whisper prompt format: <|startofprev|> prompt_tokens <|startoftranscript|> lang task notimestamps
    sop = torch.tensor([[50362]], device="cuda")  # <|startofprev|>
    decoder_ids = torch.cat([sop, prompt, sot, lang, task, notimestamps], dim=1)

    return encoder_outputs, decoder_ids


def step_decode(model, encoder_outputs, decoder_ids):
    """One decoder step, return logits for next token."""
    with torch.no_grad():
        out = model(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_ids)
    return out.logits[0, -1, :]  # (vocab_size,)


def greedy_decode_stepwise(model, encoder_outputs, decoder_prefix, tokenizer, max_tokens=200):
    """Step-by-step greedy decode with full control."""
    generated = decoder_prefix.clone()
    tokens = []
    log_probs = []

    for _ in range(max_tokens):
        logits = step_decode(model, encoder_outputs, generated)
        probs = torch.softmax(logits.float(), dim=-1)
        tid = probs.argmax().item()
        lp = torch.log(probs[tid] + 1e-10).item()

        if tid == TID_EOS or tid >= 50257:
            break

        tokens.append(tid)
        log_probs.append(lp)
        generated = torch.cat([generated, torch.tensor([[tid]], device="cuda")], dim=1)

    return tokens, log_probs, generated


def rollout_from(model, encoder_outputs, prefix_ids, n_steps, tokenizer):
    """Greedy rollout from a prefix, return tokens and log-probs."""
    generated = prefix_ids.clone()
    tokens = []
    log_probs = []

    for _ in range(n_steps):
        logits = step_decode(model, encoder_outputs, generated)
        probs = torch.softmax(logits.float(), dim=-1)
        tid = probs.argmax().item()
        lp = torch.log(probs[tid] + 1e-10).item()

        if tid == TID_EOS or tid >= 50257:
            break

        tokens.append(tid)
        log_probs.append(lp)
        generated = torch.cat([generated, torch.tensor([[tid]], device="cuda")], dim=1)

    return tokens, log_probs


def punct_reward(token_ids, tokenizer):
    """Score a sequence's punctuation quality.
    Penalize long stretches without punctuation in CJK text."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    cjk_since_punct = 0
    penalty = 0.0
    for ch in text:
        if re.match(r"[，,。、！？!?；;]", ch):
            cjk_since_punct = 0
        elif CJK_RE.match(ch):
            cjk_since_punct += 1
            if cjk_since_punct > 15:
                penalty -= 0.1 * (cjk_since_punct - 15)
    return penalty


def phase0_diagnostic(model, processor, tokenizer, problem_wavs, prompt_ids):
    """Phase 0: At near-miss points, compare continuation log-prob
    with vs without comma. Answers: does the model prefer the world
    where comma was inserted?"""
    print("\n" + "=" * 60)
    print("Phase 0: 診斷 — 逗號 vs Winner 續寫品質比較")
    print("=" * 60)

    results = []
    for wav_path in problem_wavs:
        audio = load_audio(wav_path)
        input_features = prepare_audio(processor, audio)
        encoder_outputs, decoder_prefix = build_decoder_prefix(model, input_features, prompt_ids)

        # Step-by-step greedy, find near-miss points
        generated = decoder_prefix.clone()
        chars_since_punct = 0
        near_misses = []

        for step in range(200):
            logits = step_decode(model, encoder_outputs, generated)
            probs = torch.softmax(logits.float(), dim=-1)

            comma_prob = probs[TID_COMMA].item()
            comma_rank = (probs > comma_prob).sum().item() + 1
            winner_tid = probs.argmax().item()
            winner_prob = probs[winner_tid].item()

            if winner_tid == TID_EOS or winner_tid >= 50257:
                break

            winner_token = tokenizer.decode([winner_tid])
            is_punct = winner_tid in PUNCT_TIDS
            is_cjk = bool(CJK_RE.search(winner_token))

            if is_punct:
                chars_since_punct = 0
            elif is_cjk:
                chars_since_punct += len(CJK_RE.findall(winner_token))

            if comma_rank <= 10 and chars_since_punct > 8:
                near_misses.append({
                    "step": step,
                    "prefix_ids": generated.clone(),
                    "winner_tid": winner_tid,
                    "winner_token": winner_token,
                    "comma_prob": comma_prob,
                    "comma_rank": comma_rank,
                    "winner_prob": winner_prob,
                    "chars_since_punct": chars_since_punct,
                })

            generated = torch.cat(
                [generated, torch.tensor([[winner_tid]], device=generated.device)], dim=1
            )

        # For each near-miss, compare rollouts
        seg_results = []
        rollout_len = 15
        for nm in near_misses[:5]:  # max 5 per segment
            prefix = nm["prefix_ids"]

            # Path A: winner → greedy rollout
            prefix_a = torch.cat(
                [prefix, torch.tensor([[nm["winner_tid"]]], device=prefix.device)], dim=1
            )
            tokens_a, lp_a = rollout_from(model, encoder_outputs, prefix_a, rollout_len, tokenizer)
            total_lp_a = sum(lp_a) + math.log(nm["winner_prob"] + 1e-10)

            # Path B: comma → greedy rollout
            prefix_b = torch.cat(
                [prefix, torch.tensor([[TID_COMMA]], device=prefix.device)], dim=1
            )
            tokens_b, lp_b = rollout_from(model, encoder_outputs, prefix_b, rollout_len, tokenizer)
            total_lp_b = sum(lp_b) + math.log(nm["comma_prob"] + 1e-10)

            text_a = tokenizer.decode([nm["winner_tid"]] + tokens_a)
            text_b = tokenizer.decode([TID_COMMA] + tokens_b)

            seg_results.append({
                "step": nm["step"],
                "winner": nm["winner_token"],
                "comma_rank": nm["comma_rank"],
                "comma_prob": round(nm["comma_prob"], 4),
                "chars_since_punct": nm["chars_since_punct"],
                "path_winner_lp": round(total_lp_a, 3),
                "path_comma_lp": round(total_lp_b, 3),
                "comma_wins": total_lp_b > total_lp_a,
                "lp_diff": round(total_lp_b - total_lp_a, 3),
                "continuation_winner": text_a[:60],
                "continuation_comma": text_b[:60],
            })

        comma_win_count = sum(1 for r in seg_results if r["comma_wins"])
        print(f"  {Path(wav_path).parent.name}/{Path(wav_path).name}: "
              f"{comma_win_count}/{len(seg_results)} 逗號路徑勝出")

        results.append({
            "wav": wav_path,
            "near_miss_count": len(near_misses),
            "tested": len(seg_results),
            "comma_wins": comma_win_count,
            "details": seg_results,
        })

    total_tested = sum(r["tested"] for r in results)
    total_comma_wins = sum(r["comma_wins"] for r in results)
    print(f"\n  總計: {total_comma_wins}/{total_tested} 逗號路徑勝出 "
          f"({total_comma_wins/max(total_tested,1)*100:.0f}%)")

    return results


def mcts_decode(model, encoder_outputs, decoder_prefix, tokenizer, config):
    """MCTS-lite decode: at near-miss positions, branch and use rollout + reward to choose."""
    branch_k = config["branch_k"]
    rollout_len = config["rollout_len"]
    reward_lambda = config["reward_lambda"]
    comma_rank_threshold = config["comma_rank_threshold"]
    chars_gate = config["chars_gate"]

    generated = decoder_prefix.clone()
    all_token_ids = []
    chars_since_punct = 0
    branch_points = 0

    for step in range(250):
        logits = step_decode(model, encoder_outputs, generated)
        probs = torch.softmax(logits.float(), dim=-1)

        winner_tid = probs.argmax().item()
        if winner_tid == TID_EOS or winner_tid >= 50257:
            break

        comma_prob = probs[TID_COMMA].item()
        comma_rank = (probs > comma_prob).sum().item() + 1

        winner_token = tokenizer.decode([winner_tid])
        is_cjk = bool(CJK_RE.search(winner_token))
        is_punct = winner_tid in PUNCT_TIDS

        should_branch = (
            comma_rank <= comma_rank_threshold
            and chars_since_punct >= chars_gate
            and winner_tid != TID_COMMA
            and reward_lambda > 0
        )

        if should_branch:
            # Get top-K candidates (always include comma)
            top_k_probs, top_k_ids = probs.topk(branch_k)
            candidates = list(zip(top_k_ids.tolist(), top_k_probs.tolist()))

            # Ensure comma is in candidates
            comma_in = any(tid == TID_COMMA for tid, _ in candidates)
            if not comma_in:
                candidates.append((TID_COMMA, comma_prob))

            best_score = float("-inf")
            best_tid = winner_tid

            for cand_tid, cand_prob in candidates:
                prefix_c = torch.cat(
                    [generated, torch.tensor([[cand_tid]], device=generated.device)], dim=1
                )
                r_tokens, r_lps = rollout_from(
                    model, encoder_outputs, prefix_c, rollout_len, tokenizer
                )

                model_score = math.log(cand_prob + 1e-10) + sum(r_lps)
                full_seq = all_token_ids + [cand_tid] + r_tokens
                reward = punct_reward(full_seq, tokenizer)
                total_score = model_score + reward_lambda * reward

                if total_score > best_score:
                    best_score = total_score
                    best_tid = cand_tid

            if best_tid != winner_tid:
                branch_points += 1

            chosen_tid = best_tid
        else:
            chosen_tid = winner_tid

        all_token_ids.append(chosen_tid)
        chosen_token = tokenizer.decode([chosen_tid])

        if chosen_tid in PUNCT_TIDS:
            chars_since_punct = 0
        elif bool(CJK_RE.search(chosen_token)):
            chars_since_punct += len(CJK_RE.findall(chosen_token))

        generated = torch.cat(
            [generated, torch.tensor([[chosen_tid]], device=generated.device)], dim=1
        )

    text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    return text, branch_points


def phase1_mcts(model, processor, tokenizer, problem_wavs, normal_wavs, prompt_ids):
    """Phase 1: Run MCTS-lite with different configs."""
    print("\n" + "=" * 60)
    print("Phase 1: MCTS-lite 實驗")
    print("=" * 60)

    configs = [
        {"name": "baseline", "branch_k": 3, "rollout_len": 15,
         "reward_lambda": 0.0, "comma_rank_threshold": 5, "chars_gate": 10},
        {"name": "mcts_r5_g10_l1", "branch_k": 3, "rollout_len": 15,
         "reward_lambda": 1.0, "comma_rank_threshold": 5, "chars_gate": 10},
        {"name": "mcts_r5_g10_l3", "branch_k": 3, "rollout_len": 15,
         "reward_lambda": 3.0, "comma_rank_threshold": 5, "chars_gate": 10},
        {"name": "mcts_r10_g10_l1", "branch_k": 3, "rollout_len": 15,
         "reward_lambda": 1.0, "comma_rank_threshold": 10, "chars_gate": 10},
        {"name": "mcts_r10_g10_l3", "branch_k": 3, "rollout_len": 15,
         "reward_lambda": 3.0, "comma_rank_threshold": 10, "chars_gate": 10},
        {"name": "mcts_r10_g15_l2", "branch_k": 3, "rollout_len": 15,
         "reward_lambda": 2.0, "comma_rank_threshold": 10, "chars_gate": 15},
    ]

    all_wavs = [(w, "problem") for w in problem_wavs] + [(w, "normal") for w in normal_wavs]

    # Pre-load audio, encode, and build decoder prefixes
    prep_cache = {}
    for wav_path, _ in all_wavs:
        audio = load_audio(wav_path)
        input_features = prepare_audio(processor, audio)
        enc_out, dec_prefix = build_decoder_prefix(model, input_features, prompt_ids)
        prep_cache[wav_path] = (enc_out, dec_prefix, input_features)

    # Get baseline texts using model.generate (reliable)
    print("\n  Baseline decode...", flush=True)
    baseline_texts = {}
    for wav_path, cat in all_wavs:
        _, _, input_features = prep_cache[wav_path]
        text, _ = baseline_generate(model, input_features, prompt_ids, tokenizer)
        baseline_texts[wav_path] = text

    results = []
    for cfg in configs:
        print(f"\n  Config: {cfg['name']}", flush=True)
        cfg_results = []
        t0 = time.time()

        for wav_path, cat in all_wavs:
            enc_out, dec_prefix, _ = prep_cache[wav_path]
            baseline_text = baseline_texts[wav_path]

            if cfg["reward_lambda"] == 0.0:
                new_text = baseline_text
                branches = 0
            else:
                new_text, branches = mcts_decode(
                    model, enc_out, dec_prefix, tokenizer, cfg
                )

            baseline_commas = baseline_text.count(",") + baseline_text.count("，")
            new_commas = new_text.count(",") + new_text.count("，")
            content_base = re.sub(r"[，,。、！？!?\s]", "", baseline_text)
            content_new = re.sub(r"[，,。、！？!?\s]", "", new_text)

            cfg_results.append({
                "wav": wav_path,
                "category": cat,
                "baseline_commas": baseline_commas,
                "new_commas": new_commas,
                "comma_delta": new_commas - baseline_commas,
                "content_match": content_base == content_new,
                "branches_taken": branches,
                "baseline_text": baseline_text,
                "new_text": new_text,
            })

        elapsed = time.time() - t0

        # Summary
        prob_results = [r for r in cfg_results if r["category"] == "problem"]
        norm_results = [r for r in cfg_results if r["category"] == "normal"]

        prob_improved = sum(1 for r in prob_results if r["comma_delta"] > 0)
        prob_total = len(prob_results)
        norm_changed = sum(1 for r in norm_results if not r["content_match"])
        norm_extra_comma = sum(1 for r in norm_results if r["comma_delta"] > 0)
        norm_total = len(norm_results)

        summary = {
            "problem_improved": f"{prob_improved}/{prob_total}",
            "problem_improved_pct": round(prob_improved / max(prob_total, 1) * 100, 1),
            "normal_content_changed": f"{norm_changed}/{norm_total}",
            "normal_extra_comma": f"{norm_extra_comma}/{norm_total}",
            "elapsed_sec": round(elapsed, 1),
        }

        print(f"    問題段改善: {summary['problem_improved']} ({summary['problem_improved_pct']}%)")
        print(f"    正常段內容變化: {summary['normal_content_changed']}")
        print(f"    正常段多逗號: {summary['normal_extra_comma']}")
        print(f"    耗時: {summary['elapsed_sec']}s")

        results.append({
            "config": cfg,
            "summary": summary,
            "segments": cfg_results,
        })

    return results


def main():
    print("=" * 60)
    print("MCTS-lite 逗號實驗")
    print("=" * 60)

    problem_wavs, normal_wavs = select_segments()

    print("\n載入模型...", flush=True)
    t0 = time.time()
    processor_obj = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo", dtype=torch.float16
    ).to("cuda")
    model.eval()
    tokenizer = processor_obj.tokenizer
    prompt_ids = processor_obj.get_prompt_ids("繁體中文，台灣用語。", return_tensors="pt").to("cuda")
    print(f"模型載入: {time.time() - t0:.1f}s", flush=True)

    # Phase 0: Diagnostic
    phase0_results = phase0_diagnostic(
        model, processor_obj, tokenizer, problem_wavs, prompt_ids
    )

    # Phase 1: MCTS with different configs
    phase1_results = phase1_mcts(
        model, processor_obj, tokenizer, problem_wavs, normal_wavs, prompt_ids
    )

    # Save all results
    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "model": "openai/whisper-large-v3-turbo",
            "problem_segments": len(problem_wavs),
            "normal_segments": len(normal_wavs),
            "problem_filter": "cjk>40, zero commas, no other punct",
        },
        "phase0_diagnostic": phase0_results,
        "phase1_mcts": phase1_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果已存: {OUTPUT_PATH}")

    # Print Phase 0 summary
    print("\n" + "=" * 60)
    print("Phase 0 摘要 — 逗號 vs Winner 續寫品質")
    print("=" * 60)
    for r in phase0_results:
        wav_name = f"{Path(r['wav']).parent.name}/{Path(r['wav']).name}"
        print(f"\n{wav_name}: {r['comma_wins']}/{r['tested']} 逗號路徑勝出")
        for d in r["details"]:
            marker = "✓" if d["comma_wins"] else "✗"
            print(f"  {marker} step {d['step']:2d} | rank={d['comma_rank']} "
                  f"chars={d['chars_since_punct']:2d} | "
                  f"diff={d['lp_diff']:+.2f}")
            print(f"    winner: {d['continuation_winner'][:50]}")
            print(f"    comma:  {d['continuation_comma'][:50]}")

    # Print Phase 1 summary table
    print("\n" + "=" * 60)
    print("Phase 1 摘要 — MCTS 各組態比較")
    print("=" * 60)
    print(f"{'Config':<25} {'問題改善':>8} {'正常變化':>8} {'正常多逗':>8} {'耗時':>6}")
    print("-" * 60)
    for r in phase1_results:
        cfg_name = r["config"]["name"]
        s = r["summary"]
        print(f"{cfg_name:<25} {s['problem_improved']:>8} {s['normal_content_changed']:>8} "
              f"{s['normal_extra_comma']:>8} {s['elapsed_sec']:>5.0f}s")


if __name__ == "__main__":
    main()
