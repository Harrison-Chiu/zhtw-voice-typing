"""Experiment v2: test 4 compensation strategies on stubborn hallucination sub-segments.

Strategies:
  A) RMS / Peak normalization — normalize sub-segment volume before transcription
  B) Bandpass / Highpass filter — keep human voice frequencies only
  C) Speed perturbation — slightly speed up or slow down audio
  D) Temperature — increase Whisper temperature to break deterministic beam path

Flow:
  1. Run full audio through streaming pipeline → find hallucination segments
  2. For each hallucination segment, run current fallback (resegment)
  3. Identify sub-segments that still hallucinate
  4. Apply all 4 strategies to those stubborn sub-segments
  5. Also test strategies on the original full segment (before resegment)

Usage:
    .venv\\Scripts\\python.exe experiments/experiment_hallucination_v2.py [audio_path]
"""

import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import butter, sosfilt, resample  # noqa: E402

from asr_input.asr import build_engine  # noqa: E402
from asr_input.audio.streaming_vad import StreamingVAD  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline  # noqa: E402
from asr_input.streaming import StreamingSession  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HALLUCINATION_THRESHOLD_SEC = 1.5
OUTPUT_PATH = Path(__file__).resolve().parent / "results/experiment_hallucination_v2.json"

# ---------------------------------------------------------------------------
# Audio transforms
# ---------------------------------------------------------------------------


def rms_normalize(audio: np.ndarray, target_rms: float) -> dict:
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms == 0:
        return {"audio": audio, "method": f"rms_norm(target={target_rms})", "rms_before": 0,
                "rms_after": 0, "gain": 0}
    gain = target_rms / rms
    out = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
    return {"audio": out, "method": f"rms_norm(target={target_rms})",
            "rms_before": round(rms, 5), "rms_after": round(float(np.sqrt(np.mean(out**2))), 5),
            "gain": round(gain, 2)}


def peak_normalize(audio: np.ndarray, target_peak: float = 0.9) -> dict:
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        return {"audio": audio, "method": f"peak_norm(target={target_peak})",
                "peak_before": 0, "peak_after": 0, "gain": 0}
    gain = target_peak / peak
    out = (audio * gain).astype(np.float32)
    return {"audio": out, "method": f"peak_norm(target={target_peak})",
            "peak_before": round(peak, 5), "peak_after": round(float(np.max(np.abs(out))), 5),
            "gain": round(gain, 2)}


def bandpass_filter(audio: np.ndarray, sr: int, low: int, high: int) -> dict:
    sos = butter(5, [low, high], btype="band", fs=sr, output="sos")
    out = sosfilt(sos, audio).astype(np.float32)
    return {"audio": out, "method": f"bandpass({low}-{high}Hz)",
            "low_hz": low, "high_hz": high}


def highpass_filter(audio: np.ndarray, sr: int, cutoff: int) -> dict:
    sos = butter(5, cutoff, btype="high", fs=sr, output="sos")
    out = sosfilt(sos, audio).astype(np.float32)
    return {"audio": out, "method": f"highpass({cutoff}Hz)", "cutoff_hz": cutoff}


def speed_perturb(audio: np.ndarray, sr: int, factor: float) -> dict:
    new_len = int(len(audio) / factor)
    out = resample(audio, new_len).astype(np.float32)
    return {"audio": out, "method": f"speed({factor}x)",
            "factor": factor, "original_len": len(audio), "new_len": new_len}


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------


def transcribe_once(engine, audio, sample_rate):
    t0 = time.time()
    text = engine.transcribe(audio, sample_rate)
    dt = time.time() - t0
    return text, dt


def transcribe_with_temperature(model, audio, sample_rate, temperature, language="zh",
                                 initial_prompt="繁體中文，台灣用語。", beam_size=5):
    t0 = time.time()
    if temperature == 0.0:
        segments, _ = model.transcribe(
            audio.astype(np.float32), language=language, beam_size=beam_size,
            initial_prompt=initial_prompt,
        )
    else:
        segments, _ = model.transcribe(
            audio.astype(np.float32), language=language, beam_size=beam_size,
            initial_prompt=initial_prompt, temperature=temperature,
        )
    text = "".join(seg.text for seg in segments).strip()
    dt = time.time() - t0
    return text, dt


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------


def run_strategy_a(engine, audio, sr):
    """Strategy A: volume normalization variants."""
    variants = []

    for target_rms in [0.03, 0.05, 0.1]:
        info = rms_normalize(audio, target_rms)
        text, dt = transcribe_once(engine, info["audio"], sr)
        variants.append({
            "variant": info["method"],
            "params": {k: v for k, v in info.items() if k != "audio"},
            "text": text, "transcribe_sec": round(dt, 3),
            "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
        })

    info = peak_normalize(audio, 0.9)
    text, dt = transcribe_once(engine, info["audio"], sr)
    variants.append({
        "variant": info["method"],
        "params": {k: v for k, v in info.items() if k != "audio"},
        "text": text, "transcribe_sec": round(dt, 3),
        "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
    })

    return variants


def run_strategy_b(engine, audio, sr):
    """Strategy B: frequency filtering variants."""
    variants = []

    for low, high in [(300, 3000), (200, 4000), (80, 5000)]:
        info = bandpass_filter(audio, sr, low, high)
        text, dt = transcribe_once(engine, info["audio"], sr)
        variants.append({
            "variant": info["method"],
            "params": {k: v for k, v in info.items() if k != "audio"},
            "text": text, "transcribe_sec": round(dt, 3),
            "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
        })

    info = highpass_filter(audio, sr, 300)
    text, dt = transcribe_once(engine, info["audio"], sr)
    variants.append({
        "variant": info["method"],
        "params": {k: v for k, v in info.items() if k != "audio"},
        "text": text, "transcribe_sec": round(dt, 3),
        "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
    })

    return variants


def run_strategy_c(engine, audio, sr):
    """Strategy C: speed perturbation variants."""
    variants = []

    for factor in [0.9, 0.95, 1.05, 1.1]:
        info = speed_perturb(audio, sr, factor)
        text, dt = transcribe_once(engine, info["audio"], sr)
        variants.append({
            "variant": info["method"],
            "params": {k: v for k, v in info.items() if k != "audio"},
            "text": text, "transcribe_sec": round(dt, 3),
            "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
        })

    return variants


def run_strategy_d(whisper_model, audio, sr, initial_prompt="繁體中文，台灣用語。"):
    """Strategy D: temperature variants (needs raw whisper model)."""
    variants = []

    for temp in [0.0, 0.2, 0.4, 0.6, 0.8]:
        text, dt = transcribe_with_temperature(
            whisper_model, audio, sr, temperature=temp,
            initial_prompt=initial_prompt,
        )
        variants.append({
            "variant": f"temp({temp})",
            "params": {"temperature": temp},
            "text": text, "transcribe_sec": round(dt, 3),
            "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
        })

    return variants


# ---------------------------------------------------------------------------
# Resegment helper (mirrors current fallback logic)
# ---------------------------------------------------------------------------


def run_fallback_resegment(engine, audio, probs, sr, vad_threshold=0.5,
                            min_speech_ms=250, speech_pad_ms=100, min_energy=0.005):
    """Run the current fallback: step down from 500ms to 300ms."""
    prev_seg_lens = (len(audio),)
    best_results = None

    for silence_ms in range(500, 299, -100):
        sub_segments = StreamingVAD.resegment(
            audio, probs, silence_trigger_ms=silence_ms,
            sample_rate=sr, threshold=vad_threshold,
            min_speech_ms=min_speech_ms, speech_pad_ms=speech_pad_ms,
            min_energy=min_energy,
        )
        if not sub_segments:
            continue

        cur_lens = tuple(len(s) for s in sub_segments)
        if cur_lens == prev_seg_lens:
            continue
        prev_seg_lens = cur_lens

        results = []
        any_hall = False
        for i, sub in enumerate(sub_segments):
            sub_sec = len(sub) / sr
            text, dt = transcribe_once(engine, sub, sr)
            is_hall = dt > HALLUCINATION_THRESHOLD_SEC
            if is_hall:
                any_hall = True
            results.append({
                "sub_index": i,
                "audio_sec": round(sub_sec, 2),
                "rms": round(float(np.sqrt(np.mean(sub**2))), 5),
                "peak": round(float(np.max(np.abs(sub))), 5),
                "transcribe_sec": round(dt, 3),
                "text": text,
                "is_hallucination": is_hall,
                "audio": sub,  # keep for strategy testing
            })

        best_results = {
            "silence_ms": silence_ms,
            "n_sub_segments": len(sub_segments),
            "any_hallucination": any_hall,
            "sub_segments": results,
        }

        if not any_hall:
            return best_results

    return best_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    config = load_config()
    asr_cfg = config["asr"]
    sample_rate = config["audio"]["sample_rate"]
    streaming_cfg = config.get("streaming", {})
    vad_cfg = config.get("vad", {})

    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        files = sorted((PROJECT_ROOT / "data/test_audio").glob("*.m4a"))
        if not files:
            print("No audio files found in data/test_audio/")
            return
        audio_path = files[0]

    print("=== 幻覺補償策略實驗 v2 ===")
    print(f"音檔: {audio_path.name}")
    print(f"幻覺判定閾值: 轉錄時間 > {HALLUCINATION_THRESHOLD_SEC}s\n")

    # Load audio
    print("載入音檔...", flush=True)
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    print(f"音檔長度: {len(audio) / sample_rate:.1f}s")

    # Load models
    print("載入 ASR 模型...", flush=True)
    engine = build_engine(asr_cfg, vad_cfg=None)
    engine.load()
    whisper_model = engine._model  # direct access for temperature experiments
    pipeline = build_pipeline(config, include_output=False)

    print("載入 VAD 模型...", flush=True)
    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    print("模型載入完成!\n")

    # -----------------------------------------------------------------------
    # Phase 1: streaming → find hallucination segments (capture audio + probs)
    # -----------------------------------------------------------------------
    print("=== Phase 1: 串流辨識，找出幻覺段落 ===\n")

    captured: list[tuple[np.ndarray, list[float]]] = []
    original_on_seg = None

    session = StreamingSession(
        engine=engine, pipeline=pipeline, vad_model=vad_model,
        sample_rate=sample_rate,
        silence_trigger_ms=streaming_cfg.get("silence_trigger_ms", 1000),
        silence_min_ms=streaming_cfg.get("silence_min_ms", 300),
        ramp_start_sec=streaming_cfg.get("ramp_start_sec", 10.0),
        ramp_end_sec=streaming_cfg.get("ramp_end_sec", 25.0),
        vad_threshold=vad_cfg.get("threshold", 0.5),
        min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
        speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
        max_segment_sec=streaming_cfg.get("max_segment_sec", 30.0),
        min_energy=streaming_cfg.get("min_energy", 0.005),
        hallucination_threshold_sec=999,  # disable built-in fallback
    )

    original_on_seg = session._on_speech_segment

    def capture_on_seg(audio_seg: np.ndarray, probs: list[float]):
        captured.append((audio_seg.copy(), list(probs)))
        original_on_seg(audio_seg, probs)

    session._vad._on_speech_segment = capture_on_seg

    session.start_from_file(audio)
    session.stop()

    stats = session.segment_stats
    hallucination_indices = []

    print(f"共 {len(stats)} 段:\n")
    for i, stat in enumerate(stats):
        is_hall = stat["transcribe_sec"] > HALLUCINATION_THRESHOLD_SEC
        marker = " *** 幻覺" if is_hall else ""
        print(f"  段 {i+1:2d}: {stat['audio_sec']:.1f}s → {stat['transcribe_sec']:.2f}s{marker}")
        if is_hall:
            hallucination_indices.append(i)

    if not hallucination_indices:
        print("\n沒有偵測到幻覺段落！")
        engine.unload()
        return

    print(f"\n找到 {len(hallucination_indices)} 個幻覺段落: "
          f"{[i+1 for i in hallucination_indices]}\n")

    # -----------------------------------------------------------------------
    # Phase 2: for each hallucination segment → fallback → strategies
    # -----------------------------------------------------------------------
    all_experiments = []

    for idx in hallucination_indices:
        seg_audio, seg_probs = captured[idx]
        stat = stats[idx]
        seg_sec = len(seg_audio) / sample_rate
        seg_rms = float(np.sqrt(np.mean(seg_audio**2)))
        seg_peak = float(np.max(np.abs(seg_audio)))

        print(f"{'='*60}")
        print(f"段 {idx+1} ({seg_sec:.1f}s, RMS={seg_rms:.4f}, peak={seg_peak:.4f})")
        print(f"原始轉錄: {stat['transcribe_sec']:.2f}s")
        print(f"原始文字: {stat['raw'][:80]}...")
        print()

        experiment = {
            "segment_index": idx + 1,
            "audio_sec": round(seg_sec, 2),
            "rms": round(seg_rms, 5),
            "peak": round(seg_peak, 5),
            "original": {
                "text": stat["raw"],
                "processed": stat["processed"],
                "transcribe_sec": stat["transcribe_sec"],
            },
            "full_segment_strategies": {},
            "fallback": None,
            "sub_segment_strategies": {},
        }

        # --- Test strategies on full segment (before resegment) ---
        print("  [整段] 策略 A: 音量正規化...")
        experiment["full_segment_strategies"]["A_volume"] = run_strategy_a(
            engine, seg_audio, sample_rate
        )
        for v in experiment["full_segment_strategies"]["A_volume"]:
            marker = "幻覺" if v["is_hallucination"] else "✓"
            print(f"    {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

        print("  [整段] 策略 B: 頻率濾波...")
        experiment["full_segment_strategies"]["B_filter"] = run_strategy_b(
            engine, seg_audio, sample_rate
        )
        for v in experiment["full_segment_strategies"]["B_filter"]:
            marker = "幻覺" if v["is_hallucination"] else "✓"
            print(f"    {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

        print("  [整段] 策略 C: 語速微調...")
        experiment["full_segment_strategies"]["C_speed"] = run_strategy_c(
            engine, seg_audio, sample_rate
        )
        for v in experiment["full_segment_strategies"]["C_speed"]:
            marker = "幻覺" if v["is_hallucination"] else "✓"
            print(f"    {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

        print("  [整段] 策略 D: Temperature...")
        experiment["full_segment_strategies"]["D_temperature"] = run_strategy_d(
            whisper_model, seg_audio, sample_rate
        )
        for v in experiment["full_segment_strategies"]["D_temperature"]:
            marker = "幻覺" if v["is_hallucination"] else "✓"
            print(f"    {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

        # --- Run fallback resegment ---
        print(f"\n  [Fallback] 重切...")
        fb = run_fallback_resegment(
            engine, seg_audio, seg_probs, sample_rate,
            vad_threshold=vad_cfg.get("threshold", 0.5),
            min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
            min_energy=streaming_cfg.get("min_energy", 0.005),
        )

        if fb is None:
            print("    重切失敗（無子段）")
            experiment["fallback"] = {"error": "no sub-segments"}
            all_experiments.append(experiment)
            continue

        fb_serializable = {
            "silence_ms": fb["silence_ms"],
            "n_sub_segments": fb["n_sub_segments"],
            "any_hallucination": fb["any_hallucination"],
            "sub_segments": [{k: v for k, v in s.items() if k != "audio"}
                            for s in fb["sub_segments"]],
        }
        experiment["fallback"] = fb_serializable

        for s in fb["sub_segments"]:
            marker = "⚠幻覺" if s["is_hallucination"] else "✓"
            print(f"    sub[{s['sub_index']}]: {s['audio_sec']:.1f}s → "
                  f"{s['transcribe_sec']:.2f}s RMS={s['rms']:.4f} ({marker})")

        # --- Test strategies on still-hallucinating sub-segments ---
        hall_subs = [s for s in fb["sub_segments"] if s["is_hallucination"]]

        if not hall_subs:
            print("    所有子段正常，跳過子段策略測試")
            all_experiments.append(experiment)
            continue

        print(f"\n  {len(hall_subs)} 個頑固子段，測試策略...\n")

        sub_strategies = {}
        for sub_info in hall_subs:
            sub_idx = sub_info["sub_index"]
            sub_audio = [s["audio"] for s in fb["sub_segments"]
                         if s["sub_index"] == sub_idx][0]
            sub_key = f"sub_{sub_idx}"

            print(f"    --- sub[{sub_idx}] ({sub_info['audio_sec']:.1f}s, "
                  f"RMS={sub_info['rms']:.4f}) ---")

            sub_result = {"control": sub_info}

            print(f"      策略 A: 音量正規化...")
            sub_result["A_volume"] = run_strategy_a(engine, sub_audio, sample_rate)
            for v in sub_result["A_volume"]:
                marker = "幻覺" if v["is_hallucination"] else "✓"
                print(f"        {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

            print(f"      策略 B: 頻率濾波...")
            sub_result["B_filter"] = run_strategy_b(engine, sub_audio, sample_rate)
            for v in sub_result["B_filter"]:
                marker = "幻覺" if v["is_hallucination"] else "✓"
                print(f"        {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

            print(f"      策略 C: 語速微調...")
            sub_result["C_speed"] = run_strategy_c(engine, sub_audio, sample_rate)
            for v in sub_result["C_speed"]:
                marker = "幻覺" if v["is_hallucination"] else "✓"
                print(f"        {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

            print(f"      策略 D: Temperature...")
            sub_result["D_temperature"] = run_strategy_d(
                whisper_model, sub_audio, sample_rate
            )
            for v in sub_result["D_temperature"]:
                marker = "幻覺" if v["is_hallucination"] else "✓"
                print(f"        {v['variant']}: {v['transcribe_sec']:.2f}s ({marker})")

            # Remove non-serializable control audio ref
            sub_result["control"] = {k: v for k, v in sub_info.items() if k != "audio"}
            sub_strategies[sub_key] = sub_result
            print()

        experiment["sub_segment_strategies"] = sub_strategies
        all_experiments.append(experiment)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    report = {
        "meta": {
            "audio_file": audio_path.name,
            "hallucination_threshold_sec": HALLUCINATION_THRESHOLD_SEC,
            "total_segments": len(stats),
            "hallucination_segments": [i + 1 for i in hallucination_indices],
            "strategies": {
                "A_volume": "RMS/Peak normalization",
                "B_filter": "Bandpass/Highpass frequency filter",
                "C_speed": "Speed perturbation (scipy resample)",
                "D_temperature": "Whisper temperature parameter",
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "all_segments_overview": [
            {
                "index": i + 1,
                "audio_sec": stat["audio_sec"],
                "transcribe_sec": stat["transcribe_sec"],
                "is_hallucination": stat["transcribe_sec"] > HALLUCINATION_THRESHOLD_SEC,
                "text_preview": stat["processed"][:60],
            }
            for i, stat in enumerate(stats)
        ],
        "experiments": all_experiments,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n報告已輸出: {OUTPUT_PATH}")

    engine.unload()
    print("Done!")


if __name__ == "__main__":
    main()
