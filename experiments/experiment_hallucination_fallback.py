"""Experiment: test three hallucination fallback strategies.

Runs the streaming pipeline on a test audio file, identifies segments where
Whisper transcription takes >1.5s (hallucination indicator), then tests
three fallback strategies on each:

  A) Re-segment with shorter silence thresholds (split into smaller pieces)
  B) Pad audio with silence to shift alignment
  C) Direct retry (rely on Whisper's beam search randomness)

Usage:
    .venv\\Scripts\\python.exe experiment_hallucination_fallback.py [audio_path]
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

from asr_input.asr import build_engine  # noqa: E402
from asr_input.audio.streaming_vad import StreamingVAD  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline  # noqa: E402
from asr_input.streaming import StreamingSession  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HALLUCINATION_THRESHOLD_SEC = 1.5
OUTPUT_PATH = Path(__file__).resolve().parent / "results/experiment_hallucination_fallback.json"


def transcribe_once(engine, audio, sample_rate):
    """Transcribe and return (text, elapsed_sec)."""
    t0 = time.time()
    text = engine.transcribe(audio, sample_rate)
    return text, time.time() - t0


def strategy_c_retry(engine, audio, sample_rate, n_retries=3):
    """Strategy C: direct retry, rely on beam search randomness."""
    results = []
    for i in range(n_retries):
        text, dt = transcribe_once(engine, audio, sample_rate)
        results.append({
            "attempt": i + 1,
            "text": text,
            "transcribe_sec": round(dt, 3),
            "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
        })
    return results


def strategy_a_resegment(engine, pipeline, audio, sample_rate, vad_model):
    """Strategy A: re-segment with shorter silence thresholds, transcribe each piece."""
    silence_ms_options = [500, 300, 200]
    results = []

    for silence_ms in silence_ms_options:
        segments_audio: list[np.ndarray] = []

        def capture(seg_audio, _out=segments_audio):
            _out.append(seg_audio.copy())

        vad = StreamingVAD(
            on_speech_segment=capture,
            sample_rate=sample_rate,
            threshold=0.5,
            min_speech_ms=250,
            silence_trigger_ms=silence_ms,
            speech_pad_ms=100,
            max_segment_sec=30.0,
            min_energy=0.003,
        )
        vad.load(vad_model)
        vad.reset()

        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            vad.feed(audio[i : i + chunk_size])
        vad.flush()

        sub_results = []
        total_dt = 0
        texts = []
        for j, seg in enumerate(segments_audio):
            text, dt = transcribe_once(engine, seg, sample_rate)
            processed = pipeline.run(text) if text.strip() else ""
            total_dt += dt
            texts.append(processed)
            sub_results.append({
                "sub_index": j + 1,
                "audio_sec": round(len(seg) / sample_rate, 2),
                "transcribe_sec": round(dt, 3),
                "text": processed,
                "is_hallucination": dt > HALLUCINATION_THRESHOLD_SEC,
            })

        combined = "".join(texts)
        results.append({
            "silence_ms": silence_ms,
            "n_sub_segments": len(segments_audio),
            "total_transcribe_sec": round(total_dt, 3),
            "any_hallucination": any(s["is_hallucination"] for s in sub_results),
            "combined_text": combined,
            "sub_segments": sub_results,
        })

    return results


def strategy_b_pad(engine, audio, sample_rate):
    """Strategy B: pad audio with silence to shift alignment."""
    pad_ms_options = [100, 250, 500, 1000]
    results = []

    for pad_ms in pad_ms_options:
        pad_samples = int(pad_ms * sample_rate / 1000)
        silence = np.zeros(pad_samples, dtype=np.float32)

        # Pad at the beginning
        padded_front = np.concatenate([silence, audio])
        text_front, dt_front = transcribe_once(engine, padded_front, sample_rate)

        # Pad at the end
        padded_back = np.concatenate([audio, silence])
        text_back, dt_back = transcribe_once(engine, padded_back, sample_rate)

        # Pad both sides
        padded_both = np.concatenate([silence, audio, silence])
        text_both, dt_both = transcribe_once(engine, padded_both, sample_rate)

        results.append({
            "pad_ms": pad_ms,
            "front_pad": {
                "text": text_front,
                "transcribe_sec": round(dt_front, 3),
                "is_hallucination": dt_front > HALLUCINATION_THRESHOLD_SEC,
            },
            "back_pad": {
                "text": text_back,
                "transcribe_sec": round(dt_back, 3),
                "is_hallucination": dt_back > HALLUCINATION_THRESHOLD_SEC,
            },
            "both_pad": {
                "text": text_both,
                "transcribe_sec": round(dt_both, 3),
                "is_hallucination": dt_both > HALLUCINATION_THRESHOLD_SEC,
            },
        })

    return results


def strategy_d_volume(engine, audio, sample_rate):
    """Strategy D: adjust volume — peak normalize + amplify by fixed factors."""
    results = []

    peak = np.max(np.abs(audio))
    rms = float(np.sqrt(np.mean(audio**2)))

    # Peak normalize to 0.9
    if peak > 0:
        normalized = audio * (0.9 / peak)
        text_norm, dt_norm = transcribe_once(engine, normalized, sample_rate)
        results.append({
            "method": f"peak_normalize (原 peak={peak:.4f}→0.9)",
            "text": text_norm,
            "transcribe_sec": round(dt_norm, 3),
            "is_hallucination": dt_norm > HALLUCINATION_THRESHOLD_SEC,
        })

    # Fixed amplification factors
    for factor in [2.0, 4.0]:
        amplified = np.clip(audio * factor, -1.0, 1.0).astype(np.float32)
        text_amp, dt_amp = transcribe_once(engine, amplified, sample_rate)
        clipped_pct = float(np.mean(np.abs(audio * factor) > 1.0)) * 100
        results.append({
            "method": f"{factor}x 放大 (clip {clipped_pct:.1f}%)",
            "text": text_amp,
            "transcribe_sec": round(dt_amp, 3),
            "is_hallucination": dt_amp > HALLUCINATION_THRESHOLD_SEC,
        })

    # RMS normalize to target 0.05 (typical speech level)
    if rms > 0:
        target_rms = 0.05
        gain = target_rms / rms
        rms_normalized = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        text_rms, dt_rms = transcribe_once(engine, rms_normalized, sample_rate)
        results.append({
            "method": f"RMS_normalize (原 RMS={rms:.4f}→{target_rms}, gain={gain:.1f}x)",
            "text": text_rms,
            "transcribe_sec": round(dt_rms, 3),
            "is_hallucination": dt_rms > HALLUCINATION_THRESHOLD_SEC,
        })

    return results


def main():
    config = load_config()
    asr_cfg = config["asr"]
    sample_rate = config["audio"]["sample_rate"]
    streaming_cfg = config.get("streaming", {})
    vad_cfg = config.get("vad", {})

    # Find audio file
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        files = sorted((PROJECT_ROOT / "data/test_audio").glob("*.m4a"))
        if not files:
            print("No audio files found in data/test_audio/")
            return
        audio_path = files[0]

    print("=== 幻覺 Fallback 實驗 ===")
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
    pipeline = build_pipeline(config, include_output=False)

    print("載入 VAD 模型...", flush=True)
    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    print("模型載入完成!\n")

    # Step 1: Run streaming to capture segment audio and find hallucination segments
    print("=== Phase 1: 找出幻覺段落 ===\n")

    captured_segments: list[np.ndarray] = []

    def capture_segment(seg_audio):
        captured_segments.append(seg_audio.copy())

    session = StreamingSession(
        engine=engine,
        pipeline=pipeline,
        vad_model=vad_model,
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
    )

    # Monkey-patch to also capture raw audio
    original_on_segment = session._on_speech_segment

    def patched_on_segment(seg_audio):
        captured_segments.append(seg_audio.copy())
        original_on_segment(seg_audio)

    session._on_speech_segment = patched_on_segment
    session._vad._on_speech_segment = patched_on_segment

    session.start_from_file(audio)
    session.stop()

    stats = session.segment_stats
    hallucination_indices = []

    for i, (stat, _seg_audio) in enumerate(zip(stats, captured_segments, strict=False)):
        marker = " *** 幻覺" if stat["transcribe_sec"] > HALLUCINATION_THRESHOLD_SEC else ""
        print(
            f"  段 {i+1}: {stat['audio_sec']:.1f}s → {stat['transcribe_sec']:.2f}s{marker}"
        )
        if stat["transcribe_sec"] > HALLUCINATION_THRESHOLD_SEC:
            hallucination_indices.append(i)

    if not hallucination_indices:
        print("\n沒有偵測到幻覺段落！")
        engine.unload()
        return

    print(f"\n找到 {len(hallucination_indices)} 個幻覺段落: "
          f"{[i+1 for i in hallucination_indices]}\n")

    # Step 2: Test fallback strategies on each hallucination segment
    experiment_results = []

    for idx in hallucination_indices:
        seg_audio = captured_segments[idx]
        stat = stats[idx]
        seg_sec = len(seg_audio) / sample_rate

        print(f"=== 段 {idx+1} ({seg_sec:.1f}s, 原始轉錄 {stat['transcribe_sec']:.2f}s) ===")
        print(f"  原始文字: {stat['raw'][:100]}...")
        print()

        # Strategy C: retry
        print("  策略 C: 直接重試（3次）...")
        retry_results = strategy_c_retry(engine, seg_audio, sample_rate)
        for r in retry_results:
            marker = " *** 仍幻覺" if r["is_hallucination"] else " ✓ 正常"
            print(f"    嘗試 {r['attempt']}: {r['transcribe_sec']:.2f}s{marker}")
        print()

        # Strategy A: re-segment
        print("  策略 A: 重新切段...")
        # Need a fresh VAD model for each test
        vad_model_a, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        resegment_results = strategy_a_resegment(
            engine, pipeline, seg_audio, sample_rate, vad_model_a
        )
        for r in resegment_results:
            status = "有幻覺" if r["any_hallucination"] else "✓ 全正常"
            print(
                f"    silence={r['silence_ms']}ms → {r['n_sub_segments']} 子段, "
                f"總辨識 {r['total_transcribe_sec']:.2f}s ({status})"
            )
        print()

        # Strategy B: pad
        print("  策略 B: 靜音平移...")
        pad_results = strategy_b_pad(engine, seg_audio, sample_rate)
        for r in pad_results:
            for pos in ["front_pad", "back_pad", "both_pad"]:
                p = r[pos]
                marker = "幻覺" if p["is_hallucination"] else "✓"
                label = {"front_pad": "前", "back_pad": "後", "both_pad": "前後"}[pos]
                print(
                    f"    pad={r['pad_ms']}ms ({label}): "
                    f"{p['transcribe_sec']:.2f}s ({marker})"
                )
        print()

        # Strategy D: volume adjustment
        print("  策略 D: 音量調整...")
        volume_results = strategy_d_volume(engine, seg_audio, sample_rate)
        for r in volume_results:
            marker = "幻覺" if r["is_hallucination"] else "✓"
            print(f"    {r['method']}: {r['transcribe_sec']:.2f}s ({marker})")
        print()

        experiment_results.append({
            "segment_index": idx + 1,
            "audio_sec": round(seg_sec, 2),
            "original": {
                "text": stat["raw"],
                "processed": stat["processed"],
                "transcribe_sec": stat["transcribe_sec"],
            },
            "strategy_c_retry": retry_results,
            "strategy_a_resegment": resegment_results,
            "strategy_b_pad": pad_results,
            "strategy_d_volume": volume_results,
        })

    # Save results
    report = {
        "meta": {
            "audio_file": audio_path.name,
            "hallucination_threshold_sec": HALLUCINATION_THRESHOLD_SEC,
            "total_segments": len(stats),
            "hallucination_segments": [i + 1 for i in hallucination_indices],
        },
        "experiments": experiment_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"報告已輸出: {OUTPUT_PATH}")

    engine.unload()
    print("Done!")


if __name__ == "__main__":
    main()
