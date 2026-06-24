"""一次性評測：whisper-large-v3-turbo 的 zh-TW 微調 vs 現況 faster-whisper baseline。

目的只是「眼睛看品質值不值得換」，不接進主程式。
- baseline：現有 config（faster-whisper large-v3-turbo + VAD 切段 + initial_prompt）。
- 候選：JacobLinCool/whisper-large-v3-turbo-common_voice_19_0-zh-TW，走 transformers
  pipeline、chunk_length_s=30（8 分鐘音訊不切會 OOM）。
兩者都印原始輸出 + 過現有後處理（OpenCC+詞表）的對照，並掃簡體字殘留。

注意：候選走固定 30s 分塊、baseline 走 VAD 切段，切法不同，比較的是「字體/標點/
品質」層級，不是逐段對齊。

用法：.venv\\Scripts\\python.exe scripts/test_hf_whisper_zhtw.py [audio_path]
"""

import gc
import io
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from asr_input.asr import build_engine  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.main import build_pipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO = PROJECT_ROOT / "data/test_audio/簡報日.m4a"
CANDIDATE_MODEL = "JacobLinCool/whisper-large-v3-turbo-common_voice_19_0-zh-TW"

# 常見簡體字取樣，用來快速掃殘留（非窮舉，給眼睛判斷的輔助）
SIMPLIFIED_SAMPLE = set("软视启发这说应说话时间题问题为业产权术语种类别样东")


def scan_simplified(text: str) -> list[str]:
    return sorted({c for c in text if c in SIMPLIFIED_SAMPLE})


def decode_audio(path: Path, sr: int) -> np.ndarray:
    """用 ffmpeg 解碼成 16k mono float32。

    繞過 librosa：目前 .venv 的 numpy 2.5 與 numba 不相容（numba 要 ≤2.4），
    librosa 會 import 失敗。ffmpeg 直出 PCM 不依賴 numba。
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_baseline(audio, sample_rate: int, config: dict) -> str:
    asr_cfg = config["asr"]
    print(f"\n[baseline] {asr_cfg['engine']} / {asr_cfg['model_id']}（含 VAD 切段）")
    engine = build_engine(asr_cfg, vad_cfg=config.get("vad"))
    engine.load()
    t0 = time.time()
    raw = engine.transcribe(audio, sample_rate)
    print(f"[baseline] 轉錄耗時 {time.time() - t0:.1f}s")
    engine.unload()
    _free()
    return raw


def run_candidate(audio, sample_rate: int) -> str | None:
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        print(f"\n[候選] import transformers 失敗：{e}")
        print("[候選] 使用者環境未裝 transformers，跳過候選評測（不自行安裝）。")
        print(
            "       若要測，請自行跑：uv pip install -U transformers --python "
            '"D:\\Harrison\\code_test\\asr-input\\.venv\\Scripts\\python.exe"'
        )
        return None

    print(f"\n[候選] {CANDIDATE_MODEL}（transformers pipeline, chunk_length_s=30）")
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=CANDIDATE_MODEL,
        torch_dtype=dtype,
        device=device,
    )
    t0 = time.time()
    out = asr(
        {"raw": audio, "sampling_rate": sample_rate},
        chunk_length_s=30,
        generate_kwargs={"language": "zh", "task": "transcribe"},
    )
    print(f"[候選] 轉錄耗時 {time.time() - t0:.1f}s")
    del asr
    _free()
    return out["text"]


def report(name: str, raw: str, post) -> None:
    processed = post.run(raw)
    print(f"\n========== {name} ==========")
    print(f"[原始] {raw}")
    print(f"[後處理] {processed}")
    simp_raw = scan_simplified(raw)
    print(f"[簡體殘留·原始] {'無' if not simp_raw else ' '.join(simp_raw)}")
    print(
        f"[連續刪節號 …] 原始出現 {raw.count('…')} 次 / 全形逗號 {raw.count('，')} "
        f"半形逗號 {raw.count(',')}"
    )


def main() -> None:
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not audio_path.exists():
        print(f"找不到音檔：{audio_path}")
        return

    config = load_config()
    sample_rate = config["audio"]["sample_rate"]
    post = build_pipeline(config, include_output=False)

    print(f"音檔：{audio_path.name}")
    print("解碼中（ffmpeg）...")
    audio = decode_audio(audio_path, sample_rate)
    print(f"音訊長度：{len(audio) / sample_rate:.1f}s")

    baseline_raw = run_baseline(audio, sample_rate, config)
    candidate_raw = run_candidate(audio, sample_rate)

    report("BASELINE (faster-whisper 現況)", baseline_raw, post)
    if candidate_raw is not None:
        report("CANDIDATE (whisper zh-TW 微調)", candidate_raw, post)

    print(
        "\n判斷準則：候選是否原生繁體（簡體殘留=無）、標點正常全形、無大量 …、"
        "短句穩定度 ≥ baseline。達標才考慮換，否則保留現況。"
    )


if __name__ == "__main__":
    main()
