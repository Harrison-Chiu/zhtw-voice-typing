"""一次性評測：Fun-ASR-Nano-2512 的辨識品質與「能否引導繁體」。

獨立腳本，跑在隔離環境 .venv-funasr（funasr + torch 都裝在那）：
    .venv-funasr\\Scripts\\python.exe scripts/test_funasr_nano.py [audio]

不 import asr_input（隔離環境沒裝本專案套件）。用 ffmpeg 把音檔轉成 16k mono
wav 餵給 funasr。重點看：
  1. 原生輸出是簡體還繁體（language="中文"）。
  2. 試著用 hotword 餵「台灣繁體中文」看能不能被引導（funasr 部分模型吃 hotword）。
若原生簡體且無法引導 → 跟 Qwen 一樣只能靠 OpenCC 後處理，對現況無增益。

已知結論（2026-06-25）：用 `pip install funasr`（1.3.14）載入此模型會噴一堆
`miss key in ckpt: ctc_decoder.*`，輸出退化成單字+逗號重複數百次的垃圾。研判是
funasr 版本與 Fun-ASR-Nano-2512（2025/12 模型）對不上。重試方向：依 FunAudioLLM
官方 GitHub 的 requirements 裝對應 funasr 版本，而非通用 pip funasr。
"""

import io
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO = PROJECT_ROOT / "data/test_audio/隊伍簡報日_8分鐘.m4a"
MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"

SIMPLIFIED_SAMPLE = set("软视启发这说应话时题问为业产权术语种类别样东马达机现")


def scan_simplified(text: str) -> list[str]:
    return sorted({c for c in text if c in SIMPLIFIED_SAMPLE})


def to_wav(src: Path, sr: int = 16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # noqa: SIM115
    tmp.close()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sr),
            tmp.name,
        ],
        check=True,
    )
    return tmp.name


def main() -> None:
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not audio_path.exists():
        print(f"找不到音檔：{audio_path}")
        return

    import torch
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"裝置：{device}")
    print(f"音檔：{audio_path.name}")
    wav = to_wav(audio_path)

    print(f"載入模型 {MODEL} ...")
    model = AutoModel(model=MODEL, hub="hf", trust_remote_code=True, device=device)

    def run(label: str, **kw):
        try:
            res = model.generate(
                input=[wav], cache={}, batch_size=1, language="中文", itn=True, **kw
            )
            text = res[0].get("text", "")
        except Exception as e:  # noqa: BLE001
            print(f"\n[{label}] generate 失敗：{type(e).__name__}: {e}")
            return
        simp = scan_simplified(text)
        print(f"\n========== {label} ==========")
        print(f"[原始] {text}")
        print(f"[簡體殘留] {'無' if not simp else ' '.join(simp)}")
        print(
            f"[標點] 全形逗號 {text.count('，')} 半形 {text.count(',')} / 刪節號 {text.count('…')}"
        )

    run("native (language=中文)")
    run("hotword=台灣繁體中文", hotword="台灣繁體中文")

    print(
        "\n判斷：原生繁體+品質≥現況才值得；若簡體且無法引導，等於要再加 OpenCC，"
        "對現況 faster-whisper 路線無增益。"
    )


if __name__ == "__main__":
    main()
