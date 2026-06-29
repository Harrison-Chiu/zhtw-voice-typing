"""Fun-ASR-Nano-2512 評測 v2 — 修正載入路徑（2026-06-29）。

v1 的失敗根因（已釐清）：Fun-ASR-Nano 是 LLM-ASR（SenseVoice encoder + Qwen3-0.6B
backbone），不是傳統 funasr 的 CTC/Paraformer 模型。HF 模型快照裡**沒有 model.py**，
通用 `pip install funasr`（1.3.14）的 AutoModel 找不到 `FunASRNano` 類別，退化用 CTC
路徑硬載 model.pt → 噴一堆 `miss key: ctc_decoder.*`，輸出退化成單字重複垃圾。

修法（依官方 https://github.com/FunAudioLLM/Fun-ASR README）：把 `remote_code` 指到
官方 repo 根目錄的 model.py（內含 FunASRNano 實作），並把該 repo 目錄加進 sys.path
讓 model.py 的相對 import（ctc 等）找得到。funasr>=1.3.0 即可（現有 1.3.14 符合）。

用法（隔離環境）：
    .venv-funasr\\Scripts\\python.exe scripts/test_funasr_nano_v2.py <repo_dir> [audio]
其中 <repo_dir> 是 clone 下來的 Fun-ASR repo 路徑（含 model.py）。
"""

import io
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_AUDIO = PROJECT_ROOT / "data/test_audio/機器人展示，語音轉錄測試.m4a"
SIMPLIFIED_SAMPLE = set("软视启发这说应话时题问为业产权术语种类别样东马达机现们个")


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
    if len(sys.argv) < 2:
        print("用法：python test_funasr_nano_v2.py <Fun-ASR_repo_dir> [audio]")
        return
    repo_dir = Path(sys.argv[1]).resolve()
    model_py = repo_dir / "model.py"
    if not model_py.exists():
        print(f"找不到 model.py：{model_py}（請先 git clone FunAudioLLM/Fun-ASR）")
        return
    audio_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_AUDIO

    # 讓 model.py 的相對 import 找得到 repo 內其他模組
    sys.path.insert(0, str(repo_dir))

    import torch
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"裝置：{device} / 音檔：{audio_path.name}")
    wav = to_wav(audio_path)

    print(f"載入 {MODEL}（remote_code={model_py}）...")
    model = AutoModel(
        model=MODEL,
        hub="hf",
        trust_remote_code=True,
        remote_code=str(model_py),
        device=device,
        disable_update=True,
    )

    def run(label: str, **kw):
        try:
            res = model.generate(
                input=[wav], cache={}, batch_size=1, language="中文", itn=True, **kw
            )
            text = res[0].get("text", "")
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"\n[{label}] generate 失敗：{type(e).__name__}: {e}")
            traceback.print_exc()
            return
        simp = scan_simplified(text)
        print(f"\n========== {label} ==========")
        print(f"[原始] {text}")
        print(f"[簡體殘留] {'無' if not simp else ' '.join(simp)}")
        print(
            f"[標點] 全形逗號 {text.count('，')} 半形 {text.count(',')} / 句號 {text.count('。')}"
        )

    run("native (language=中文)")
    run("hotword=台灣 繁體 中文", hotwords=["台灣", "繁體", "中文"])

    print("\n判斷：原生繁體+品質≥現況才值得；若簡體且無法引導 → 等於要再加 OpenCC。")


if __name__ == "__main__":
    main()
