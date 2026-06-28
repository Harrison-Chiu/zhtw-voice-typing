"""冷啟動各階段細分計時 — 釐清 tray 首次載入 ~27s 花在哪。

背景：tray 首啟計時顯示「CUDA 暖機 0.1s / 載 Whisper ~27.7s / 載 VAD 1.2s」，
瓶頸全在「載 Whisper」。但 faster-whisper 推論走 CTranslate2（非 PyTorch），
故 _setup() 的 PyTorch 暖機暖不到它。本腳本把「載 Whisper」再拆成：
  import faster_whisper（載 CTranslate2 / cuDNN 等原生 DLL）
  vs WhisperModel() 建構（CTranslate2 CUDA 初始化 + 讀權重）
並在同進程跑「第二次建構」對照 tray 卸載/重載為何快很多。

跑法（在自己的終端，乾淨進程）：
    .venv\\Scripts\\python.exe experiments/profile_cold_start.py

注意：OS 檔案快取會讓二次量測偏快。要量真正冷啟動，重開機後再跑一次。
"""

import time


def stamp(label: str, t0: float) -> float:
    dt = time.perf_counter() - t0
    print(f"[{dt:6.2f}s] {label}", flush=True)
    return dt


t0 = time.perf_counter()
import torch  # noqa: E402

stamp("import torch", t0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"         device={device}, cuda.is_available()={torch.cuda.is_available()}", flush=True)

t0 = time.perf_counter()
if device == "cuda":
    torch.zeros(1).to(device)
    torch.cuda.synchronize()
stamp("PyTorch CUDA 暖機（torch.zeros(1).to(cuda)）", t0)

t0 = time.perf_counter()
from faster_whisper import WhisperModel  # noqa: E402

stamp("import faster_whisper（載 CTranslate2/cuDNN 原生 DLL）", t0)

t0 = time.perf_counter()
model = WhisperModel("large-v3-turbo", device=device, compute_type="float16")
stamp("WhisperModel 建構 #1（CT2 CUDA 初始化 + 讀權重，冷）", t0)

del model
if device == "cuda":
    torch.cuda.empty_cache()

t0 = time.perf_counter()
model2 = WhisperModel("large-v3-turbo", device=device, compute_type="float16")
stamp("WhisperModel 建構 #2（對照 tray 卸載→重載）", t0)

print("\n結論判讀：", flush=True)
print("  - import faster_whisper 偏大 → 首載大宗是載原生 DLL（首次冷讀磁碟）", flush=True)
print("  - 建構 #1 偏大、#2 明顯變快 → 大宗是讀權重 + CT2 初始化（一次性，重載跳過）", flush=True)
