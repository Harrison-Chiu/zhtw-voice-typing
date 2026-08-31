"""Backend-agnostic Silero VAD v5 runners for the recording path.

`StreamingVAD` only needs two things from the model: a probability for one
512-sample window, and a way to reset the recurrent state between recordings.
Expressing that as a small protocol here lets the recording path run on ONNX
Runtime, which does not import torch.

Why that matters: on this project's Windows dev machine `import torch` is the
single largest startup cost (measured 2026-09-01: 18.1s for the first import
after a long idle, 1.7s with a warm OS file cache) and, at startup, torch is
loaded *only* for the VAD -- faster-whisper runs on CTranslate2 and the Silero
model runs on CPU.  A torch-free VAD therefore moves "hotkey can actually
record" much earlier in a cold start.

Both backends run the same upstream Silero v5 weights: the `silero-vad`
distribution ships `silero_vad.jit` and `silero_vad.onnx` side by side.  ONNX is
the default since 2026-09-01, after `experiments/compare_vad_backends.py` found
the two numerically interchangeable on 498s of real test audio (max per-window
probability difference 4.7e-06, zero threshold-decision flips, identical
`StreamingVAD` segment boundaries -- 32/32 on the 8-minute file).  This is a
runtime swap of identical weights, not the VAD-replacement question that
`docs/vad-experiment-plan.md` gates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class VadModel(Protocol):
    """What `StreamingVAD` requires of a VAD backend."""

    def __call__(self, window: np.ndarray, sample_rate: int) -> float: ...

    def reset_states(self) -> None: ...


def silero_data_dir() -> Path:
    """Locate the installed `silero_vad` package data without importing it.

    `importlib.resources` would execute the package's `__init__`, which pulls in
    torch -- exactly what the ONNX backend exists to avoid.  `find_spec` on a
    top-level package resolves its location without running any of its code.
    """
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("silero-vad package not found; cannot locate VAD weights")
    return Path(next(iter(spec.submodule_search_locations))) / "data"


class SileroOnnxVAD:
    """Silero VAD v5 on ONNX Runtime — no torch import anywhere in this path.

    The v5 ONNX graph is not a plain "512 samples in, probability out" model:
    the caller must prepend the previous window's trailing `context_size`
    samples, because the exported graph drops the padding its internal STFT
    would otherwise need.  Upstream does this inside `OnnxWrapper.__call__`
    (`silero_vad/utils_vad.py`); feeding a bare 512-sample window instead
    produces plausible-looking but meaningless probabilities.  This class
    mirrors the upstream state and context handling exactly.
    """

    STATE_SHAPE = (2, 1, 128)

    def __init__(self, model_path: str | Path | None = None, num_threads: int = 1) -> None:
        import onnxruntime as ort

        path = Path(model_path) if model_path else silero_data_dir() / "silero_vad.onnx"
        if not path.is_file():
            raise FileNotFoundError(f"Silero ONNX weights not found: {path}")
        options = ort.SessionOptions()
        # One window is 32 ms of 16 kHz audio; extra threads only add scheduling
        # overhead and compete with the PortAudio callback thread.
        options.inter_op_num_threads = num_threads
        options.intra_op_num_threads = num_threads
        self.model_path = path
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros(self.STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, 0), dtype=np.float32)

    @staticmethod
    def _context_size(sample_rate: int) -> int:
        if sample_rate == 16000:
            return 64
        if sample_rate == 8000:
            return 32
        raise ValueError(f"Silero VAD supports 8000/16000 Hz, got {sample_rate}")

    def __call__(self, window: np.ndarray, sample_rate: int) -> float:
        context_size = self._context_size(sample_rate)
        if self._context.shape[-1] != context_size:
            self._context = np.zeros((1, context_size), dtype=np.float32)
        audio = np.asarray(window, dtype=np.float32).reshape(1, -1)
        audio = np.concatenate([self._context, audio], axis=1)
        output, state = self._session.run(
            None,
            {
                "input": audio,
                "state": self._state,
                "sr": np.array(sample_rate, dtype=np.int64),
            },
        )
        self._state = state
        self._context = audio[..., -context_size:]
        return float(output[0][0])

    def reset_states(self) -> None:
        self._state = np.zeros(self.STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, 0), dtype=np.float32)


class TorchSileroVAD:
    """Silero VAD v5 on the torch JIT model, wrapped in the shared protocol."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @property
    def raw_model(self) -> Any:
        return self._model

    def __call__(self, window: np.ndarray, sample_rate: int) -> float:
        import torch

        return float(self._model(torch.from_numpy(window), sample_rate).item())

    def reset_states(self) -> None:
        self._model.reset_states()


def load_torch_vad() -> TorchSileroVAD:
    import torch

    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return TorchSileroVAD(model)


def build_vad_model(vad_cfg: dict | None = None) -> VadModel:
    """Construct the VAD backend named by `vad.backend` (default `onnx`)."""
    cfg = vad_cfg or {}
    backend = cfg.get("backend", "onnx")
    if backend == "onnx":
        return SileroOnnxVAD(model_path=cfg.get("onnx_model_path"))
    if backend == "torch":
        return load_torch_vad()
    raise ValueError(f"Unknown vad.backend: {backend!r}. Supported: torch, onnx")


def as_vad_model(model: Any) -> VadModel:
    """Accept either a protocol backend or a bare torch JIT model."""
    if isinstance(model, SileroOnnxVAD | TorchSileroVAD):
        return model
    return TorchSileroVAD(model)
