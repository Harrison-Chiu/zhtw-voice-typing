"""VAD backend contract and ONNX equivalence.

The ONNX backend exists to keep torch off the recording path, so these tests
deliberately avoid importing torch: they check the contract `StreamingVAD`
relies on, and pin the two upstream details that silently produce wrong
probabilities when they are missed (state carry-over and context prepending).

Cross-backend agreement on real audio is not covered here -- that needs the
torch model and multi-minute files; see `experiments/compare_vad_backends.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from asr_input.audio.streaming_vad import StreamingVAD
from asr_input.audio.vad_backend import (
    SileroOnnxVAD,
    TorchSileroVAD,
    as_vad_model,
    build_vad_model,
    silero_data_dir,
)

SAMPLE_RATE = 16000
WINDOW = StreamingVAD.SILERO_CHUNK_SAMPLES


def onnx_model() -> SileroOnnxVAD:
    path = silero_data_dir() / "silero_vad.onnx"
    if not path.is_file():
        pytest.skip(f"Silero ONNX weights not installed: {path}")
    return SileroOnnxVAD()


def test_onnx_backend_returns_probabilities_in_range():
    model = onnx_model()
    rng = np.random.default_rng(0)
    probs = [
        model(rng.standard_normal(WINDOW).astype(np.float32) * 0.05, SAMPLE_RATE) for _ in range(10)
    ]

    assert all(isinstance(p, float) for p in probs)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_onnx_backend_is_stateful_and_reset_restores_the_first_output():
    """Silero v5 is recurrent: without carried state every window would score
    the same, and without reset a new recording would inherit the old one."""
    model = onnx_model()
    window = np.full(WINDOW, 0.1, dtype=np.float32)

    first = model(window, SAMPLE_RATE)
    second = model(window, SAMPLE_RATE)
    model.reset_states()
    first_again = model(window, SAMPLE_RATE)

    assert first != second
    assert first_again == pytest.approx(first)


def test_onnx_backend_prepends_the_previous_windows_context():
    model = onnx_model()
    window = np.full(WINDOW, 0.1, dtype=np.float32)
    model.reset_states()
    model(window, SAMPLE_RATE)

    assert model._context.shape == (1, 64)
    assert np.allclose(model._context, window[-64:])


def test_onnx_backend_rejects_unsupported_sample_rates():
    model = onnx_model()

    with pytest.raises(ValueError, match="8000/16000"):
        model(np.zeros(WINDOW, dtype=np.float32), 44100)


def test_streaming_vad_accepts_a_backend_directly():
    vad = StreamingVAD(on_speech_segment=lambda audio, probs: None)
    model = onnx_model()

    vad.load(model)

    assert vad._model is model


def test_streaming_vad_wraps_a_bare_jit_model_for_older_callers():
    """`streaming.py` and scripts still hand over `torch.hub.load(...)` output."""

    class FakeJitModel:
        def reset_states(self) -> None: ...

    vad = StreamingVAD(on_speech_segment=lambda audio, probs: None)
    vad.load(FakeJitModel())

    assert isinstance(vad._model, TorchSileroVAD)


def test_as_vad_model_is_idempotent_for_backends():
    model = onnx_model()

    assert as_vad_model(model) is model


def test_build_vad_model_rejects_unknown_backends():
    with pytest.raises(ValueError, match="Unknown vad.backend"):
        build_vad_model({"backend": "webrtc"})


def test_build_vad_model_selects_onnx_without_importing_torch():
    onnx_model()  # skip early if the weights are missing

    model = build_vad_model({"backend": "onnx"})

    assert isinstance(model, SileroOnnxVAD)
