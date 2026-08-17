"""OpenRouter engine tests without network access or real credentials."""

import base64
import io
import json
import wave

import numpy as np
import pytest

from asr_input.asr.openrouter import OpenRouterASREngine


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_transcription_encodes_wav_and_records_usage(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "text": "測試成功",
                "usage": {
                    "seconds": 0.25,
                    "cost": 0.0001,
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    engine = OpenRouterASREngine("test/model", language="zh", timeout_sec=12, zdr=True)
    engine.load()
    text = engine.transcribe(np.array([-1.0, 0.0, 1.0], dtype=np.float32), 16000)

    assert text == "測試成功"
    assert captured["timeout"] == 12
    assert captured["request"].get_header("Authorization") == "Bearer test-secret"
    payload = json.loads(captured["request"].data)
    assert payload["model"] == "test/model"
    assert payload["provider"] == {"zdr": True}
    assert payload["language"] == "zh"
    with wave.open(io.BytesIO(base64.b64decode(payload["input_audio"]["data"]))) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3
    assert engine.last_usage.cost_usd == pytest.approx(0.0001)
    assert engine.last_usage.total_tokens == 7


def test_load_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(OpenRouterASREngine, "_read_credential", lambda self: None)
    engine = OpenRouterASREngine("test/model")

    with pytest.raises(RuntimeError, match="API key not found"):
        engine.load()
