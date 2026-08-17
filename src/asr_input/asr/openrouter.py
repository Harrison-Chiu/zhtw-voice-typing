"""OpenRouter speech-to-text engine.

The engine is deliberately dormant until selected in config.yaml. API keys are
resolved at load time and are never accepted from the repository configuration.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

from asr_input.asr.base import ASREngine


@dataclass(frozen=True)
class OpenRouterUsage:
    """Billing metadata returned by the most recent transcription."""

    seconds: float | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class OpenRouterASREngine(ASREngine):
    """Send PCM audio to OpenRouter's dedicated transcription endpoint."""

    transcription_latency_is_quality_signal = False

    def __init__(
        self,
        model_id: str,
        *,
        language: str | None = None,
        timeout_sec: float = 45.0,
        api_key_env: str = "OPENROUTER_API_KEY",
        credential_service: str = "asr-input/openrouter",
        credential_username: str = "api-key",
        zdr: bool = True,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.timeout_sec = timeout_sec
        self.api_key_env = api_key_env
        self.credential_service = credential_service
        self.credential_username = credential_username
        self.zdr = zdr
        self.base_url = base_url.rstrip("/")
        self._api_key: str | None = None
        self._last_usage: OpenRouterUsage | None = None

    @property
    def last_usage(self) -> OpenRouterUsage | None:
        return self._last_usage

    def load(self) -> None:
        self._api_key = os.environ.get(self.api_key_env) or self._read_credential()
        if not self._api_key:
            raise RuntimeError(
                f"OpenRouter API key not found. Set {self.api_key_env} or store it in "
                f"Windows Credential Manager as {self.credential_service!r}."
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._api_key is None:
            raise RuntimeError("OpenRouter engine not loaded. Call load() first.")

        body: dict[str, Any] = {
            "model": self.model_id,
            "input_audio": {
                "data": base64.b64encode(_encode_wav(audio, sample_rate)).decode("ascii"),
                "format": "wav",
            },
            "provider": {"zdr": self.zdr},
        }
        if self.language:
            body["language"] = self.language

        request = urllib.request.Request(
            f"{self.base_url}/audio/transcriptions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter transcription failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter transcription connection failed: {exc.reason}") from exc

        usage = payload.get("usage") or {}
        self._last_usage = OpenRouterUsage(
            seconds=_number(usage.get("seconds")),
            cost_usd=_number(usage.get("cost")),
            input_tokens=_integer(usage.get("input_tokens")),
            output_tokens=_integer(usage.get("output_tokens")),
            total_tokens=_integer(usage.get("total_tokens")),
        )
        return str(payload.get("text", ""))

    def unload(self) -> None:
        self._api_key = None
        self._last_usage = None

    def _read_credential(self) -> str | None:
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError(
                "Credential Manager support requires the optional 'keyring' package."
            ) from exc
        return keyring.get_password(self.credential_service, self.credential_username)


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return output.getvalue()


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _integer(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None
