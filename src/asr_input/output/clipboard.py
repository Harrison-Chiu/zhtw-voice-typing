"""Copy transcription result to clipboard."""

import subprocess

from asr_input.platform import ClipboardBackend, select_clipboard_backend
from asr_input.processing.pipeline import TextProcessor


class ClipboardOutput(TextProcessor):
    def __init__(self, backend: ClipboardBackend | None = None) -> None:
        self._backend = select_clipboard_backend() if backend is None else backend

    def process(self, text: str) -> str:
        try:
            self._backend.copy(text)
        except (subprocess.CalledProcessError, OSError) as e:
            # Clipboard is a side-effect, not the result — never crash the pipeline over it.
            print(f"    ⚠️ 複製到剪貼簿失敗（已略過）: {e}")
        return text
