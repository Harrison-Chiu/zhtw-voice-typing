"""Copy transcription result to clipboard."""

import subprocess

from asr_input.processing.pipeline import TextProcessor


class ClipboardOutput(TextProcessor):
    def process(self, text: str) -> str:
        # PowerShell single-quoted literal: the only special char is ' itself,
        # escaped by doubling it. Without this, text like "that's" breaks parsing.
        escaped = text.replace("'", "''")
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Set-Clipboard -Value '{escaped}'"],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError) as e:
            # Clipboard is a side-effect, not the result — never crash the pipeline over it.
            print(f"    ⚠️ 複製到剪貼簿失敗（已略過）: {e}")
        return text
