"""Copy transcription result to clipboard."""

from asr_input.processing.pipeline import TextProcessor


class ClipboardOutput(TextProcessor):
    def process(self, text: str) -> str:
        import subprocess

        subprocess.run(
            ["powershell", "-Command", f"Set-Clipboard -Value '{text}'"],
            check=True,
            capture_output=True,
        )
        return text
