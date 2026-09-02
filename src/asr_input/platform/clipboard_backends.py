"""Platform-specific clipboard backends.

Each backend shells out to whatever the host OS provides. They raise on failure;
deciding whether a clipboard failure is fatal belongs to the caller
(`output/clipboard.py` treats it as a warning, since the transcription itself
already succeeded).

The `runner` seam exists so tests can assert on the exact command composed for
each platform without touching a real clipboard.
"""

import subprocess
import sys
from typing import Protocol


class ClipboardBackend(Protocol):
    def copy(self, text: str) -> None: ...


class WindowsClipboard:
    """PowerShell `Set-Clipboard`. Semantics unchanged from the pre-platform-layer code."""

    name = "windows"

    def __init__(self, runner=subprocess.run) -> None:
        self._run = runner

    def copy(self, text: str) -> None:
        # PowerShell single-quoted literal: the only special char is ' itself,
        # escaped by doubling it. Without this, text like "that's" breaks parsing.
        escaped = text.replace("'", "''")
        self._run(
            ["powershell", "-NoProfile", "-Command", f"Set-Clipboard -Value '{escaped}'"],
            check=True,
            capture_output=True,
        )


class MacClipboard:
    """`pbcopy`, fed through stdin.

    stdin avoids the quoting problem entirely — there is no shell literal to escape,
    so no input can break the command the way an apostrophe can on the Windows path.
    """

    name = "macos"

    def __init__(self, runner=subprocess.run) -> None:
        self._run = runner

    def copy(self, text: str) -> None:
        self._run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )


class NullClipboard:
    """No known clipboard command for this platform.

    Raises rather than silently succeeding, so the caller reports it instead of
    leaving the user to wonder why nothing was pasted.
    """

    name = "null"

    def __init__(self, platform_name: str = sys.platform) -> None:
        self._platform_name = platform_name

    def copy(self, text: str) -> None:
        raise OSError(f"no clipboard backend for platform {self._platform_name!r}")


def select_clipboard_backend(platform_name: str | None = None, runner=None) -> ClipboardBackend:
    """Pick a backend by `sys.platform` value."""
    platform_name = sys.platform if platform_name is None else platform_name
    kwargs = {} if runner is None else {"runner": runner}
    if platform_name == "win32":
        return WindowsClipboard(**kwargs)
    if platform_name == "darwin":
        return MacClipboard(**kwargs)
    return NullClipboard(platform_name=platform_name)
