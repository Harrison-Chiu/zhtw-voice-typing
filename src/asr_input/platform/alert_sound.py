"""Audible alerts for capture faults, kept behind a protocol.

Tray notifications are deliberately silent (`_silent_notify`), because most of
them report ordinary progress. One case is different: the microphone delivering
no audio at all while the user is already speaking. That costs the user the
whole utterance, and a toast they are not looking at does not reach them in
time. Only that signal gets a sound.

Backends never raise: a missing sound device must not turn a microphone warning
into a crashed watchdog thread. `play()` returns whether the sound was emitted,
so callers can log the difference between "no sound wanted" and "sound failed".
"""

import subprocess
import sys
from typing import Protocol


class AlertSound(Protocol):
    name: str

    def play(self) -> bool: ...


class NullAlert:
    """Used when alerts are disabled, or on a platform with no known beep."""

    name = "null"

    def play(self) -> bool:
        return False


class WindowsAlert:
    """`MessageBeep` — the OS-themed alert sound, no audio file to ship."""

    name = "windows"

    def play(self) -> bool:
        try:
            import winsound

            # MB_ICONEXCLAMATION: the system "warning" theme sound. It is
            # asynchronous, so it never blocks the watchdog timer thread.
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            return True
        except Exception:
            return False


class MacAlert:
    """`osascript -e beep` — the user's configured alert sound."""

    name = "macos"

    def __init__(self, runner=subprocess.run) -> None:
        self._run = runner

    def play(self) -> bool:
        try:
            self._run(
                ["osascript", "-e", "beep"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            return True
        except Exception:
            return False


def select_alert_sound(platform: str | None = None) -> AlertSound:
    """Pick the backend for the host OS; unknown platforms stay silent."""
    name = platform or sys.platform
    if name.startswith("win"):
        return WindowsAlert()
    if name == "darwin":
        return MacAlert()
    return NullAlert()
