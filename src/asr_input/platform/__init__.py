"""Platform-specific implementations, kept behind protocols.

Everything above this package (pipeline, streaming, ASR, processing) is
platform-agnostic; anything that shells out to an OS-specific command or probes
OS-specific hardware belongs here. See `docs/macos-port-plan.md`.
"""

from asr_input.platform.clipboard_backends import (
    ClipboardBackend,
    MacClipboard,
    NullClipboard,
    WindowsClipboard,
    select_clipboard_backend,
)
from asr_input.platform.device import cuda_device_count, resolve_device

__all__ = [
    "cuda_device_count",
    "resolve_device",
    "ClipboardBackend",
    "MacClipboard",
    "NullClipboard",
    "WindowsClipboard",
    "select_clipboard_backend",
]
