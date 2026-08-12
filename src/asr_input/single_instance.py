"""Windows single-instance guard for the tray process."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from ctypes import wintypes

ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    """Own a named Windows mutex for the lifetime of this object.

    Non-Windows platforms are treated as acquired so unit tests and developer
    tools remain portable.  Injectable functions keep the Windows behavior
    testable without creating a real process-global mutex.
    """

    def __init__(
        self,
        name: str,
        *,
        create_mutex: Callable[[str], int] | None = None,
        get_last_error: Callable[[], int] | None = None,
        close_handle: Callable[[int], None] | None = None,
    ) -> None:
        self._handle: int | None = None
        self._close_handle = close_handle
        self.acquired = True

        if create_mutex is None:
            if sys.platform != "win32":
                return
            create_mutex, get_last_error, close_handle = _windows_mutex_functions()
            self._close_handle = close_handle

        assert get_last_error is not None
        assert self._close_handle is not None

        handle = create_mutex(name)
        if not handle:
            raise OSError("CreateMutexW failed")
        if get_last_error() == ERROR_ALREADY_EXISTS:
            self._close_handle(handle)
            self.acquired = False
            return
        self._handle = handle

    def close(self) -> None:
        if self._handle is not None and self._close_handle is not None:
            self._close_handle(self._handle)
            self._handle = None

    def __enter__(self) -> SingleInstance:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def notify_already_running() -> None:
    """Give visible feedback when a desktop shortcut is clicked twice."""
    if sys.platform != "win32":
        return
    MB_ICONINFORMATION = 0x40
    ctypes.windll.user32.MessageBoxW(
        None,
        "ASR Input 已經在執行。請使用系統匣圖示。",
        "ASR Input",
        MB_ICONINFORMATION,
    )


def _windows_mutex_functions() -> tuple[
    Callable[[str], int],
    Callable[[], int],
    Callable[[int], None],
]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    def create_mutex(name: str) -> int:
        return int(kernel32.CreateMutexW(None, False, name))

    def get_last_error() -> int:
        return ctypes.get_last_error()

    def close_handle(handle: int) -> None:
        kernel32.CloseHandle(handle)

    return create_mutex, get_last_error, close_handle
