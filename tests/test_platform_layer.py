"""Platform layer: backend selection, command composition, device resolution, paths.

No real clipboard, no real CUDA probe, no real subprocess — every seam is injected,
so these stay in the millisecond-scale test tier alongside processing/.
"""

import os
import subprocess
from pathlib import Path

import pytest

from asr_input.output.clipboard import ClipboardOutput
from asr_input.paths import data_dir, logs_dir, project_root
from asr_input.platform import (
    MacAlert,
    MacClipboard,
    NullAlert,
    NullClipboard,
    WindowsClipboard,
    resolve_device,
    select_alert_sound,
    select_clipboard_backend,
)


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return None


# --- backend selection ---------------------------------------------------------


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [
        ("win32", WindowsClipboard),
        ("darwin", MacClipboard),
        ("linux", NullClipboard),
        ("freebsd13", NullClipboard),
    ],
)
def test_select_clipboard_backend(platform_name, expected):
    assert isinstance(select_clipboard_backend(platform_name), expected)


# --- command composition -------------------------------------------------------


def test_windows_backend_doubles_single_quotes():
    runner = RecordingRunner()
    WindowsClipboard(runner=runner).copy("that's it")
    cmd, kwargs = runner.calls[0]
    assert cmd[:3] == ["powershell", "-NoProfile", "-Command"]
    assert cmd[3] == "Set-Clipboard -Value 'that''s it'"
    assert kwargs["check"] is True


def test_mac_backend_feeds_stdin_not_shell_literal():
    runner = RecordingRunner()
    text = "that's it — 中文，全形。"
    MacClipboard(runner=runner).copy(text)
    cmd, kwargs = runner.calls[0]
    assert cmd == ["pbcopy"]
    # The whole point of stdin: the text never becomes part of the command line,
    # so no quoting rule can be broken by its content.
    assert kwargs["input"] == text.encode("utf-8")
    assert not any(text in part for part in cmd)


def test_null_backend_raises_rather_than_silently_dropping():
    with pytest.raises(OSError, match="haiku-os"):
        NullClipboard(platform_name="haiku-os").copy("text")


# --- ClipboardOutput contract --------------------------------------------------


def test_clipboard_output_returns_text_unchanged():
    class Backend:
        def __init__(self):
            self.seen = []

        def copy(self, text):
            self.seen.append(text)

    backend = Backend()
    assert ClipboardOutput(backend=backend).process("hello") == "hello"
    assert backend.seen == ["hello"]


@pytest.mark.parametrize(
    "error",
    [
        OSError("no such command"),
        subprocess.CalledProcessError(1, "powershell"),
    ],
)
def test_clipboard_failure_never_breaks_the_pipeline(error, capsys):
    class Failing:
        def copy(self, text):
            raise error

    # The transcription is the result; the clipboard is a side effect.
    assert ClipboardOutput(backend=Failing()).process("keep me") == "keep me"
    assert "剪貼簿" in capsys.readouterr().out


# --- device resolution ---------------------------------------------------------


def test_explicit_config_is_respected_without_probing():
    def exploding_probe():
        raise AssertionError("probe must not run when device is explicit")

    device, compute, note = resolve_device("cuda", "float16", "win32", exploding_probe)
    assert (device, compute, note) == ("cuda", "float16", None)


@pytest.mark.parametrize(
    ("platform_name", "cuda_count", "expected"),
    [
        ("win32", 1, ("cuda", "float16")),
        ("win32", 0, ("cpu", "int8")),
        ("linux", 2, ("cuda", "float16")),
        ("darwin", 0, ("cpu", "int8")),
    ],
)
def test_auto_device_matrix(platform_name, cuda_count, expected):
    device, compute, note = resolve_device(None, None, platform_name, lambda: cuda_count)
    assert (device, compute) == expected
    assert note is not None


def test_darwin_never_probes_for_cuda():
    def exploding_probe():
        raise AssertionError("macOS has no CUDA; probing it wastes a torch import")

    assert resolve_device(None, None, "darwin", exploding_probe)[0] == "cpu"


def test_cuda_on_macos_falls_back_instead_of_failing_at_load_time():
    device, compute, note = resolve_device("cuda", "float16", "darwin", lambda: 0)
    assert (device, compute) == ("cpu", "int8")
    assert note and "macOS" in note


def test_float16_on_cpu_is_corrected():
    device, compute, note = resolve_device("cpu", "float16", "win32", lambda: 1)
    assert (device, compute) == ("cpu", "int8")
    assert note is not None


def test_device_value_is_case_and_space_tolerant():
    assert resolve_device(" CUDA ", "float16", "win32", lambda: 1)[0] == "cuda"


# --- path resolution -----------------------------------------------------------


def test_paths_are_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("ASR_INPUT_HOME", raising=False)
    before = project_root()
    monkeypatch.chdir(tmp_path)
    assert project_root() == before


def test_paths_point_at_the_real_checkout():
    root = project_root()
    assert (root / "pyproject.toml").is_file()
    assert logs_dir() == root / "data" / "logs"
    assert data_dir() == root / "data"


def test_home_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_INPUT_HOME", str(tmp_path))
    assert project_root() == Path(os.path.realpath(tmp_path))
    assert logs_dir() == Path(os.path.realpath(tmp_path)) / "data" / "logs"


# --- alert sound ---------------------------------------------------------------


def test_alert_backend_matches_the_platform():
    assert select_alert_sound("win32").name == "windows"
    assert select_alert_sound("darwin").name == "macos"
    assert select_alert_sound("linux").name == "null"


def test_mac_alert_uses_osascript_beep():
    calls = []
    MacAlert(runner=lambda *a, **kw: calls.append((a, kw))).play()
    assert calls[0][0][0] == ["osascript", "-e", "beep"]


def test_alert_failure_is_reported_not_raised():
    """A missing sound device must not take down the caller's watchdog thread."""

    def explode(*_args, **_kwargs):
        raise OSError("no audio device")

    assert MacAlert(runner=explode).play() is False


def test_null_alert_reports_that_nothing_played():
    assert NullAlert().play() is False
