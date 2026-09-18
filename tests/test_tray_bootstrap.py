"""Lightweight tray bootstrap behavior."""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from asr_input import tray
from asr_input.lifecycle import CaptureState, EffectKind, ModelState
from asr_input.tray import TrayApp


class FakeIcon:
    def __init__(self) -> None:
        self.visible = False


class FakeListener:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def _silent_launcher_source() -> str:
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "start_tray.vbs"
    return launcher.read_bytes().decode("ascii")


def test_silent_launcher_remains_ascii_for_windows_script_host():
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "start_tray.vbs"

    assert launcher.read_bytes().isascii()


def test_silent_launcher_defaults_to_manual_but_honours_autostart():
    """開機啟動的捷徑靠 /autostart 走 lazy 模式；桌面捷徑不帶參數時必須維持 manual 預載。"""
    source = _silent_launcher_source()

    assert 'Named.Exists("autostart")' in source
    assert '--startup-mode " & _' in source
    assert "--startup-mode manual" not in source


def test_tray_becomes_visible_before_heavy_setup(monkeypatch):
    app = TrayApp()
    icon = FakeIcon()
    setup_visibility = []
    monkeypatch.setattr(app, "_setup", lambda: setup_visibility.append(icon.visible))

    app._on_tray_ready(icon)

    assert setup_visibility == [True]


def test_hotkey_listener_starts_without_blocking_setup(monkeypatch):
    listener = FakeListener()
    monkeypatch.setattr(tray.keyboard, "Listener", lambda **kwargs: listener)
    app = TrayApp()

    app._listen_hotkey()

    assert listener.started
    assert app._keyboard_listener is listener


def test_toggle_before_capture_is_ready_reports_progress_instead_of_recording(monkeypatch):
    """A press during startup must explain itself, not fail or vanish silently."""
    app = TrayApp()
    notices = []
    monkeypatch.setattr(
        tray, "_silent_notify", lambda icon, message, title="": notices.append(message)
    )

    app._toggle()

    assert app._lifecycle.snapshot.capture is CaptureState.IDLE
    assert len(notices) == 1
    assert "啟動中" in notices[0]


def test_startup_notice_is_throttled_for_repeated_presses(monkeypatch):
    app = TrayApp()
    notices = []
    monkeypatch.setattr(
        tray, "_silent_notify", lambda icon, message, title="": notices.append(message)
    )

    app._toggle()
    app._toggle()
    app._toggle()

    assert len(notices) == 1


def test_startup_notice_surfaces_setup_error(monkeypatch):
    app = TrayApp()
    app._setup_error = "RuntimeError: boom"
    notices = []
    monkeypatch.setattr(
        tray, "_silent_notify", lambda icon, message, title="": notices.append(message)
    )

    app._toggle()

    assert "RuntimeError: boom" in notices[0]


def test_hotkey_callback_error_does_not_prevent_the_next_toggle(monkeypatch):
    app = TrayApp()
    toggles = []

    def toggle():
        toggles.append(True)
        if len(toggles) == 1:
            raise UnicodeEncodeError("cp950", "🎤", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr(app, "_toggle", toggle)
    keys = list(app._hotkey)
    for key in keys:
        app._on_key_press(key)

    app._on_key_release(keys[-1])
    app._on_key_press(keys[-1])

    assert toggles == [True, True]


def test_quit_stops_hotkeys_and_finishes_on_background_thread():
    class StoppableIcon:
        stopped = False

        def stop(self):
            self.stopped = True

    app = TrayApp()
    listener = FakeListener()
    icon = StoppableIcon()
    app._keyboard_listener = listener

    app._on_quit(icon, None)
    app._shutdown_thread.join(timeout=2)

    assert listener.stopped
    assert icon.stopped
    assert app._closing


def test_near_zero_input_warns_as_possible_mute_and_clears_on_signal(monkeypatch):
    app = TrayApp()
    times = iter([0.0, 6.0, 7.0])
    monkeypatch.setattr(tray.time, "monotonic", lambda: next(times))

    app._on_input_level(0.0)
    assert app._input_warning is None
    app._on_input_level(0.0)
    assert app._input_warning == "未偵測到聲音／可能靜音"
    app._on_input_level(0.01)
    assert app._input_warning is None


def test_composite_icon_supports_warning_and_hard_error_symbols():
    image = tray._make_icon(
        "#555555",
        count=2,
        ring_color="#ff0000",
        warning=True,
        hard_error=True,
        level=1.0,
    )

    assert image.size == (64, 64)
    center_pixels = [image.getpixel((x, y))[:3] for x in range(16, 48) for y in range(16, 48)]
    assert (255, 255, 255) in center_pixels


def test_recording_ended_white_ring_overrides_volume_ring_without_a_minimum_timer():
    image = tray._make_icon(
        tray.COLORS[tray.State.TRANSCRIBING],
        count=2,
        recording_ended=True,
        ring_color="#E53935",
        level=1.0,
    )

    assert image.getpixel((32, 2))[:3] == (255, 255, 255)


def test_recording_ended_ring_is_independent_from_error_symbol():
    image = tray._make_icon(
        tray.COLORS[tray.State.ERROR],
        recording_ended=True,
        hard_error=True,
    )

    assert image.getpixel((32, 2))[:3] == (255, 255, 255)
    assert image.getpixel((32, 32))[:3] == (255, 255, 255)


def test_stop_feedback_stays_visible_until_this_recording_finishes(monkeypatch):
    result = SimpleNamespace(duration_sec=1.0, segments=(object(), object()))
    app = TrayApp()
    observed = []

    class FakeRecording:
        segments = result.segments

        def stop(self):
            observed.append(("capture-stop", app._stop_feedback))
            return result

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            observed.append(("finalizer-start", app._stop_feedback))

    app._recording = FakeRecording()
    monkeypatch.setattr(tray.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        app,
        "_update_runtime_icon",
        lambda: observed.append(("icon", app._stop_feedback)),
    )

    app._stop_capture()

    assert observed[0] == ("icon", True)
    assert ("capture-stop", True) in observed
    assert ("finalizer-start", True) in observed
    assert app._stop_feedback is True


def test_loading_and_unloaded_symbols_are_drawn_without_font_glyphs():
    loading = tray._make_icon(tray.COLORS[tray.State.LOADING], symbol="loading")
    unloaded = tray._make_icon(tray.COLORS[tray.State.UNLOADED], symbol="unloaded")

    assert loading.getpixel((32, 31))[:3] == (255, 255, 255)
    assert unloaded.getpixel((32, 32))[:3] == (255, 255, 255)
    assert tray.COLORS[tray.State.STARTING] == tray.COLORS[tray.State.LOADING]


def test_main_disc_uses_the_same_larger_geometry_for_every_state():
    for state in tray.State:
        image = tray._make_icon(tray.COLORS[state])
        assert image.getpixel((3, 32))[3] == 255
        assert image.getpixel((2, 32))[3] == 0


def test_rms_mapping_uses_the_observed_zero_to_point_zero_two_range():
    assert tray._rms_to_icon_level(0.0) == 0.0
    assert tray._rms_to_icon_level(0.005) == 0.25
    assert tray._rms_to_icon_level(0.01) == 0.5
    assert tray._rms_to_icon_level(0.015) == 0.75
    assert tray._rms_to_icon_level(0.02) == 1.0
    assert tray._rms_to_icon_level(0.5) == 1.0


def test_live_asr_flashes_amber_while_microphone_activity_remains_visible():
    app = TrayApp()
    app._tray = FakeIcon()
    app._recording = SimpleNamespace(segments=())
    snapshot_type = app._lifecycle.snapshot.__class__
    app._lifecycle._snapshot = snapshot_type(
        model=ModelState.READY,
        capture=CaptureState.RECORDING,
    )
    app._input_level = 0.02

    app._segment_transcribing = False
    app._update_runtime_icon()
    without_asr = app._tray.icon

    app._segment_transcribing = True
    app._update_runtime_icon()
    with_asr = app._tray.icon

    assert without_asr.getpixel((32, 32))[:3] == (229, 57, 53)
    assert with_asr.getpixel((32, 32))[:3] == (251, 140, 0)
    assert without_asr.getpixel((32, 0)) != with_asr.getpixel((32, 0))
    assert "錄音中" in app._tray.title
    assert "辨識中" in app._tray.title


def test_win32_icon_replacement_error_never_escapes_audio_callback_path(capsys):
    class FailingTray:
        @property
        def icon(self):
            return None

        @icon.setter
        def icon(self, value):
            error = OSError("invalid icon handle")
            error.winerror = 1402
            raise error

    app = TrayApp()
    app._tray = FailingTray()

    app._set_tray_visual(icon=tray._make_icon("#555555"))

    assert "Tray icon update failed" in capsys.readouterr().out


class _FakeTimer:
    daemon = False

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def cancel(self):
        pass


class _FakeRecording:
    recording = True

    def __init__(self, callback_count=0):
        self.callback_count = callback_count


def _watchdog_app(monkeypatch, *, callback_count=0):
    """TrayApp wired so the watchdog can be stepped without real timers."""
    app = TrayApp()
    app._recording = _FakeRecording(callback_count)
    app._alert_sound_enabled = False
    monkeypatch.setattr(tray.threading, "Timer", _FakeTimer)
    return app


def _stall(app, seconds):
    """Pretend the last callback arrived `seconds` ago."""
    app._last_callback_at = time.monotonic() - seconds


def test_stalled_capture_warns_before_the_error_threshold(monkeypatch):
    app = _watchdog_app(monkeypatch)
    stopped = []
    monkeypatch.setattr(app, "_stop_capture", lambda: stopped.append(True))

    _stall(app, app._no_data_warning_sec + 0.1)
    app._check_capture_health()

    # Warned, but the capture is still running: the user may still be mid-sentence
    # and a device that resumes should not have cost them the recording.
    assert app._input_warning is not None
    assert stopped == []
    assert app._lifecycle.snapshot.capture is not CaptureState.ERROR


def test_stalled_capture_becomes_hard_error_and_preserves_capture(monkeypatch):
    app = _watchdog_app(monkeypatch)
    stopped = []
    monkeypatch.setattr(app, "_stop_capture", lambda: stopped.append(True))

    _stall(app, app._no_data_error_sec + 0.1)
    app._check_capture_health()

    assert stopped == [True]
    assert app._lifecycle.snapshot.capture is CaptureState.ERROR


def test_capture_that_never_delivered_data_is_reported_differently(monkeypatch):
    never = _watchdog_app(monkeypatch, callback_count=0)
    monkeypatch.setattr(never, "_stop_capture", lambda: None)
    _stall(never, never._no_data_error_sec + 0.1)
    never._check_capture_health()

    midway = _watchdog_app(monkeypatch, callback_count=7)
    midway._last_callback_count = 7
    monkeypatch.setattr(midway, "_stop_capture", lambda: None)
    _stall(midway, midway._no_data_error_sec + 0.1)
    midway._check_capture_health()

    assert "自始未送出" in never._lifecycle.snapshot.error
    assert "中斷" in midway._lifecycle.snapshot.error


def test_resumed_callbacks_clear_the_warning(monkeypatch):
    app = _watchdog_app(monkeypatch)
    monkeypatch.setattr(app, "_stop_capture", lambda: None)

    _stall(app, app._no_data_warning_sec + 0.1)
    app._check_capture_health()
    assert app._input_warning is not None

    app._recording.callback_count = 1
    app._check_capture_health()

    assert app._input_warning is None
    assert app._lifecycle.snapshot.capture is not CaptureState.ERROR


def test_capture_fault_plays_the_alert_sound_once_enabled(monkeypatch):
    app = _watchdog_app(monkeypatch)
    app._alert_sound_enabled = True
    played = []
    app._alert_sound = SimpleNamespace(name="fake", play=lambda: played.append(True) or True)
    monkeypatch.setattr(app, "_stop_capture", lambda: None)

    _stall(app, app._no_data_error_sec + 0.1)
    app._check_capture_health()

    assert played == [True]


def test_near_zero_level_never_plays_the_alert_sound(monkeypatch):
    """Ordinary silence is not a fault; only a dead data stream makes noise."""
    app = _watchdog_app(monkeypatch)
    app._alert_sound_enabled = True
    played = []
    app._alert_sound = SimpleNamespace(name="fake", play=lambda: played.append(True) or True)

    app._near_zero_warning_sec = 0.0
    app._on_input_level(0.0)
    app._on_input_level(0.0)

    assert played == []


def test_model_ready_mid_capture_catches_up_then_accepts_live_segment_once(monkeypatch):
    class FakeRecording:
        recording = True
        segments = (SimpleNamespace(audio=np.ones(4, dtype=np.float32), probabilities=(0.9,)),)

    class FakeTranscriber:
        def __init__(self):
            self.started = False
            self.segments = []

        def start_segment_stream(self):
            self.started = True

        def feed_segment(self, audio, probabilities):
            self.segments.append((audio.copy(), probabilities))

    app = TrayApp()
    app._recording = FakeRecording()
    app._lifecycle._snapshot = app._lifecycle.snapshot.__class__(model=ModelState.READY)
    transcriber = FakeTranscriber()
    monkeypatch.setattr(app, "_new_transcriber", lambda: transcriber)

    app._maybe_start_live_transcription()
    app._on_captured_segment(FakeRecording.segments[0], 0)
    second = SimpleNamespace(audio=np.ones(4, dtype=np.float32) * 2, probabilities=(0.8,))
    app._on_captured_segment(second, 1)

    assert transcriber.started
    assert [int(audio[0]) for audio, _ in transcriber.segments] == [1, 2]


def test_startup_automatically_requeues_previously_failed_audio(tmp_path, monkeypatch):
    audio_path = tmp_path / "failed.wav"
    audio_path.write_bytes(b"audio")
    stored = SimpleNamespace(
        id="failed-job",
        state="failed",
        audio_path=audio_path,
        received_frames=16000,
        sample_rate=16000,
    )

    class FakeStore:
        def __init__(self):
            self.marked = []

        def recoverable(self):
            return [stored]

        def mark_pending(self, job_id):
            self.marked.append(job_id)

    app = TrayApp()
    app._history_store = FakeStore()
    result = SimpleNamespace(duration_sec=1.0)
    monkeypatch.setattr(app, "_decode_stored_capture", lambda value: result)

    effects = app._restore_pending_jobs()

    assert app._history_store.marked == ["failed-job"]
    assert app._pending_jobs[0][0] is result
    assert EffectKind.LOAD_MODEL in {effect.kind for effect in effects}


def test_queue_badge_is_independent_from_center_counter():
    without_queue = tray._make_icon(tray.COLORS[tray.State.STREAMING], count=3)
    with_queue = tray._make_icon(tray.COLORS[tray.State.STREAMING], count=3, queued=True)

    assert without_queue.getpixel((55, 55)) != with_queue.getpixel((55, 55))
    assert without_queue.getpixel((32, 32)) == with_queue.getpixel((32, 32))


def test_manual_startup_preloads_model_but_autostart_remains_lazy():
    manual = TrayApp(startup_mode="manual")
    autostart = TrayApp(startup_mode="autostart")

    manual_effects = manual._startup_effects(())
    autostart_effects = autostart._startup_effects(())

    assert EffectKind.LOAD_MODEL in {effect.kind for effect in manual_effects}
    assert autostart_effects == ()


def test_persistence_failure_becomes_visible_and_releases_stop_feedback(monkeypatch):
    app = TrayApp()
    result = SimpleNamespace(duration_sec=1.0)
    app._capture_generation = 7
    app._stop_feedback = True
    app._stop_feedback_generation = 7
    app._captured_jobs.append((result, None, 7))
    app._finalizer_running = True
    monkeypatch.setattr(
        app,
        "_persist_pending",
        lambda value: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    app._finalize_captured_jobs()

    assert app._stop_feedback is False
    assert app._failed_jobs[0] == (result, None)
    assert app._lifecycle.snapshot.capture is CaptureState.ERROR
