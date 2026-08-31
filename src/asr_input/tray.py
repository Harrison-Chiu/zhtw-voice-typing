"""System tray app with global hotkey for ASR input."""

from __future__ import annotations

import argparse
import contextlib
import enum
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any

import pystray
from PIL import Image, ImageDraw, ImageFont
from pynput import keyboard

from asr_input.lifecycle import (
    AudioQueueBudget,
    CaptureState,
    Effect,
    EffectKind,
    HotkeyLatch,
    LifecycleMachine,
    ModelState,
)
from asr_input.single_instance import SingleInstance, notify_already_running


class State(enum.Enum):
    STARTING = "starting"
    LOADING = "loading"
    IDLE = "idle"
    STREAMING = "streaming"
    TRANSCRIBING = "transcribing"
    UNLOADED = "unloaded"
    ERROR = "error"


# 配色刻意拉開色相對比，讓 16-32px 的系統匣圖示一眼可辨：
# 灰(待命載入) / 綠(就緒) / 紅(錄音中) / 琥珀(辨識運算中)。
COLORS = {
    State.STARTING: "#607D8B",
    State.LOADING: "#607D8B",
    State.IDLE: "#43A047",
    State.STREAMING: "#E53935",
    State.TRANSCRIBING: "#FB8C00",
    State.UNLOADED: "#455A64",
    State.ERROR: "#C62828",
}

# Microphone activity uses a lighter tint of the current red/amber state.
# It must not shrink the main disc or compete with the central counter.
AUDIO_ACTIVITY_COLORS = {
    State.STREAMING: "#FF8A80",
    State.TRANSCRIBING: "#FFD180",
}

LABELS = {
    State.STARTING: "啟動中...",
    State.LOADING: "載入模型中...",
    State.IDLE: "待機（按快捷鍵錄音）",
    State.STREAMING: "串流辨識中...",
    State.TRANSCRIBING: "辨識中...",
    State.UNLOADED: "錄音就緒（模型將按需載入）",
    State.ERROR: "啟動失敗（請查看通知）",
}

DEFAULT_HOTKEY = "ctrl+shift+space"

_KEY_MAP = {
    "ctrl": keyboard.Key.ctrl_l,
    "alt": keyboard.Key.alt_l,
    "shift": keyboard.Key.shift_l,
    "win": keyboard.Key.cmd,
    "space": keyboard.Key.space,
    "esc": keyboard.Key.esc,
    "tab": keyboard.Key.tab,
    **{f"f{i}": getattr(keyboard.Key, f"f{i}") for i in range(1, 13)},
}


def _parse_hotkey(combo: str) -> set:
    keys: set = set()
    for part in combo.lower().split("+"):
        part = part.strip()
        if part in _KEY_MAP:
            keys.add(_KEY_MAP[part])
        elif len(part) == 1:
            keys.add(keyboard.KeyCode.from_vk(ord(part.upper())))
        else:
            raise ValueError(f"Unknown key in hotkey: {part!r}")
    return keys


def _make_icon(
    color: str,
    count: int | None = None,
    recording_ended: bool = False,
    *,
    ring_color: str | None = None,
    warning: bool = False,
    hard_error: bool = False,
    level: float = 0.0,
    symbol: str | None = None,
    queued: bool = False,
) -> Image.Image:
    img = Image.new("RGBA", (64, 64))
    draw = ImageDraw.Draw(img)
    # Keep a nearly full-size 15 px disc. Microphone activity grows only in the
    # three source pixels outside it, so feedback never steals counter space.
    if recording_ended:
        draw.ellipse((0, 0, 63, 63), fill="white")
    elif ring_color and level > 0:
        activity_edge = max(0, 3 - round(level * 3))
        draw.ellipse(
            (activity_edge, activity_edge, 63 - activity_edge, 63 - activity_edge),
            fill=ring_color,
        )
    draw.ellipse((3, 3, 60, 60), fill=color)
    text = None
    if hard_error:
        draw.line((21, 21, 42, 42), fill="white", width=6)
        draw.line((42, 21, 21, 42), fill="white", width=6)
    elif warning:
        text = "!"
    elif count is not None:
        text = "9+" if count > 9 else str(count)
    elif symbol == "loading":
        for center_x in (24, 32, 40):
            draw.ellipse((center_x - 2, 29, center_x + 2, 33), fill="white")
    elif symbol == "unloaded":
        draw.line((24, 32, 40, 32), fill="white", width=4)
    elif symbol:
        text = symbol
    if text is not None:
        try:
            # This is a Windows-only tray app.  Use the system's screen font
            # instead of Pillow's thin default bitmap face: after 64 -> 16 px
            # reduction, Segoe UI Bold keeps digit counters and stems distinct.
            font = ImageFont.truetype("segoeuib.ttf", 44 if len(text) == 1 else 28)
        except OSError:
            try:
                font = ImageFont.load_default(size=44 if len(text) == 1 else 28)
            except TypeError:  # 舊版 Pillow 的 load_default 不吃 size 參數
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (31.5 - tw / 2 - bbox[0], 31.0 - th / 2 - bbox[1]),
            text,
            fill="white",
            font=font,
        )
    if queued:
        # Keep exact queue size in the tooltip/menu: a second numeric counter
        # is illegible at 16 px, while this badge remains independently visible.
        draw.ellipse((48, 48, 62, 62), fill="white")
        draw.ellipse((51, 51, 59, 59), fill="#7E57C2")
    return img


def _rms_to_icon_level(rms: float, *, ceiling: float = 0.02) -> float:
    """Map the observed microphone range onto the full visible ring range."""
    if rms <= 0:
        return 0.0
    return min(1.0, rms / ceiling)


NIIF_NOSOUND = 0x10


def _silent_notify(icon: pystray.Icon | None, message: str, title: str = "") -> None:
    """Windows notification without the default chime."""
    if icon is None:
        return
    from pystray._util import win32

    icon._message(
        win32.NIM_MODIFY,
        win32.NIF_INFO,
        szInfo=message,
        szInfoTitle=title or icon.title or "",
        dwInfoFlags=NIIF_NOSOUND,
    )


class TrayApp:
    def __init__(self, startup_mode: str = "manual") -> None:
        if startup_mode not in {"manual", "autostart"}:
            raise ValueError(f"unknown startup mode: {startup_mode!r}")
        # Keep construction deliberately lightweight.  The tray must become
        # visible before importing torch, OpenCC, or ASR backends.
        self._state = State.STARTING
        self._lock = threading.Lock()
        self._tray_ui_lock = threading.Lock()
        self._pressed_keys: set = set()
        self._hotkey_latch = HotkeyLatch()
        self._lifecycle = LifecycleMachine()
        self._startup_mode = startup_mode

        self._config: dict[str, Any] = {}
        self._device = "cuda"
        self._sample_rate = 16000
        self._hotkey = _parse_hotkey(DEFAULT_HOTKEY)
        self._hotkey_label = DEFAULT_HOTKEY.upper()
        self._silence_trigger_ms = 1000
        self._verbose = False
        self._cuda_warmup = False
        self._idle_unload_sec = 30 * 60
        # Capacity policy is deferred until codec/retention measurements.
        self._queue_budget = AudioQueueBudget(float("inf"))
        self._logging_mode = "full"
        self._audio_budget_bytes = 2 * 1024**3
        self._recovery_enabled = True
        self._recovery_max_age_hours: float | None = None
        self._recent_results_enabled = True
        self._recent_results_limit = 20
        self._recent_menu_limit = 5

        self._engine: Any = None
        self._pipeline: Any = None
        self._clipboard: Any = None
        self._history_store: Any = None
        self._vad_model: Any = None
        self._recording: Any = None
        self._transcriber: Any = None
        self._live_transcriber: Any = None
        self._live_segment_cursor = 0
        self._live_lock = threading.Lock()
        self._segment_transcribing = False
        self._capture_generation = 0
        self._stop_feedback_generation: int | None = None
        self._stop_feedback_segment_count = 0
        self._asr_thread: threading.Thread | None = None
        self._keyboard_listener: Any = None
        self._shutdown_thread: threading.Thread | None = None
        self._closing = False
        self._captured_jobs: deque[Any] = deque()
        self._finalizer_running = False
        self._finalizer_thread: threading.Thread | None = None
        self._pending_jobs: deque[tuple[Any, Any, str | None, int | None]] = deque()
        self._failed_jobs: deque[tuple[Any, str | None]] = deque()
        self._idle_timer: threading.Timer | None = None
        self._capture_watchdog: threading.Timer | None = None
        self._model_status_timer: threading.Timer | None = None
        self._model_load_started_at: float | None = None
        self._last_callback_count = 0
        self._missed_callback_checks = 0
        self._input_level = 0.0
        self._input_warning: str | None = None
        self._near_zero_threshold = 0.0001
        self._near_zero_warning_sec = 5.0
        self._near_zero_since: float | None = None
        self._tray: pystray.Icon | None = None
        self._stop_feedback = False
        self._setup_error: str | None = None
        # The hotkey listener starts before the capture path exists, so presses
        # during startup must be answered instead of raising or being ignored.
        self._capture_ready = False
        self._startup_started_at: float | None = None
        self._startup_notice_at = 0.0

    def run(self) -> None:
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: LABELS[self._state], None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: (
                    "重試初始化"
                    if self._setup_error
                    else "載入／重試模型"
                    if self._lifecycle.snapshot.model in (ModelState.UNLOADED, ModelState.ERROR)
                    else "載入完成後立即卸載"
                    if self._state is State.LOADING
                    else "卸載模型（釋放顯卡）"
                ),
                self._on_toggle_model,
                enabled=lambda item: (
                    self._state in (State.IDLE, State.LOADING, State.UNLOADED, State.ERROR)
                ),
            ),
            pystray.MenuItem(
                "近期結果",
                pystray.Menu(self._recent_menu_items),
                visible=lambda item: self._recent_results_enabled,
            ),
            pystray.MenuItem(
                lambda item: f"重試失敗工作（{len(self._failed_jobs)}）",
                self._on_retry_failed,
                visible=lambda item: bool(self._failed_jobs),
            ),
            pystray.MenuItem("自動卸載", pystray.Menu(self._idle_menu_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("結束", self._on_quit),
        )
        self._tray = pystray.Icon(
            "asr-input",
            icon=_make_icon(COLORS[State.STARTING]),
            title=f"ASR Input — {LABELS[State.STARTING]}",
            menu=menu,
        )

        def _handle_sigint(sig, frame):
            print("\nCtrl+C — 安全退出中...", flush=True)
            self._on_quit(self._tray, None)

        signal.signal(signal.SIGINT, _handle_sigint)

        self._tray.run(setup=self._on_tray_ready)

    def _on_tray_ready(self, icon: pystray.Icon) -> None:
        """Reveal the icon before importing or initializing heavy components."""
        icon.visible = True
        self._setup()

    def _setup(self) -> None:
        try:
            self._initialize()
        except Exception as exc:
            self._setup_error = f"{type(exc).__name__}: {exc}"
            print(f"啟動失敗：{self._setup_error}", flush=True)
            self._set_state(State.ERROR)
            _silent_notify(self._tray, self._setup_error, "ASR Input 啟動失敗")

    def _initialize(self) -> None:
        self._startup_started_at = time.perf_counter()
        print(
            f"=== 啟動 {datetime.now():%Y-%m-%d %H:%M:%S} — 模式 {self._startup_mode} ===",
            flush=True,
        )

        # Config parsing is pure YAML.  Reading it before any heavy import lets
        # the hotkey listener start while PyTorch and the VAD model still load,
        # so a press during startup gets an explicit answer instead of silence.
        from asr_input.config import load_config

        self._config = load_config()
        asr_cfg = self._config["asr"]
        self._device = asr_cfg.get("device", "cuda")
        self._sample_rate = self._config["audio"]["sample_rate"]
        hotkey_str = self._config.get("hotkey", {}).get("combination", DEFAULT_HOTKEY)
        self._hotkey = _parse_hotkey(hotkey_str)
        self._hotkey_label = hotkey_str.upper()

        streaming_cfg = self._config.get("streaming", {})
        self._silence_trigger_ms = streaming_cfg.get("silence_trigger_ms", 1000)
        self._verbose = streaming_cfg.get("verbose", False)
        self._cuda_warmup = self._config.get("startup", {}).get("cuda_warmup", False)
        self._idle_unload_sec = (
            self._config.get("lifecycle", {}).get("idle_unload_minutes", 30) * 60
        )
        self._queue_budget = AudioQueueBudget(float("inf"))
        logging_cfg = self._config.get("logging", {})
        self._logging_mode = logging_cfg.get(
            "mode", "full" if logging_cfg.get("save_audio", True) else "off"
        )
        if self._logging_mode not in {"off", "errors", "full"}:
            raise ValueError(
                f"logging.mode must be off, errors, or full; got {self._logging_mode!r}"
            )
        self._audio_budget_bytes = int(logging_cfg.get("max_audio_gib", 2) * 1024**3)
        self._recovery_enabled = self._config.get("recovery", {}).get("enabled", True)
        self._recovery_max_age_hours = self._config.get("recovery", {}).get("max_age_hours")
        microphone_cfg = self._config.get("microphone", {})
        self._near_zero_threshold = microphone_cfg.get("near_zero_rms", 0.0001)
        self._near_zero_warning_sec = microphone_cfg.get("near_zero_warning_sec", 5.0)
        history_cfg = self._config.get("history", {})
        self._recent_results_enabled = history_cfg.get("recent_results_enabled", True)
        self._recent_results_limit = history_cfg.get("recent_results_limit", 20)
        self._recent_menu_limit = history_cfg.get("tray_recent_limit", 5)

        self._listen_hotkey()
        print(
            f"快捷鍵已註冊: {self._hotkey_label}；錄音就緒前按下會回報啟動進度",
            flush=True,
        )

        # Heavy imports belong here: the tray icon is visible and the hotkey now
        # answers.  `import torch` alone dominates a cold start and is highly
        # cache-dependent -- measured on the dev machine 2026-09-01: 18.1s for
        # the first import after a long idle, 1.7s once the OS file cache was
        # warm.  It is timed on its own so future launches record which case
        # they hit instead of leaving the largest cost untimed.
        torch_t0 = time.perf_counter()
        import torch

        torch_sec = time.perf_counter() - torch_t0
        print(f"[計時] import torch: {torch_sec:.1f}s", flush=True)

        from asr_input.asr import build_engine
        from asr_input.main import build_pipeline
        from asr_input.output.clipboard import ClipboardOutput

        self._engine = build_engine(asr_cfg, vad_cfg=None)
        self._pipeline = build_pipeline(self._config, include_output=False)
        self._clipboard = ClipboardOutput()
        from asr_input.output.history_store import HistoryStore

        self._history_store = HistoryStore()
        if self._recovery_max_age_hours is not None:
            self._history_store.expire_recovery(self._recovery_max_age_hours)
        saved_idle_minutes = self._history_store.get_setting("idle_unload_minutes")
        if saved_idle_minutes is not None:
            self._idle_unload_sec = int(saved_idle_minutes) * 60

        print("載入 VAD 模型...", flush=True)
        t0 = time.perf_counter()
        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._vad_model = model
        vad_sec = time.perf_counter() - t0
        print("VAD 模型載入完成!", flush=True)

        self._recording = self._build_recording_session()
        self._capture_ready = True
        recovery_effects = self._restore_pending_jobs()
        ready_sec = time.perf_counter() - self._startup_started_at
        print(
            f"[計時] import torch: {torch_sec:.1f}s / 載 CPU VAD: {vad_sec:.1f}s / "
            f"可錄音就緒: {ready_sec:.1f}s / Whisper: 尚未載入",
            flush=True,
        )

        self._refresh_state()
        model_message = (
            "Whisper 正在背景預載"
            if self._startup_mode == "manual"
            else "Whisper 將在首次使用時載入"
        )
        _silent_notify(
            self._tray,
            f"錄音已就緒 — {self._hotkey_label}；{model_message}",
            "ASR Input",
        )
        self._execute_effects(self._startup_effects(recovery_effects))
        self._listen_hotkey()

    def _startup_effects(self, recovery_effects: tuple[Effect, ...]) -> tuple[Effect, ...]:
        effects = list(recovery_effects)
        if self._startup_mode == "manual":
            effects.extend(self._lifecycle.request_model_load())
        return tuple(effects)

    def _build_recording_session(self):
        from asr_input.audio.streaming_vad import StreamingVAD
        from asr_input.recording import RecordingSession

        vad_cfg = self._config.get("vad", {})
        streaming_cfg = self._config.get("streaming", {})

        def make_segmenter(callback, event_callback):
            vad = StreamingVAD(
                on_speech_segment=callback,
                on_decision_event=event_callback,
                sample_rate=self._sample_rate,
                threshold=vad_cfg.get("threshold", 0.5),
                min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
                silence_trigger_ms=self._silence_trigger_ms,
                silence_min_ms=streaming_cfg.get("silence_min_ms", 300),
                ramp_start_sec=streaming_cfg.get("ramp_start_sec", 10.0),
                ramp_end_sec=streaming_cfg.get("ramp_end_sec", 25.0),
                speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
                max_segment_sec=streaming_cfg.get("max_segment_sec", 30.0),
                min_energy=streaming_cfg.get("min_energy", 0.005),
                verbose=self._verbose,
            )
            vad.load(self._vad_model)
            return vad

        return RecordingSession(
            make_segmenter,
            sample_rate=self._sample_rate,
            on_level=self._on_input_level,
            on_status=self._on_capture_status,
            on_segment=self._on_captured_segment,
        )

    def _listen_hotkey(self) -> None:
        if self._keyboard_listener is not None:
            return
        listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        listener.start()
        self._keyboard_listener = listener

    def _on_key_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        try:
            self._pressed_keys.add(self._normalize_key(key))
            if self._hotkey_latch.update(self._hotkey.issubset(self._pressed_keys)):
                self._toggle()
        except Exception as exc:
            # pynput terminates its listener when a callback lets an exception
            # escape.  A presentation-only failure (for example, printing an
            # emoji to a CP950 redirected stream) must never disable the only
            # way the user has to stop an active recording.
            with contextlib.suppress(OSError, UnicodeError):
                print(
                    f"Hotkey callback failed: {type(exc).__name__}: {exc!r}",
                    flush=True,
                )

    def _on_key_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        self._pressed_keys.discard(self._normalize_key(key))
        self._hotkey_latch.update(self._hotkey.issubset(self._pressed_keys))

    @staticmethod
    def _normalize_key(key: keyboard.Key | keyboard.KeyCode | None):
        if isinstance(key, keyboard.KeyCode):
            if key.vk is not None:
                return keyboard.KeyCode.from_vk(key.vk)
            if key.char is not None:
                return keyboard.KeyCode.from_vk(ord(key.char.upper()))
        return key

    def _toggle(self) -> None:
        if self._closing:
            return
        if not self._capture_ready:
            self._notify_startup_pending()
            return
        with self._lock:
            if self._lifecycle.snapshot.capture is CaptureState.RECORDING:
                effects = self._lifecycle.end_capture()
            else:
                if self._lifecycle.snapshot.capture is CaptureState.ERROR:
                    self._lifecycle.clear_capture_error()
                effects = self._lifecycle.begin_capture()
        self._refresh_state()
        self._execute_effects(effects)

    def _notify_startup_pending(self) -> None:
        """Answer a hotkey press that arrives before the capture path exists.

        The listener is deliberately live during startup so the user learns why
        nothing is recording; without this the press is silently dropped and the
        app looks broken.  Repeated presses are throttled so a held or retried
        chord does not queue a burst of notifications.
        """
        now = time.monotonic()
        if now - self._startup_notice_at < 3.0:
            return
        self._startup_notice_at = now
        if self._setup_error:
            message = f"啟動失敗，尚無法錄音：{self._setup_error}"
        else:
            elapsed = (
                time.perf_counter() - self._startup_started_at
                if self._startup_started_at is not None
                else 0.0
            )
            message = f"啟動中（已 {elapsed:.0f} 秒），錄音尚未就緒；就緒時會另行通知。"
        print(message, flush=True)
        _silent_notify(self._tray, message, "ASR Input")

    def _on_toggle_model(self, icon, item) -> None:
        if self._setup_error:
            self._setup_error = None
            self._set_state(State.STARTING)
            threading.Thread(target=self._setup, daemon=True).start()
            return
        with self._lock:
            if self._lifecycle.snapshot.model in (ModelState.UNLOADED, ModelState.ERROR):
                effects = self._lifecycle.request_model_load()
            else:
                effects = self._lifecycle.request_unload()
        self._refresh_state()
        self._execute_effects(effects)

    def _execute_effects(self, effects: tuple[Effect, ...]) -> None:
        for effect in effects:
            if effect.kind is EffectKind.CANCEL_IDLE_UNLOAD:
                self._cancel_idle_timer()
            elif effect.kind is EffectKind.START_CAPTURE:
                self._start_capture()
            elif effect.kind is EffectKind.STOP_CAPTURE:
                # This only closes the microphone and freezes the captured job;
                # ASR runs elsewhere.  Keeping it synchronous prevents a fast
                # next press from overlapping the previous stream teardown.
                self._stop_capture()
            elif effect.kind is EffectKind.LOAD_MODEL:
                threading.Thread(
                    target=self._load_model,
                    args=(effect.generation,),
                    daemon=True,
                ).start()
            elif effect.kind is EffectKind.UNLOAD_MODEL:
                threading.Thread(
                    target=self._unload_model,
                    args=(effect.generation,),
                    daemon=True,
                ).start()
            elif effect.kind is EffectKind.TRANSCRIBE_NEXT:
                self._asr_thread = threading.Thread(target=self._transcribe_next, daemon=True)
                self._asr_thread.start()
            elif effect.kind is EffectKind.SCHEDULE_IDLE_UNLOAD:
                self._schedule_idle_timer(effect.generation)

    def _start_capture(self) -> None:
        self._input_warning = None
        self._near_zero_since = None
        with self._live_lock:
            self._capture_generation += 1
            self._stop_feedback = False
            self._stop_feedback_generation = None
            self._stop_feedback_segment_count = 0
            self._live_transcriber = None
            self._live_segment_cursor = 0
            self._segment_transcribing = False
        try:
            self._recording.start()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._lifecycle.capture_failed(message)
            self._refresh_state()
            _silent_notify(self._tray, message, "ASR Input 麥克風啟動失敗")
            return
        self._refresh_state()
        self._maybe_start_live_transcription()
        self._start_capture_watchdog()
        print("錄音中；再按一次快捷鍵結束", flush=True)

    def _stop_capture(self) -> None:
        with self._live_lock:
            generation = self._capture_generation
            self._stop_feedback = True
            self._stop_feedback_generation = generation
            self._stop_feedback_segment_count = len(self._recording.segments)
        self._update_runtime_icon()
        try:
            self._stop_capture_watchdog()
            result = self._recording.stop()
            self._stop_feedback_segment_count = len(result.segments)
            with self._live_lock:
                live_transcriber = self._live_transcriber
                self._live_transcriber = None
                if live_transcriber is not None:
                    for segment in result.segments[self._live_segment_cursor :]:
                        live_transcriber.feed_segment(segment.audio, list(segment.probabilities))
                    self._live_segment_cursor = len(result.segments)
                    live_transcriber.finish_segment_stream()
            with self._lock:
                self._captured_jobs.append((result, live_transcriber, generation))
                start_finalizer = not self._finalizer_running
                self._finalizer_running = True
            print(
                f"錄音完成：{result.duration_sec:.1f}s，正在保存並排隊 "
                f"({len(result.segments)} 個語音片段)",
                flush=True,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._lifecycle.capture_failed(message)
            self._clear_stop_feedback(generation)
            self._refresh_state()
            _silent_notify(self._tray, message, "ASR Input 錄音收尾失敗")
            return
        self._refresh_state()
        if start_finalizer:
            self._finalizer_thread = threading.Thread(
                target=self._finalize_captured_jobs, daemon=True
            )
            self._finalizer_thread.start()

    def _clear_stop_feedback(self, generation: int) -> None:
        with self._live_lock:
            if self._stop_feedback_generation != generation:
                return
            self._stop_feedback = False
            self._stop_feedback_generation = None
            self._stop_feedback_segment_count = 0

    def _finalize_captured_jobs(self) -> None:
        while True:
            with self._lock:
                if not self._captured_jobs:
                    self._finalizer_running = False
                    return
                result, transcriber, generation = self._captured_jobs.popleft()
            try:
                job_id = self._persist_pending(result)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._failed_jobs.append((result, None))
                    self._lifecycle.capture_failed(message)
                self._clear_stop_feedback(generation)
                self._refresh_state()
                _silent_notify(self._tray, message, "ASR Input recording save failed")
                continue
            with self._lock:
                accepted, warning = self._queue_budget.try_add(result.duration_sec)
                if accepted:
                    self._pending_jobs.append((result, transcriber, job_id, generation))
                    effects = self._lifecycle.queue_job()
            if not accepted:
                message = "待辨識音訊已達安全上限；本段已保存在復原資料庫，未被刪除。"
                if job_id is not None:
                    self._history_store.fail(job_id, message)
                _silent_notify(self._tray, message, "ASR Input 佇列已滿")
                self._clear_stop_feedback(generation)
                self._refresh_state()
                continue
            if warning:
                _silent_notify(
                    self._tray,
                    "待辨識音訊已接近安全上限，請讓佇列完成或檢查模型錯誤。",
                    "ASR Input 佇列提醒",
                )
            self._refresh_state()
            self._execute_effects(effects)

    def _recent_menu_items(self):
        if self._history_store is None:
            return (pystray.MenuItem("尚未完成初始化", None, enabled=False),)
        recent = self._history_store.recent(
            limit=min(self._recent_menu_limit, self._recent_results_limit)
        )
        if not recent:
            return (pystray.MenuItem("尚無辨識結果", None, enabled=False),)

        def copy_result(text: str):
            def action(icon, item):
                self._clipboard.process(text)

            return action

        prefixes = ("① 最新", "②", "③", "④", "⑤")

        def prefix(index: int) -> str:
            return prefixes[index] if index < len(prefixes) else f"{index + 1}."

        return tuple(
            pystray.MenuItem(
                f"{prefix(index)}｜{(job.text or '')[:28]}"
                + ("…" if len(job.text or "") > 28 else ""),
                copy_result(job.text or ""),
            )
            for index, job in enumerate(recent)
        )

    def _idle_menu_items(self):
        choices = (("永不", 0), ("15 分鐘", 15), ("30 分鐘", 30), ("60 分鐘", 60))

        def select(minutes: int):
            def action(icon, item):
                self._idle_unload_sec = minutes * 60
                self._history_store.set_setting("idle_unload_minutes", str(minutes))
                with self._lock:
                    effects = self._lifecycle.reschedule_idle_unload()
                self._execute_effects(effects)

            return action

        return tuple(
            pystray.MenuItem(
                label,
                select(minutes),
                checked=lambda item, minutes=minutes: self._idle_unload_sec == minutes * 60,
                radio=True,
            )
            for label, minutes in choices
        )

    def _on_retry_failed(self, icon, item) -> None:
        effects: list[Effect] = []
        while True:
            with self._lock:
                if not self._failed_jobs:
                    break
                captured_or_stored, job_id = self._failed_jobs[0]
                duration_sec = (
                    captured_or_stored.duration_sec
                    if hasattr(captured_or_stored, "duration_sec")
                    else captured_or_stored.received_frames / captured_or_stored.sample_rate
                )
                accepted, _warning = self._queue_budget.try_add(duration_sec)
                if not accepted:
                    break
                self._failed_jobs.popleft()
            try:
                result = (
                    self._decode_stored_capture(captured_or_stored)
                    if hasattr(captured_or_stored, "audio_path")
                    else captured_or_stored
                )
            except Exception as exc:
                self._queue_budget.remove(duration_sec)
                message = f"recovery decode failed: {type(exc).__name__}: {exc}"
                if job_id is not None:
                    self._history_store.fail(job_id, message)
                _silent_notify(self._tray, message, "ASR Input 復原失敗")
                continue
            with self._lock:
                self._pending_jobs.append((result, None, job_id, None))
                if job_id is not None:
                    self._history_store.mark_pending(job_id)
                effects.extend(self._lifecycle.queue_job())
        self._refresh_state()
        self._execute_effects(tuple(effects))

    def _persist_pending(self, result) -> str | None:
        try:
            return self._history_store.create_pending(
                result.audio,
                sample_rate=result.sample_rate,
                received_frames=result.received_frames,
                persist_audio=self._recovery_enabled or self._logging_mode != "off",
                metadata={
                    "callback_count": result.callback_count,
                    "asr_config": self._config.get("asr", {}),
                    "vad_config": self._config.get("vad", {}),
                    "streaming_config": self._config.get("streaming", {}),
                    "status_events": [
                        {
                            "sample_offset": event.sample_offset,
                            "message": event.message,
                        }
                        for event in result.status_events
                    ],
                    "gap_events": [
                        {
                            "sample_offset": event.sample_offset,
                            "expected_adc_time": event.expected_adc_time,
                            "actual_adc_time": event.actual_adc_time,
                            "gap_sec": event.gap_sec,
                        }
                        for event in result.gap_events
                    ],
                    "vad_events": [
                        {
                            "kind": event.kind,
                            "start_sample": event.start_sample,
                            "end_sample": event.end_sample,
                            "reason": event.reason,
                            "rms": event.rms,
                            "probability_count": event.probability_count,
                            "probability_min": event.probability_min,
                            "probability_max": event.probability_max,
                            "probability_mean": event.probability_mean,
                        }
                        for event in result.vad_events
                    ],
                },
            )
        except Exception as exc:
            message = f"無法建立復原快取：{type(exc).__name__}: {exc}"
            print(message, flush=True)
            _silent_notify(self._tray, message, "ASR Input 記錄警告")
            return None

    def _restore_pending_jobs(self) -> tuple[Effect, ...]:
        if not self._recovery_enabled:
            return ()
        effects: list[Effect] = []
        restored = 0
        deferred = 0
        for stored in self._history_store.recoverable():
            if stored.audio_path is None or not stored.audio_path.exists():
                continue
            duration_sec = stored.received_frames / stored.sample_rate
            accepted, _warning = self._queue_budget.try_add(duration_sec)
            if not accepted:
                deferred += 1
                continue
            try:
                result = self._decode_stored_capture(stored)
            except Exception as exc:
                self._queue_budget.remove(duration_sec)
                self._history_store.fail(
                    stored.id, f"recovery decode failed: {type(exc).__name__}: {exc}"
                )
                continue
            self._pending_jobs.append((result, None, stored.id, None))
            if stored.state == "failed":
                self._history_store.mark_pending(stored.id)
            effects.extend(self._lifecycle.queue_job())
            restored += 1
        if deferred:
            _silent_notify(
                self._tray,
                f"另有 {deferred} 筆復原工作超過記憶體安全上限，仍保留在資料庫。",
                "ASR Input 復原佇列",
            )
        if restored:
            effects.extend(self._lifecycle.request_model_load())
            print(f"復原 {restored} 筆未完成錄音，將依序辨識。", flush=True)
        return tuple(effects)

    def _decode_stored_capture(self, stored):
        import soundfile as sf

        audio, sample_rate = sf.read(stored.audio_path, dtype="float32")
        if sample_rate != self._sample_rate:
            raise ValueError(f"sample rate {sample_rate} does not match {self._sample_rate}")
        self._recording.begin()
        for start in range(0, len(audio), 1024):
            self._recording.feed(audio[start : start + 1024])
        return self._recording.stop()

    def _load_model(self, generation: int) -> None:
        self._model_load_started_at = time.monotonic()
        self._start_model_status_timer()
        try:
            print("背景載入 Whisper 模型中...", flush=True)
            self._engine.load()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._lifecycle.model_failed(generation, message)
            print(f"[計時] Whisper 載入失敗於 {self._model_load_elapsed():.1f}s", flush=True)
            self._stop_model_status_timer()
            self._refresh_state()
            _silent_notify(self._tray, message, "ASR Input 模型載入失敗；可重試")
            return
        with self._lock:
            effects = self._lifecycle.model_ready(generation)
        load_sec = self._model_load_elapsed()
        self._stop_model_status_timer()
        print(f"Whisper 模型載入完成 [計時] {load_sec:.1f}s", flush=True)
        self._refresh_state()
        self._execute_effects(effects)

        self._maybe_start_live_transcription()

    def _model_load_elapsed(self) -> float:
        started = self._model_load_started_at
        return 0.0 if started is None else time.monotonic() - started

    def _unload_model(self, generation: int) -> None:
        import torch

        try:
            self._engine.unload()
            if self._device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._lifecycle.model_unload_failed(generation, message)
            self._refresh_state()
            _silent_notify(self._tray, message, "ASR Input 模型卸載失敗；可重試")
            return
        with self._lock:
            effects = self._lifecycle.model_unloaded(generation)
        self._refresh_state()
        print("模型已卸載，顯卡記憶體已釋放。", flush=True)
        self._execute_effects(effects)

    def _new_transcriber(self, logger=None):
        from asr_input.streaming import StreamingSession

        vad_cfg = self._config.get("vad", {})
        streaming_cfg = self._config.get("streaming", {})
        return StreamingSession(
            engine=self._engine,
            pipeline=self._pipeline,
            vad_model=self._vad_model,
            sample_rate=self._sample_rate,
            silence_trigger_ms=self._silence_trigger_ms,
            silence_min_ms=streaming_cfg.get("silence_min_ms", 300),
            ramp_start_sec=streaming_cfg.get("ramp_start_sec", 10.0),
            ramp_end_sec=streaming_cfg.get("ramp_end_sec", 25.0),
            vad_threshold=vad_cfg.get("threshold", 0.5),
            min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
            max_segment_sec=streaming_cfg.get("max_segment_sec", 30.0),
            min_energy=streaming_cfg.get("min_energy", 0.005),
            hallucination_threshold_sec=streaming_cfg.get("hallucination_threshold_sec", 1.5),
            min_hallucination_audio_sec=streaming_cfg.get("min_hallucination_audio_sec", 3.0),
            fallback_silence_ms=streaming_cfg.get("fallback_silence_ms", [500, 300]),
            fallback_rms_target=streaming_cfg.get("fallback_rms_target", 0.05),
            on_partial=self._on_partial_result,
            on_transcribing=self._on_transcribing,
            on_segment_done=self._on_segment_done,
            verbose=self._verbose,
            session_logger=logger,
        )

    def _maybe_start_live_transcription(self) -> None:
        """Attach ASR to the active capture as soon as the engine is available."""
        if not self._recording or not self._recording.recording:
            return
        snapshot = self._lifecycle.snapshot
        if snapshot.model is not ModelState.READY or snapshot.transcribing:
            return
        with self._lock:
            if self._pending_jobs or self._captured_jobs or self._finalizer_running:
                return
        with self._live_lock:
            if self._live_transcriber is not None or not self._recording.recording:
                return
            transcriber = self._new_transcriber()
            transcriber.start_segment_stream()
            segments = self._recording.segments
            for segment in segments:
                transcriber.feed_segment(segment.audio, list(segment.probabilities))
            self._live_segment_cursor = len(segments)
            self._live_transcriber = transcriber
        self._update_runtime_icon()

    def _on_captured_segment(self, segment, index: int) -> None:
        """Feed each primary VAD segment live, skipping catch-up duplicates."""
        with self._live_lock:
            if self._live_transcriber is None or index < self._live_segment_cursor:
                return
            self._live_transcriber.feed_segment(segment.audio, list(segment.probabilities))
            self._live_segment_cursor = index + 1
        self._update_runtime_icon()

    def _transcribe_next(self) -> None:
        with self._lock:
            result, existing_transcriber, job_id, generation = self._pending_jobs.popleft()
            self._queue_budget.remove(result.duration_sec)
        self._transcriber = existing_transcriber or self._new_transcriber()
        try:
            if existing_transcriber is None:
                segments = [
                    (segment.audio, list(segment.probabilities)) for segment in result.segments
                ]
                self._transcriber.start_from_segments(segments)
            full_text = self._transcriber.stop()
            segment_count = self._transcriber.segment_count
            segment_stats = self._transcriber.segment_stats
            for stat, captured in zip(segment_stats, result.segments, strict=False):
                stat["start_sample"] = captured.start_sample
                stat["end_sample"] = captured.end_sample
            if full_text.strip():
                self._clipboard.process(full_text)
                print(f"完成: {segment_count} 段, 結果: {full_text}", flush=True)
                notify_text = full_text[:250] + "…" if len(full_text) > 250 else full_text
                _silent_notify(self._tray, notify_text, "ASR Input")
            else:
                print("（沒有辨識到文字）", flush=True)
            if job_id is not None:
                self._history_store.record_segments(job_id, segment_stats)
                self._label_diagnostic_candidates(job_id, segment_stats, result)
                self._history_store.complete(
                    job_id,
                    full_text,
                    retain_audio=self._logging_mode == "full",
                )
                self._history_store.enforce_audio_budget(self._audio_budget_bytes)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if job_id is not None:
                self._history_store.fail(job_id, message)
            with self._lock:
                self._failed_jobs.append((result, job_id))
                effects = self._lifecycle.job_failed(message)
            _silent_notify(self._tray, message, "ASR Input 辨識失敗")
        else:
            with self._lock:
                effects = self._lifecycle.job_completed()
        finally:
            self._transcriber = None
            self._segment_transcribing = False
            if generation is not None:
                self._clear_stop_feedback(generation)
        self._refresh_state()
        self._execute_effects(effects)
        self._maybe_start_live_transcription()

    def _label_diagnostic_candidates(self, job_id: str, stats: list[dict], result) -> None:
        threshold = self._config.get("streaming", {}).get("hallucination_threshold_sec", 1.5)
        for index, stat in enumerate(stats):
            if stat.get("transcribe_sec", 0) > threshold:
                self._history_store.add_corpus_label(
                    job_id,
                    "slow-transcription",
                    segment_index=index,
                    source="derived",
                    reason=f"absolute transcription time exceeded {threshold}s",
                )
            if stat.get("fallback"):
                self._history_store.add_corpus_label(
                    job_id,
                    "fallback",
                    segment_index=index,
                    source="observed",
                    reason="runtime fallback path executed",
                )
            if stat.get("fallback_exhausted"):
                self._history_store.add_corpus_label(
                    job_id,
                    "fallback-exhausted",
                    segment_index=index,
                    source="observed",
                    reason="all configured resegmentation thresholds remained slow",
                )
        if result.status_events or result.gap_events:
            self._history_store.add_corpus_label(
                job_id,
                "capture-warning",
                source="observed",
                reason="; ".join(
                    [event.message for event in result.status_events]
                    + [f"callback gap {event.gap_sec:.3f}s" for event in result.gap_events]
                ),
            )

    def _on_transcribing(self, audio_sec: float) -> None:
        self._segment_transcribing = True
        """Called when a segment starts being transcribed — flash amber."""
        self._update_runtime_icon()
        if self._tray and self._lifecycle.snapshot.capture is not CaptureState.RECORDING:
            self._set_tray_visual(title=f"ASR Input — 辨識 {audio_sec:.0f}s 音訊中")

    def _on_partial_result(self, latest_segment: str, accumulated: str) -> None:
        """Called from worker thread when a segment is transcribed."""
        self._update_runtime_icon()
        if self._tray and self._lifecycle.snapshot.capture is not CaptureState.RECORDING:
            seg_count = self._transcriber.segment_count if self._transcriber else 0
            char_count = len(accumulated)
            self._set_tray_visual(
                title=f"ASR Input — {seg_count}段 {char_count}字 | {latest_segment[:50]}"
            )

    def _on_segment_done(self) -> None:
        self._segment_transcribing = False
        self._update_runtime_icon()

    def _on_input_level(self, rms: float) -> None:
        """Expose live input without treating ordinary silence as a hard error."""
        self._input_level = rms
        near_zero_message = "未偵測到聲音／可能靜音"
        now = time.monotonic()
        if rms <= self._near_zero_threshold:
            if self._near_zero_since is None:
                self._near_zero_since = now
            elif (
                now - self._near_zero_since >= self._near_zero_warning_sec
                and self._input_warning is None
            ):
                self._input_warning = near_zero_message
                _silent_notify(self._tray, near_zero_message, "ASR Input 麥克風提醒")
        else:
            self._near_zero_since = None
            if self._input_warning == near_zero_message:
                self._input_warning = None
        self._update_runtime_icon()

    def _on_capture_status(self, event) -> None:
        message = event.message
        if "input overflow" in message.lower():
            message = "麥克風資料一度來不及接收，該瞬間可能有少量音訊遺失"
        self._input_warning = message
        print(f"麥克風 callback 警告：{event.message}", flush=True)
        self._refresh_state()
        _silent_notify(self._tray, message, "ASR Input 麥克風警告")

    def _start_capture_watchdog(self) -> None:
        self._stop_capture_watchdog()
        self._last_callback_count = self._recording.callback_count
        self._missed_callback_checks = 0
        self._capture_watchdog = threading.Timer(2.0, self._check_capture_health)
        self._capture_watchdog.daemon = True
        self._capture_watchdog.start()

    def _check_capture_health(self) -> None:
        if not self._recording or not self._recording.recording:
            return
        callback_count = self._recording.callback_count
        if callback_count == self._last_callback_count:
            self._missed_callback_checks += 1
            if self._missed_callback_checks >= 2:
                message = "麥克風 callback 已中斷；錄音已停止並保留"
                with self._lock:
                    self._lifecycle.capture_failed(message)
                self._stop_capture()
                self._refresh_state()
                _silent_notify(self._tray, message, "ASR Input 麥克風錯誤")
                return
            self._input_warning = "兩秒內未收到麥克風資料；裝置可能中斷"
            self._refresh_state()
            _silent_notify(self._tray, self._input_warning, "ASR Input 麥克風警告")
        else:
            self._last_callback_count = callback_count
            self._missed_callback_checks = 0
            if self._input_warning == "兩秒內未收到麥克風資料；裝置可能中斷":
                self._input_warning = None
                self._refresh_state()
        self._capture_watchdog = threading.Timer(2.0, self._check_capture_health)
        self._capture_watchdog.daemon = True
        self._capture_watchdog.start()

    def _stop_capture_watchdog(self) -> None:
        timer, self._capture_watchdog = self._capture_watchdog, None
        if timer is not None:
            timer.cancel()

    def _refresh_state(self) -> None:
        snapshot = self._lifecycle.snapshot
        if (
            snapshot.capture is CaptureState.ERROR
            or snapshot.model is ModelState.ERROR
            or snapshot.error is not None
        ):
            state = State.ERROR
        elif snapshot.capture is CaptureState.RECORDING:
            state = State.STREAMING
        elif snapshot.transcribing:
            state = State.TRANSCRIBING
        elif snapshot.model is ModelState.LOADING:
            state = State.LOADING
        elif snapshot.model is ModelState.READY:
            state = State.IDLE
        else:
            state = State.UNLOADED
        self._set_state(state)
        self._update_runtime_icon()

    def _update_runtime_icon(self) -> None:
        if not self._tray:
            return
        snapshot = self._lifecycle.snapshot
        center_state = {
            ModelState.UNLOADED: State.UNLOADED,
            ModelState.LOADING: State.LOADING,
            ModelState.READY: State.IDLE,
            ModelState.UNLOADING: State.LOADING,
            ModelState.ERROR: State.ERROR,
        }[snapshot.model]
        recording = snapshot.capture is CaptureState.RECORDING
        active_asr = self._segment_transcribing or snapshot.transcribing
        if recording:
            center_state = State.TRANSCRIBING if active_asr else State.STREAMING
        elif active_asr:
            center_state = State.TRANSCRIBING
        ring_color = AUDIO_ACTIVITY_COLORS[center_state] if recording else None
        queued = snapshot.queued_jobs
        if recording:
            count = len(self._recording.segments)
        elif self._stop_feedback:
            count = self._stop_feedback_segment_count
        else:
            count = None
        symbol = None
        if count is None:
            if snapshot.model in (ModelState.LOADING, ModelState.UNLOADING):
                symbol = "loading"
            elif snapshot.model is ModelState.UNLOADED:
                symbol = "unloaded"
        icon = _make_icon(
            COLORS[center_state],
            count=count,
            recording_ended=self._stop_feedback and not recording,
            ring_color=ring_color,
            warning=self._input_warning is not None,
            hard_error=(
                snapshot.capture is CaptureState.ERROR
                or snapshot.model is ModelState.ERROR
                or snapshot.error is not None
            ),
            level=_rms_to_icon_level(self._input_level) if recording else 0.0,
            symbol=symbol,
            queued=queued > 0,
        )
        statuses = []
        if recording:
            statuses.append("錄音中")
            if active_asr:
                statuses.append("辨識中")
        if snapshot.model is ModelState.LOADING and self._model_load_started_at is not None:
            elapsed = int(time.monotonic() - self._model_load_started_at)
            statuses.append(f"模型載入中 {elapsed}s")
        elif not recording:
            statuses.append(LABELS[center_state])
        if queued:
            statuses.append(f"工作 {queued}")
        if self._input_warning:
            statuses.append(f"麥克風警告：{self._input_warning}")
        self._set_tray_visual(icon=icon, title="ASR Input — " + "｜".join(statuses))

    def _set_tray_visual(
        self,
        *,
        icon: Image.Image | None = None,
        title: str | None = None,
        update_menu: bool = False,
    ) -> None:
        """Serialize Win32 icon handle replacement and never leak into audio callbacks."""
        if not self._tray or self._closing:
            return
        try:
            with self._tray_ui_lock:
                if self._closing:
                    return
                if icon is not None:
                    self._tray.icon = icon
                if title is not None:
                    self._tray.title = title
                if update_menu:
                    self._tray.update_menu()
        except OSError as exc:
            # pystray may race Windows Explorer while a notification icon is
            # being recreated.  This must never escape a PortAudio C callback.
            print(f"Tray icon update failed: {type(exc).__name__}: {exc}", flush=True)

    def _start_model_status_timer(self) -> None:
        self._stop_model_status_timer(clear_started=False)
        self._model_status_timer = threading.Timer(1.0, self._tick_model_status)
        self._model_status_timer.daemon = True
        self._model_status_timer.start()

    def _tick_model_status(self) -> None:
        if self._closing or self._lifecycle.snapshot.model is not ModelState.LOADING:
            self._stop_model_status_timer()
            return
        self._update_runtime_icon()
        self._start_model_status_timer()

    def _stop_model_status_timer(self, *, clear_started: bool = True) -> None:
        timer, self._model_status_timer = self._model_status_timer, None
        if timer is not None:
            timer.cancel()
        if clear_started:
            self._model_load_started_at = None

    def _cancel_idle_timer(self) -> None:
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None:
            timer.cancel()

    def _schedule_idle_timer(self, generation: int) -> None:
        self._cancel_idle_timer()
        if self._idle_unload_sec <= 0:
            return
        self._stop_capture_watchdog()
        self._idle_timer = threading.Timer(
            self._idle_unload_sec,
            self._on_idle_expired,
            args=(generation,),
        )
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _on_idle_expired(self, generation: int) -> None:
        with self._lock:
            effects = self._lifecycle.idle_expired(generation)
        self._refresh_state()
        self._execute_effects(effects)

    def _set_state(self, state: State) -> None:
        self._state = state
        if self._tray:
            self._set_tray_visual(
                icon=_make_icon(COLORS[state], hard_error=state is State.ERROR),
                title=f"ASR Input — {LABELS[state]}",
                update_menu=True,
            )

    def _on_quit(self, icon, item) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
        print("安全結束中。", flush=True)
        self._cancel_idle_timer()
        self._stop_model_status_timer()
        self._stop_capture_watchdog()
        listener, self._keyboard_listener = self._keyboard_listener, None
        if listener is not None:
            listener.stop()
        self._shutdown_thread = threading.Thread(
            target=self._finish_shutdown,
            args=(icon,),
            daemon=True,
        )
        self._shutdown_thread.start()

    def _finish_shutdown(self, icon) -> None:
        if self._recording and self._recording.recording:
            self._stop_capture()
        finalizer = self._finalizer_thread
        if finalizer and finalizer.is_alive():
            finalizer.join(timeout=5)
            if finalizer.is_alive():
                _silent_notify(
                    self._tray,
                    "錄音仍在寫入復原快取；完成後將自動結束。",
                    "ASR Input 保存中",
                )
                finalizer.join()
        # A transcription job is already durable before its worker starts.  Do
        # not hold shutdown hostage to CUDA/ASR; an interrupted pending job is
        # recovered on the next launch.
        asr_thread = self._asr_thread
        if (
            self._engine is not None
            and self._lifecycle.snapshot.model is not ModelState.LOADING
            and not (asr_thread and asr_thread.is_alive())
        ):
            self._engine.unload()
        with self._tray_ui_lock:
            icon.stop()


def _optout_ecoqos() -> None:
    """退出 Win11 EcoQoS（效率模式），避免背景行程被節流。

    從 start_tray.vbs 以隱藏視窗（WindowStyle 0、無前景視窗）啟動時，Win11 會把
    本行程判為背景並套 EcoQoS，把驅動 CUDA 的 CPU 執行緒趕到 E-core / 降頻。
    實測 faster-whisper 解碼中位數從 ~0.46s 惡化到 ~1.1s 且抖動大；opt-out 後恢復
    全速且變異幾乎消失。前景（終端）啟動時本呼叫無害。僅 Windows 有效。
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes as wt

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [
            ("Version", wt.DWORD),
            ("ControlMask", wt.DWORD),
            ("StateMask", wt.DWORD),
        ]

    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        # ControlMask 選 EXECUTION_SPEED、StateMask=0 → 明確關閉該項節流
        state = _PowerThrottlingState(1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
        k32.SetProcessInformation(
            k32.GetCurrentProcess(),
            ProcessPowerThrottling,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
    except OSError:
        pass  # 舊版 Windows 無此 API，忽略


def main() -> None:
    # Preserve the launcher's native encoding, but make every diagnostic write
    # non-fatal when a message contains a character outside that code page.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="ASR Input tray application")
    parser.add_argument(
        "--startup-mode",
        choices=("manual", "autostart"),
        default="manual",
        help="manual preloads Whisper; autostart stays lazy until first capture",
    )
    args = parser.parse_args()
    _optout_ecoqos()
    with SingleInstance("Local\\ASRInput.Tray.v1") as instance:
        if not instance.acquired:
            notify_already_running()
            return
        app = TrayApp(startup_mode=args.startup_mode)
        app.run()


if __name__ == "__main__":
    main()
