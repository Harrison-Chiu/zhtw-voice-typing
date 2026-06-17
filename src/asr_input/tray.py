"""System tray app with global hotkey for ASR input."""

from __future__ import annotations

import enum
import threading

import pystray
import torch
from PIL import Image, ImageDraw
from pynput import keyboard

from asr_input.asr import build_engine
from asr_input.config import load_config
from asr_input.main import build_pipeline
from asr_input.output.clipboard import ClipboardOutput
from asr_input.output.transcript_log import log_transcript
from asr_input.streaming import StreamingSession


class State(enum.Enum):
    LOADING = "loading"
    IDLE = "idle"
    STREAMING = "streaming"


COLORS = {
    State.LOADING: "#888888",
    State.IDLE: "#4CAF50",
    State.STREAMING: "#F44336",
}

LABELS = {
    State.LOADING: "載入模型中...",
    State.IDLE: "待機（按快捷鍵錄音）",
    State.STREAMING: "串流辨識中...",
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


def _make_icon(color: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 56, 56), fill=color)
    return img


NIIF_NOSOUND = 0x10


def _silent_notify(icon: pystray.Icon, message: str, title: str = "") -> None:
    """Windows notification without the default chime."""
    from pystray._util import win32

    icon._message(
        win32.NIM_MODIFY,
        win32.NIF_INFO,
        szInfo=message,
        szInfoTitle=title or icon.title or "",
        dwInfoFlags=NIIF_NOSOUND,
    )


class TrayApp:
    def __init__(self) -> None:
        self._state = State.LOADING
        self._lock = threading.Lock()
        self._pressed_keys: set = set()

        self._config = load_config()
        asr_cfg = self._config["asr"]
        self._sample_rate = self._config["audio"]["sample_rate"]
        hotkey_str = self._config.get("hotkey", {}).get("combination", DEFAULT_HOTKEY)
        self._hotkey = _parse_hotkey(hotkey_str)
        self._hotkey_label = hotkey_str.replace("+", "+").upper()

        streaming_cfg = self._config.get("streaming", {})
        self._silence_trigger_ms = streaming_cfg.get("silence_trigger_ms", 1000)

        # Build ASR engine WITHOUT VadSegmentedEngine wrapper —
        # streaming mode handles segmentation via StreamingVAD.
        self._engine = build_engine(asr_cfg, vad_cfg=None)
        # Text processing without clipboard (streaming copies once at the end)
        self._pipeline = build_pipeline(self._config, include_output=False)
        self._clipboard = ClipboardOutput()

        self._vad_model: torch.jit.ScriptModule | None = None
        self._session: StreamingSession | None = None
        self._tray: pystray.Icon | None = None

    def run(self) -> None:
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: LABELS[self._state], None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("結束", self._on_quit),
        )
        self._tray = pystray.Icon(
            "asr-input",
            icon=_make_icon(COLORS[State.LOADING]),
            title="ASR Input",
            menu=menu,
        )
        threading.Thread(target=self._setup, daemon=True).start()
        self._tray.run()

    def _setup(self) -> None:
        print("載入模型中...", flush=True)
        self._engine.load()
        print("ASR 模型載入完成!", flush=True)

        print("載入 VAD 模型...", flush=True)
        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._vad_model = model
        print("VAD 模型載入完成!", flush=True)

        self._set_state(State.IDLE)
        print(f"快捷鍵: {self._hotkey_label}（串流錄音切換）", flush=True)
        _silent_notify(self._tray, f"就緒 — {self._hotkey_label} 開始錄音", "ASR Input")
        self._listen_hotkey()

    def _listen_hotkey(self) -> None:
        with keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        ) as listener:
            listener.join()

    def _on_key_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        self._pressed_keys.add(self._normalize_key(key))
        if self._hotkey.issubset(self._pressed_keys):
            self._toggle()

    def _on_key_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        self._pressed_keys.discard(self._normalize_key(key))

    @staticmethod
    def _normalize_key(key: keyboard.Key | keyboard.KeyCode | None):
        if isinstance(key, keyboard.KeyCode):
            if key.vk is not None:
                return keyboard.KeyCode.from_vk(key.vk)
            if key.char is not None:
                return keyboard.KeyCode.from_vk(ord(key.char.upper()))
        return key

    def _toggle(self) -> None:
        with self._lock:
            if self._state == State.IDLE:
                self._start_streaming()
            elif self._state == State.STREAMING:
                threading.Thread(target=self._stop_streaming, daemon=True).start()

    def _start_streaming(self) -> None:
        vad_cfg = self._config.get("vad", {})
        streaming_cfg = self._config.get("streaming", {})

        raw_adaptive = streaming_cfg.get("adaptive_thresholds")
        adaptive = (
            [(e["after_sec"], e["silence_ms"]) for e in raw_adaptive]
            if raw_adaptive
            else None
        )

        self._session = StreamingSession(
            engine=self._engine,
            pipeline=self._pipeline,
            vad_model=self._vad_model,
            sample_rate=self._sample_rate,
            silence_trigger_ms=self._silence_trigger_ms,
            vad_threshold=vad_cfg.get("threshold", 0.5),
            min_speech_ms=vad_cfg.get("min_speech_duration_ms", 250),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
            max_segment_sec=streaming_cfg.get("max_segment_sec", 30.0),
            adaptive_thresholds=adaptive,
            min_energy=streaming_cfg.get("min_energy", 0.005),
            on_partial=self._on_partial_result,
        )
        self._session.start()
        self._set_state(State.STREAMING)
        print("🎤 串流辨識中... 按快捷鍵停止", flush=True)

    def _stop_streaming(self) -> None:
        if self._session is None:
            self._set_state(State.IDLE)
            return

        full_text = self._session.stop()
        segment_count = self._session.segment_count
        self._session = None

        if not full_text.strip():
            print("（沒有辨識到文字）", flush=True)
            self._set_state(State.IDLE)
            return

        self._clipboard.process(full_text)
        log_transcript("(streaming)", full_text, audio_duration_sec=0)
        print(f"完成: {segment_count} 段, 結果: {full_text}", flush=True)
        notify_text = full_text[:250] + "…" if len(full_text) > 250 else full_text
        _silent_notify(self._tray, notify_text, "ASR Input")
        self._set_state(State.IDLE)

    def _on_partial_result(self, latest_segment: str, accumulated: str) -> None:
        """Called from worker thread when a segment is transcribed."""
        if self._tray:
            self._tray.title = f"ASR Input — {latest_segment[:60]}"

    def _set_state(self, state: State) -> None:
        self._state = state
        if self._tray:
            self._tray.icon = _make_icon(COLORS[state])
            self._tray.title = f"ASR Input — {LABELS[state]}"

    def _on_quit(self, icon, item) -> None:
        print("結束。", flush=True)
        if self._session:
            self._session.stop()
            self._session = None
        self._engine.unload()
        icon.stop()


def main() -> None:
    app = TrayApp()
    app.run()


if __name__ == "__main__":
    main()
