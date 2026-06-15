"""System tray app with global hotkey for ASR input."""

from __future__ import annotations

import enum
import threading

import pystray
from PIL import Image, ImageDraw
from pynput import keyboard

from asr_input.asr import build_engine
from asr_input.audio.capture import MicrophoneCapture
from asr_input.config import load_config
from asr_input.main import build_pipeline


class State(enum.Enum):
    LOADING = "loading"
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


COLORS = {
    State.LOADING: "#888888",
    State.IDLE: "#4CAF50",
    State.RECORDING: "#F44336",
    State.PROCESSING: "#2196F3",
}

LABELS = {
    State.LOADING: "載入模型中...",
    State.IDLE: "待機（按快捷鍵錄音）",
    State.RECORDING: "錄音中...",
    State.PROCESSING: "辨識中...",
}

DEFAULT_HOTKEY = "ctrl+alt+r"

_KEY_MAP = {
    "ctrl": keyboard.Key.ctrl_l,
    "alt": keyboard.Key.alt_l,
    "shift": keyboard.Key.shift_l,
    "win": keyboard.Key.cmd,
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

        self._engine = build_engine(asr_cfg)
        self._source = MicrophoneCapture(sample_rate=self._sample_rate)
        self._pipeline = build_pipeline(self._config)

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
        print("模型載入完成!", flush=True)
        self._set_state(State.IDLE)
        print(f"快捷鍵: {self._hotkey_label}（錄音切換）", flush=True)
        self._tray.notify(f"就緒 — {self._hotkey_label} 開始錄音", "ASR Input")
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
                self._set_state(State.RECORDING)
                self._source.start()
                print("🎤 錄音中...", flush=True)
            elif self._state == State.RECORDING:
                self._set_state(State.PROCESSING)
                threading.Thread(target=self._finish_recording, daemon=True).start()

    def _finish_recording(self) -> None:
        audio = self._source.stop()
        if len(audio) == 0:
            print("（沒有收到音訊）", flush=True)
            self._set_state(State.IDLE)
            return

        print("辨識中...", flush=True)
        raw_text = self._engine.transcribe(audio, self._sample_rate)
        if not raw_text.strip():
            print("（沒有辨識到文字）", flush=True)
            self._set_state(State.IDLE)
            return

        processed_text = self._pipeline.run(raw_text)
        print(f"原始: {raw_text}", flush=True)
        print(f"結果: {processed_text}", flush=True)
        self._tray.notify(processed_text, "ASR Input")
        self._set_state(State.IDLE)

    def _set_state(self, state: State) -> None:
        self._state = state
        if self._tray:
            self._tray.icon = _make_icon(COLORS[state])
            self._tray.title = f"ASR Input — {LABELS[state]}"

    def _on_quit(self, icon, item) -> None:
        print("結束。", flush=True)
        self._engine.unload()
        icon.stop()


def main() -> None:
    app = TrayApp()
    app.run()


if __name__ == "__main__":
    main()
