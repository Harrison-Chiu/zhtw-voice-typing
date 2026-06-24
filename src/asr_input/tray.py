"""System tray app with global hotkey for ASR input."""

from __future__ import annotations

import enum
import signal
import threading
import time

import pystray
import torch
from PIL import Image, ImageDraw, ImageFont
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
    TRANSCRIBING = "transcribing"


# 配色刻意拉開色相對比，讓 16-32px 的系統匣圖示一眼可辨：
# 灰(待命載入) / 綠(就緒) / 紅(錄音中) / 琥珀(辨識運算中)。
COLORS = {
    State.LOADING: "#9E9E9E",
    State.IDLE: "#43A047",
    State.STREAMING: "#E53935",
    State.TRANSCRIBING: "#FB8C00",
}

LABELS = {
    State.LOADING: "載入模型中...",
    State.IDLE: "待機（按快捷鍵錄音）",
    State.STREAMING: "串流辨識中...",
    State.TRANSCRIBING: "辨識中...",
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


def _make_icon(color: str, count: int | None = None) -> Image.Image:
    img = Image.new("RGBA", (64, 64))
    draw = ImageDraw.Draw(img)
    draw.ellipse((4, 4, 60, 60), fill=color)
    if count is not None:
        text = str(count)
        try:
            font = ImageFont.load_default(size=40)
        except TypeError:  # 舊版 Pillow 的 load_default 不吃 size 參數
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (32 - tw / 2 - bbox[0], 32 - th / 2 - bbox[1]),
            text,
            fill="white",
            font=font,
        )
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
        self._device = asr_cfg.get("device", "cuda")
        self._sample_rate = self._config["audio"]["sample_rate"]
        hotkey_str = self._config.get("hotkey", {}).get("combination", DEFAULT_HOTKEY)
        self._hotkey = _parse_hotkey(hotkey_str)
        self._hotkey_label = hotkey_str.replace("+", "+").upper()

        streaming_cfg = self._config.get("streaming", {})
        self._silence_trigger_ms = streaming_cfg.get("silence_trigger_ms", 1000)
        self._verbose = streaming_cfg.get("verbose", False)

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

        def _handle_sigint(sig, frame):
            print("\nCtrl+C — 安全退出中...", flush=True)
            self._on_quit(self._tray, None)

        signal.signal(signal.SIGINT, _handle_sigint)

        threading.Thread(target=self._setup, daemon=True).start()
        self._tray.run()

    def _setup(self) -> None:
        # CUDA 暖機：第一次把 tensor 丟到 GPU 會觸發 CUDA context 初始化，
        # 跟「載權重」是兩段不同成本，分開計時才看得出時間花在哪。
        t0 = time.perf_counter()
        if self._device != "cpu" and torch.cuda.is_available():
            torch.zeros(1).to(self._device)
            torch.cuda.synchronize()
        warmup_sec = time.perf_counter() - t0

        print("載入模型中...", flush=True)
        t0 = time.perf_counter()
        self._engine.load()
        whisper_sec = time.perf_counter() - t0
        print("ASR 模型載入完成!", flush=True)

        print("載入 VAD 模型...", flush=True)
        t0 = time.perf_counter()
        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._vad_model = model
        vad_sec = time.perf_counter() - t0
        print("VAD 模型載入完成!", flush=True)

        print(
            f"[計時] CUDA 暖機: {warmup_sec:.1f}s / "
            f"載 Whisper: {whisper_sec:.1f}s / 載 VAD: {vad_sec:.1f}s",
            flush=True,
        )

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

        self._session = StreamingSession(
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
            verbose=self._verbose,
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

    def _on_transcribing(self, audio_sec: float) -> None:
        """Called when a segment starts being transcribed — flash amber."""
        if self._tray:
            seg_count = self._session.segment_count if self._session else 0
            self._tray.icon = _make_icon(COLORS[State.TRANSCRIBING], count=seg_count)
            self._tray.title = f"ASR — 辨識 {audio_sec:.0f}s 音訊中..."

    def _on_partial_result(self, latest_segment: str, accumulated: str) -> None:
        """Called from worker thread when a segment is transcribed."""
        if self._tray:
            seg_count = self._session.segment_count if self._session else 0
            char_count = len(accumulated)
            self._tray.icon = _make_icon(COLORS[State.STREAMING], count=seg_count)
            self._tray.title = f"ASR — {seg_count}段 {char_count}字 | {latest_segment[:50]}"

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
