"""Play the capture-fault alert once, using the live config.

The alert only fires when the microphone stops delivering data, which cannot be
staged on demand — so hearing nothing during normal use is expected, and says
nothing about whether the sound works. This script separates the two questions:
it reports the thresholds currently in effect and plays the sound itself.

    uv run python scripts/test_alert_sound.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_input.config import load_config  # noqa: E402
from asr_input.platform import select_alert_sound  # noqa: E402


def main() -> int:
    microphone = load_config().get("microphone", {})
    enabled = bool(microphone.get("alert_sound", True))
    warning_sec = float(microphone.get("no_data_warning_sec", 1.0))
    error_sec = float(microphone.get("no_data_error_sec", 3.0))
    backend = select_alert_sound()

    print(f"警示音設定：{'開啟' if enabled else '關閉'}（microphone.alert_sound）")
    print(f"後端：{backend.name}")
    print(f"提醒門檻：連續 {warning_sec:g} 秒沒有麥克風資料（錄音繼續）")
    print(f"停止門檻：連續 {error_sec:g} 秒沒有麥克風資料（停止並保留已錄內容）")

    if not enabled:
        print("\n設定為關閉，不播放。要試聽請把 config.yaml 的 microphone.alert_sound 設為 true。")
        return 0

    print("\n現在播放一次警示音……")
    emitted = backend.play()
    print(f"已呼叫音效後端，未發生例外：{emitted}")
    print("注意：這個值只代表呼叫成功，不代表喇叭真的出聲——請以你聽到的為準。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
