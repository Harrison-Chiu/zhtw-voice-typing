"""Entry point for ASR input system — Terminal CLI version."""

import io
import sys
import time

print("正在載入 PyTorch（首次可能需要 30-60 秒）...", flush=True)

from asr_input.asr.qwen import QwenASREngine  # noqa: E402
from asr_input.audio.capture import MicrophoneCapture  # noqa: E402
from asr_input.config import load_config  # noqa: E402
from asr_input.output.clipboard import ClipboardOutput  # noqa: E402
from asr_input.processing.opencc_conv import OpenCCConverter  # noqa: E402
from asr_input.processing.pipeline import ProcessingPipeline  # noqa: E402
from asr_input.processing.tw_terms import TaiwanTermReplacer  # noqa: E402

print("PyTorch 載入完成。", flush=True)


def _ensure_utf8_stdout() -> None:
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def build_pipeline(config: dict) -> ProcessingPipeline:
    pipeline = ProcessingPipeline()
    pipeline.add(OpenCCConverter(config["processing"]["opencc_config"]))
    pipeline.add(TaiwanTermReplacer())
    if config["output"]["method"] == "clipboard":
        pipeline.add(ClipboardOutput())
    return pipeline


def main() -> None:
    _ensure_utf8_stdout()
    config = load_config()
    asr_cfg = config["asr"]

    print("=== ASR Input — 台灣繁體中文語音輸入 ===")
    print(f"模型: {asr_cfg['model_id']}")
    print(f"裝置: {asr_cfg['device']}")
    print()

    print("載入模型中...")
    engine = QwenASREngine(
        model_id=asr_cfg["model_id"],
        device=asr_cfg["device"],
        language=asr_cfg.get("language"),
        context=asr_cfg.get("context", ""),
    )
    engine.load()
    print("模型載入完成!")
    print()

    mic = MicrophoneCapture(sample_rate=config["audio"]["sample_rate"])
    pipeline = build_pipeline(config)

    print("操作方式: 按 Enter 開始錄音，再按 Enter 停止錄音")
    print("         輸入 q 然後 Enter 退出")
    print()

    try:
        while True:
            user_input = input(">>> 按 Enter 開始錄音 (q=退出): ").strip()
            if user_input.lower() == "q":
                break

            mic.start()
            print("    🎤 錄音中... 按 Enter 停止")
            start_time = time.time()
            input()
            duration = time.time() - start_time
            audio = mic.stop()

            if len(audio) == 0:
                print("    (沒有收到音訊)")
                continue

            print(f"    錄音 {duration:.1f} 秒，辨識中...")
            raw_text = engine.transcribe(audio, config["audio"]["sample_rate"])

            if not raw_text.strip():
                print("    (沒有辨識到文字)")
                continue

            processed_text = pipeline.run(raw_text)

            print(f"    原始: {raw_text}")
            print(f"    結果: {processed_text}")
            if config["output"]["method"] == "clipboard":
                print("    (已複製到剪貼簿)")
            print()

    except KeyboardInterrupt:
        print("\n中斷。")
    finally:
        print("釋放模型資源...")
        engine.unload()
        print("結束。")


if __name__ == "__main__":
    main()
