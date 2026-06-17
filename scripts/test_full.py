"""Full pipeline test: record from mic, run through ASR + processing."""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch

from asr_input.asr.qwen import QwenASREngine
from asr_input.processing.opencc_conv import OpenCCConverter
from asr_input.processing.tw_terms import TaiwanTermReplacer
from asr_input.processing.pipeline import ProcessingPipeline
from asr_input.audio.capture import MicrophoneCapture

print("=== Full Pipeline Test (Mic) ===")
print()

pipeline = ProcessingPipeline()
pipeline.add(OpenCCConverter("s2twp"))
pipeline.add(TaiwanTermReplacer())

print("Loading Qwen3-ASR-1.7B...")
engine = QwenASREngine(
    model_id="Qwen/Qwen3-ASR-1.7B",
    device="cuda",
    language="Chinese",
    prompt="以下是台灣繁體中文的語音轉錄。",
)
engine.load()
print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
print()

mic = MicrophoneCapture(sample_rate=16000)

print("Press Enter to start recording...")
input()
mic.start()
print("Recording... Press Enter to stop.")
input()
audio = mic.stop()

print(f"Recorded {len(audio)/16000:.1f} seconds of audio")
print("Transcribing...")

raw_text = engine.transcribe(audio, 16000)
print(f"Raw ASR output : {raw_text}")

if raw_text.strip():
    processed = pipeline.run(raw_text)
    print(f"After processing: {processed}")
else:
    print("(no speech detected)")

print()
engine.unload()
print("Done!")
