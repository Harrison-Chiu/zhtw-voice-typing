"""ASR engines and the config-driven factory."""

from asr_input.asr.base import ASREngine


def build_engine(asr_cfg: dict, vad_cfg: dict | None = None) -> ASREngine:
    """Construct an ASR engine from the `asr` section of config.yaml.

    If vad_cfg is provided and vad_cfg["enabled"] is True, the engine is wrapped
    with VadSegmentedEngine for automatic long-audio segmentation.
    """
    engine_name = asr_cfg.get("engine", "qwen")

    if engine_name == "qwen":
        from asr_input.asr.qwen import QwenASREngine

        engine = QwenASREngine(
            model_id=asr_cfg["model_id"],
            device=asr_cfg["device"],
            language=asr_cfg.get("language"),
            context=asr_cfg.get("context", ""),
        )

    elif engine_name == "whisper":
        from asr_input.asr.whisper_fw import WhisperFWEngine

        engine = WhisperFWEngine(
            model_id=asr_cfg["model_id"],
            device=asr_cfg["device"],
            compute_type=asr_cfg.get("compute_type", "float16"),
            language=asr_cfg.get("language", "zh"),
            initial_prompt=asr_cfg.get("initial_prompt", "繁體中文，台灣用語。"),
            hotwords=asr_cfg.get("hotwords"),
            beam_size=asr_cfg.get("beam_size", 5),
        )

    else:
        raise ValueError(f"Unknown ASR engine: {engine_name!r}. Supported: qwen, whisper")

    if vad_cfg and vad_cfg.get("enabled", False):
        from asr_input.asr.vad_engine import VadSegmentedEngine

        engine = VadSegmentedEngine(
            inner=engine,
            threshold=vad_cfg.get("threshold", 0.5),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 250),
            min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 800),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
            max_segment_sec=vad_cfg.get("max_segment_sec", 60.0),
            fallback_silence_ms=tuple(vad_cfg.get("fallback_silence_ms", [500, 300])),
            min_audio_len_sec=vad_cfg.get("min_audio_len_sec", 30.0),
        )

    return engine
