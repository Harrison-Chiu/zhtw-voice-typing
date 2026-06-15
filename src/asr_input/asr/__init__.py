"""ASR engines and the config-driven factory."""

from asr_input.asr.base import ASREngine


def build_engine(asr_cfg: dict) -> ASREngine:
    """Construct an ASR engine from the `asr` section of config.yaml."""
    engine = asr_cfg.get("engine", "qwen")

    if engine == "qwen":
        from asr_input.asr.qwen import QwenASREngine

        return QwenASREngine(
            model_id=asr_cfg["model_id"],
            device=asr_cfg["device"],
            language=asr_cfg.get("language"),
            context=asr_cfg.get("context", ""),
        )

    if engine == "whisper":
        from asr_input.asr.whisper_fw import WhisperFWEngine

        return WhisperFWEngine(
            model_id=asr_cfg["model_id"],
            device=asr_cfg["device"],
            compute_type=asr_cfg.get("compute_type", "float16"),
            language=asr_cfg.get("language", "zh"),
            initial_prompt=asr_cfg.get("initial_prompt", "繁體中文，台灣用語。"),
            beam_size=asr_cfg.get("beam_size", 5),
        )

    raise ValueError(f"Unknown ASR engine: {engine!r}. Supported: qwen, whisper")
