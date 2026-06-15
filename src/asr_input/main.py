"""Entry point for ASR input system."""

from asr_input.config import load_config


def main() -> None:
    config = load_config()
    print(f"ASR Input v0.1.0 — engine: {config['asr']['engine']}")
    # TODO: wire up audio → asr → processing → output pipeline


if __name__ == "__main__":
    main()
