from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
