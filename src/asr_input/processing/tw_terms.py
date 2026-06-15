"""Custom Taiwan terminology replacements."""

from pathlib import Path

import yaml

from asr_input.processing.pipeline import TextProcessor

DEFAULT_DICT_PATH = Path(__file__).parent.parent.parent.parent / "data" / "tw_dict.yaml"


class TaiwanTermReplacer(TextProcessor):
    def __init__(self, dict_path: Path = DEFAULT_DICT_PATH) -> None:
        with open(dict_path, encoding="utf-8") as f:
            self._replacements: dict[str, str] = yaml.safe_load(f) or {}

    def process(self, text: str) -> str:
        for cn_term, tw_term in self._replacements.items():
            text = text.replace(cn_term, tw_term)
        return text
