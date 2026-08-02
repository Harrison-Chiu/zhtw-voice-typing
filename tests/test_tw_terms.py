"""Tests for custom Taiwan terminology replacement."""

from pathlib import Path

import pytest

from asr_input.processing.tw_terms import TaiwanTermReplacer


@pytest.fixture
def replacer_from(tmp_path: Path):
    """Factory: build a replacer from an inline dict written to a temp YAML."""

    def _make(mapping: dict[str, str]) -> TaiwanTermReplacer:
        dict_path = tmp_path / "tw_dict.yaml"
        lines = "\n".join(f'"{k}": "{v}"' for k, v in mapping.items())
        dict_path.write_text(lines + "\n", encoding="utf-8")
        return TaiwanTermReplacer(dict_path=dict_path)

    return _make


def test_basic_replacement(replacer_from):
    r = replacer_from({"人工智能": "人工智慧"})
    assert r.process("這是人工智能技術") == "這是人工智慧技術"


def test_multiple_occurrences_all_replaced(replacer_from):
    r = replacer_from({"代碼": "程式碼"})
    assert r.process("寫代碼，改代碼") == "寫程式碼，改程式碼"


def test_unmatched_text_unchanged(replacer_from):
    r = replacer_from({"代碼": "程式碼"})
    assert r.process("完全沒有關鍵詞") == "完全沒有關鍵詞"


def test_opencc_over_conversion_fix(replacer_from):
    # The real dict's primary job: undo OpenCC's 平台→平臺 over-conversion.
    r = replacer_from({"平臺": "平台"})
    assert r.process("開發平臺") == "開發平台"


def test_empty_dict_is_noop(replacer_from):
    r = replacer_from({})
    assert r.process("任何文字") == "任何文字"


def test_asr_corrections_use_global_external_parameter_and_phrase_only_punctuation():
    r = TaiwanTermReplacer()

    assert r.process("手動外餐的計算") == "手動外參的計算"
    assert r.process("工廠提供的外餐資訊") == "工廠提供的外參資訊"
    assert r.process("外餐") == "外參"
    assert r.process("這個手動表點的技術") == "這個手動標點的技術"
    assert r.process("空間的坐表點的位置") == "空間的座標點的位置"

    # 外餐是明確的個人詞彙取捨；表點仍避免跨詞界誤傷。
    assert r.process("戶外餐廳與報表點選") == "戶外參廳與報表點選"
    assert r.process("表點") == "表點"


@pytest.mark.parametrize(
    ("term", "replacement"),
    [
        ("視頻", "影片"),
        ("默認", "預設"),
        ("插件", "外掛"),
    ],
)
def test_parametrized_terms(replacer_from, term, replacement):
    r = replacer_from({term: replacement})
    assert r.process(f"設定{term}選項") == f"設定{replacement}選項"


def test_real_shipped_dict_loads_and_runs():
    # Smoke test against the actual data/tw_dict.yaml to catch YAML breakage.
    r = TaiwanTermReplacer()
    assert r.process("人工智能") == "人工智慧"
    assert r.process("平臺") == "平台"
