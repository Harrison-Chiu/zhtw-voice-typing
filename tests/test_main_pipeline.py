"""Production processing policy tests."""

from asr_input.main import build_pipeline


def make_config(*, localize: bool) -> dict:
    return {
        "processing": {
            "opencc_config": "s2t",
            "localize_tw_terms": localize,
        },
        "output": {"method": "clipboard"},
    }


def test_default_policy_converts_script_without_rewriting_spoken_terms():
    pipeline = build_pipeline(make_config(localize=False), include_output=False)

    assert pipeline.run("设备接口支持串口") == "設備接口支持串口"


def test_localization_can_be_enabled_independently():
    pipeline = build_pipeline(make_config(localize=True), include_output=False)

    assert pipeline.run("默認插件") == "預設外掛"
