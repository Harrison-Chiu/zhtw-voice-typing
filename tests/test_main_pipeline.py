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


def test_tw_dict_path_from_config_is_honoured(tmp_path, monkeypatch):
    """config 的 tw_dict_path 要真的被讀，否則它會退化成沒人用的死設定。"""
    import asr_input.main as main_mod

    custom = tmp_path / "custom_dict.yaml"
    custom.write_text("默認: 客製結果\n", encoding="utf-8")
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)

    config = make_config(localize=True)
    config["processing"]["tw_dict_path"] = "custom_dict.yaml"
    pipeline = build_pipeline(config, include_output=False)

    assert pipeline.run("默認") == "客製結果"


def test_tw_dict_path_absent_falls_back_to_bundled_dict():
    config = make_config(localize=True)
    config["processing"].pop("tw_dict_path", None)
    pipeline = build_pipeline(config, include_output=False)

    assert pipeline.run("默認插件") == "預設外掛"
