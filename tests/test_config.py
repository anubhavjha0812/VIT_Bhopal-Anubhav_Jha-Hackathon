import pytest

from riskengine.config import Config, load_config


def test_default_config_is_baseline():
    assert load_config() == Config()


def test_bad_switch_value_is_rejected():
    with pytest.raises(ValueError, match="sentiment"):
        Config(sentiment="gpt")


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('sentimnet = "own"\n')
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_impact_trigger_must_be_1_to_10():
    with pytest.raises(ValueError, match="impact_trigger"):
        Config(impact_trigger=11)
