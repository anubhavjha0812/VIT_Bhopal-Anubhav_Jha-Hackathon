import pytest

from riskengine.config import Config, load_config


def test_default_config_runs_our_models():
    # The file switches on our models (DECISIONS E11); Config() itself stays the baseline.
    assert load_config() == Config(sentiment="own", events="model")


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


def test_window_must_be_in_order():
    from datetime import date

    with pytest.raises(ValueError, match="window_start"):
        Config(window_start=date(2020, 6, 10), window_end=date(2020, 4, 9))
