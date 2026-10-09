import pandas as pd
import pytest

from riskengine.module_b.shock import EPISODES, market, parse_notches, scenario

RATINGS = pd.Series({"BA": "A-", "DIS": "A", "KO": "A+", "TSLA": "B-"})


@pytest.mark.parametrize(
    "text,event,expected",
    [
        ("Moodys downgrading #Boeing to Baa2 from Baa1.", "Credit downgrade", 1),
        ("Fitch downgrades Disney's IDR to 'A-' from 'A' and maintains", "Credit downgrade", 1),
        ("S&P cuts Ford to junk BB+ from BBB-", "Credit downgrade", 1),
        ("Moody's cuts Boeing from A3 to Baa2", "Credit downgrade", 2),
        ("S&P lowers Kraft to BBB- from BBB+ on weak cash flow", "Credit downgrade", 2),
        ("$DIS - Disney's credit rating lowered at S&P", "Credit downgrade", 1),  # no ratings
        ("Upgraded to A from A- by S&P", "Credit downgrade", 1),  # an upgrade is not a cut
        ("Hertz files for Chapter 11", "Default", 99),
    ],
)
def test_parse_notches(text, event, expected):
    assert parse_notches(text, event) == expected


def test_crisis_episodes_are_bad_markets():
    for episode in EPISODES:
        m = market(episode)
        assert m.z > 0 and m.d_baa > 0 and m.fx_move > 0
    assert 1.5 < market("covid2020").z < 2.5  # one-year scale, not the episode's own length


def test_obligor_is_cut_and_others_widen_less():
    s, _, _ = scenario(RATINGS, "BA", 1, multiple=0.0)  # alert only, no market move
    assert s.loc["BA", "rating_after"] == "BBB+"
    assert (s.drop("BA")["rating_after"] == s.drop("BA")["rating_before"]).all()
    own = s.loc["BA", "spread_after"] - s.loc["BA", "spread_before"]
    others = s.drop("BA")["spread_after"] - s.drop("BA")["spread_before"]
    assert own > 0
    assert ((others >= 0) & (others < own)).all()


def test_default_goes_to_d():
    s, _, _ = scenario(RATINGS, "TSLA", 99, multiple=0.0)
    assert s.loc["TSLA", "rating_after"] == "D"
    assert s.loc["TSLA", "pd_after"] == 1.0


def test_market_only_has_no_contagion_and_zero_multiple_changes_nothing():
    s, m, fx = scenario(RATINGS, None, 0, multiple=0.0)
    assert (s["contagion"] == 0).all()
    assert s["spread_after"].tolist() == pytest.approx(s["spread_before"].tolist())
    assert s["pd_after"].tolist() == pytest.approx(s["pd_before"].tolist())
    assert fx == pytest.approx(m.fx0)


def test_bigger_multiple_bigger_stress_and_weaker_rupee():
    s1, _, fx1 = scenario(RATINGS, "BA", 1, multiple=1.0)
    s2, _, fx2 = scenario(RATINGS, "BA", 1, multiple=2.0)
    assert (s2["pd_after"] > s1["pd_after"]).all()
    assert (s2["spread_after"] > s1["spread_after"]).all()
    assert (s1["pd_after"] >= s1["pd_before"]).all()
    assert fx2 > fx1
