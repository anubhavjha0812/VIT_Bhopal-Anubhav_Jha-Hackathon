import numpy as np
import pandas as pd
import pytest

from riskengine.module_b.book import build_book
from riskengine.module_b.montecarlo import (
    before_after,
    loss_given_default,
    nearest_correlation,
    simulate,
)
from riskengine.module_b.revalue import revalue

NAMES = list("ABCDEFGHIJ")
PD = pd.Series(0.02, index=NAMES)
LGD = pd.Series(10.0, index=NAMES)


def corr(rho: float) -> pd.DataFrame:
    c = np.full((len(NAMES), len(NAMES)), rho)
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, NAMES, NAMES)


def test_zero_correlation_mean_matches_pd_times_lgd():
    d = simulate(PD, LGD, corr(0.0), draws=200_000)
    assert d.expected == pytest.approx((PD * LGD).sum(), rel=0.02)


def test_correlation_fattens_the_tail_not_the_mean():
    calm, crisis = simulate(PD, LGD, corr(0.0)), simulate(PD, LGD, corr(0.6))
    assert crisis.expected == pytest.approx(calm.expected, rel=0.1)
    assert crisis.es > 2 * calm.es
    assert calm.var <= calm.es and crisis.var <= crisis.es


def test_same_seed_same_answer():
    assert simulate(PD, LGD, corr(0.3)).es == simulate(PD, LGD, corr(0.3)).es


def test_nearest_correlation_is_valid():
    bad = pd.DataFrame([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]], list("xyz"), list("xyz"))
    fixed = nearest_correlation(bad).to_numpy()
    assert np.linalg.eigvalsh(fixed).min() > 0
    assert np.diag(fixed) == pytest.approx(np.ones(3))


def test_bought_protection_cuts_the_loss_if_tesla_defaults():
    book = build_book()
    lgd = loss_given_default(book)
    funded = book[(book["ticker"] == "TSLA") & book["instrument"].isin(["bond", "revolver"])]
    exposure = funded["ead"].where(funded["instrument"] == "revolver", funded["notional"])
    gross = 0.45 * exposure.sum()
    assert 0 < lgd["TSLA"] < gross


def test_stress_raises_tail_loss():
    book = build_book()
    alert = revalue(book, "BA", 1, multiple=0.0)
    before, after_alert = before_after(book, alert.scenario, market_shock=False)
    covid = revalue(book, "BA", 1, multiple=1.0)
    _, after_covid = before_after(book, covid.scenario)
    assert before.expected <= after_alert.expected < after_covid.expected
    assert after_alert.es < after_covid.es
