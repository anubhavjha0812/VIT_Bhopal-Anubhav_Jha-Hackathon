"""Correlated default simulation: one-year loss distribution, VaR and ES at 99% (E14).

Gaussian copula: each obligor defaults when its asset draw falls below N^-1(PD). The draws are
correlated with the 15 stocks' daily return correlation: calm 2008-2019 before, stressed
14 Feb - 23 Mar 2020 after (in a crisis everything falls together). PDs: the long-run PD of the
rating before; the scenario PD (downgrade + Vasicek stress) after. 20,000 draws, seed 42.

Loss when an obligor defaults: LGD × (revolver EAD + bond notional + swap exposure) + sold
protection × LGD - bought protection × LGD.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskengine.module_b.book import LGD
from riskengine.module_b.revalue import sa_ccr_ead
from riskengine.module_b.shock import CALM, EPISODES, returns

DRAWS = 20_000
SEED = 42
LEVEL = 0.99
STRESSED = EPISODES["covid2020"]


@dataclass(frozen=True)
class LossDistribution:
    losses: np.ndarray  # USD, one per draw
    expected: float
    var: float
    es: float


def nearest_correlation(c: pd.DataFrame) -> pd.DataFrame:
    """Clip negative eigenvalues, rescale to a unit diagonal (pairwise estimates may not be PSD)."""
    vals, vecs = np.linalg.eigh(c.to_numpy())
    fixed = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return pd.DataFrame(fixed / np.outer(d, d), c.index, c.columns)


def asset_correlation(window: tuple[str, str], tickers: list[str]) -> pd.DataFrame:
    corr = returns().loc[window[0] : window[1], tickers].corr().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    return nearest_correlation(pd.DataFrame(corr, tickers, tickers))


def loss_given_default(book: pd.DataFrame) -> pd.Series:
    """USD loss per obligor if it defaults (hedges net it down)."""
    sign = {"revolver": 1, "bond": 1, "irs_pay_fixed": 1, "cds_sold": 1, "cds_bought": -1}
    exposure = book["ead"].where(book["instrument"] == "revolver", book["notional"])
    irs = book["instrument"] == "irs_pay_fixed"
    exposure[irs] = [sa_ccr_ead(n, y) for n, y in book.loc[irs, ["notional", "years"]].to_numpy()]
    lgd = book["lgd"].fillna(LGD) * exposure * book["instrument"].map(sign)
    return lgd.groupby(book["ticker"]).sum()


def simulate(
    pd_by_ticker: pd.Series,
    lgd_by_ticker: pd.Series,
    corr: pd.DataFrame,
    draws: int = DRAWS,
    seed: int = SEED,
    level: float = LEVEL,
) -> LossDistribution:
    tickers = list(pd_by_ticker.index)
    chol = np.linalg.cholesky(corr.loc[tickers, tickers].to_numpy())
    z = np.random.default_rng(seed).standard_normal((draws, len(tickers))) @ chol.T
    threshold = norm.ppf(np.clip(pd_by_ticker.to_numpy(), 1e-12, 1 - 1e-12))
    losses = (z < threshold) @ lgd_by_ticker.reindex(tickers).to_numpy()
    var = float(np.quantile(losses, level))
    worst = np.sort(losses)[-max(int(round((1 - level) * draws)), 1) :]  # the worst 1% of draws
    return LossDistribution(losses, float(losses.mean()), var, float(worst.mean()))


def before_after(book: pd.DataFrame, scenario: pd.DataFrame, market_shock: bool = True) -> tuple:
    """(before, after) loss distributions for a scenario from shock.scenario / revalue.

    After uses the stressed correlation only when a market episode is applied; an alert alone
    (multiple 0) keeps the calm one, so only the PDs change.
    """
    tickers = list(scenario.index)
    lgd = loss_given_default(book).reindex(tickers).fillna(0)
    calm = asset_correlation(CALM, tickers)
    after_corr = asset_correlation(STRESSED, tickers) if market_shock else calm
    before = simulate(scenario["pd_before"], lgd, calm)
    after = simulate(scenario["pd_after"], lgd, after_corr)
    return before, after
