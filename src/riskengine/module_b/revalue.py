"""Value the book before and after a scenario: losses, IFRS 9 stages, IRB RWA and CET1 (E14).

Where each position's loss goes:
- revolvers (amortised cost): IFRS 9 expected credit loss. Stage 1 = 12-month PD; stage 2 =
  lifetime PD after a cut of 3+ notches or a fall from investment grade to below BBB-; stage 3 = D.
- bonds (fair value through OCI, which counts in CET1): full repricing of a par bullet bond at
  Treasury + spread; at D the bond is worth its recovery (1 - LGD).
- CDS (trading book): mark-to-market (spread change × risky annuity); at D, ± notional × LGD.
  Bought protection gains when spreads widen, sold protection loses.
- interest rate swaps (trading book, pay fixed at the book-date rate): with `rates="fixed"` only
  CVA moves; with `rates="moved"` (E22) also the mark-to-market N·(r1 − r0)·annuity(r1, T). CVA
  exposure = SA-CCR EAD (1.4 × 0.5% × supervisory duration × notional), not the mark-to-market.
- rates (E22): `fixed` holds the 10-year Treasury at the book date; `moved` shifts every tenor by
  the episode's 10-year change × multiple (a parallel shift), floored at 0. Revolvers are taken
  as floating-rate, so rates move only bonds, swaps and the CDS annuity.
Capital (E14 as amended after the first run):
- credit: Basel F-IRB corporate RWA on revolvers, bonds and swap exposures, at the normal one-year
  PD of the rating (Basel uses long-run PDs; a downgrade raises RWA, a bad market alone does not).
  Defaulted = 0, covered by provisions. The Vasicek stressed PD drives IFRS 9 ECL only.
- market: bonds also carry Basel standardised market-risk capital (specific risk by rating and
  maturity + general interest-rate risk by maturity band), as a bank's bond book does.
RWA is converted to INR at the scenario USD/INR. CET1 starts at 13% of RWA; losses after 25% tax.
Market risk of CDS and swaps is not modelled (stated).
"""

import math
from dataclasses import dataclass

import pandas as pd
from scipy.stats import norm

from riskengine.module_b.params import DEFAULT, basel_rho, lifetime_pd, notch, pd_1y
from riskengine.module_b.shock import DEFAULT_EPISODE, Market, scenario

Episode = str | tuple[str, str] | Market

CET1_START = 0.13
CET1_FLOOR = 0.08  # RBI: CET1 5.5% + capital conservation buffer 2.5%
TAX = 0.25
STAGE2_NOTCHES = 3
INVESTMENT_GRADE_LAST = notch("BBB-")
SA_CCR_FACTOR = 1.4 * 0.005  # alpha × interest-rate supervisory factor
Q999 = norm.ppf(0.999)
# Basel standardised market risk for debt (1996 amendment, kept in Basel 2.5).
# General interest-rate risk, maturity method, coupon ≥ 3%: (band upper bound in years, weight).
GENERAL_RISK = [(1, 0.007), (2, 0.0125), (3, 0.0175), (4, 0.0225), (5, 0.0275), (7, 0.0325),
                (10, 0.0375), (15, 0.045), (20, 0.0525), (math.inf, 0.06)]  # fmt: skip


# --- Pricing ---------------------------------------------------------------------------------


def annuity(rate: float, years: float) -> float:
    """Value of 1 a year for `years` (annual, last part-year pro rata) at a flat rate."""
    n = max(int(math.ceil(years)), 1)
    return sum(min(1.0, years - t + 1) / (1 + rate) ** t for t in range(1, n + 1))


def bond_price(coupon: float, yld: float, years: float) -> float:
    """Price per 1 of face: annual coupons, face at maturity, flat yield."""
    return coupon * annuity(yld, years) + 1 / (1 + yld) ** max(years, 1.0)


def duration_convexity(coupon: float, yld: float, years: float, h: float = 1e-4) -> tuple:
    """Modified duration and convexity by finite differences (for display and tests)."""
    p0, up, dn = (bond_price(coupon, yld + d, years) for d in (0.0, h, -h))
    return (dn - up) / (2 * h * p0), (up + dn - 2 * p0) / (h * h * p0)


def sa_ccr_ead(notional: float, years: float) -> float:
    supervisory_duration = (1 - math.exp(-0.05 * years)) / 0.05
    return SA_CCR_FACTOR * supervisory_duration * notional


def irb_rwa(pd: float, lgd: float, years: float, ead: float) -> float:
    """Basel F-IRB corporate risk-weighted assets (no SME adjustment)."""
    if pd >= 1:
        return 0.0
    rho = basel_rho(pd)
    m = min(max(years, 1.0), 5.0)
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    tail = norm.cdf((norm.ppf(pd) + math.sqrt(rho) * Q999) / math.sqrt(1 - rho))
    k = lgd * (tail - pd) * (1 + (m - 2.5) * b) / (1 - 1.5 * b)
    return 12.5 * k * ead


def market_risk_rwa(rating: str, years: float, value: float) -> float:
    """Standardised market-risk RWA of a long bond position: 12.5 × (specific + general) × value.

    Specific risk: investment grade 0.25% (≤ 6 months), 1.0% (≤ 2 years), 1.6% (longer);
    BB+ to BB- 8%; below BB- 12%.
    """
    n = notch(rating)
    if n <= INVESTMENT_GRADE_LAST:
        specific = 0.0025 if years <= 0.5 else 0.01 if years <= 2 else 0.016
    else:
        specific = 0.08 if n <= notch("BB-") else 0.12
    general = next(w for upper, w in GENERAL_RISK if years <= upper)
    return 12.5 * (specific + general) * value


def stage(rating_before: str, rating_after: str) -> int:
    nb, na = notch(rating_before), notch(rating_after)
    if na == DEFAULT:
        return 3
    fallen_angel = nb <= INVESTMENT_GRADE_LAST < na
    return 2 if na - nb >= STAGE2_NOTCHES or fallen_angel else 1


def ecl(pd: float, lgd: float, ead: float, years: float, stg: int) -> float:
    if stg == 3:
        return lgd * ead
    return (lifetime_pd(pd, years) if stg == 2 else pd) * lgd * ead


# --- Book valuation --------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    positions: pd.DataFrame  # one row per position: value and capital before/after, loss
    scenario: pd.DataFrame  # one row per ticker (shock.scenario)
    market: Market
    fx_before: float
    fx_after: float
    loss_usd: float  # before tax; positive = loss
    rwa_before_inr: float
    rwa_after_inr: float
    capital_before_inr: float
    capital_after_inr: float

    @property
    def cet1_before(self) -> float:
        return self.capital_before_inr / self.rwa_before_inr

    @property
    def cet1_after(self) -> float:
        return self.capital_after_inr / self.rwa_after_inr


def _position(
    p: pd.Series, s: pd.Series, rate0: float, rate: float, when: str
) -> tuple[float, float, int]:
    """(value or -ECL in USD, RWA in USD, stage) of one position in the before or after state.
    rate0 = the book-date Treasury (sets coupons and swap fixed rates); rate = the one now."""
    rating, sprd, pd_ = s[f"rating_{when}"], s[f"spread_{when}"] / 100, s[f"pd_{when}"]
    pd_capital = pd_1y(rating)  # long-run PD of the (new) rating
    defaulted = notch(rating) == DEFAULT
    coupon = rate0 + s["spread_before"] / 100  # priced at par at the book date
    inst, n, years, lgd = p["instrument"], p["notional"], p["years"], p["lgd"]
    if inst == "revolver":
        stg = stage(s["rating_before"], rating) if when == "after" else 1
        rwa = irb_rwa(pd_capital, lgd, years, p["ead"])
        return -ecl(pd_, lgd, p["ead"], years, stg), rwa, stg
    if inst == "bond":
        value = n * (1 - lgd) if defaulted else n * bond_price(coupon, rate + sprd, years)
        rwa = irb_rwa(pd_capital, lgd, years, n) + market_risk_rwa(rating, years, value)
        return value, rwa, 0
    if inst in ("cds_bought", "cds_sold"):
        sign = 1 if inst == "cds_bought" else -1
        if defaulted:
            return sign * n * lgd, 0.0, 0
        mtm = (sprd - s["spread_before"] / 100) * annuity(rate + sprd, years)
        return sign * n * mtm, 0.0, 0
    if inst == "irs_pay_fixed":
        ead = sa_ccr_ead(n, years)
        hazard = sprd / lgd
        cva = lgd * ead if defaulted else lgd * ead * (1 - math.exp(-hazard * years))
        mtm = n * (rate - rate0) * annuity(rate, years)  # pay fixed: gains when rates rise
        return mtm - cva, irb_rwa(pd_capital, lgd, years, ead), 0
    raise ValueError(inst)


def treasury_after(mk: Market, multiple: float = 1.0, rates: str = "fixed") -> float:
    """The 10-year Treasury after the scenario, as a fraction (E22: parallel shift, floor 0)."""
    rate0 = mk.rate0 / 100
    return max(rate0 + multiple * mk.d_rate / 100, 0.0) if rates == "moved" else rate0


def revalue(
    book: pd.DataFrame,
    obligor: str | None = None,
    notches: int = 0,
    episode: Episode = DEFAULT_EPISODE,
    multiple: float = 1.0,
    rates: str = "fixed",
) -> Result:
    ratings = book.drop_duplicates("ticker").set_index("ticker")["rating"]
    sc, mk, fx_after = scenario(ratings, obligor, notches, episode, multiple)
    rate0, rate1 = mk.rate0 / 100, treasury_after(mk, multiple, rates)
    rows = []
    for _, p in book.iterrows():
        s = sc.loc[p["ticker"]]
        v0, rwa0, _ = _position(p, s, rate0, rate0, "before")
        v1, rwa1, stg = _position(p, s, rate0, rate1, "after")
        rows.append({**p.to_dict(), "value_before": v0, "value_after": v1, "loss": v0 - v1,
                     "rwa_before": rwa0, "rwa_after": rwa1, "stage": stg})  # fmt: skip
    pos = pd.DataFrame(rows)
    loss = float(pos["loss"].sum())
    rwa0_inr, rwa1_inr = pos["rwa_before"].sum() * mk.fx0, pos["rwa_after"].sum() * fx_after
    cap0 = CET1_START * rwa0_inr
    cap1 = cap0 - (1 - TAX) * loss * fx_after
    return Result(pos, sc, mk, mk.fx0, fx_after, loss, rwa0_inr, rwa1_inr, cap0, cap1)


def reverse_stress(
    book: pd.DataFrame,
    obligor: str | None = None,
    notches: int = 0,
    episode: Episode = DEFAULT_EPISODE,
    floor: float = CET1_FLOOR,
    high: float = 20.0,
    tol: float = 1e-3,
    rates: str = "fixed",
) -> float:
    """Smallest multiple of the episode that takes CET1 down to `floor` (inf if not by `high`)."""

    def cet1(k: float) -> float:
        return revalue(book, obligor, notches, episode, k, rates).cet1_after

    if cet1(high) > floor:
        return math.inf
    low = 0.0
    while high - low > tol:
        mid = (low + high) / 2
        low, high = (mid, high) if cet1(mid) > floor else (low, mid)
    return high
