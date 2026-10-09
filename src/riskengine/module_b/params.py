"""Rating scale and credit parameters for Module B (docs/DECISIONS.md, E14).

- PD: S&P long-run one-year default rates by letter grade (1981-2019), log-interpolated by notch,
  floored at 0.03% (Basel). D = 100%.
- Spread: Moody's Aaa and Baa spreads over the 10-year Treasury (FRED), log-linear by notch with
  Aaa = AAA and Baa = BBB, extrapolated below BBB (this understates junk spreads; stated).
- Vasicek stressed PD and the Basel corporate asset correlation.
"""

import math

from scipy.stats import norm

# S&P long-term scale, best to worst; the index is the notch (AAA = 0).
SCALE = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]  # fmt: skip
DEFAULT = SCALE.index("D")

# S&P 2019 Annual Global Corporate Default and Rating Transition Study, table 24: average
# cumulative default rates 1981-2019, year 1 (%). Placed on the letter's flat notch.
LETTER_PD = {"AAA": 0.00, "AA": 0.02, "A": 0.05, "BBB": 0.16, "BB": 0.61, "B": 3.33, "CCC": 27.08}
PD_SOURCE = "S&P 2019 Annual Global Corporate Default and Rating Transition Study, table 24"
PD_FLOOR = 0.0003  # Basel floor for corporates
AAA_NOTCH, BAA_NOTCH = SCALE.index("AAA"), SCALE.index("BBB")  # Moody's Aaa and Baa indices


def notch(rating: str) -> int:
    return SCALE.index(rating)


def rating_at(n: int) -> str:
    return SCALE[min(max(n, 0), DEFAULT)]


def pd_1y(rating: str) -> float:
    """One-year PD as a fraction. Log-linear between letter anchors; below CCC stays CCC/C."""
    n = notch(rating)
    if n == DEFAULT:
        return 1.0
    anchors = [(notch(r), max(p / 100, PD_FLOOR)) for r, p in LETTER_PD.items()]
    if n >= anchors[-1][0]:
        return anchors[-1][1]
    for (n0, p0), (n1, p1) in zip(anchors, anchors[1:], strict=False):
        if n0 <= n <= n1:
            w = (n - n0) / (n1 - n0)
            return math.exp((1 - w) * math.log(p0) + w * math.log(p1))
    raise ValueError(rating)


def lifetime_pd(pd: float, years: float) -> float:
    """IFRS 9 stage 2: chance of default over the remaining life, constant yearly PD."""
    return 1 - (1 - pd) ** max(years, 1.0)


def spread(rating: str, aaa: float, baa: float) -> float:
    """Credit spread (in % points) for a rating, from the Aaa and Baa index spreads that day.

    D has no spread (the bond is worth its recovery); the worst live rating C is used for it.
    """
    n = min(notch(rating), DEFAULT - 1)
    slope = math.log(baa / aaa) / (BAA_NOTCH - AAA_NOTCH)
    return aaa * math.exp(slope * (n - AAA_NOTCH))


def basel_rho(pd: float) -> float:
    """Basel II/III corporate asset correlation: 24% for the best names, 12% for the worst."""
    w = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def stressed_pd(pd: float, z: float, rho: float | None = None) -> float:
    """Vasicek PD given a bad market of z standard deviations (z > 0 is bad)."""
    if pd >= 1:
        return 1.0
    rho = basel_rho(pd) if rho is None else rho
    return float(norm.cdf((norm.ppf(pd) + math.sqrt(rho) * z) / math.sqrt(1 - rho)))
