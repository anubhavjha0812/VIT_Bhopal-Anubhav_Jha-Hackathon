import pytest

from riskengine.module_b.params import (
    PD_FLOOR,
    SCALE,
    basel_rho,
    lifetime_pd,
    pd_1y,
    rating_at,
    spread,
    stressed_pd,
)


def test_pd_never_falls_as_rating_gets_worse():
    pds = [pd_1y(r) for r in SCALE]
    assert pds == sorted(pds)
    assert pds[-1] == 1.0


def test_pd_hits_the_sp_table_on_letter_grades_and_the_floor():
    assert pd_1y("BBB") == pytest.approx(0.0016)
    assert pd_1y("B") == pytest.approx(0.0333)
    assert pd_1y("AAA") == pytest.approx(PD_FLOOR)  # S&P table says 0.00
    assert pd_1y("BBB") < pd_1y("BBB-") < pd_1y("BB+") < pd_1y("BB")


def test_rating_at_clamps_to_scale():
    assert rating_at(-3) == "AAA"
    assert rating_at(99) == "D"


def test_spread_matches_moodys_indices_and_rises():
    aaa, baa = 1.12, 1.98  # FRED, 31 Dec 2019
    assert spread("AAA", aaa, baa) == pytest.approx(aaa)
    assert spread("BBB", aaa, baa) == pytest.approx(baa)
    spreads = [spread(r, aaa, baa) for r in SCALE]
    assert spreads == sorted(spreads)


def test_basel_rho_bounds_and_known_value():
    assert basel_rho(0.0003) == pytest.approx(0.2382, abs=1e-4)
    assert basel_rho(0.2) == pytest.approx(0.12, abs=1e-4)


def test_vasicek_rises_with_bad_market():
    pd = 0.01
    calm, bad, worse = (stressed_pd(pd, z) for z in (0.0, 2.0, 3.0))
    assert calm < pd < bad < worse  # z = 0 is the median market, a bit below the average PD
    assert stressed_pd(1.0, 3.0) == 1.0


def test_vasicek_at_999_matches_basel_capital_quantile():
    # Basel IRB uses the same formula at z = N^-1(0.999) = 3.0902. By hand for PD 1%:
    # rho 0.19278; (-2.32635 + 0.43907 * 3.0902) / 0.89846 = -1.07912; N(-1.07912) = 0.14027
    assert stressed_pd(0.01, 3.0902) == pytest.approx(0.14027, abs=1e-4)


def test_lifetime_pd():
    assert lifetime_pd(0.01, 1) == pytest.approx(0.01)
    assert lifetime_pd(0.01, 5) == pytest.approx(1 - 0.99**5)
    assert lifetime_pd(0.01, 0.5) == pytest.approx(0.01)  # at least one year
