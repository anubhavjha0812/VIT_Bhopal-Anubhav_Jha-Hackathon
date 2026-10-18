import numpy as np
import pandas as pd

from riskengine.module_a_placebo import placebo_metrics


def test_placebo_is_seeded_and_keeps_the_values():
    days = pd.bdate_range("2020-04-09", periods=6)
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(100 + rng.normal(size=(6, 3)).cumsum(0), index=days, columns=cols)
    signal = pd.DataFrame(rng.normal(size=(6, 3)), index=days, columns=cols)
    a = placebo_metrics(signal, closes, 5, seed=1)
    b = placebo_metrics(signal, closes, 5, seed=1)
    assert a.equals(b) and len(a) == 5
    assert (a["total_return"] > -1).all()
