"""Impact v3: robustness candidates chosen on rolling folds only (docs/IMPACT_V3.md, E28).

uv run python -m riskengine.impact.train_v3 --select   # folds only, writes the frozen choice
uv run python -m riskengine.impact.train_v3 --test     # once, after --select
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from riskengine.impact.features import CATEGORICAL, build_dataset
from riskengine.impact.train_v2 import FIRST_YEAR, LAST_FIT_YEAR, expected_move, features, rho
from riskengine.paths import DATA, ROOT

FOLDS = (2011, 2015, 2016, 2017, 2018)
FROZEN = DATA / "outputs" / "impact_v3_frozen.json"
RESULTS = DATA / "outputs" / "impact_v3_results.json"
DROP = ("vol20", "vix_prev")
MAX_LOSS_VS_A = 0.02
ROUNDS, ETA, DRO_ALPHA = 5, 0.5, 10.0


def ridge(cols: list[str], alpha: float = 1.0) -> Pipeline:
    num = [c for c in cols if c not in CATEGORICAL]
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), num),
        ]
    )
    return Pipeline([("prep", prep), ("model", Ridge(alpha=alpha))])


class Clipped:
    """Clip every numeric column to the training 1st..99th percentile, then fit."""

    def __init__(self, inner: Pipeline, cols: list[str]):
        self.inner, self.num = inner, [c for c in cols if c not in CATEGORICAL]

    def fit(self, x, y, sample_weight=None):
        self.lo, self.hi = x[self.num].quantile(0.01), x[self.num].quantile(0.99)
        self.inner.fit(self._clip(x), y)
        return self

    def _clip(self, x):
        x = x.copy()
        x[self.num] = x[self.num].clip(self.lo, self.hi, axis=1)
        return x

    def predict(self, x):
        return self.inner.predict(self._clip(x))


class GroupReweighted:
    """Group DRO idea: raise the weight of the training year with the largest loss, 5 rounds."""

    def __init__(self, cols: list[str], years_col: str = "year"):
        self.cols, self.years_col = cols, years_col

    def fit(self, x, y, sample_weight=None):
        years = x[self.years_col].to_numpy()
        groups = np.unique(years)
        w = dict.fromkeys(groups, 1.0 / len(groups))
        for _ in range(ROUNDS):
            sw = np.array([w[g] for g in years])
            sw = sw / sw.mean()
            self.m = ridge(self.cols, DRO_ALPHA)
            self.m.fit(x[self.cols], y, model__sample_weight=sw)
            err = (self.m.predict(x[self.cols]) - np.asarray(y)) ** 2
            loss = {g: err[years == g].mean() for g in groups}
            w = {g: w[g] * np.exp(ETA * loss[g]) for g in groups}
            s = sum(w.values())
            w = {g: v / s for g, v in w.items()}
        return self

    def predict(self, x):
        return self.m.predict(x[self.cols])


def build(name: str, cols: list[str]):
    slim = [c for c in cols if c not in DROP]
    if name == "a_v2_ridge":
        return ridge(cols)
    if name == "b_clipped":
        return Clipped(ridge(cols), cols)
    if name == "c_no_vol_regressors":
        return ridge(slim)
    if name == "d_clipped_no_vol":
        return Clipped(ridge(slim), slim)
    if name == "e_group_reweighted":
        return _Cols(GroupReweighted(slim), slim)
    raise ValueError(name)


class _Cols:
    """Keep the full frame for scaling but let the inner model see only its columns."""

    def __init__(self, inner, cols):
        self.inner, self.cols = inner, cols

    def fit(self, x, y):
        self.inner.fit(x, y)
        return self

    def predict(self, x):
        return self.inner.predict(x)


NAMES = ["a_v2_ridge", "b_clipped", "c_no_vol_regressors", "d_clipped_no_vol", "e_group_reweighted"]


def fit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    model = build(name, cols)
    fit_x = tr if name == "e_group_reweighted" else tr[cols]
    pred_x = te if name == "e_group_reweighted" else te[cols]
    if name in ("b_clipped", "d_clipped_no_vol"):
        fit_x, pred_x = tr[model.num + CATEGORICAL], te[model.num + CATEGORICAL]
    model.fit(fit_x, np.log1p(tr["z"]))
    return expected_move(model.predict(pred_x), te)


def select() -> dict:
    df = build_dataset()
    cols = features(df)
    table = {n: [] for n in NAMES}
    vol = []
    for year in FOLDS:
        tr, va = df[df["year"].between(FIRST_YEAR, year - 1)], df[df["year"] == year]
        vol.append(rho(va["vol20"], va["abs_car01"]))
        for n in NAMES:
            table[n].append(rho(fit_predict(n, tr, va, cols), va["abs_car01"]))
    gain = {n: [a - b for a, b in zip(table[n], vol, strict=True)] for n in NAMES}
    mean = {n: float(np.mean(table[n])) for n in NAMES}
    worst = {n: float(min(gain[n])) for n in NAMES}
    ok = [n for n in NAMES if mean["a_v2_ridge"] - mean[n] <= MAX_LOSS_VS_A]
    best = max(ok, key=worst.get)
    chosen = best if worst[best] > worst["a_v2_ridge"] else "a_v2_ridge"
    code = (ROOT / "src/riskengine/impact/train_v3.py").read_bytes()
    out = {
        "folds": list(FOLDS), "spearman": table, "volatility_only": vol, "gain_vs_vol": gain,
        "mean": mean, "worst_gain": worst, "chosen": chosen,
        "code_sha256": hashlib.sha256(code).hexdigest(),
        "rule": "max worst-fold gain over vol-only; mean loss vs v2 <= 0.02; else keep v2",
    }  # fmt: skip
    FROZEN.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("mean", "worst_gain", "chosen")}, indent=1))
    return out


def test() -> None:
    from riskengine.impact import train_v2 as v2

    frozen = json.loads(FROZEN.read_text())
    assert not RESULTS.exists(), "the 2019/2020 test is run once"
    df = build_dataset()
    cols = features(df)
    top = sorted(set(df["publisher"]) - {"other"})
    fit_rows = df[df["year"].between(FIRST_YEAR, LAST_FIT_YEAR)]
    fresh_df = v2.fresh_dataset(top)
    sets = {f"F_fresh_{y}": fresh_df[fresh_df["year"] == y] for y in (2019, 2020)}
    sets |= {f"S_seen_{y}": df[df["year"] == y] for y in (2019, 2020)}
    res = {"chosen": frozen["chosen"], "sets": {}}
    for name, t in sets.items():
        row = {"n": len(t), "volatility_only": round(rho(t["vol20"], t["abs_car01"]), 4)}
        for n in dict.fromkeys(["a_v2_ridge", frozen["chosen"]]):
            row[n] = round(rho(fit_predict(n, fit_rows, t, cols), t["abs_car01"]), 4)
        res["sets"][name] = row
        print(name, row, flush=True)
    RESULTS.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.select:
        select()
    if a.test:
        test()
