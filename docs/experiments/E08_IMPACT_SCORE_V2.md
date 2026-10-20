# E8: Impact score v2

Rules fixed 2026-10-09, before any v2 model is fitted. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** in v1 (E6) the stock's recent volatility alone ranked moves better than the chosen
  model (Spearman 0.27–0.34 vs 0.17). Volatile stocks move more whatever the news says, so
  volatility-only is the fair naive baseline. v1 stays reported as it is.
- **Bug found in v1:** pandas 3 stores these timestamps in microseconds, but the "headlines in
  the previous 24 h" feature assumed nanoseconds, so in v1 it counted roughly the previous
  1,000 days. v1 results stay reported as they are, with this caveat. v2 uses the fixed feature
  (with a unit test).
- **Rows, event day, CAR:** as E6. Headline features use the same E5 sentiment and topic models
  as v1, so v1 and v2 differ only in target, features and model.
- **Target:** z = |CAR[0,+1]| / (√2 × vol20), the move in units of the stock's normal two-day
  move. Fit on log(1 + z). **Impact** = predicted z × √2 × vol20, i.e. the expected size of
  the move; deciles of it on 2009–2018 give the 1–10 scale.
- **Features:** E6 features plus **novelty** = 1 − the highest cosine similarity between the
  headline and any headline for the same stock in the previous 72 hours (word 1–2-grams,
  hashing vectoriser, no fitting); 1 when there is none. |sentiment| is already in E6.
- **Baselines:** volatility-only (impact = vol20, the official baseline) and the E6 formula.
- **Candidates:** ridge, gradient boosting (E6 settings), and their average on the log(1 + z)
  scale (ensemble).
- **Choice:** rolling-origin validation. Train on 2009 up to year Y − 1, validate on Y, for
  Y = 2015, 2016, 2017, 2018. Score = mean Spearman between impact and |CAR| over the four years.
  The highest mean wins. The chosen model is refitted on 2009–2018.
- **Test F (fresh, primary):** S&P 100 members (2020 list) other than our 15, headlines
  2019-01-01 to 2020-06-10 from FNSPID by their `Stock_symbol`, prices from yfinance. Every
  symbol is included; rows without enough price history for the market model drop out as in E6.
  No row of test F is used for fitting or choosing. Scored once; 2019 and 2020 reported
  separately.
- **Test S (seen):** our 15 stocks in 2019 and 2020. These years were already looked at in v1,
  so test S is reported for continuity only and marked as seen.
- **Metrics:** E6 metrics (Spearman vs |CAR| with 95% bootstrap interval, mean |CAR| for impact
  ≥ 8 vs ≤ 3), plus Spearman vs z, which shows how much a model knows beyond volatility.
- **Success:** the chosen model beats volatility-only on Spearman vs |CAR| in both years of
  test F. Reported either way.
- **Engine default:** v2 replaces v1 in the engine if its mean rolling-validation Spearman beats
  volatility-only. Decided before testing.

## Result

- **Question:** Impact score v2
- **Outcome:** fresh 2019 Spearman 0.294 vs volatility 0.145; fresh 2020 −0.041 vs 0.307
- **Verdict:** win 2019, fail 2020
- **Stored results:** `impact_v2_results.json`, `docs/experiments/IMPACT_V2.md`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
