# E6: Learned impact score

Rules fixed 2026-10-06, before any impact model is fitted. Written down before the experiment was run; unchanged here.

## Rules

- **Rows:** FNSPID headlines for our 15 stocks after cleaning, one row per (headline, stock).
  Gold-set headlines (E3) are dropped.
- **Event day:** first trading day whose New York close (16:00) is after the headline time.
- **Target:** |CAR[0,+1]|, the absolute sum of market-model abnormal returns on the event day and
  the next. Market = S&P 500 (^GSPC). Alpha and beta per stock from daily returns over trading
  days -280 to -31 before the event day (min 120 days, else the row is dropped).
- **Features:** only what is known at publication time: event class and 20 topic
  probabilities (our topic model), sentiment and |sentiment| (our sentiment model), source,
  publisher (top 10, else "other"), headline length, number of headlines for the stock in the
  previous 24 h, the stock's 20-day realised volatility and the VIX close, both from the day
  before the event day.
- **Time split:** train 2009–2016, validation 2017–2018, test A 2019, test B 2020-01-01 to
  2020-06-10. Test years are used once, after the model is chosen.
- **Candidates:** the fixed formula (baseline), ridge regression, gradient-boosted trees
  (scikit-learn `HistGradientBoostingRegressor`, used in place of LightGBM so the image needs
  no extra system library). Chosen by validation Spearman. Targets are log(1 + |CAR| in %).
- **Refit:** the chosen model is refitted on train + validation (2009–2018) before testing.
- **Scale 1–10:** deciles of the refitted model's predictions on 2009–2018.
- **Metrics (per test year):** Spearman between impact and |CAR| with a 95% bootstrap interval
  (1,000 resamples, seed 0); mean |CAR| for impact ≥ 8 vs ≤ 3.

## Result

- **Question:** Learned impact score v1
- **Outcome:** lost to volatility-only baseline (Spearman 0.17 vs 0.27–0.34)
- **Verdict:** fail
- **Stored results:** `impact_results.json`, `docs/experiments/IMPACT.md`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
