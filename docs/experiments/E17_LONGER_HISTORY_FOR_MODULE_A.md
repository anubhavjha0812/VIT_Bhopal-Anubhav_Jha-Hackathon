# E17: Longer history for Module A

Rules fixed 2026-10-10, before any label or return is seen. Written down before the experiment was run; unchanged here.

## Rules

- **Why:** Module A has only been tested on the 43-day replay window, the one period with both news
  and tweets. That is too short to tell skill from luck. Our FNSPID slice (headlines filed under
  our 15 symbols) is thin before 2017 (289 in 2010, 1,245 in 2016), but the same file has 17,000
  to 37,000 headlines a year that name one of our companies under another symbol.
- **Name-linked headlines:** `fetch_fnspid --by-name` keeps a headline filed under another symbol
  when it names one of our companies (`Linker`): 262,927 such rows. A sample of 100 (seed 7) showed
  about half only mention the company. The relevance rules in `ingest/relevance.py` (sha256
  `adc911bc335fd87d…`) were written from that sample and frozen before the samples below were
  drawn. In order: not English (or mis-encoded), another company carrying the name (per-company
  veto list), fund notice, holdings list, market list, former employer, for JPM/BAC also bank
  conference and bank as analyst (E12 rule plus research-desk patterns), and the company not named
  within the first 6 words. Headlines filed under our own symbol keep only the language rule.
- **Measuring the rules (labels never train anything):**
  - S1: 200 (headline, company) pairs drawn at random from all name-linked pairs dated 2010-01-01
    to 2020-06-10, before the rules, seed 17, excluding the 100 design headlines.
  - S2: 100 pairs drawn the same way from headlines filed under our own symbol, excluding the
    gold-set headlines (E3).
  - The label file shows only id, company and headline: no rule outcome, no sentiment, no price.
  - Label: is the headline about this company, news that matters for its stock (Y/N)? Three
    open-licence models label each pair with one fixed prompt, scored by the next-token
    probability of Y vs N: Qwen3-8B (Apache 2.0), OLMo-2-1124-7B-Instruct (Apache 2.0) and
    Phi-4-mini-instruct (MIT). The majority is the label. No closed model is used, so anyone can
    rebuild the labels with `notebooks/relevance_labels/`.
  - Amended 2026-10-10, after the first labelling run and before any precision was computed:
    OLMo-2 never answered Y (its Y probability was below 0.07 on all 300 pairs), so reading the
    next-token probability failed for it. All three models now answer by greedy generation of up
    to 4 tokens, read as the first Y or N; the next-token probability is the fallback when neither
    appears. A labeller that answers Y on fewer than 5% or more than 95% of pairs is reported as
    degenerate and left out; the label then needs both remaining models to agree, and pairs where
    they disagree are marked unsure, reported, and added to the human audit.
  - Human audit: 30 random labelled pairs (seed 0) checked by hand; agreement is reported.
  - Metrics: precision before and after the rules, recall of the rules (correct pairs kept / all
    correct pairs), each with a 95% Wilson interval, overall, by period (2010–2016, 2017–2020) and
    for JPM+BAC vs the rest; S2 precision.
  - Target: S1 precision after the rules ≥ 0.85, reported with its interval whatever it is. A
    period is labelled "weak data" only if its precision after the rules is below 0.80.
  - The rules do not change after the labels are seen. A change would need a new sample.
- **History set:** headlines under our symbol (language rule only) plus name-linked pairs that pass
  the rules, 2010-01-01 to 2020-06-10, rows where the company is the subject (E12). Sentiment from
  the engine's fine-tuned bge-small sentiment model. A headline is usable from the first trading
  day after its New York date (as Module A).
  - Amended 2026-10-10, before any headline was scored: the ~172,000 headlines are scored on a
    Kaggle GPU with the full-precision model (the engine runs its int8 copy, E9, which is CPU-only
    and would take about an hour). The same job scores a random 5,000 with the int8 copy and
    reports the correlation and sign agreement between the two (`notebooks/history_sentiment/`).
- **Signal test (IC):** on each trading day with at least 5 companies that have a usable headline in
  the last 3 trading days, the Spearman correlation across those companies between mean sentiment
  and the next day's return. Mean daily IC per period, with a 95% moving-block bootstrap interval
  (blocks of 20 days, 1,000 resamples, seed 0).
- **Portfolio test:** Module A as now (tilt 0.5 per z-score, cap 15%, 3-day lookback, 5 bp
  costs). Each day the universe is the companies with a close that day and the next; equal weight
  is 1/n over that universe. Four candidates: rebalance every 1 or 5 trading days × weights moved
  fully (λ = 1) or 20% of the way (λ = 0.2) to the target on rebalance days.
- **Periods:** choose on 2010–2016 (dev), by Sharpe minus equal-weight Sharpe. Score the chosen
  candidate once on 2017-01-01 to 2020-06-10 (test). The test period contains the replay window
  (9 Apr – 10 Jun 2020) whose result was already seen, so the test is also reported without it.
  Fresh test, scored once: equinxx tweets 30 Sep 2021 – 29 Sep 2022 (linked by their text),
  prices from yfinance.
- **Pass rules:** signal: test mean IC > 0 with the interval above 0. Portfolio: test Sharpe of the
  chosen candidate above equal weight; the Sharpe difference is reported with a moving-block
  interval, plus yearly return, volatility, drawdown and turnover.
- **Effect of the rules:** IC and the chosen candidate are also run on symbol-only headlines and on
  unfiltered name-linked headlines. Reported, never used to choose; the filtered set is primary.
- **Engine:** if both tests pass, the chosen candidate becomes Module A's default and the replay is
  re-reported. Otherwise Module A stays as it is and the README says the tilt has no proven edge.
- **Leaks guarded:** no labels train a model; labellers never see returns; design and measure
  samples are disjoint; gold-set headlines are excluded; dev and test years are separate.

## Result

- **Question:** Module A over 10 years (2010 – Jun 2020) + tweets 2021–22
- **Outcome:** test IC 0.011 [−0.012, 0.034]; Sharpe diff −0.071 [−0.265, 0.103]; relevance-rule precision 0.849 [0.76, 0.91]
- **Verdict:** fail: no proven edge
- **Stored results:** `module_a_history.json`, `relevance_results.json`, `data/relevance/`

Ledger: [EXPERIMENTS.md](EXPERIMENTS.md). Rules also kept in [DECISIONS.md](../DECISIONS.md).
