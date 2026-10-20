# Transaction-seeded synthetic sleeve

These rules are fixed before building or stressing the sleeve. The SEC book remains
the main Module B book. This is a separate synthetic proxy, not a bank's actual
portfolio or an estimate of merchants' debt, credit ratings or default risk.

Source: [Financial Transactions Dataset: Analytics](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets).
License: see the dataset page. The local files are used without network access.
Only `transactions_data.csv` and the MCC description lookup are needed. No client,
card, address or fraud-label fields enter the sample or the book.

## Fixed mapping

1. Read the raw CSV in chunks, first for date coverage and then for aggregation.
   The reference window is the last 12 full calendar months present. Include the
   last observed month only if its last calendar day is present; otherwise end in
   the preceding month. Require all 12 months to have observations and the data
   to start no later than the window. This checks observed coverage, not feed
   completeness. Dates without a valid timestamp are excluded.
2. Within that window, keep positive purchases only. Parse dollar strings such as
   `$-77.00` and `$1,234.56` into integer cents. Reject malformed amounts, refunds,
   zero amounts, missing merchant IDs and missing MCCs. If `errors` exists, keep
   only empty/blank values. Sum cents so chunk size cannot change the result.
3. A merchant ID is an obligor. Require positive flow in at least 10 of 12 months;
   missing months count as zero. Rank by total purchase flow and select at most
   60. Names are exactly `M<merchant_id>`, with no real-company mapping. Sector is
   the MCC description; choose the merchant's MCC with the largest purchase flow
   over the window. An absent lookup entry is `Unknown MCC <code>`.
4. All ties use a stable SHA-256 key of `0|<identifier>` (fixed seed 0), with the
   identifier as a final tie-break. This applies to flow, dominant MCC, volatility
   and hedge rankings; it does not depend on CSV order or chunk boundaries.
5. Calculate population standard deviation / mean of each selected merchant's
   12 monthly flows, including zero months. Sort by this coefficient of variation,
   lowest first; equal-count rank quantiles `floor(5 * rank / count)` receive
   `A`, `BBB`, `BB`, `B`, `CCC`. These are synthetic buckets on the existing
   `params.py` scale, not agency ratings. With 60 merchants there are 12 per bucket.
6. Each merchant receives a five-year revolver: commitment = twice average monthly
   flow, EAD = 75% of commitment. The top 30 by flow also receive a three-year
   fully drawn term loan = six times average monthly flow. The top 10 also receive
   a five-year bond = three times average monthly flow. Use existing LGD 45% and
   book conventions. Currency is USD; amounts and EAD are rounded to cents. Do not
   rescale the sleeve. Report cash EAD separately from derivative notionals.
7. Follow the SEC book's hedge rule on cash position notionals per merchant:
   buy five-year CDS protection on the three lowest-rated merchants at 25%; sell
   it on the three highest-rated at 10%; add five-year pay-fixed IRS with the five
   largest at 20%. Rating ties prefer larger cash notionals, then the seed-0 key.
   Derivative EAD in the book column is zero, as in the SEC book; IRS revaluation
   still calculates SA-CCR exposure. Hedge shares are imported from `book.py`.

## Files and rebuild

`data/reference/transactions_monthly_sample.csv` contains only the selected
merchant/month aggregates (normally 720 rows), MCC/sector and synthetic flags.
This is the only transaction-source data intended for tracking; raw files stay
ignored. The book has the SEC book's columns plus `book=transactions` and
`synthetic=true`. Its source fields identify the proxy and MCC sector.

```bash
uv run python -m riskengine.module_b.book_transactions --from-raw
uv run python -m riskengine.module_b.book_transactions
uv run python -m riskengine.module_b.book_transactions --stress
```

The first command streams raw data and writes the sample and book. The default
command reads the sample, so a clean clone needs no raw download. Both builds
must produce byte-identical book files. No customer/card data is exported.

## Fixed stress method and limits

Use existing `revalue.py` and the three named historical episodes from `shock.py`:
GFC 2008, euro crisis 2011 and COVID 2020. Apply the market-only scenario at
multiple 1, no named-obligor downgrade, and fixed Treasury rates (the existing
report default). Keep existing PD, LGD, spread, tax, RWA and CET1 conventions.
The extra `term_loan` pricing branch values funded principal minus IFRS 9 ECL
and uses the same credit RWA/staging as revolvers. Existing instrument branches
are unchanged.

These are counterfactual stresses of the transaction-window sleeve using the
existing 31 December 2019 market base and stock-basket market factor. They are
not contemporaneous merchant observations or a validated merchant risk model.
Only `data/outputs/module_b_transactions_stress.json` receives the results:
cash EAD, position counts, value, loss, RWA, capital and CET1 before/after.

The SEC book, stored reports, backtests, alert checks and dashboard defaults stay
separate. The dashboard is left on the SEC book because its company/ticker views
and news alerts do not map to these synthetic merchant IDs.
