# Gold set: how to label

Open `gold_to_label.csv` and fill the `label` column with **one** of these 12 classes (DECISIONS E16), exactly as
written. Read only the headline; don't look anything up. When unsure, use `Other`.
Save the result as `gold.csv` (same columns). Don't change any other column.

| Label | Use it when the headline reports… | Not this |
|---|---|---|
| `Default` | a missed payment, bankruptcy filing, Chapter 11, insolvency, debt restructuring forced by distress | "could go bankrupt" opinions |
| `Credit downgrade` | a **credit** rating agency (Moody's, S&P, Fitch, DBRS) lowering a company's or its debt's rating, or moving its outlook to negative | a **stock** analyst downgrading shares ("Cowen downgrades Exxon to Market Perform" → `Analyst rating`) |
| `Covenant breach` | breaking or nearly breaking loan terms, asking lenders for a waiver or amendment | ordinary new loans |
| `M&A` | a merger, acquisition, takeover, buyout, stake purchase, asset sale, spin-off or IPO | a fund buying shares in a portfolio ("X LLC Buys Vanguard…" → `Other`) |
| `Geopolitical` | trade wars, tariffs, sanctions, elections, wars, China/US tensions, regulation driven by politics | |
| `Macroeconomic` | central banks and rates, inflation, jobs, GDP, recession, stimulus, Treasuries and bond markets, oil, metals, currencies | one company's earnings |
| `Product news` | a company's own product or business news: launches, unveilings, releases, recalls, deals with customers | its stock price moves |
| `Earnings` | quarterly results, earnings beats or misses, guidance, dividends | |
| `Analyst rating` | a stock analyst's rating or price target change (upgrade, downgrade, initiate, target cut) | a credit rating agency's action (→ `Credit downgrade`) |
| `Legal/Regulatory` | lawsuits, regulators (SEC, FTC, EU), fines, probes, court rulings, settlements | politics-driven trade measures (→ `Geopolitical`) |
| `Management change` | a CEO, CFO, chair or other executive appointed, leaving or replaced | |
| `Other` | everything else: price moves, market commentary, fund holdings, lists of movers | |

About 1 hour for 300 headlines. Label them all before looking at any model output.
