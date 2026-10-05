# Data coverage

Replay window 2020-04-09 to 2020-06-10 (43 trading days). Counts after cleaning and company linking.

|       |   tweets |   tweet_days |   news |   news_days |   price_days |
|:------|---------:|-------------:|-------:|------------:|-------------:|
| BA    |     2521 |           42 |     29 |          19 |           43 |
| MSFT  |    11882 |           42 |     53 |          28 |           43 |
| AMD   |     2082 |           42 |     77 |          33 |           43 |
| XOM   |     3476 |           42 |     87 |          40 |           43 |
| BAC   |     3830 |           42 |    111 |          45 |           43 |
| KO    |     1544 |           42 |    111 |          34 |           43 |
| NVDA  |     1863 |           42 |    119 |          38 |           43 |
| NFLX  |     4670 |           42 |    197 |          48 |           43 |
| JPM   |     6021 |           42 |    298 |          54 |           43 |
| META  |    15725 |           42 |    317 |          49 |           43 |
| AMZN  |    22685 |           42 |    331 |          47 |           43 |
| AAPL  |    24570 |           42 |    344 |          53 |           43 |
| TSLA  |     5602 |           42 |    344 |          52 |           43 |
| DIS   |    10708 |           42 |    377 |          53 |           43 |
| GOOGL |     8027 |           42 |    387 |          53 |           43 |

`news_days` and `tweet_days` count calendar days. The 2020 tweet dataset has gaps:
no tweets on 30 Apr - 2 May and 10 - 27 May 2020 (missing in the source file). News covers those days.
