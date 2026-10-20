# Real Macro/Geo headline sources

Checked 17 October 2026. Research and counts only; no external training corpus built.

Start with official monetary-policy titles for Macro and foreign-office titles for Geo,
then add independent newsroom headlines. Synthetic data stays last. These recommendations
are for a separate later model; the frozen C candidate is unchanged.

## What remains in local FNSPID

The existing extra manifest records 37,085 clean rows outside the pool, of which 1,212 were
selected for extra. That leaves 35,873 in that historical frame. **Zero match the old Macro/Geo
keywords beyond extra:** the entire selected union was below each group's 2,500-row cap, and
the final pass dropped zero, so all old-keyword matches were already selected. This is derived
from the prior manifest and shortlist code, not a newly repeated leak check.

An additive keyword screen was fixed before inspecting annotation labels. Additions include
FOMC, quantitative easing, nonfarm, PCE, ISM, industrial production, OPEC, Brent, referendum,
NATO, ceasefire, trade agreement and foreign minister. After normalized pool/extra removal
and normalized deduplication, 35,311 raw-frame titles remain:

| Screen | Macro | Geo | Union |
|---|---:|---:|---:|
| Old keyword hits, before fresh exclusions | 2 | 6 | 8 |
| Old plus added keyword hits, before fresh exclusions | 13 | 15 | 28 |
| Hits introduced by additions | 11 | 9 | 20 |

**28 is an upper bound, not 28 clean or correctly labelled new examples.** The raw frame can
contain reserved-text matches; the 8 old-keyword hits do not contradict the zero count in the
historically cleaned frame. No full clean-remainder cache exists. Protected examples were not
reopened, so expanded candidates have not passed `remove_seen` / `assert_clean`. The old
manifest also predates R3. A later ingestion step must exclude all current reserved sets,
including the new credit held-out set, before using any candidate.

Counts, exact added expressions, input hashes and the small reproducible count script:
[fnspid_counts.json](../../../private_notes/annotation_uploads/task1/real_sources_20261017/fnspid_counts.json).
No titles were sampled or labelled in this sources task.

## Five preferred external sources

Ranking balances class fit, explicit research-use terms and a small dated-title fetch.
All five have English headlines/titles. Their existing topic tags are **not** E24 labels.

| Rank / source | Size and dates | Title / existing labels | Research-use terms and fit |
|---|---|---|---|
| 1. [Federal Reserve Board releases](https://www.federalreserve.gov/newsevents/pressreleases/press-release-archive.htm) | Ongoing year archives, back to 1996; total not enumerated here. Fetch 2010–2016 indexes only. | Separate release titles, dates and categories such as Monetary Policy; no E24 labels. | Board material is public domain unless marked otherwise; cite the Board. Best focused Macro source. Exclude enforcement and bank-application titles from any automatic Macro assumption. [Terms](https://www.federalreserve.gov/disclaimer.htm) |
| 2. [GOV.UK foreign-office news](https://www.gov.uk/search/news-and-communications?organisations%5B%5D=foreign-commonwealth-office) | Validated API: 8,320 legacy FCO news/press records, 6,588 with update timestamps in 2010–2016. These are index counts, not unique English or clean counts. | Titles, links and dates; document types and organisation tags, no E24 labels. [2016 example](https://www.gov.uk/government/news/uk-condemns-north-koreas-ballistic-missile-launch). | GOV.UK reuse under OGL, with attribution and stated exceptions. Best focused Geo source for diplomacy/conflict/sanctions. Use legacy FCO for historical data; current FCDO is a different period. [Terms](https://www.gov.uk/help/reuse-govuk-content) |
| 3. [ECB press releases](https://www.ecb.europa.eu/press/pr/date/2016/html/index.en.html) | Dated archive includes 1998 onwards; full count not enumerated. Select 2010–2016 English titles. | Release titles, dates and publication types; no E24 labels. | ECB permits free use with source attribution, accurate reproduction and disclosure of modifications. Named-author papers have a separate permission exception; use institutional releases. Second monetary-policy source. [Terms](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html) |
| 4. [News Category Dataset, original publisher](https://www.kaggle.com/datasets/rmisra/news-category-dataset) | 210,294 HuffPost records, 2012–2022; listed JSON size 87.3 MB, so not downloaded. Select dates through 2016 for the historical recipe. | Separate `headline`, date, original URL, description and 42 categories. POLITICS/BUSINESS are shortlist cues. | Publisher declares CC BY 4.0; retain attribution, citation, URLs and change notes. Good independent newsroom Geo coverage, less targeted Macro. Dataset declaration does not remove third-party rights. [Author paper](https://arxiv.org/abs/2209.11429) |
| 5. [AG News](https://www.tensorflow.org/datasets/catalog/ag_news_subset) | 120,000 train / 7,600 test; TFDS lists 11.24 MiB download and 35.79 MiB prepared. Source collection began July 2004; row publication dates are absent in the standard subset. | Separate `title` and description in TFDS/raw CSV; World, Business, Sports, Sci/Tech categories. Use train only, title only. | Documented permission is research/noncommercial, not Apache or CC BY. [HF card](https://huggingface.co/datasets/fancyzhx/ag_news) says licence unknown; preserve upstream terms. Useful World/Business examples, but strict per-row time eligibility cannot be demonstrated from this subset. |

Official search documentation: [GOV.UK Search API](https://docs.publishing.service.gov.uk/repos/search-api/using-the-search-api.html).
Its `public_timestamp` is an update time: verify original publication date from content
metadata before enforcing the training cutoff. The successful request URLs and small-response
sizes are saved in [source_metadata.json](../../../private_notes/annotation_uploads/task1/real_sources_20261017/source_metadata.json).
Initial document-type queries returned zero; documented format filters returned real counts.
Do not treat the initial zeros as an empty archive. A direct automated Fed archive request
returned HTTP 403; its public archive was verified through web browsing, without bypassing it.

## Other candidates and licence limits

- Local [FNSPID](https://github.com/zGeneral/fnspid) remains authentic finance news: the publisher
  reports 15.7 million records for 1999–2023. Its [current raw licence](https://raw.githubusercontent.com/zGeneral/fnspid/main/LICENSE)
  is CC BY-NC 4.0. Research must stay within the noncommercial terms and retain attribution;
  no new bulk download is justified by the small local keyword yield.
- [GDELT](https://www.gdeltproject.org/data.html) is useful for discovering articles and their
  URLs; its project calls its database free and open. That does not establish a blanket
  licence to train on every linked publisher's article/title. It is a discovery fallback,
  not one of the five preferred source recommendations.
- A Reuters/Bloomberg-style mirror on Kaggle/Hugging Face is not sufficient evidence of
  publisher permission. No unverified mirror is recommended for ingestion.

Do not auto-map all central-bank titles to Macro or all government/World titles to Geo:
regulatory actions, appointments, corporate stories and commentary need other classes.
Preserve titles verbatim, source URL, original date, language, licence and content hashes.
Before later training, run the established exact/near-duplicate filter against all reserved
sets and existing training data in an authorized ingestion step, and get E24-guide labels.
Broad category labels are only candidate-selection aids.

No synthetic generation, annotation, model/gate run, training, large download, protected-set
inspection, frozen-file modification, commit or push occurred in this sources task.
