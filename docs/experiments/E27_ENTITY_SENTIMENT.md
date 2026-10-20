# E27: company-specific (entity-aware) sentiment (fixed 2026-10-17, before any training or scoring)

`docs/DECISIONS.md` is not edited. Written before any model is trained on this split.

## Why
- The engine scores a headline once and copies that sentiment to every linked company. A headline like "A gains, B falls" gets the same score for both. This is also what Module A trades on (placebo test: no proof the tilt adds value).
- SEntFiN (`data/eval/sentfin_v1.1.csv`, 10,753 headlines, human-labelled sentiment per named entity) has the labels to train and test this.

## Data and split (built by `data/cache/e27/*.csv`, seed 27)
- Split by headline: train 70%, dev 15%, test 15%. Rows are (headline, entity, label). Train 10,071 rows, dev 2,186, test 2,152 rows.
- Conflicting headlines: entities with different sentiments. Test has 463 entity rows in conflicting headlines.
- E20 used the single-entity SEntFiN headlines as a "fresh" confirmation set. Training on SEntFiN spends that role. E20's result stays as reported; this split is a new experiment with its own test.

## Model
- bge-small (E21 recipe), input is the entity name and the headline as a text pair, 3 classes (negative, neutral, positive), int8 ONNX. Seeds 42, 43, 44; best seed by dev macro-F1.
- No tweets in this experiment; tweet-style text is not tested here (state it).

## Comparison and rule (fixed now)
- **A** = the current engine's headline sentiment copied to every entity (class by the same mapping as E18/E20 scoring).
- **B** = the entity-aware model.
- Primary metric: macro-F1 over the conflicting-headline entity rows of the test split, paired bootstrap of B - A.
- Secondary: macro-F1 over all test rows.
- **Switch rule:** the engine switches to B for per-company sentiment only if the primary gain is at least 0.05 and the all-test macro-F1 of B is not more than 0.02 below A's. Otherwise A stays and both results are reported.
- One test run, after the choice of seed on dev. No threshold or retraining after seeing the test.

## Disclosure
- SEntFiN headlines are Indian business news; US stock news and tweets may differ. Labels are human-made (dataset authors). Report the test as news-headline sentiment only.
