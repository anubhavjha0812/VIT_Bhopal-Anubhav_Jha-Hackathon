# Shipped model structures

Each diagram is a 1600 × 1000 PNG with a light background, using the colours of the existing system architecture. Dimensions and label IDs come from each shipped model's config; routing and impact features come from code and the stored impact packs. No training, prediction, evaluation or network request is performed to draw them.

| Diagram | Model structure | Shipped file |
|---|---|---|
| [Headline sentiment](sentiment.png) | A 12-layer bge-small BERT encoder and three-class pooled classifier produce headline sentiment, including the fallback for missing company names. | [sentiment/model.onnx](../../models/sentiment/model.onnx) |
| [Topic / event classifier](topic.png) | The same backbone has a nine-class event head; separate credit rules can override its label and add three credit classes. | [topic/model.onnx](../../models/topic/model.onnx) |
| [Event student v3](e25-student-v3.png) | Teacher-labelled real news trains the nine-class student, which the running engine combines with frozen C rules. | [e25-student-v3/model.onnx](../../models/e25-student-v3/model.onnx) |
| [Company-specific sentiment](entity-sentiment.png) | A company-name/headline pair enters a three-class BERT classifier to produce one sentiment score per company. | [E27 entity model.onnx](../../models/e27-entity-bge-small-en-v1.5-int8/model.onnx) |
| [Learned impact v3](impact-v3.png) | Separate news and tweet feature sets pass through numeric clipping, one-hot encoding/scaling and Ridge regression, then volatility rescaling and training-decile scoring. | [News joblib](../../models/impact-v3/impact_v3.joblib) · [Tweet joblib](../../models/impact-v3/impact_v3_tweets.joblib) |

## Redraw offline

From the repository root:

```bash
python3 docs/models/draw_models.py
```

The drawing interpreter needs Matplotlib. The script uses the existing `.venv/bin/python` and installed project dependencies to inspect the small local impact packs; if that interpreter is absent, it uses the drawing interpreter. It never installs packages, loads ONNX inference sessions or calls `fit()` or `predict()`.

[draw_models.py](draw_models.py) also regenerates [model_facts.json](model_facts.json), which records the four text configs, source training metrics, exact impact feature names, source hashes and image dimensions. Training-time development scores in that JSON are not new test results. The script checks text bounds and rejects unsupported architecture or impact-pack assumptions.

## Details checked against code

- All four text models use `BertForSequenceClassification`: 12 layers, hidden size 384, 12 attention heads, feed-forward width 1,536 and maximum position capacity 512. The 0.1 dropout belongs to training; the ONNX models run in inference mode. Runtime token truncation is separate from position capacity.
- Label IDs differ: headline sentiment is negative/positive/neutral; entity sentiment is negative/neutral/positive. Topic starts with Geopolitical, while student v3 starts with M&A. The diagrams read these orders directly from `id2label`.
- Entity tokenization uses `(company name, headline)` in that order. Company A/B scores in its diagram are symbols, not fabricated predictions.
- The student diagram's “running engine, E44b winner” designation follows the model metadata and `events = "student_rules"`. E44b used AI-only labels, and that same test selected v3; it is not independent human validation.
- `impact-v3/` has no `config.json` or `source_metrics.json`. Its actual joblib objects and `impact/train_v3.py`, `train_v2.py`, `tweets_v3.py` and `learned.py` confirm clipped Ridge with `alpha=1`, numeric training-percentile clipping, one-hot categorical columns and StandardScaler numeric columns.
- The news impact pack expects 20 cached topic-probability columns; the tweet pack expects nine. Their full names are in `model_facts.json`. Neither fitted Ridge uses `vol20` or `vix_prev` as a regressor; prior volatility rescales the predicted absolute move. Nine training boundaries produce scores 1–10.
- The current `learned.apply()` fills `impact` from the learned move score when available and preserves `impact_formula`; credit gating keeps formula severity. This follows the function body rather than the older module-header description.

## What this task does not confirm

Fresh-news rescoring compatibility with the news pack's 20 probability columns was not exercised. The current shipped topic model has nine classes, while the news pack preserves an older 20-column feature schema. The diagrams document the stored artifacts and current code; no cache rebuild or NLP inference was run to validate that integration.

Primary metadata: [models/models-metadata.md](../../models/models-metadata.md). Existing architecture images, model artifacts, data and source code are unchanged.
