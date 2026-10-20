# Decision: Research-Grade AI-Assisted Annotation Protocol

**Project:** Financial NLP Risk Engine

**Purpose:** Establish a reproducible, evidence-based annotation standard for future automated labeling of financial news and social-media text.

**Status:** Adopted as the proposed annotation standard (2026-10-14). 
Individual implementation details and quality thresholds remain subject to validation.

> This file is separate from `docs/DECISIONS.md` on purpose. 
`DECISIONS.md` is hash-checked by the E24 evaluation gate and must not be edited. 
This protocol does not change any existing label definition, preregistered experiment or held-out set (see section 4).

## 1. Core methodology

Adopt AI-assisted, rubric-based labeling with risk-based human review and independent quality validation.

The system must:

1. **Use a versioned annotation rubric.** Define positive, negative, neutral, and skip labels, including financial-context examples, edge cases, and explicit decision rules.
2. **Generate provisional labels using an LLM.** Require structured output containing the predicted label, concise text-grounded rationale, ambiguity flag, and model/prompt version.
3. **Avoid treating confidence as correctness.** Do not accept model self-reported confidence as a validated probability.
4. **Route uncertain cases for review.** Review ambiguous, conflicting, unsupported, and potentially misleading predictions. Also audit a random sample of apparently straightforward predictions.
5. **Validate independently.** Create a reference dataset reviewed using documented guidelines, preferably with a qualified financial annotator. Measure agreement and class-wise performance.
6. **Resolve disagreements transparently.** Preserve initial predictions, individual annotations, final adjudicated labels, and reasons for changes.
7. **Prevent evaluation leakage.** Keep training, validation, and held-out test sets separate. Do not use test labels to tune prompts, thresholds, or models.
8. **Maintain reproducibility.** Record dataset versions, annotation guidelines, prompts, model identifiers, timestamps, review decisions, and label provenance.

## 2. Required quality reporting

Where applicable, report:

- Number of examples and class distribution.
- Annotation method and annotator qualifications.
- Inter-annotator agreement when independent annotators are available.
- Agreement metric and justification for its selection.
- Per-class precision, recall, and F1 against independently validated reference labels.
- Disagreement rate, review rate, unresolved cases, and limitations.
- Results separately for news and social-media text where appropriate.

Never claim expert validation, inter-annotator agreement, or a measured accuracy improvement unless supported by actual results.

## 3. Supporting literature

Use these sources to justify the methodology. **Not yet verified:** check bibliographic details and the exact claims against the original publications before citing them in a paper.

1. Aguda et al. (2024). *Large Language Models as Financial Data Annotators: A Study on Effectiveness and Efficiency.* LREC-COLING 2024.
   https://aclanthology.org/2024.lrec-main.885/
2. Deng et al. (2023). *What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis.*
   https://research.google/pubs/what-do-llms-know-about-financial-markets-a-case-study-on-reddit-market-sentiment-analysis/
3. Federal Reserve (2026). *Validating Large Language Model Annotations.*
   https://www.federalreserve.gov/econres/feds/validating-large-language-model-annotations.htm

## 4. Implementation constraints

- Do not change the existing labeling definitions, preregistered experiments, or held-out evaluation datasets without explicit authorization.
- Do not invent citations, experimental results, thresholds, or validation claims.
- Do not introduce automatic acceptance thresholds without empirical calibration on independent reference data.
- Treat LLM labels as provisional until the chosen quality-control procedure supports their intended use.
- Preserve a complete audit trail for every automatically generated label.
- Distinguish verified literature findings from project-specific hypotheses and measured results.

## 5. Future implementation rule

When asked to implement an automated labeler, first inspect the existing project documentation and annotation decisions. Propose changes consistent with this protocol, explain their research justification, and identify which quality claims still require empirical verification.
