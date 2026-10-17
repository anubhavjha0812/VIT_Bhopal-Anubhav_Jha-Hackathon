"""Event classifier choice for the running engine, including the frozen E24 candidate C.

`model_rules` is candidate C of docs/DECISIONS.md E24, chosen on D2 and confirmed once on the
final sets: the credit rules, the holdings, analyst, earnings and product rules of e24_rules.py,
and the event model's class where no rule applies. The rules and the per-row model scoring are
the ones that were frozen and evaluated.
"""

from collections.abc import Callable, Sequence

from riskengine.nlp.events import get_event_classifier

Classifier = Callable[[Sequence[str]], list[str]]


def get_engine_classifier(kind: str) -> Classifier:
    if kind != "model_rules":
        return get_event_classifier(kind)
    from riskengine.eval.e24_dev import Config, build_predictor, engine_probs

    predict = build_predictor(Config("C"), engine_probs())
    return lambda texts: predict(list(texts))[0]
