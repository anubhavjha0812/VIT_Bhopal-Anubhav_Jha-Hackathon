import json

import pandas as pd

from riskengine.eval.e24_dev import build_predictor, engine_probs
from riskengine.eval.e25_retest_score import CANDIDATES

D = "private_notes/annotation_uploads/task1/credit_8k_full/"
t = pd.read_csv(D + "to_label.csv", dtype=str)
a = pd.read_csv(D + "labels_chatgpt.csv")
b = pd.read_csv(D + "labels_claude.csv")
assert t.id.tolist() == a.item_id.tolist() == b.item_id.tolist()
agree = a.label == b.label
final = b.label.copy()  # DevxAnu: Claude's label is final on disagreements
src = pd.Series("ai_agreed", index=b.index)
src[~agree] = "ai_adjudicated"
src[agree & ((a.ambiguous == "Y") | (b.ambiguous == "Y"))] = "ai_agreed_ai_audited"
pred = build_predictor(CANDIDATES["C"], engine_probs())(t.text.tolist())[0]
CR = {"Default", "Covenant breach", "Credit downgrade"}
y = final.isin(CR)
p = pd.Series(pred).isin(CR)
tp = int((y & p).sum())
fp = int((~y & p).sum())
fn = int((y & ~p).sum())
out = {
    "rows": len(t),
    "label_source_counts": src.value_counts().to_dict(),
    "credit_true": int(y.sum()),
    "pred_credit": int(p.sum()),
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "precision": round(tp / max(tp + fp, 1), 3),
    "recall": round(tp / max(tp + fn, 1), 3),
    "sensitivity_chatgpt_labels": {
        "recall": round(float((a.label.isin(CR) & p).sum() / a.label.isin(CR).sum()), 3),
        "precision": round(float((a.label.isin(CR) & p).sum() / p.sum()), 3),
    },
    "note": (
        "Sample is anchored on 8-K bankruptcy items, so Default is over-represented: "
        "recall is meaningful, precision is not comparable to E23. AI labels, no human "
        "check; Claude's label used where labellers differ (ai_adjudicated; the "
        "adjudicator is also a labeller)."
    ),
}
pd.DataFrame(
    {
        "item_id": t.id,
        "text": t.text,
        "label_chatgpt": a.label,
        "label_claude": b.label,
        "final": final,
        "label_source": src,
        "pred_C": pred,
    }
).to_csv("data/outputs/e25_credit_8k_full_rows.csv", index=False)
json.dump(out, open("data/outputs/e25_credit_8k_full_gate.json", "w"), indent=2)
print(json.dumps(out, indent=1))
