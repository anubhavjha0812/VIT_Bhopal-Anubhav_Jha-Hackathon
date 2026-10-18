"""E28 candidate G: frozen encoder vectors + TF-IDF -> logistic regression (Kaggle GPU kernel).

Reads train.csv / valid.csv (teacher-agreed rows, 9 learned classes) and eval_texts.csv (D1, D2, R3
texts only). Hyper-parameters are chosen on valid log-loss; temperature on valid. Never reads an
E24 final file. Writes probabilities for eval_texts.csv per encoder.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from transformers import AutoModel, AutoTokenizer

ENCODERS = [  # name, prefix
    ("intfloat/multilingual-e5-large", "query: "),
    ("intfloat/e5-base-v2", "query: "),
    ("BAAI/bge-small-en-v1.5", ""),
]
GRID_C, GRID_ALPHA, GRID_T = (0.5, 1.0, 3.0), (1.0, 2.0), (0.5, 0.75, 1.0, 1.5, 2.0)
OUT = Path("/kaggle/working/e28")
OUT.mkdir(parents=True, exist_ok=True)


def find(name):
    hits = sorted(Path("/kaggle/input").rglob(name))
    assert hits, f"{name} not found under /kaggle/input"
    return hits[0]


def embed(model_name, prefix, texts, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).half().eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), 128):
            enc = tok(
                [prefix + t for t in texts[s : s + 128]],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            h = model(**enc).last_hidden_state
            if "bge" in model_name:
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v.float(), dim=1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def temper(p, t):
    z = np.log(np.clip(p, 1e-12, 1.0)) / t
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, valid = pd.read_csv(find("train.csv")), pd.read_csv(find("valid.csv"))
    ev = pd.read_csv(find("eval_texts.csv"))
    classes = [c[2:] for c in train.columns if c.startswith("p_")]
    ytr, yva = train.label.map(classes.index).to_numpy(), valid.label.map(classes.index).to_numpy()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train.text)
    xt, xv, xe = (tfidf.transform(d.text) for d in (train, valid, ev))
    summary = {
        "classes": classes,
        "rows": {"train": len(train), "valid": len(valid), "eval": len(ev)},
    }
    for name, prefix in ENCODERS:
        t0 = time.time()
        et, evv, ee = (embed(name, prefix, d.text.tolist(), device) for d in (train, valid, ev))
        enc_s = time.time() - t0
        best = None
        for a in GRID_ALPHA:
            ft = hstack([xt, csr_matrix(a * et)]).tocsr()
            fv = hstack([xv, csr_matrix(a * evv)]).tocsr()
            for c in GRID_C:
                lr = LogisticRegression(C=c, max_iter=300).fit(ft, ytr)
                ll = log_loss(yva, lr.predict_proba(fv), labels=list(range(len(classes))))
                if best is None or ll < best[0]:
                    best = (ll, a, c, lr)
        ll, a, c, lr = best
        pv = lr.predict_proba(hstack([xv, csr_matrix(a * evv)]).tocsr())
        t_best = min(
            GRID_T, key=lambda t: log_loss(yva, temper(pv, t), labels=list(range(len(classes))))
        )
        pe = temper(lr.predict_proba(hstack([xe, csr_matrix(a * ee)]).tocsr()), t_best)
        tag = name.split("/")[-1]
        np.savez_compressed(
            OUT / f"{tag}_eval_probs.npz",
            probs=pe,
            ids=ev.id.to_numpy(),
            texts=ev.text.to_numpy(),
            sets=ev.set.to_numpy(),
        )
        summary[tag] = {
            "alpha": a,
            "C": c,
            "temperature": t_best,
            "valid_log_loss": round(ll, 4),
            "valid_macro_f1": round(float(f1_score(yva, pv.argmax(1), average="macro")), 4),
            "encode_seconds_total": round(enc_s, 1),
            "encode_seconds_per_1000_texts": round(
                enc_s / (len(train) + len(valid) + len(ev)) * 1000, 2
            ),
        }
        print(tag, summary[tag], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
