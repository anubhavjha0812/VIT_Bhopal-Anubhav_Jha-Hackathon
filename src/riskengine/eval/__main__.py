"""Score the configured models on the public test sets and write docs/RESULTS.md.

uv run python -m riskengine.eval
uv run python -m riskengine.eval --only topic-bge-small-en-v1.5-int8   # score one new model,
    keep the stored rows of all others (each test is touched once per model)
uv run python -m riskengine.eval --e16 <current> <new> <new-int8>   # E16 9-class rows, once
uv run python -m riskengine.eval --e18 <model folder or "finbert"> ...   # E18 public sets, once
"""

import argparse
import json
import resource
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from riskengine.eval.datasets import (
    E18_FILES,
    E18_SOURCES,
    EVAL_DIR,
    EVENT9_TEST_CLASSES,
    EVENT_TEST_CLASSES,
    e18_tests,
    tfns_test,
    topic_test,
)
from riskengine.eval.leaks import seen
from riskengine.nlp.classifier import TextClassifier
from riskengine.nlp.events import TOPIC_TO_EVENT_E2, classify_keywords, topic_to_event
from riskengine.nlp.multihead import MultiHeadModel
from riskengine.nlp.sentiment import FINBERT, TransformerSentiment
from riskengine.paths import DATA, MODELS, ROOT

RESULTS_MD = ROOT / "docs" / "others" / "RESULTS.md"
RESULTS_JSON = DATA / "outputs" / "eval_results.json"
EVENTS9_KEY = "Events (TFN-topic, 9 classes)"
E18_KEY = "Sentiment, more public sets (E18)"
E18_MANIFEST = DATA / "outputs" / "e18_manifest.json"
E18_PREDICTIONS = DATA / "outputs" / "e18_predictions.csv.gz"  # per row, for paired tests
E18_CORPUS = {  # model folder prefix -> what it was trained on (E18 leak check)
    "sentiment-bge-small": "E5 models (TFNS train)",
    "sentiment-finbert": "E5 FinBERT (TFNS train + PhraseBank)",
    "finbert": "FinBERT baseline (PhraseBank)",
    "sentiment-distil": "E7/E10 models (TFNS + PhraseBank 75Agree + FiQA train/val)",
    "sentiment-teacher": "E7/E10 models (TFNS + PhraseBank 75Agree + FiQA train/val)",
    "multihead": "E7/E10 models (TFNS + PhraseBank 75Agree + FiQA train/val)",
}


def e2_events(topics: list[str]) -> list[str]:
    """TFN topics -> the 4 E2 classes."""
    return [topic_to_event(t, TOPIC_TO_EVENT_E2) for t in topics]


def e2_keywords(texts: Sequence[str]) -> list[str]:
    """The keyword baseline seen through E2: classes E2 does not score count as Other."""
    return [c if c in EVENT_TEST_CLASSES else "Other" for c in map(classify_keywords, texts)]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB


def timed(predict: Callable[[Sequence[str]], list[str]], texts: list[str]) -> tuple[list, float]:
    predict(texts[:64])  # warm-up batch (DECISIONS E4)
    t0 = time.perf_counter()
    out = predict(texts)
    return out, len(texts) / (time.perf_counter() - t0)


def bootstrap_ci(y, pred, classes, n_boot: int = 1000, seed: int = 0, per_class: bool = False):
    """95% bootstrap intervals for accuracy and macro-F1 (DECISIONS E7), and per class (E16)."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    acc, f1, each = [], [], []
    for i in rng.integers(0, len(y), (n_boot, len(y))):
        acc.append(accuracy_score(y[i], pred[i]))
        f1.append(f1_score(y[i], pred[i], labels=classes, average="macro", zero_division=0))
        if per_class:
            each.append(f1_score(y[i], pred[i], labels=classes, average=None, zero_division=0))
    out = {
        "accuracy_ci95": np.percentile(acc, [2.5, 97.5]).tolist(),
        "macro_f1_ci95": np.percentile(f1, [2.5, 97.5]).tolist(),
    }
    if per_class:
        lo, hi = np.percentile(np.array(each), [2.5, 97.5], axis=0)
        out["per_class_f1_ci95"] = {c: [a, b] for c, a, b in zip(classes, lo, hi, strict=True)}
    return out


def evaluate(
    name, test, predict, classes, size_mb, label: str = "label", keep_pred: bool = False
) -> dict:
    """One model on one test set. The 9-class test (label9) also gets per-class P/R and CIs."""
    pred, speed = timed(predict, test["text"].tolist())
    y = test[label]
    report = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
    detail = label != "label"
    row = {
        "model": name,
        "n": len(test),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0),
        **bootstrap_ci(y, pred, classes, per_class=detail),
        "per_class_f1": {c: report[c]["f1-score"] for c in classes},
        "texts_per_s": speed,
        "model_size_mb": size_mb,
        "peak_rss_mb": peak_rss_mb(),
        "_pred": list(pred),
    }
    if not keep_pred:
        row.pop("_pred")
    if detail:
        row["per_class_precision"] = {c: report[c]["precision"] for c in classes}
        row["per_class_recall"] = {c: report[c]["recall"] for c in classes}
        row["per_class_support"] = {c: int(report[c]["support"]) for c in classes}
    return row


def to_markdown(results: dict) -> str:
    lines = ["# Results", "", "Generated by `uv run python -m riskengine.eval`. "
             "Test rules: [DECISIONS.md](DECISIONS.md). CPU only.", ""]  # fmt: skip
    for task, rows in results.items():
        lines += [f"## {task}", ""]
        if task == E18_KEY:
            lines += e18_markdown(rows)
            continue
        df = pd.DataFrame(rows)
        for col in ("accuracy_ci95", "macro_f1_ci95"):
            df[col] = df[col].map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
        table = df[["model", "n", "accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95",
                    "texts_per_s", "model_size_mb"]]  # fmt: skip
        lines += [table.round(3).to_markdown(index=False), ""]
        per_class = pd.DataFrame({r["model"]: r["per_class_f1"] for r in rows}).round(3)
        lines += ["Per-class F1:", "", per_class.to_markdown(), ""]
        for r in rows:
            if "per_class_f1_ci95" not in r:
                continue
            detail = pd.DataFrame(
                {
                    "precision": r["per_class_precision"],
                    "recall": r["per_class_recall"],
                    "f1": r["per_class_f1"],
                    "f1_ci95": {
                        c: f"[{a:.3f}, {b:.3f}]" for c, (a, b) in r["per_class_f1_ci95"].items()
                    },
                    "support": r["per_class_support"],
                }
            )
            lines += [f"{r['model']}, per class:", "", detail.round(3).to_markdown(), ""]
    return "\n".join(lines)


def e18_markdown(rows: list[dict]) -> list[str]:
    """One table per set: all rows, and the never-seen rows where they differ (primary)."""
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731
    out = []
    for test_name in dict.fromkeys(r["test_set"] for r in rows):
        table = []
        for r in (r for r in rows if r["test_set"] == test_name):
            c = r.get("clean", r if r["status"] == "clean" else None)
            table.append({
                "model": r["model"],
                "status": r["status"],
                "n": r["n"],
                "accuracy": round(r["accuracy"], 3),
                "macro_f1": round(r["macro_f1"], 3),
                "macro_f1_ci95": ci(r["macro_f1_ci95"]),
                "clean n": c["n"] if c else 0,
                "clean macro_f1": round(c["macro_f1"], 3) if c else None,
                "clean macro_f1_ci95": ci(c["macro_f1_ci95"]) if c else None,
                "texts_per_s": round(r["texts_per_s"], 1) if r["texts_per_s"] else None,
            })  # fmt: skip
        out += [f"### {test_name}", "", pd.DataFrame(table).to_markdown(index=False), ""]
    return out


def trained(task: str) -> list[Path]:
    """Fine-tuned model folders, torch or ONNX, e.g. models/*/topic-*, models/*/*-int8."""
    found = {
        p.parent for ext in ("safetensors", "onnx") for p in MODELS.glob(f"**/{task}-*/model.{ext}")
    }
    return sorted(found)


def score_only(names: list[str]) -> dict[str, list[dict]]:
    """Score just the named model folders and merge them into the stored results."""
    results = json.loads(RESULTS_JSON.read_text())
    sentiment_key, events_key = list(results)[:2]
    for name in names:
        path = next((p for p in trained("topic") + trained("sentiment") if p.name == name), None)
        if path is None:
            raise FileNotFoundError(f"no model folder named {name!r} under {MODELS}")
        if name.startswith("topic-"):
            clf = TextClassifier(str(path))
            key, row = (
                events_key,
                evaluate(
                    name,
                    topic_test(),
                    lambda ts, clf=clf: e2_events(clf.predict(ts)),
                    EVENT_TEST_CLASSES,
                    clf.size_mb,
                ),
            )
        else:
            clf = TransformerSentiment(path, name=name)
            labels = ["negative", "neutral", "positive"]
            key, row = sentiment_key, evaluate(name, tfns_test(), clf.classify, labels, clf.size_mb)
        results[key] = [r for r in results[key] if r["model"] != name] + [row]
    return results


def score_all() -> dict[str, list[dict]]:
    sentiment_key, events_key = "Sentiment (TFNS, 3 classes)", "Events (TFN-topic)"
    results: dict[str, list[dict]] = {sentiment_key: [], events_key: []}

    events = topic_test()
    results[events_key].append(
        evaluate(
            "keywords (baseline)",
            events,
            e2_keywords,
            EVENT_TEST_CLASSES,
            0.0,
        )
    )
    for path in trained("topic"):
        clf = TextClassifier(str(path))
        results[events_key].append(
            evaluate(
                path.name,
                events,
                lambda ts, clf=clf: e2_events(clf.predict(ts)),
                EVENT_TEST_CLASSES,
                clf.size_mb,
            )
        )

    sentiment = tfns_test()
    labels = ["negative", "neutral", "positive"]
    for name, model in [
        ("finbert (baseline)", FINBERT),
        *[(p.name, p) for p in trained("sentiment")],
    ]:
        clf = TransformerSentiment(model, name=name)
        results[sentiment_key].append(evaluate(name, sentiment, clf.classify, labels, clf.size_mb))

    # E10: one encoder, two heads. Each test runs the full pass, so speed covers both heads.
    for path in sorted(p.parent for p in MODELS.glob("**/multihead-*/heads.pt")):
        mh = MultiHeadModel(path)
        results[sentiment_key].append(
            evaluate(
                f"{path.name} (sentiment head)",
                sentiment,
                lambda ts, mh=mh: mh.predict(ts)[0],
                labels,
                mh.size_mb,
            )  # fmt: skip
        )
        results[events_key].append(
            evaluate(
                f"{path.name} (topic head)",
                events,
                lambda ts, mh=mh: e2_events(mh.predict(ts)[1]),
                EVENT_TEST_CLASSES,
                mh.size_mb,
            )  # fmt: skip
        )
    return results


def score_e16(folders: list[Path]) -> dict[str, list[dict]]:
    """E16: the keyword baseline and each model folder on the 9-class test, merged into the
    stored results. Each is scored once; the label says which E16 role it plays."""
    results = json.loads(RESULTS_JSON.read_text())
    test = topic_test()
    rows = [
        evaluate(
            "keywords (baseline)",
            test,
            lambda ts: list(map(classify_keywords, ts)),
            EVENT9_TEST_CLASSES,
            0.0,
            label="label9",
        )
    ]
    for folder in folders:
        clf = TextClassifier(str(folder))
        rows.append(
            evaluate(
                folder.name,
                test,
                lambda ts, clf=clf: [topic_to_event(t) for t in clf.predict(ts)],
                EVENT9_TEST_CLASSES,
                clf.size_mb,
                label="label9",
            )
        )
    results[EVENTS9_KEY] = rows
    return results


def e18_model(name: str):
    """(display name, predict, size MB, leak-check corpus) for a folder name or "finbert"."""
    if name == "finbert":
        clf = TransformerSentiment(FINBERT, name="finbert (baseline)")
        return "finbert (baseline)", clf.classify, clf.size_mb, E18_CORPUS["finbert"]
    corpus = next(c for k, c in E18_CORPUS.items() if name.startswith(k))
    if name.startswith("multihead"):
        path = next(p.parent for p in MODELS.glob(f"**/{name}/heads.pt"))
        mh = MultiHeadModel(path)
        return f"{name} (sentiment head)", lambda ts: mh.predict(ts)[0], mh.size_mb, corpus
    path = next((p for p in trained("sentiment") if p.name == name), None)
    if path is None:
        raise FileNotFoundError(f"no model folder named {name!r} under {MODELS}")
    clf = TransformerSentiment(path, name=name)
    return name, clf.classify, clf.size_mb, corpus


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def e18_manifest(names: list[str]) -> dict:
    """Freeze what E18 scored: test files (revision, licence, rows, sha256), model files
    (sha256), library versions and the code commit."""
    import subprocess
    from importlib.metadata import version

    tests = e18_tests()
    data = {}
    for name, (file, repo) in E18_FILES.items():
        rev, licence = E18_SOURCES[repo]
        data[name] = {
            "file": f"data/eval/{file}",
            "source": f"huggingface.co/datasets/{repo}",
            "revision": rev,
            "licence": licence,
            "rows": len(tests[name]),
            "labels": tests[name]["label"].value_counts().to_dict(),
            "sha256": sha256(EVAL_DIR / file),
        }
    models = {}
    for name in names:
        if name == "finbert":
            models[name] = {"hub_id": FINBERT}
            continue
        folder = next(p.parent for p in MODELS.glob(f"**/{name}/config.json"))
        models[name] = {
            "folder": str(folder.relative_to(ROOT)),
            "files": {
                p.name: sha256(p)
                for p in sorted(folder.iterdir())
                if p.suffix in {".onnx", ".safetensors", ".pt", ".json"}
            },
        }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    libs = ["torch", "transformers", "onnxruntime", "scikit-learn", "numpy", "pandas"]
    return {
        "rule": "docs/DECISIONS.md E18",
        "test_sets": data,
        "models": models,
        "libraries": {lib: version(lib) for lib in libs},
        "code_commit": commit.stdout.strip(),
        "bootstrap": {"resamples": 1000, "seed": 0},
    }


def score_e18(names: list[str]) -> dict[str, list[dict]]:
    """E18: each named model once on each set, on all rows and on the rows it never saw in
    training (the primary figure). Rows are merged into the stored results."""
    results = json.loads(RESULTS_JSON.read_text())
    tests = e18_tests()
    flags = seen(tests)
    labels = ["negative", "neutral", "positive"]
    rows = results.get(E18_KEY, [])
    preds = []
    for name in names:
        model, predict, size, corpus = e18_model(name)
        for test_name, test in tests.items():
            row = evaluate(model, test, predict, labels, size, keep_pred=True)
            clean = ~flags[corpus][test_name]
            preds.append(
                pd.DataFrame({"model": model, "test_set": test_name, "row": range(len(test)),
                              "label": test["label"], "pred": row["_pred"], "seen": ~clean})
            )  # fmt: skip
            row |= {"test_set": test_name, "trained_on": corpus, "seen_rows": int((~clean).sum())}
            if clean.all():
                row["status"] = "clean"
            elif not clean.any():
                row["status"] = "in-sample (never counted as a result)"
            else:
                row["status"] = "partly seen: clean rows are primary"
                y = test["label"].to_numpy()[clean]
                pred = np.asarray(row.pop("_pred"))[clean]
                row["clean"] = {
                    "n": int(clean.sum()),
                    "accuracy": accuracy_score(y, pred),
                    "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
                    **bootstrap_ci(y, pred, labels),
                }
            row.pop("_pred", None)
            rows = [r for r in rows if (r["model"], r["test_set"]) != (model, test_name)] + [row]
    results[E18_KEY] = rows
    new = pd.concat(preds, ignore_index=True)
    if E18_PREDICTIONS.exists():
        old = pd.read_csv(E18_PREDICTIONS)
        old = old[~old["model"].isin(new["model"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(E18_PREDICTIONS, index=False)
    E18_MANIFEST.write_text(json.dumps(e18_manifest(names), indent=2))
    return results


E18_ROW_CORPUS = {"finbert (baseline)": E18_CORPUS["finbert"]}


def corpus_of(model: str) -> str:
    if model in E18_ROW_CORPUS:
        return E18_ROW_CORPUS[model]
    return next(c for k, c in E18_CORPUS.items() if model.startswith(k))


def e18_from_predictions(path: Path, meta_path: Path | None = None) -> dict[str, list[dict]]:
    """E18 rows from per-row predictions made elsewhere (the Kaggle job): the same metrics and
    leak split as score_e18; speed is not measured there, so it is left empty."""
    results = json.loads(RESULTS_JSON.read_text())
    tests = e18_tests()
    flags = seen(tests)
    meta = json.loads(meta_path.read_text())["models"] if meta_path else {}
    labels = ["negative", "neutral", "positive"]
    preds = pd.read_csv(path)
    rows = results.get(E18_KEY, [])
    out_preds = []
    for (model, test_name), g in preds.groupby(["model", "test_set"], sort=False):
        test = tests[test_name]
        g = g.sort_values("row")
        if len(g) != len(test) or (g["label"].to_numpy() != test["label"].to_numpy()).any():
            raise ValueError(f"{model} / {test_name}: rows or labels differ from the frozen test")
        y, pred = test["label"].to_numpy(), g["pred"].to_numpy()
        corpus = corpus_of(model)
        clean = ~flags[corpus][test_name]
        report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
        row = {
            "model": model,
            "n": len(y),
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
            **bootstrap_ci(y, pred, labels),
            "per_class_f1": {c: report[c]["f1-score"] for c in labels},
            "texts_per_s": None,
            "ran_on": meta.get(model, {}).get("ran_on"),
            "test_set": test_name,
            "trained_on": corpus,
            "seen_rows": int((~clean).sum()),
        }
        if clean.all():
            row["status"] = "clean"
        elif not clean.any():
            row["status"] = "in-sample (never counted as a result)"
        else:
            row["status"] = "partly seen: clean rows are primary"
            row["clean"] = {
                "n": int(clean.sum()),
                "accuracy": accuracy_score(y[clean], pred[clean]),
                "macro_f1": f1_score(
                    y[clean], pred[clean], labels=labels, average="macro", zero_division=0
                ),
                **bootstrap_ci(y[clean], pred[clean], labels),
            }
        rows = [r for r in rows if (r["model"], r["test_set"]) != (model, test_name)] + [row]
        out_preds.append(g.assign(seen=~clean))
    results[E18_KEY] = rows
    pd.concat(out_preds, ignore_index=True).to_csv(E18_PREDICTIONS, index=False)
    return results


def paired_diff(a: str, b: str, test_set: str, clean_only: bool = True) -> dict:
    """E18 amendment: paired bootstrap of macro-F1(a) - macro-F1(b) on the same rows."""
    preds = pd.read_csv(E18_PREDICTIONS)
    pa = preds[(preds["model"] == a) & (preds["test_set"] == test_set)].sort_values("row")
    pb = preds[(preds["model"] == b) & (preds["test_set"] == test_set)].sort_values("row")
    keep = np.ones(len(pa), bool)
    if clean_only:
        keep = ~(pa["seen"].to_numpy() | pb["seen"].to_numpy())
    y = pa["label"].to_numpy()[keep]
    xa, xb = pa["pred"].to_numpy()[keep], pb["pred"].to_numpy()[keep]
    labels = ["negative", "neutral", "positive"]
    f1 = lambda p, i: f1_score(y[i], p[i], labels=labels, average="macro", zero_division=0)  # noqa: E731
    rng = np.random.default_rng(0)
    draws = [f1(xa, i) - f1(xb, i) for i in rng.integers(0, len(y), (1000, len(y)))]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    full = np.arange(len(y))
    verdict = "higher" if lo > 0 else "lower" if hi < 0 else "tie"
    return {"a": a, "b": b, "test_set": test_set, "n": int(len(y)),
            "diff": f1(xa, full) - f1(xb, full), "ci95": [lo, hi], "verdict": verdict}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", metavar="FOLDER", help="model folder names to score")
    p.add_argument("--e16", nargs="+", type=Path, metavar="PATH", help="model folders, E16 test")
    p.add_argument("--e18", nargs="+", metavar="NAME", help="model folders or finbert, E18 sets")
    p.add_argument("--freeze-e18", nargs="+", metavar="NAME", help="write the E18 manifest only")
    p.add_argument("--e18-preds", type=Path, metavar="FILE", help="E18 rows from predictions")
    p.add_argument("--e18-meta", type=Path, metavar="FILE", help="its meta json (where it ran)")
    args = p.parse_args()
    if args.freeze_e18:
        E18_MANIFEST.write_text(json.dumps(e18_manifest(args.freeze_e18), indent=2))
        print(E18_MANIFEST.read_text())
        return
    if args.e18_preds:
        results = e18_from_predictions(args.e18_preds, args.e18_meta)
    elif args.e18:
        results = score_e18(args.e18)
    elif args.e16:
        results = score_e16(args.e16)
    else:
        results = score_only(args.only) if args.only else score_all()
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    RESULTS_MD.write_text(to_markdown(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
