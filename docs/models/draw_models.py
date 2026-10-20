"""Redraw the five shipped-model diagrams offline; never fit or run inference.

From the repository root: python3 docs/models/draw_models.py
The drawing interpreter needs matplotlib; .venv/bin/python reads the small, trusted
impact joblib packs using the project's installed dependencies. No ONNX session is opened.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/helix-model-diagrams-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT, DPI = 1600, 1000, 100
NAVY, TEAL, BLUE, AMBER = "#193e6b", "#10837c", "#3378b7", "#ac681b"
PURPLE, MUTED = "#7555aa", "#64788e"
BG, WHITE, BORDER = "#f8fbfd", "#ffffff", "#d7e3ee"
PALE_TEAL, PALE_BLUE, PALE_AMBER = "#e9f5f1", "#ecf3fb", "#fff3e1"
FOLDERS = ("sentiment", "topic", "e25-student-v3", "e27-entity-bge-small-en-v1.5-int8")
CHECKS = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constant(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing constant {name}: {path}")


def read_facts():
    result = {"models": {}, "sources_sha256": {}}
    defaults = tomllib.loads((ROOT / "config/default.toml").read_text())
    assert defaults["events"] == "student_rules", "Review the running-engine badge after a change"
    sources = [
        "models/models-metadata.md",
        "config/default.toml",
        "src/riskengine/nlp/sentiment.py",
        "src/riskengine/nlp/classifier.py",
        "src/riskengine/nlp/event_engine.py",
        "src/riskengine/nlp/events.py",
        "src/riskengine/eval/e24_dev.py",
        "src/riskengine/eval/e24_inference.py",
        "src/riskengine/eval/e27_entity_eval.py",
        "src/riskengine/impact/features.py",
        "src/riskengine/impact/train_v2.py",
        "src/riskengine/impact/train_v3.py",
        "src/riskengine/impact/tweets_v3.py",
        "src/riskengine/impact/learned.py",
        "src/riskengine/impact/scores.py",
    ]
    for name in FOLDERS:
        folder = ROOT / "models" / name
        cfg = json.loads((folder / "config.json").read_text())
        metrics = json.loads((folder / "source_metrics.json").read_text())
        assert cfg["architectures"] == ["BertForSequenceClassification"]
        assert cfg["hidden_dropout_prob"] == 0.1
        assert cfg.get("classifier_dropout") in (None, 0.1)
        assert (
            cfg["num_hidden_layers"],
            cfg["hidden_size"],
            cfg["num_attention_heads"],
            cfg["intermediate_size"],
        ) == (12, 384, 12, 1536)
        labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
        result["models"][name] = {"config": cfg, "labels": labels, "source_metrics": metrics}
        sources += [f"models/{name}/config.json", f"models/{name}/source_metrics.json"]
    # Load only the small, local impact packs to inspect estimator types and feature names.
    # This does not call fit(), predict(), any NLP model, or any data/cache writer.
    python = ROOT / ".venv/bin/python"
    if not python.exists():
        python = Path(sys.executable)
    code = r"""
import json, joblib
from pathlib import Path
result = {}
for name in ["impact_v3.joblib", "impact_v3_tweets.joblib"]:
    pack = joblib.load(Path("models/impact-v3") / name)
    model = pack["model"]
    prep = model.inner.named_steps["prep"]
    result[name] = {
        "kind": pack["kind"], "used": pack["used"], "numeric": model.num,
        "wrapper": type(model).__name__,
        "pipeline": [[name, type(step).__name__] for name, step in model.inner.steps],
        "alpha": model.inner.named_steps["model"].alpha,
        "categorical": list(prep.transformers_[0][2]),
        "categorical_transform": type(prep.transformers_[0][1]).__name__,
        "numeric_transform": type(prep.transformers_[1][1]).__name__,
        "edges": len(pack["edges"]),
        "probability_features": [c for c in pack["used"] if c.startswith("p_")],
    }
print(json.dumps(result))
"""
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    raw = subprocess.check_output([str(python), "-c", code], cwd=ROOT, env=env, text=True)
    result["impact"] = json.loads(raw)
    for name, item in result["impact"].items():
        assert item["kind"] == "d_clipped_no_vol"
        assert item["wrapper"] == "Clipped" and item["pipeline"][-1][1] == "Ridge"
        assert item["alpha"] == 1.0 and item["edges"] == 9
        assert item["categorical"] == ["event", "publisher"]
        assert item["categorical_transform"] == "OneHotEncoder"
        assert item["numeric_transform"] == "StandardScaler"
        assert not {"vol20", "vix_prev"} & set(item["used"])
        sources.append(f"models/impact-v3/{name}")
    assert len(result["impact"]["impact_v3.joblib"]["probability_features"]) == 20
    assert len(result["impact"]["impact_v3_tweets.joblib"]["probability_features"]) == 9
    assert constant(ROOT / "src/riskengine/impact/train_v3.py", "DROP") == ("vol20", "vix_prev")
    result["scoring_max_tokens"] = constant(
        ROOT / "src/riskengine/eval/e24_inference.py", "MAX_LENGTH"
    )
    for relative in sources:
        result["sources_sha256"][relative] = digest(ROOT / relative)
    return result


class Diagram:
    def __init__(self, title, subtitle, badge, accent=TEAL):
        self.fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.texts = []
        self.card(40, 36, 5, 119, accent, accent, radius=2)
        self.text(62, 34, "HELIX  /  SHIPPED MODEL STRUCTURE", 11.8, accent, bold=True)
        self.text(62, 70, title, 28, NAVY, bold=True)
        self.text(62, 127, subtitle, 13.4, MUTED, max_width=1450)
        self.pill(1120, 34, 432, 32, badge, accent)
        self.ax.plot([48, 1552], [178, 178], color=BORDER, linewidth=0.8)

    def text(
        self,
        x,
        y,
        value,
        size=14,
        color=NAVY,
        bold=False,
        max_width=None,
        max_height=None,
        align="left",
    ):
        prop = FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal")
        lines = []
        for paragraph in value.split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                width = self.renderer.get_text_width_height_descent(candidate, prop, False)[0]
                if max_width is not None and width > max_width and line:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        line_height = size * DPI / 72 * 1.36
        height = len(lines) * line_height
        if max_height is not None and height > max_height + 2:
            raise ValueError(f"Text too tall ({height:.1f}>{max_height}): {value}")
        for line in lines:
            width = self.renderer.get_text_width_height_descent(line, prop, False)[0]
            if max_width is not None and width > max_width + 1:
                raise ValueError(f"Text too wide: {line}")
        artist = self.ax.text(
            x,
            y,
            "\n".join(lines),
            fontproperties=prop,
            color=color,
            va="top",
            ha=align,
            linespacing=1.36,
        )
        self.texts.append(artist)
        return height

    def card(self, x, y, w, h, fill=WHITE, edge=BORDER, radius=14):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.9,
                zorder=1,
            )
        )

    def pill(self, x, y, w, h, label, accent=TEAL):
        self.card(x, y, w, h, PALE_AMBER if accent == AMBER else PALE_TEAL, edge=BG, radius=7)
        self.text(x + w / 2, y + 7, label, 10.8, accent, True, align="center", max_width=w - 20)

    def arrow(self, start, end, color=NAVY, dashed=False, points=None):
        if points:
            xs, ys = zip(start, *points, end, strict=True)
            self.ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.5,
                linestyle=(0, (4, 3)) if dashed else "-",
                zorder=2,
            )
            start = points[-1]
        self.ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color=color,
                linestyle=(0, (4, 3)) if dashed else "-",
                zorder=2,
            )
        )

    def panel(
        self, x, y, w, h, title, body, accent=TEAL, fill=WHITE, body_size=13.8, title_size=16
    ):
        self.card(x, y, w, h, fill)
        title_h = self.text(x + 20, y + 18, title, title_size, accent, True, max_width=w - 40)
        if body:
            self.text(
                x + 20,
                y + title_h + 30,
                body,
                body_size,
                max_width=w - 40,
                max_height=h - title_h - 44,
            )

    def footer(self, left, right="Facts from local configs, model packs and code · no model run"):
        self.ax.plot([48, 1552], [945, 945], color=BORDER, linewidth=0.8)
        self.text(48, 962, left, 10.2, MUTED, max_width=1480)
        self.text(1552, 981, right, 8.9, MUTED, align="right", max_width=1480)

    def save(self, name):
        self.fig.canvas.draw()
        for item in self.texts:
            bbox = item.get_window_extent(self.fig.canvas.get_renderer())
            if bbox.x0 < 0 or bbox.x1 > WIDTH + 1 or bbox.y0 < 0 or bbox.y1 > HEIGHT + 1:
                raise ValueError(f"Off-canvas text in {name}: {item.get_text()}")
        path = OUT / name
        self.fig.savefig(
            path,
            dpi=DPI,
            facecolor=BG,
            metadata={"Title": name, "Software": "Helix offline model diagram script"},
        )
        CHECKS.append(
            {
                "file": name,
                "width": WIDTH,
                "height": HEIGHT,
                "text_boxes": len(self.texts),
                "sha256": digest(path),
            }
        )
        plt.close(self.fig)


def backbone(d, x, y, w, h, cfg):
    d.card(x, y, w, h, PALE_BLUE)
    d.text(x + 20, y + 18, "bge-small BERT encoder", 17, BLUE, True, max_width=w - 40)
    d.card(x + 18, y + 72, w - 36, 96)
    d.text(x + 34, y + 84, "Input embeddings", 13.5, BLUE, True, max_width=w - 68)
    d.text(
        x + 34,
        y + 116,
        "Token + position + segment\n"
        f"Vocabulary {cfg['vocab_size']:,} · width {cfg['hidden_size']}",
        12.0,
        max_width=w - 68,
        max_height=50,
    )
    d.arrow((x + w / 2, y + 168), (x + w / 2, y + 192), BLUE)
    d.card(x + 18, y + 202, w - 36, 123)
    d.text(
        x + 34,
        y + 214,
        f"Transformer block × {cfg['num_hidden_layers']}",
        13.5,
        BLUE,
        True,
        max_width=w - 68,
    )
    d.text(
        x + 34,
        y + 246,
        f"{cfg['num_attention_heads']}-head self-attention\n"
        f"FFN width {cfg['intermediate_size']:,}\nResiduals + layer normalization",
        12.0,
        max_width=w - 68,
        max_height=74,
    )
    d.text(
        x + 22,
        y + h - 37,
        f"Position capacity: {cfg['max_position_embeddings']} tokens",
        11.6,
        MUTED,
        max_width=w - 44,
    )


def label_rows(d, x, y, labels, line=31, size=13.3):
    for i, label in enumerate(labels):
        d.text(x, y + i * line, f"{i}", size, TEAL, True)
        d.text(x + 30, y + i * line, label, size, NAVY, max_width=325)


def sentiment(facts):
    m = facts["models"]["sentiment"]
    c = m["config"]
    d = Diagram(
        "Headline sentiment",
        "models/sentiment/ · a three-class BERT sequence classifier",
        "3 learned labels",
    )
    d.panel(
        48,
        232,
        250,
        156,
        "Headline or tweet",
        "One financial text\nHeadline-only scoring",
        body_size=13.2,
        title_size=14.4,
    )
    d.arrow((173, 391), (173, 425), TEAL)
    d.panel(
        48,
        430,
        250,
        173,
        "Tokenizer",
        "Token IDs + mask\nSegment IDs\nDefault truncation: 64",
        body_size=13.0,
    )
    backbone(d, 334, 232, 410, 371, c)
    d.arrow((300, 465), (331, 465))
    d.panel(
        780,
        232,
        300,
        371,
        "Classification head",
        "First token [CLS]\nBERT pooled output: 384\nDropout 0.1 (training)\n"
        "Linear layer: 384 → 3\nLogits → softmax",
        body_size=14.0,
    )
    d.arrow((747, 415), (777, 415))
    d.panel(1116, 232, 436, 371, "Class probabilities", "", body_size=14)
    label_rows(d, 1140, 310, m["labels"], line=48, size=16)
    d.text(1140, 494, "Sentiment score", 14, TEAL, True)
    d.text(1140, 530, "P(positive) − P(negative)\nRange: −1 to +1", 14.4, max_width=385)
    d.arrow((1083, 415), (1113, 415))
    d.panel(
        48,
        650,
        690,
        224,
        "How the engine uses it",
        'Selectable as sentiment = "own".\nFallback when no company name is '
        "available.\nThe saved credit replay keeps own-model scores.",
        body_size=15.0,
    )
    d.panel(
        766,
        650,
        786,
        224,
        "Local export and runtime",
        "model.onnx · INT8 quantized export\nONNX Runtime · "
        "CPUExecutionProvider\nTokenizer and config load from "
        "models/sentiment/.",
        BLUE,
        body_size=15.0,
    )
    d.footer(
        "Sources: sentiment/config.json + source_metrics.json · "
        "nlp/sentiment.py · nlp/classifier.py"
    )
    d.save("sentiment.png")


def topic(facts):
    m = facts["models"]["topic"]
    c = m["config"]
    d = Diagram(
        "Topic / event classifier",
        "models/topic/ · nine learned classes; credit classes come from rules",
        "9 learned classes + credit rules",
        BLUE,
    )
    d.panel(
        48,
        232,
        250,
        371,
        "Text → tokens",
        "Financial headline\nor social post\n\nToken IDs + mask\nSegment IDs\n\nONNX INT8 · CPU",
        body_size=14.0,
    )
    backbone(d, 334, 232, 410, 371, c)
    d.arrow((300, 415), (331, 415))
    d.panel(
        780,
        232,
        300,
        371,
        "Pooled classifier",
        "[CLS] → pooler (384)\nDropout 0.1 (training)\nLinear: 384 → 9\n"
        "Softmax → class index\n\nLabel order from config",
        BLUE,
        body_size=14.0,
    )
    d.arrow((747, 415), (777, 415))
    d.panel(1116, 232, 436, 371, "Learned label IDs", "", BLUE)
    label_rows(d, 1140, 297, m["labels"], line=30, size=13.1)
    d.arrow((1083, 415), (1113, 415))
    d.panel(
        48,
        677,
        925,
        205,
        "Credit rules run beside the model",
        "Default · Credit downgrade · Covenant breach\ncredit_event(text) "
        "overrides the learned label when a rule matches.\nNine learned "
        "classes + three credit classes = 12 possible events.",
        AMBER,
        PALE_AMBER,
        14.0,
    )
    d.panel(
        1000,
        677,
        552,
        205,
        "Routing options",
        'events = "model": credit rules + model\nevents = "model_rules": '
        "frozen C rules\nBoth use this shipped topic artifact.",
        BLUE,
        body_size=13.4,
    )
    d.arrow((170, 605), (170, 673), AMBER, True)
    d.text(217, 632, "Raw text also reaches the credit rules", 12.3, AMBER, max_width=700)
    d.footer("Sources: topic/config.json + source_metrics.json · nlp/events.py · eval/e24_dev.py")
    d.save("topic.png")


def student(facts):
    m = facts["models"]["e25-student-v3"]
    c = m["config"]
    d = Diagram(
        "Event student v3",
        "models/e25-student-v3/ · event classifier with frozen C routing",
        "Running engine · E44b winner",
    )
    d.text(48, 202, "TRAINING PROVENANCE", 11.5, MUTED, True)
    for x, w, title, body in [
        (48, 290, "Real-news pool", "Teacher-labelled texts"),
        (376, 340, "Teacher labels", "Targets for 9 learned classes"),
        (754, 390, "Student fine-tuning", "Same BERT backbone + 9-class head"),
        (1182, 370, "Shipped model", "INT8 ONNX export"),
    ]:
        d.panel(x, 237, w, 119, title, body, BLUE, PALE_BLUE, 12.8, 14.5)
    for a, b in [(340, 373), (718, 751), (1146, 1179)]:
        d.arrow((a, 296), (b, 296), BLUE)
    d.text(48, 390, "RUNTIME INFERENCE", 11.5, MUTED, True)
    d.panel(
        48,
        424,
        250,
        286,
        "Headline → tokens",
        "Token IDs + mask\nSegment IDs\n"
        f"Truncate at {facts['scoring_max_tokens']} tokens\n\n"
        "One-row ONNX scoring\nCPU execution",
        body_size=13.1,
    )
    d.panel(
        334,
        424,
        442,
        286,
        "bge-small BERT + head",
        f"{c['num_hidden_layers']} layers · hidden {c['hidden_size']}\n"
        f"{c['num_attention_heads']} heads · FFN {c['intermediate_size']:,}\n"
        "[CLS] → pooler → dropout\nLinear 384 → 9 → softmax\n\nOne shared model artifact",
        BLUE,
        body_size=14.0,
    )
    d.panel(
        812,
        424,
        352,
        286,
        "Frozen rules C",
        "Credit rules first\nHoldings / analyst routing\nEarnings / product "
        "rules\nThen model fallback\n\nRules can override the model.",
        AMBER,
        PALE_AMBER,
        13.4,
    )
    d.panel(
        1200,
        424,
        352,
        286,
        "One event class",
        "9 learned event labels\n+ 3 credit classes\n= 12 possible outputs\n\n"
        "Default / Credit downgrade /\nCovenant breach come from rules.",
        body_size=13.0,
    )
    for a, b in [(300, 331), (778, 809), (1166, 1197)]:
        d.arrow((a, 562), (b, 562))
    d.text(48, 750, "STUDENT LABEL IDS — order differs from topic/", 11.5, MUTED, True)
    for i, label in enumerate(m["labels"]):
        column, row = divmod(i, 3)
        x = 48 + column * 510
        y = 793 + row * 34
        d.text(x, y, str(i), 13.3, TEAL, True)
        d.text(x + 31, y, label, 13.3, max_width=450)
    d.text(
        48,
        905,
        "E44b designation: AI-only reference; the same test selected v3. "
        "No human validation is claimed.",
        11.7,
        MUTED,
        max_width=1504,
    )
    d.footer(
        "Sources: e25-student-v3/config.json + source_metrics.json · "
        "nlp/event_engine.py · eval/e24_dev.py"
    )
    d.save("e25-student-v3.png")


def entity(facts):
    m = facts["models"]["e27-entity-bge-small-en-v1.5-int8"]
    c = m["config"]
    d = Diagram(
        "Company-specific sentiment",
        "E27 · models/e27-entity-bge-small-en-v1.5-int8/",
        "One score per company",
    )
    d.panel(
        48,
        232,
        310,
        166,
        "One shared headline",
        "One headline mentions\nboth Company A and B.",
        body_size=13.4,
    )
    d.panel(
        48,
        443,
        143,
        108,
        "Company A",
        "",
        title_size=14.0,
        body_size=11.8,
        fill=PALE_TEAL,
    )
    d.panel(
        215,
        443,
        143,
        108,
        "Company B",
        "",
        title_size=14.0,
        body_size=11.8,
        fill=PALE_TEAL,
    )
    d.panel(
        394,
        232,
        324,
        371,
        "Pair tokenizer",
        "Company name first\nHeadline second\n\n[CLS] company [SEP]\nheadline "
        "[SEP]\n\nToken + segment IDs + mask",
        BLUE,
        body_size=13.2,
    )
    d.arrow((360, 315), (391, 315))
    d.arrow((119.5, 553), (556, 606), TEAL, points=[(119.5, 627), (556, 627)])
    d.arrow((286.5, 553), (556, 606), TEAL, points=[(286.5, 627), (556, 627)])
    d.panel(
        754,
        232,
        410,
        371,
        "Shared BERT + classifier",
        f"{c['num_hidden_layers']} layers · hidden {c['hidden_size']}\n"
        f"{c['num_attention_heads']} heads · FFN {c['intermediate_size']:,}\n"
        f"Position capacity: {c['max_position_embeddings']}\n\n"
        "[CLS] → pooler (384)\nDropout 0.1 (training)\nLinear 384 → 3 → softmax",
        BLUE,
        body_size=14.1,
    )
    d.arrow((720, 415), (751, 415))
    d.panel(1200, 232, 352, 371, "Label IDs", "")
    label_rows(d, 1224, 309, m["labels"], line=48, size=15.3)
    d.text(1224, 507, "P(positive) − P(negative)\nRange: −1 to +1", 13.4, max_width=300)
    d.arrow((1166, 415), (1197, 415))
    d.panel(
        48,
        664,
        670,
        215,
        "Missing company name?",
        "The engine uses the headline-only sentiment model.\nThe pair model "
        "is called with (company, headline);\ncompany identity is supplied "
        "by the linker.",
        AMBER,
        PALE_AMBER,
        14.1,
    )
    d.panel(
        754,
        664,
        798,
        215,
        "Same headline, separate company scores",
        "(Company A, headline) → score s_A\n(Company B, headline) → score "
        "s_B\nSame model weights; each company is scored separately.",
        body_size=15.4,
    )
    d.text(
        48,
        908,
        "s_A and s_B are symbols illustrating separate outputs, not measured example predictions.",
        11.7,
        MUTED,
        max_width=1504,
    )
    d.footer(
        "Sources: entity config.json + source_metrics.json · "
        "eval/e27_entity_eval.py · nlp/sentiment.py"
    )
    d.save("entity-sentiment.png")


def impact(facts):
    news = facts["impact"]["impact_v3.joblib"]
    tweets = facts["impact"]["impact_v3_tweets.joblib"]
    d = Diagram(
        "Learned impact v3",
        "models/impact-v3/ · two small scikit-learn packs; no transformer layers",
        "Clipped Ridge · learned move score",
        AMBER,
    )
    news_base = [c for c in news["numeric"] if not c.startswith("p_")]
    tweet_base = [c for c in tweets["numeric"] if not c.startswith("p_")]
    assert news_base == ["sentiment", "abs_sentiment", "title_words", "n_prev_24h", "novelty"]
    assert tweet_base == ["sentiment", "abs_sentiment", "title_words"]
    d.panel(
        48,
        223,
        733,
        221,
        "NEWS · impact_v3.joblib",
        "Numeric: sentiment, abs_sentiment, title_words,\nn_prev_24h, "
        "novelty + 20 cached topic p_* columns\nCategorical: event, "
        "publisher\nExact feature names are saved in model_facts.json.",
        BLUE,
        body_size=14.3,
    )
    d.panel(
        815,
        223,
        737,
        221,
        "TWEETS · impact_v3_tweets.joblib",
        "Numeric: sentiment, abs_sentiment, title_words\n+ 9 topic p_* "
        "probability columns\nCategorical: event, publisher (publisher = "
        '"other")\nNo headline-count or novelty regressors.',
        BLUE,
        body_size=14.3,
    )
    d.arrow((414, 447), (194, 492), BLUE, points=[(414, 472), (194, 472)])
    d.arrow((1182, 447), (194, 492), BLUE, points=[(1182, 472), (194, 472)])
    d.panel(
        48,
        502,
        292,
        192,
        "1  Clip numeric values",
        "Training 1st–99th\npercentile limits\nFrozen at inference",
        AMBER,
        PALE_AMBER,
        14.0,
        15.0,
    )
    d.panel(
        375,
        502,
        335,
        192,
        "2  ColumnTransformer",
        "One-hot: event, publisher\nStandardScaler: numeric\nNumeric + categorical join",
        BLUE,
        body_size=13.5,
        title_size=14.7,
    )
    d.panel(
        745,
        502,
        314,
        192,
        "3  Ridge regression",
        "alpha = 1.0\nPredicts log(1 + z)\nz = scaled absolute move",
        AMBER,
        PALE_AMBER,
        13.5,
        14.7,
    )
    d.panel(
        1094,
        502,
        458,
        192,
        "4  Recover expected move",
        "max(exp(prediction) − 1, 0)\n× √2 × prior 20-day volatility\n"
        "Expected absolute two-day abnormal return",
        body_size=13.2,
        title_size=14.7,
    )
    for a, b in [(342, 372), (712, 742), (1061, 1091)]:
        d.arrow((a, 596), (b, 596))
    d.panel(
        48,
        744,
        482,
        164,
        "vol20 and VIX handling",
        "vol20 rescales the output above.\nvol20 and vix_prev are excluded\n"
        "from the fitted Ridge regressors.",
        BLUE,
        PALE_BLUE,
        13.0,
        14.7,
    )
    d.panel(
        557,
        744,
        482,
        164,
        "1–10 move score",
        "9 training-decile boundaries\nDigitize the expected move → 1…10\n"
        "This estimates move size.",
        body_size=13.0,
        title_size=14.7,
    )
    d.panel(
        1066,
        744,
        486,
        164,
        "Integration boundary",
        "No prices / live analysis → formula.\nThe credit gate keeps "
        "formula severity.\nSeparate news and tweet feature sets.",
        AMBER,
        PALE_AMBER,
        12.9,
        14.7,
    )
    d.arrow((1323, 696), (797, 741), TEAL, points=[(1323, 719), (797, 719)])
    d.footer(
        "Sources: stored joblib packs · impact/features.py, train_v3.py, "
        "train_v2.py, tweets_v3.py, learned.py"
    )
    d.save("impact-v3.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    facts = read_facts()
    for draw in (sentiment, topic, student, entity, impact):
        draw(facts)
    facts["rendered_images"] = CHECKS
    (OUT / "model_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps({"images": CHECKS, "no_training_or_inference": True}, indent=2))


if __name__ == "__main__":
    main()
