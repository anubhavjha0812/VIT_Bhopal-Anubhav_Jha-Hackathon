"""Label (headline, company) pairs with three open-licence models (rules: docs/DECISIONS.md, E17).

Question per pair: is the headline about this company, news that matters for its stock? Each model
gets the same fixed prompt and answers by greedy generation (up to 4 tokens, no sampling, so a
rerun gives the same labels); the first Y or N is its answer, and the next-token probability of
"Y" vs "N" is the fallback. A model that answers Y on fewer than 5% or more than 95% of pairs is
degenerate and left out; the label is the majority of the rest, or "unsure" on a tie.

    python label_relevance.py --data to_label.csv --out labels.csv
On Kaggle the label file comes from the private dataset in kernel-metadata.json.
"""

import argparse
import gc
import glob
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {  # all open licences, none gated
    "qwen3_8b": "Qwen/Qwen3-8B",  # Apache 2.0
    "olmo2_7b": "allenai/OLMo-2-1124-7B-Instruct",  # Apache 2.0
    "phi4_mini": "microsoft/Phi-4-mini-instruct",  # MIT
}
BATCH = 4
NEW_TOKENS = 4
DEGENERATE = (0.05, 0.95)  # share of Y answers outside this range -> the labeller is left out
ANSWER = re.compile(r"\b([YN])\b|^\s*([YN])", re.IGNORECASE)
GPU_GB = 12  # per GPU, leaves room for activations

SYSTEM = "You are a careful financial news analyst."
PROMPT = """Company: {company} (ticker {ticker})
Headline: "{headline}"

Question: Is this headline about {company} itself, that is, news that matters for {company}'s stock?

Answer N if:
- the news is about another company or person and {company} is only mentioned (a partner, a customer, a past employer, a conference host, a fund that owns many stocks);
- {company} is only a bank or broker giving a rating, price target or opinion on another company;
- it is a fund's buy/sell list, a fund's regulatory notice, or a list of many stocks;
- the headline is not in English.
Answer Y if the headline reports something done by, happening to, or said about {company} or its stock.

Answer with one letter: Y or N."""  # noqa: E501


def prompts(tok, df: pd.DataFrame) -> list[str]:
    out = []
    for r in df.itertuples():
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(company=r.company, ticker=r.ticker,
                                                      headline=r.headline)},
        ]  # fmt: skip
        kw = {"enable_thinking": False} if "Qwen3" in tok.name_or_path else {}
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
    return out


def letter_ids(tok, letter: str) -> list[int]:
    ids = {tok.encode(v, add_special_tokens=False)[0] for v in (letter, " " + letter)}
    return sorted(ids)


def parse(text: str) -> str | None:
    m = ANSWER.search(text.strip())
    return (m.group(1) or m.group(2)).upper() if m else None


@torch.no_grad()
def answer(name: str, repo: str, df: pd.DataFrame) -> tuple[list[float], list[str | None]]:
    """P(Y) from the next token, and the generated answer (Y, N or None) per pair."""
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    memory = {i: f"{GPU_GB}GiB" for i in range(torch.cuda.device_count())} | {"cpu": "28GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float16, device_map="auto", max_memory=memory
    )
    model.eval()
    yes, no = letter_ids(tok, "Y"), letter_ids(tok, "N")
    texts, probs, said = prompts(tok, df), [], []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i : i + BATCH], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        y = torch.logsumexp(logits[:, yes], dim=-1)
        n = torch.logsumexp(logits[:, no], dim=-1)
        probs += torch.sigmoid(y - n).cpu().tolist()
        gen = model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id)  # fmt: skip
        new = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        if i == 0:
            print(f"{name} first answers: {new}", flush=True)
        said += [parse(t) for t in new]
    print(f"{name}: {len(probs)} pairs in {time.time() - t0:.0f} s, "
          f"{sum(s is None for s in said)} without a Y/N", flush=True)  # fmt: skip
    del model, logits, enc, gen
    gc.collect()
    torch.cuda.empty_cache()
    return probs, said


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    found = glob.glob("/kaggle/input/**/to_label.csv", recursive=True)
    p.add_argument("--data", type=Path, default=Path(found[0]) if found else Path("to_label.csv"))
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/labels.csv"))
    p.add_argument("--limit", type=int, help="first N pairs only (smoke test)")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    labels = df[["id"]].copy()
    for name, repo in MODELS.items():
        probs, said = answer(name, repo, df)
        labels[f"p_{name}"] = probs
        labels[name] = [s or ("Y" if p >= 0.5 else "N") for s, p in zip(said, probs, strict=True)]
    share = {m: float((labels[m] == "Y").mean()) for m in MODELS}
    used = [m for m in MODELS if DEGENERATE[0] <= share[m] <= DEGENERATE[1]]
    print("Y share:", share, "used:", used, flush=True)
    yes = (labels[used] == "Y").sum(axis=1)
    no = len(used) - yes
    labels["label"] = [
        "Y" if a > b else "N" if b > a else "unsure" for a, b in zip(yes, no, strict=True)
    ]
    labels["unanimous"] = yes.isin([0, len(used)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.round(4).to_csv(args.out, index=False)
    (args.out.parent / "labellers.json").write_text(
        json.dumps({"y_share": share, "used": used}, indent=2)
    )
    print(labels["label"].value_counts().to_dict(), "unanimous:", labels["unanimous"].mean())


if __name__ == "__main__":
    main()
