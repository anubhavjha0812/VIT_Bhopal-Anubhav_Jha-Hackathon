"""E25 teacher labelling: 12-class event probabilities from open LLMs (docs/experiments/E25_TEACHER_STUDENT.md).

Each teacher gets the same fixed prompt (the E24 class definitions plus hand-written examples that
are not from any test set) and answers with one letter, A to L. The probability over the 12 classes
is the softmax of the next-token scores of those 12 letters. No sampling, so a rerun gives the same
numbers (batches are built in a fixed order). Texts are labelled; no label is read.

    python label_events.py --name d1 --data D1_to_label.csv --id-col item_id --text-col text
    python label_events.py --name pool --data pool.csv.gz --id-col pool_id --text-col text
    python label_events.py --name dry --data pool.csv.gz --id-col pool_id --text-col text \
        --limit 500

On Kaggle a script gets no arguments: set STAGE below ("dry", then "d1", then "pool") and the input
file is found in /kaggle/input (the private dataset in kernel-metadata.json). This script refuses
any file whose name contains "d2" or "final": D2 and the final sets are never labelled here.
Output per teacher: <name>_<teacher>.csv.gz (id + 12 probabilities) and <name>_<teacher>_meta.json.
Shards are saved every SHARD rows, so a stopped session resumes where it left off.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

CLASSES = [
    "Default", "Credit downgrade", "Covenant breach", "Geopolitical", "Macroeconomic", "M&A",
    "Product news", "Earnings", "Analyst rating", "Legal/Regulatory", "Management change", "Other",
]  # fmt: skip
LETTERS = "ABCDEFGHIJKL"
PROMPT_VERSION = "v2"

TEACHERS = {  # open licences; check each licence on Hugging Face before running
    "qwen3_8b": {"repo": "Qwen/Qwen3-8B", "four_bit": False},  # Apache 2.0
    "mistral_nemo": {
        "repo": "mistralai/Mistral-Nemo-Instruct-2407",
        "four_bit": True,
    },  # Apache 2.0
    # backup for the second teacher. GATED: accept the licence on Hugging Face and add an HF token
    # to the Kaggle secrets (HF_TOKEN) first. The licence asks for attribution.
    "llama31_8b": {"repo": "meta-llama/Llama-3.1-8B-Instruct", "four_bit": False},
}
DEFAULT_TEACHERS = ["qwen3_8b", "mistral_nemo"]
POOL_ROWS = 40453  # E25 pool size, for the projected labelling time
STAGE = "extra"  # the only line to edit on Kaggle: "dry", "d1", "pool", "extra"
STAGES = {  # name -> input file to find, id column, row limit
    "dry": {"file": "pool.csv.gz", "id_col": "pool_id", "limit": 500},
    "d1": {"file": "D1_to_label.csv", "id_col": "item_id", "limit": None},
    "pool": {"file": "pool.csv.gz", "id_col": "pool_id", "limit": None},
    "extra": {"file": "extra.csv.gz", "id_col": "extra_id", "limit": None},
}
BATCH = 4  # the longer v3 prompt ran Mistral out of memory at 8 rows
BATCH_CACHED = 8  # rows per step when the shared prompt prefix is cached
CACHE_TOLERANCE = 0.5  # max letter-score difference allowed between cached and full prompts
PROBE_ROWS = 16
SHARD = 2000
GPU_GB = 12  # per GPU, leaves room for activations
FORBIDDEN = re.compile(r"d2|final", re.IGNORECASE)

INSTRUCTIONS = """You label financial news headlines. Give each headline exactly ONE class.
Judge the headline text alone. Do not infer an event the text does not report. The class is the
main news fact the headline reports (what happened), not its likely effect or its background.

A. Default: a missed coupon or loan payment, a bankruptcy or Chapter 11 filing, insolvency, a distressed debt restructuring. It must be stated in the headline. Loan losses, credit quality, debt levels, loans, 'obligations' or a plant losing power are NOT Default. Not 'could go bankrupt' opinions.
B. Credit downgrade: a credit rating agency (Moody's, S&P Global Ratings, Fitch, DBRS) cutting a company's or its debt's rating, or a negative outlook or watch. Not a stock analyst's rating (that is I).
C. Covenant breach: breaking, or asking lenders to waive or amend, loan terms. Not ordinary new loans.
D. Geopolitical: wars, military action, sanctions, tariffs and trade disputes, elections and political shocks, diplomacy.
E. Macroeconomic: central banks and rates, inflation, jobs, GDP, recession, fiscal stimulus, bond and currency markets, oil and commodity prices. Not one company's results.
F. M&A: a merger, acquisition, takeover bid, buyout, stake purchase, asset sale, spin-off or IPO. Not a fund buying shares as an investment holding.
G. Product news: NARROW. The headline says a product, service, feature or software version itself is launched, unveiled, released, updated (a patch), priced, recalled or sold, or reports a deal to supply or buy a product. The verbs launch, release, introduce or open are not enough: a company launching an initiative, programme, challenge, business unit or fund, releasing a study or report, opening an exhibit, a bug-bounty, a milestone (downloads) or a pricing dispute is L. Other company news (strategy, stock moves, executives' remarks, plants, hiring, price cuts) is L.
H. Earnings: quarterly or annual results, revenue, profit or loss figures, beats or misses, guidance, dividends. A preview or opinion about coming results with no figures is L.
I. Analyst rating: a stock analyst's rating or price-target change (upgrade, downgrade, initiate, target cut). Not a credit agency's action.
J. Legal/Regulatory: lawsuits, regulators (SEC, FTC, EU and others), fines, probes, court rulings, settlements, approvals.
K. Management change: a CEO, CFO, chair or other executive appointed, leaving or replaced.
L. Other: everything else: price moves, market commentary, opinion, fund holdings, lists, company news that is not a product event.

Overlaps: a credit agency action plus a stock rating -> the credit class. Earnings plus guidance or a dividend -> H. A regulator's decision on a deal -> J if the decision is the news, else F. A recall plus a lawsuit -> J; a recall alone -> G.

Examples (invented headlines):
"Northwind Foods misses a bond coupon payment, shares fall" -> A
"Fitch cuts Contoso Bank's long-term rating to 'BBB' from 'A-'" -> B
"Globex asks lenders to waive a leverage test after a weak quarter" -> C
"Government announces new sanctions on a major oil exporter" -> D
"Central bank holds interest rates, inflation cools" -> E
"Initech agrees to buy rival Hooli for $4 billion" -> F
"Acme unveils a new electric van for delivery fleets" -> G
"Umbrella Corp reports quarterly profit of $1.2 billion, beating estimates" -> H
"Morgan Brook downgrades Wayne Industries to Sell, cuts price target to $40" -> I
"Regulator opens a probe into Stark Retail over pricing" -> J
"Cyberdyne names a new chief financial officer" -> K
"Smith Capital LLC Buys iShares Core Bond ETF, Vanguard Total Market, Sells ..." -> L
"Shares of several retailers are trading higher in sympathy with the market" -> L
"What to watch in Soylent's earnings report next week" -> L
"Wonka CEO says the company sees strong demand ahead" -> L
"Stark Retail to cut the price of its flagship phone by 10%" -> L
"Acme launches a scholarship programme for engineering students" -> L
"Initech releases a study on how teenagers use social media" -> L
"Contoso to launch a new consulting unit in Asia" -> L
"Hooli reports loan losses, but quarterly profit still beats estimates" -> H
"Northwind refinery loses power after storm, filing says" -> L
"Globex says it can meet its debt obligations after rating review" -> L"""  # noqa: E501

QUESTION = 'Headline: "{headline}"\n\nAnswer with one letter, A to L.'


def build_prompt(headline: str) -> str:
    return INSTRUCTIONS + "\n\n" + QUESTION.format(headline=str(headline).replace('"', "'"))


def prompt_sha256() -> str:
    return hashlib.sha256((PROMPT_VERSION + INSTRUCTIONS + QUESTION).encode()).hexdigest()


def letter_probs(scores: np.ndarray) -> np.ndarray:
    """Softmax over the 12 letter scores, row by row: (n, 12) -> (n, 12), rows sum to 1."""
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def check_input_name(path: Path) -> None:
    if FORBIDDEN.search(path.name):
        raise SystemExit(f"refusing {path.name}: D2 and final sets are never labelled here")


def order(texts: list[str]) -> list[int]:
    """Fixed batching order (length, then position), so batches do not depend on input order."""
    return sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))


def ensure_bitsandbytes() -> None:
    """4-bit loading needs bitsandbytes; Kaggle does not ship it. Must run before transformers is
    imported, so main() calls it first (restart the Kaggle session if it was imported already)."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        import importlib
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True
        )
        importlib.invalidate_caches()


def load_teacher(repo: str, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # the Mistral-Nemo tokenizer needs a regex fix (transformers warns about it)
    kw_tok = {"fix_mistral_regex": True} if "mistral" in repo.lower() else {}
    try:
        tok = AutoTokenizer.from_pretrained(repo, **kw_tok)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "auto"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
    else:
        memory = {i: f"{GPU_GB}GiB" for i in range(torch.cuda.device_count())} | {"cpu": "28GiB"}
        kw |= {"dtype": torch.float16, "max_memory": memory}
    model = AutoModelForCausalLM.from_pretrained(repo, **kw).eval()
    return tok, model


def letter_token_ids(tok) -> list[list[int]]:
    ids = []
    for ch in LETTERS:
        found = {tok.encode(v, add_special_tokens=False)[0] for v in (ch, " " + ch)}
        ids.append(sorted(found))
    return ids


def chat(tok, user: str) -> str:
    kw = {"enable_thinking": False} if "Qwen3" in tok.name_or_path else {}
    msgs = [
        {"role": "user", "content": user}
    ]  # one user turn: not every template has a system role
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def score_full(tok, model, texts: list[str]) -> np.ndarray:
    """(n, 12) letter scores, full prompt per row, in the given order."""
    import torch

    ids = letter_token_ids(tok)
    out = np.zeros((len(texts), len(LETTERS)), dtype=np.float32)
    idx = order(texts)
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            rows = idx[s : s + BATCH]
            enc = tok([chat(tok, build_prompt(texts[i])) for i in rows], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(model.device)  # fmt: skip
            logits = model(**enc).logits[:, -1, :].float()
            sc = torch.stack([torch.logsumexp(logits[:, t], dim=-1) for t in ids], dim=1)
            out[rows] = sc.cpu().numpy()
    return out


def shared_prefix(tok) -> str:
    """The part of the chat prompt that is the same for every headline (ends at a blank line)."""
    a = chat(tok, build_prompt("alpha one"))
    b = chat(tok, build_prompt("zulu two"))
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    cut = a.rfind("\n\n", 0, n)
    if cut < 0:
        raise ValueError("no shared prompt prefix")
    return a[: cut + 2]


def score_cached_ids(model, prefix_ids, suffixes, letters, pad_id, batch=BATCH_CACHED):
    """Letter scores with the shared prefix run once and its key/value cache reused per batch.

    Suffixes are right-padded, so the last real token of each row is read and padding cannot
    affect it. Equal (up to rounding) to running prefix + suffix in one go.
    """
    import copy

    import torch

    device = model.device
    out = np.zeros((len(suffixes), len(letters)), dtype=np.float32)
    P = len(prefix_ids)
    with torch.no_grad():
        base = model(input_ids=torch.tensor([prefix_ids], device=device), use_cache=True)
        base = base.past_key_values
        idx = sorted(range(len(suffixes)), key=lambda i: (len(suffixes[i]), i))
        for s in range(0, len(idx), batch):
            rows = idx[s : s + batch]
            n = len(rows)
            lens = [len(suffixes[i]) for i in rows]
            width = max(lens)
            ids = torch.full((n, width), pad_id, dtype=torch.long)
            mask = torch.zeros((n, P + width), dtype=torch.long)
            mask[:, :P] = 1
            for r, i in enumerate(rows):
                ids[r, : lens[r]] = torch.tensor(suffixes[i])
                mask[r, P : P + lens[r]] = 1
            cache = copy.deepcopy(base)
            cache.batch_repeat_interleave(n)
            pos = (P + torch.arange(width)).unsqueeze(0).expand(n, -1)
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           position_ids=pos.to(device), past_key_values=cache,
                           use_cache=True).logits  # fmt: skip
            last = logits[torch.arange(n, device=logits.device), torch.tensor(lens) - 1]
            last = last.float()
            sc = torch.stack([torch.logsumexp(last[:, t], dim=-1) for t in letters], dim=1)
            out[rows] = sc.cpu().numpy()
    return out


def make_scorer(tok, model, probe: list[str]):
    """Return (texts -> (n, 12) scores, info). The cached path is used only if it matches the
    full-prompt path on the probe rows (same top letter, scores within CACHE_TOLERANCE)."""
    from functools import partial

    full = partial(score_full, tok, model)
    try:
        prefix = shared_prefix(tok)
        letters = letter_token_ids(tok)
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]

        def cached(texts: list[str]) -> np.ndarray:
            suffixes = []
            for t in texts:
                text = chat(tok, build_prompt(t))
                if not text.startswith(prefix):
                    raise ValueError("prompt does not start with the shared prefix")
                suffixes.append(tok(text[len(prefix) :], add_special_tokens=False)["input_ids"])
            return score_cached_ids(model, prefix_ids, suffixes, letters, tok.pad_token_id)

        a, b = full(probe), cached(probe)
        diff = float(np.abs(a - b).max())
        same = bool((a.argmax(axis=1) == b.argmax(axis=1)).all())
        info = {
            "used": same and diff <= CACHE_TOLERANCE,
            "max_abs_diff": round(diff, 4),
            "top1_same": same,
            "prefix_tokens": len(prefix_ids),
            "probe_rows": len(probe),
        }
    except Exception as exc:  # noqa: BLE001 - any problem: fall back to the full prompt
        info = {"used": False, "error": repr(exc)}
    print("prefix cache:", info, flush=True)
    return (cached if info["used"] else full), info


def run_teacher(name: str, spec: dict, ids: list, texts: list[str], out_dir: Path, tag: str):
    """Label with one teacher. A failure (gated model, out of memory) is written to a
    <tag>_<name>_FAILED.json file and the next teacher still runs."""
    try:
        _run_teacher(name, spec, ids, texts, out_dir, tag)
    except Exception as exc:  # noqa: BLE001 - one teacher failing must not stop the others
        import traceback

        (out_dir / f"{tag}_{name}_FAILED.json").write_text(
            json.dumps({"teacher": name, "repo": spec["repo"], "error": repr(exc)}, indent=2)
        )
        print(f"{name} FAILED: {exc!r}", flush=True)
        traceback.print_exc()
        gc.collect()


def _run_teacher(name: str, spec: dict, ids: list, texts: list[str], out_dir: Path, tag: str):
    final = out_dir / f"{tag}_{name}.csv.gz"
    if final.exists():
        print(f"{final.name} exists, skipping", flush=True)
        return
    t0 = time.time()
    tok, model = load_teacher(spec["repo"], spec["four_bit"])
    load_s = time.time() - t0
    scorer, cache_info = make_scorer(tok, model, texts[:PROBE_ROWS])
    t1 = time.time()
    scored = 0
    parts = []
    for k, s in enumerate(range(0, len(texts), SHARD)):
        shard = out_dir / f"{tag}_{name}_shard{k:04d}.npy"
        if shard.exists():
            parts.append(np.load(shard))
            continue
        sc = scorer(texts[s : s + SHARD])
        scored += len(sc)
        np.save(shard, sc)
        parts.append(sc)
        print(f"{name}: {min(s + SHARD, len(texts))}/{len(texts)} rows, {time.time() - t0:.0f} s",
              flush=True)  # fmt: skip
    p = letter_probs(np.concatenate(parts))
    df = pd.DataFrame(p, columns=[f"p_{c}" for c in CLASSES]).round(5)
    df.insert(0, "id", ids)
    df.to_csv(final, index=False)
    import torch
    import transformers

    score_s = max(time.time() - t1, 1e-9)
    rps = scored / score_s if scored else None  # None if every shard came from an earlier run
    peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
            for i in range(torch.cuda.device_count())]  # fmt: skip
    meta = {
        "teacher": name, "repo": spec["repo"], "four_bit": spec["four_bit"],
        "revision": getattr(model.config, "_commit_hash", None),
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256(),
        "rows": len(ids), "batch": BATCH, "load_seconds": round(load_s),
        "score_seconds": round(score_s), "rows_per_second": round(rps, 2) if rps else None,
        "projected_pool_hours": round(POOL_ROWS / rps / 3600, 1) if rps else None,
        "peak_gpu_gib": peak, "prefix_cache": cache_info,
        "top_class_counts": df.drop(columns="id").idxmax(axis=1).str[2:].value_counts().to_dict(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }  # fmt: skip
    (out_dir / f"{tag}_{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(name, json.dumps(meta["top_class_counts"]), flush=True)
    del scorer, model  # the scorer closure also holds the model: free both before the next teacher
    gc.collect()
    torch.cuda.empty_cache()


def find_input(filename: str, roots: tuple[str, ...] = ("/kaggle/input", ".")) -> Path:
    """Find an input file under the roots. Kaggle may unpack a .gz upload, so `pool.csv.gz` is
    also looked for as `pool.csv`; names are matched without regard to case."""
    from glob import glob

    wanted = {filename.lower(), filename.lower().removesuffix(".gz")}
    seen = []
    for root in roots:
        for hit in sorted(glob(f"{root}/**/*", recursive=True)):
            path = Path(hit)
            if path.is_file():
                seen.append(hit)
                if path.name.lower() in wanted:
                    return path
    shown = "\n  ".join(seen[:30]) or "(nothing found)"
    raise SystemExit(
        f"{filename} not found under {roots}. Files seen:\n  {shown}\n"
        "Add the dataset to this notebook (right panel, Add Input)."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=list(STAGES), default=STAGE)
    p.add_argument("--name", help="output tag (default: the stage name)")
    p.add_argument("--data", type=Path, help="input file (default: found in /kaggle/input)")
    p.add_argument("--id-col")
    p.add_argument("--text-col", default="text")
    p.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--limit", type=int, help="first N rows only")
    p.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS, choices=list(TEACHERS))
    args, _ = p.parse_known_args()  # Kaggle may add its own arguments
    if any(TEACHERS[t]["four_bit"] for t in args.teachers):
        ensure_bitsandbytes()
    stage = STAGES[args.stage]
    name = args.name or args.stage
    data = args.data or find_input(stage["file"])
    id_col = args.id_col or stage["id_col"]
    limit = args.limit if args.limit is not None else stage["limit"]

    check_input_name(data)
    df = pd.read_csv(data, dtype=str)
    if limit:
        df = df.head(limit)
    ids, texts = df[id_col].tolist(), df[args.text_col].fillna("").tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {len(texts)} texts from {data}, prompt {PROMPT_VERSION} "
          f"{prompt_sha256()[:12]}", flush=True)  # fmt: skip
    for teacher in args.teachers:
        run_teacher(teacher, TEACHERS[teacher], ids, texts, args.out, name)


if __name__ == "__main__":
    main()
