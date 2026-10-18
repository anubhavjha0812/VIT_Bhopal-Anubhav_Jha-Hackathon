"""Sample local news using SEC 8-K anchors, without labels or model inference.

SEC_USER_AGENT='project-name contact@example.org' uv run python -m riskengine.eval.e25_sec8k --fetch
uv run python -m riskengine.eval.e25_sec8k --dry-run  # SEC cache counts only; no writes/network
uv run python -m riskengine.eval.e25_sec8k --build    # local news + standard E25 exclusions
uv run python -m riskengine.eval.e25_sec8k --build --news-full  # full CSV, cached exclusions

Only --fetch calls SEC. Build reuses the normal E25 exclusion loader, including its usual
training-corpus retrieval. Item codes select candidates; they never determine a news label.
Full-news builds are offline and require a local snapshot of the original exclusion texts.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from riskengine.eval.e25_pool import FINAL_FILES, assert_clean, exclusions, remove_seen
from riskengine.eval.leaks import norm
from riskengine.paths import DATA, ROOT

ITEMS = ("1.03", "2.04")
START, END = "2010-01-01", "2020-06-30"
SEED, CAP = 28, 150
ENDPOINT = "https://efts.sec.gov/LATEST/search-index"
INTERVAL = 0.21  # Start requests at least 210 ms apart: below five requests/second.
SEARCH_LIMIT = 10000
CACHE = DATA / "cache/e25/sec8k"
NEWS = (
    DATA / "raw/fnspid/universe_news_named.csv.gz",
    DATA / "raw/fnspid/universe_news.csv.gz",
)
FULL_NEWS = DATA / "raw/fnspid/All_external.csv"
CHUNK_ROWS = 50000
TICKERS = DATA / "raw/sec/company_tickers.json"
OUT = ROOT / "private_notes/annotation_uploads/task1/credit_8k"
MANIFEST = DATA / "outputs/e25_sec8k_manifest.json"
FULL_OUT = ROOT / "private_notes/annotation_uploads/task1/credit_8k_full"
FULL_MANIFEST = DATA / "outputs/e25_sec8k_full_manifest.json"
EXCLUSIONS_CACHE = DATA / "cache/e25/sec8k_exclusions.json"
EXTRA_EXCLUSIONS = {
    "R3": ROOT / "private_notes/annotation_uploads/task1/retest_set/to_label.csv",
    "credit held-out": ROOT / "private_notes/annotation_uploads/task1/credit_heldout/to_label.csv",
    "E25 pool": DATA / "cache/e25/pool.csv.gz",
    "E25 extra": DATA / "cache/e25/extra.csv.gz",
}
NEWS_COLUMNS = ["Date", "Article_title", "Stock_symbol"]
BLIND_COLUMNS = ["id", "text", "date", "tickers"]
ANCHOR_COLUMNS = ["id", "accession", "item", "company", "file_date"]
FILING_COLUMNS = ["accession", "item", "company", "file_date", "cik", "tickers", "items"]
WRAP_PATTERN = (
    r"\b(?:stocks?\s+to\s+watch|market[\s-]+(?:wrap|round[\s-]?up)|"
    r"wall\s+street[\s-]+(?:wrap|round[\s-]?up)|"
    r"(?:pre[\s-]?market|midday|after[\s-]?hours|morning|afternoon)"
    r"[\s-]+(?:movers|market[\s-]+(?:update|wrap|round[\s-]?up)))\b"
)
TICKER_TOKEN = re.compile(r"(?<![\w$])\$?([A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)?)(?![\w])")
NON_NEWS_RULE = (
    "Fixed before scanning: drop a row if its title matches the case-insensitive wrap regex, "
    "or title plus Stock_symbol contain at least 3 distinct literal uppercase tickers from "
    "the existing SEC ticker map and cached filing display-name tickers. No row-by-row review."
)
RULE = (
    "8-K filings 2010-01-01 through 2020-06-30; client-side exact item recheck; "
    "local FNSPID headline dated filing-2 through filing+5 days, inclusive; "
    "normalized issuer name without legal suffixes or literal uppercase ticker in text; "
    "unique normalized headline, earliest matching date; standard E25 leak exclusions "
    "plus R3, credit held-out, pool and extra; seeded hash draw, at most 150 headlines/item"
)
ANCHOR_NOTE = (
    "8-K item codes are anchors for sampling, not labels. Headline text still gets labelled "
    "blind. The labeller reads to_label.csv only, never anchors.csv or this manifest. "
    "This is a separate supplementary credit set, not E24 or a tuning set."
)


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def page_hits(payload: dict) -> tuple[list[dict], int]:
    """Reject malformed/truncated responses rather than calling them empty results."""
    hits = payload["hits"]["hits"]
    total = payload["hits"]["total"]
    total = total["value"] if isinstance(total, dict) else total
    if not isinstance(hits, list) or not isinstance(total, int) or total < len(hits):
        raise ValueError("Malformed SEC hits/total")
    return hits, total


class SecClient:
    """An identified, rate-limited SEC client; HTTP errors stop the fetch."""

    def __init__(self, user_agent: str, client=None, clock=time.monotonic, sleep=time.sleep):
        email = re.search(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", user_agent)
        if not email or not user_agent[: email.start()].strip():
            raise ValueError("SEC_USER_AGENT needs a project description and contact email")
        if "\n" in user_agent or "\r" in user_agent:
            raise ValueError("SEC_USER_AGENT must be one line")
        self.user_agent, self.client = user_agent, client
        self.clock, self.sleep, self.last_start = clock, sleep, None

    def get(self, item: str, start: str, end: str, offset: int) -> dict:
        if self.last_start is not None:
            self.sleep(max(0.0, INTERVAL - (self.clock() - self.last_start)))
        self.last_start = self.clock()
        params = {
            "forms": "8-K",
            "items": item,
            "dateRange": "custom",
            "startdt": start,
            "enddt": end,
            "from": offset,
        }
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if self.client is None:
            response = httpx.get(ENDPOINT, params=params, headers=headers, timeout=60)
        else:
            response = self.client.get(ENDPOINT, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        page_hits(payload)
        return payload


def fetch(cache: Path, client: SecClient) -> dict:
    """Cache raw JSON and an index. Bisect saturated date windows before paging."""
    if (cache / "index.json").exists():
        return read_cache(cache)[1]  # A complete snapshot is reused, not silently refreshed.
    queries = []

    def page(item, start, end, offset):
        name = f"{item}_{start}_{end}_{offset:06d}.json"
        path = cache / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = client.get(item, start, end, offset)
            write_json(path, payload)
        hits, total = page_hits(payload)
        return hits, total, {"file": name, "from": offset, "sha256": sha_file(path)}

    def window(item, start, end):
        first, total, record = page(item, start, end, 0)
        if total >= SEARCH_LIMIT:
            a, b = date.fromisoformat(start), date.fromisoformat(end)
            if a == b:
                raise RuntimeError(f"SEC search limit reached for {item} on {start}")
            middle = a + (b - a) // 2
            window(item, start, middle.isoformat())
            window(item, (middle + timedelta(days=1)).isoformat(), end)
            return
        pages, offset = [record], len(first)
        if total and not first:
            raise RuntimeError("SEC returned an empty page before the reported total")
        while offset < total:
            hits, new_total, record = page(item, start, end, offset)
            if new_total != total or not hits or offset + len(hits) > total:
                raise RuntimeError("SEC result count changed or pagination is incomplete")
            pages.append(record)
            offset += len(hits)
        queries.append({"item": item, "start": start, "end": end, "total": total, "pages": pages})

    for item in ITEMS:
        for year in range(2010, 2021):
            window(item, f"{year}-01-01", min(f"{year}-12-31", END))
    write_json(
        cache / "index.json",
        {
            "version": 1,
            "window": [START, END],
            "items": ITEMS,
            "endpoint": ENDPOINT,
            "complete": True,
            "queries": queries,
        },
    )
    return read_cache(cache)[1]


def parse_hit(hit: dict, wanted: str) -> list[dict]:
    """Parse issuer metadata only after checking the actual _source.items list."""
    source = hit.get("_source", {})
    items = source.get("items")
    if not isinstance(items, list) or wanted not in items:
        return []
    filed = date.fromisoformat(source["file_date"][:10]).isoformat()
    if not START <= filed <= END:
        raise ValueError("Filing outside the registered window")
    accession = source.get("adsh", str(hit.get("_id", "")).split(":")[0])
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        raise ValueError("Missing or invalid accession")
    names, ciks = source.get("display_names", []), source.get("ciks", [])
    if not names:
        raise ValueError("Missing company name")
    rows = []
    for i, display in enumerate(names):
        cik_match = re.search(r"\(CIK\s+(\d+)\)", display, re.I)
        if cik_match:
            cik = str(int(cik_match[1]))
        elif len(ciks) == len(names) or len(ciks) == 1:
            cik = str(int(ciks[i if len(ciks) == len(names) else 0]))
        else:
            raise ValueError("Cannot pair issuer names with CIKs")
        company = re.sub(r"\s*\(CIK\s+\d+\)", "", display, flags=re.I).strip()
        ticker = re.search(r"\s+\(([A-Z][A-Z0-9.-]{0,9})\)$", company)
        symbols = ticker[1] if ticker else ""
        if ticker:
            company = company[: ticker.start()].strip()
        if not company:
            raise ValueError("Empty issuer name")
        rows.append(
            {
                "accession": accession,
                "item": wanted,
                "company": company,
                "file_date": filed,
                "cik": cik,
                "tickers": symbols,
                "items": " ".join(sorted(set(items))),
            }
        )
    return rows


def read_cache(cache: Path) -> tuple[pd.DataFrame, dict]:
    """Cache-only: verify hashes, date coverage and pagination, then recheck every item."""
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    if (
        index.get("version") != 1
        or index.get("complete") is not True
        or index.get("window") != [START, END]
        or index.get("items") != list(ITEMS)
        or index.get("endpoint") != ENDPOINT
    ):
        raise ValueError("Incomplete or incompatible SEC cache index")
    for item in ITEMS:
        windows = sorted((q["start"], q["end"]) for q in index["queries"] if q["item"] == item)
        next_date = date.fromisoformat(START)
        for start, end in windows:
            if date.fromisoformat(start) != next_date or end < start:
                raise ValueError("SEC cache date coverage has a gap or overlap")
            next_date = date.fromisoformat(end) + timedelta(days=1)
        if next_date != date.fromisoformat(END) + timedelta(days=1):
            raise ValueError("SEC cache does not cover the registered dates")
    rows, counts, hashes = [], {}, {"index.json": sha_file(cache / "index.json")}
    for query in index["queries"]:
        wanted, offset = query["item"], 0
        if wanted not in ITEMS or not query["pages"] or query["total"] >= SEARCH_LIMIT:
            raise ValueError("Invalid SEC query in cache index")
        stats = counts.setdefault(
            wanted, {"hits": 0, "dropped_item_hits": 0, "dropped_metadata_hits": 0, "kept_hits": 0}
        )
        for record in query["pages"]:
            name = record["file"]
            if Path(name).name != name or record["from"] != offset:
                raise ValueError("Invalid SEC page path or offset")
            path = cache / name
            digest = sha_file(path)
            if digest != record["sha256"]:
                raise ValueError(f"SEC cache hash mismatch: {name}")
            hashes[name] = digest
            hits, total = page_hits(json.loads(path.read_text(encoding="utf-8")))
            if total != query["total"] or (not hits and total):
                raise ValueError("Incomplete SEC cached page")
            offset += len(hits)
            for hit in hits:
                stats["hits"] += 1
                items = hit.get("_source", {}).get("items")
                if not isinstance(items, list) or wanted not in items:
                    stats["dropped_item_hits"] += 1
                    continue
                try:
                    parsed = parse_hit(hit, wanted)
                    if any(not query["start"] <= r["file_date"] <= query["end"] for r in parsed):
                        raise ValueError("Filing date disagrees with query window")
                except (ValueError, KeyError, TypeError):
                    stats["dropped_metadata_hits"] += 1
                    continue
                stats["kept_hits"] += 1
                rows.extend(parsed)
        if offset != query["total"]:
            raise ValueError("SEC cache pagination did not reach the total")
    filings = pd.DataFrame(rows, columns=FILING_COLUMNS).drop_duplicates()
    filings = filings.sort_values(FILING_COLUMNS, kind="stable").reset_index(drop=True)
    return filings, {
        "status": "cached",
        "counts_per_item": counts,
        "filings": len(filings),
        "sha256": hashes,
    }


def company_key(company: str) -> str:
    """Normalize punctuation/accents and strip repeated legal suffixes, not business words."""
    value = unicodedata.normalize("NFKD", company).encode("ascii", "ignore").decode()
    tokens = norm(value.replace("&", " and ")).split()
    suffixes = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "llc",
        "llp",
        "lp",
        "plc",
        "sa",
        "nv",
    }
    # Punctuated L.L.C., L.P. and S.A. need collapsing before suffix removal.
    value = " ".join(tokens)
    value = re.sub(r"\b(l l c|l l p|l p|s a|n v)\b", lambda m: m[0].replace(" ", ""), value)
    tokens = value.split()
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    return " ".join(tokens)


def ticker_map(path: Path) -> dict[str, set[str]]:
    """Use the existing SEC ticker cache; historical display-name tickers also work."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.values() if isinstance(payload, dict) else payload
    out = defaultdict(set)
    for row in records:
        out[str(int(row["cik_str"]))].add(str(row["ticker"]).upper())
    return dict(out)


def matcher_index(filings: pd.DataFrame, symbols: dict) -> tuple:
    """Build the unchanged issuer/ticker matching index once for a streamed source."""
    token_index = defaultdict(set)
    prepared = []
    for row in filings.to_dict("records"):
        key = company_key(row["company"])
        tickers = sorted(set(row["tickers"].split()) | symbols.get(row["cik"], set()))
        pattern = (
            re.compile(r"(?<![\w$])\$?(?:" + "|".join(map(re.escape, tickers)) + r")(?![\w])")
            if tickers
            else None
        )
        i = len(prepared)
        filed = date.fromisoformat(row["file_date"])
        prepared.append(
            (row, key, tickers, pattern, filed - timedelta(days=2), filed + timedelta(days=5))
        )
        if key:
            token_index[max(key.split(), key=lambda t: (len(t), t))].add(i)
        for ticker in tickers:
            token_index[norm(ticker).split()[0]].add(i)
    return prepared, token_index


def match_candidates(
    news: pd.DataFrame, filings: pd.DataFrame, symbols: dict, *, index: tuple | None = None
) -> pd.DataFrame:
    """Match headline text, never Stock_symbol alone. A token index avoids a full cross join."""
    prepared, token_index = index if index is not None else matcher_index(filings, symbols)
    dates = pd.to_datetime(news["Date"], format="mixed", utc=True, errors="coerce")
    matched = []
    for text, published in zip(news["Article_title"].fillna(""), dates, strict=True):
        text = str(text).strip()
        if pd.isna(published) or not norm(text):
            continue
        day, key = published.date(), norm(text)
        normalized = norm(
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode()
            .replace("&", " and ")
        )
        possible = set()
        for token in normalized.split():
            possible.update(token_index.get(token, ()))
        for i in sorted(possible):
            row, name, tickers, pattern, first_day, last_day = prepared[i]
            if not first_day <= day <= last_day:
                continue
            if (name and f" {name} " in f" {normalized} ") or (pattern and pattern.search(text)):
                matched.append(
                    {
                        "text": text,
                        "date": day.isoformat(),
                        "tickers": " ".join(tickers),
                        "key": key,
                        **{c: row[c] for c in FILING_COLUMNS if c != "tickers"},
                    }
                )
    columns = ["text", "date", "tickers", "key", *[c for c in FILING_COLUMNS if c != "tickers"]]
    return pd.DataFrame(matched, columns=columns)


def remove_non_news(news: pd.DataFrame, known_tickers: set[str]) -> tuple[pd.DataFrame, dict]:
    """A fixed title/ticker screen, applied mechanically before issuer matching."""
    titles = news["Article_title"].fillna("").astype(str)
    symbols = news["Stock_symbol"].fillna("").astype(str)
    wraps = titles.str.contains(WRAP_PATTERN, case=False, regex=True)
    multi = pd.Series(
        [
            len(set(TICKER_TOKEN.findall(title + " " + stocks)) & known_tickers) >= 3
            for title, stocks in zip(titles, symbols, strict=True)
        ],
        index=news.index,
        dtype=bool,
    )
    drop = wraps | multi
    return news.loc[~drop], {
        "rows": len(news),
        "dropped_wrap_title": int(wraps.sum()),
        "dropped_three_or_more_tickers": int(multi.sum()),
        "dropped_total": int(drop.sum()),
    }


def headline_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """One blind text/date with all eligible filing anchors, including multiple item codes."""
    rows = []
    for key, group in matches.groupby("key", sort=True):
        group = group[group["date"].eq(group["date"].min())].sort_values(
            ["text", "item", "accession", "company"], kind="stable"
        )
        anchors = group[FILING_COLUMNS].drop(columns="tickers", errors="ignore")
        rows.append(
            {
                "key": key,
                "text": group.iloc[0]["text"],
                "date": group.iloc[0]["date"],
                "tickers": " ".join(sorted({t for v in group["tickers"] for t in v.split()})),
                "anchors": anchors.drop_duplicates().to_dict("records"),
            }
        )
    return pd.DataFrame(rows, columns=["key", "text", "date", "tickers", "anchors"])


def clean(frame: pd.DataFrame, corpus: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Use E25's exact/near filter until stable; assert again on the actual output frame."""
    corpus = {name: texts for name, texts in corpus.items() if texts}
    audit = []
    while len(frame):
        smaller, info = remove_seen(frame, corpus)
        audit.append(info)
        if len(smaller) == len(frame):
            assert_clean(smaller, corpus)
            return smaller, audit
        frame = smaller
    return frame, audit


def sample(frame: pd.DataFrame, corpus: dict, cap: int = CAP, seed: int = SEED) -> tuple:
    if cap < 1:
        raise ValueError("cap must be positive")
    frame, before = clean(frame, corpus)
    selected = defaultdict(set)
    for item in ITEMS:
        eligible = frame.loc[
            frame["anchors"]
            .map(lambda rows, wanted=item: any(a["item"] == wanted for a in rows))
            .astype(bool)
        ]
        keys = sorted(
            eligible["key"],
            key=lambda k: hashlib.sha256(f"{seed}\t{item}\t{k}".encode()).hexdigest(),
        )
        for key in keys[:cap]:
            selected[key].add(item)
    frame = frame[frame["key"].isin(selected)].copy()
    frame, after = clean(frame, corpus)
    frame = frame.sort_values(["date", "key", "text"], kind="stable").reset_index(drop=True)
    blind, anchors = [], []
    for i, row in enumerate(frame.to_dict("records"), 1):
        identifier = f"S8K{i:04d}"
        blind.append({"id": identifier, **{c: row[c] for c in BLIND_COLUMNS[1:]}})
        for anchor in row["anchors"]:
            if anchor["item"] in selected[row["key"]]:
                anchors.append({"id": identifier, **anchor})
    blind = pd.DataFrame(blind, columns=BLIND_COLUMNS)
    detailed = pd.DataFrame(anchors, columns=["id", *[c for c in FILING_COLUMNS if c != "tickers"]])
    return blind, detailed, {"before_draw": before, "after_draw": after}


def load_exclusions() -> dict:
    """Read text columns through the shared E25 exclusion path, never predictions/labels."""
    from riskengine.eval.e24 import exclusion_corpus

    corpus = {
        f"fixed_exclusions_{i:02d}": texts
        for i, texts in enumerate(exclusions(exclusion_corpus(), FINAL_FILES).values())
    }
    for name, path in EXTRA_EXCLUSIONS.items():
        corpus[name] = pd.read_csv(path, usecols=["text"], dtype=str)["text"].dropna().tolist()
    return corpus


def load_cached_exclusions(path: Path, baseline: Path) -> dict:
    """Offline only; require the exact original exclusion groups, counts and text hashes.

    A snapshot is JSON written from load_exclusions(), optionally wrapped as {"corpus": ...}.
    Matching the existing manifest prevents a partial export from weakening the leak filter.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Offline E25 exclusion snapshot missing: {path}. Supply --exclusions-cache <JSON> "
            "containing the original load_exclusions() text lists; no download is attempted."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Offline exclusion snapshot must be a JSON object")
    corpus = payload.get("corpus", payload)
    expected = json.loads(baseline.read_text(encoding="utf-8"))["exclusions"]
    if not isinstance(corpus, dict) or set(corpus) != set(expected):
        raise ValueError("Offline exclusion snapshot must contain every original exclusion group")
    for name, texts in corpus.items():
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError(f"Invalid text-only exclusion group: {name}")
        digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        if len(texts) != expected[name]["rows"] or digest != expected[name]["sha256"]:
            raise ValueError(f"Offline exclusion snapshot does not match original group: {name}")
    return corpus


def build(
    cache: Path,
    news_paths: tuple,
    tickers_path: Path,
    output: Path,
    manifest: Path,
    *,
    news_full: bool = False,
    exclusions_cache: Path = EXCLUSIONS_CACHE,
    exclusions_baseline: Path = MANIFEST,
) -> dict:
    targets = [output / "to_label.csv", output / "anchors.csv", manifest]
    if any(path.exists() for path in targets):
        raise FileExistsError("SEC sample outputs already exist; preserve the existing sample")
    filings, cached = read_cache(cache)
    if filings.empty:
        raise ValueError("No valid item-checked filings in the SEC cache")
    corpus = (
        load_cached_exclusions(exclusions_cache, exclusions_baseline)
        if news_full
        else load_exclusions()
    )
    symbols, matches, source_hashes, source_counts = ticker_map(tickers_path), [], {}, {}
    index = matcher_index(filings, symbols)
    known_tickers = {ticker for group in symbols.values() for ticker in group}
    known_tickers |= {ticker for group in filings["tickers"] for ticker in group.split()}
    non_news = {
        "rows": 0,
        "dropped_wrap_title": 0,
        "dropped_three_or_more_tickers": 0,
        "dropped_total": 0,
    }
    chunks = 0
    for path in news_paths:
        source_hashes[str(path)] = sha_file(path)
        count = 0
        for chunk in pd.read_csv(path, usecols=NEWS_COLUMNS, dtype=str, chunksize=CHUNK_ROWS):
            count += len(chunk)
            chunks += 1
            if news_full:
                chunk, dropped = remove_non_news(chunk, known_tickers)
                for name, value in dropped.items():
                    non_news[name] += value
            matches.append(match_candidates(chunk, filings, symbols, index=index))
            if news_full and chunks % 10 == 0:
                print(
                    f"Full news: {count:,} rows streamed; "
                    f"{non_news['dropped_total']:,} fixed-rule drops",
                    file=sys.stderr,
                    flush=True,
                )
        source_counts[str(path)] = count
    combined = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=["key"])
    frame = headline_frame(combined)
    blind, detailed, audit = sample(frame, corpus)
    # The final shared assertion uses precisely the texts written for labelling.
    if len(blind):
        assert_clean(blind, {name: texts for name, texts in corpus.items() if texts})
    output.mkdir(parents=True, exist_ok=True)
    blind.to_csv(targets[0], index=False, lineterminator="\n")
    detailed[ANCHOR_COLUMNS].to_csv(targets[1], index=False, lineterminator="\n")
    info = {
        "rule": RULE,
        "seed": SEED,
        "cap_per_item": CAP,
        "filing_window": [START, END],
        "headline_window_days": [-2, 5],
        "note": ANCHOR_NOTE,
        "rows": len(blind),
        "counts_per_item": {
            item: int(detailed.loc[detailed["item"].eq(item), "id"].nunique()) for item in ITEMS
        },
        "matched_unique_headlines": len(frame),
        "sec_cache": cached,
        "leak_filter": audit,
        "sources_sha256": source_hashes,
        "source_rows": source_counts,
        "sha256": {str(p): sha_file(p) for p in targets[:2]},
        "script_sha256": sha_file(Path(__file__)),
        "ticker_cache_sha256": sha_file(tickers_path) if tickers_path.exists() else None,
        "exclusions": {
            name: {
                "rows": len(texts),
                "sha256": hashlib.sha256(
                    json.dumps(texts, ensure_ascii=False).encode()
                ).hexdigest(),
            }
            for name, texts in corpus.items()
        },
        "selected_filing_metadata": detailed.to_dict("records"),
    }
    if news_full:
        info |= {
            "news_source": "full_local_csv",
            "chunk_rows": CHUNK_ROWS,
            "chunks": chunks,
            "non_news_filter": {
                "rule": NON_NEWS_RULE,
                "wrap_regex": WRAP_PATTERN,
                "ticker_lexicon_sha256": hashlib.sha256(
                    json.dumps(sorted(known_tickers)).encode()
                ).hexdigest(),
                "counts": non_news,
            },
            "exclusion_snapshot": {
                "path": str(exclusions_cache),
                "sha256": sha_file(exclusions_cache),
                "baseline_manifest": str(exclusions_baseline),
                "baseline_sha256": sha_file(exclusions_baseline),
            },
        }
    write_json(manifest, info)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "fetch", "build"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument(
        "--news-full", action="store_true", help="stream All_external.csv with --build"
    )
    parser.add_argument(
        "--exclusions-cache",
        type=Path,
        default=EXCLUSIONS_CACHE,
        help="text-only local E25 exclusion snapshot for --news-full",
    )
    args = parser.parse_args()
    if args.news_full and not args.build:
        parser.error("--news-full requires --build")
    if args.dry_run:
        info = (
            read_cache(CACHE)[1]
            if (CACHE / "index.json").exists()
            else {
                "status": "not_fetched",
                "filings": 0,
                "counts_per_item": {},
            }
        )
    elif args.fetch:
        info = fetch(CACHE, SecClient(os.environ.get("SEC_USER_AGENT", "")))
    else:
        if args.news_full:
            info = build(
                CACHE,
                (FULL_NEWS,),
                TICKERS,
                FULL_OUT,
                FULL_MANIFEST,
                news_full=True,
                exclusions_cache=args.exclusions_cache,
            )
        else:
            info = build(CACHE, NEWS, TICKERS, OUT, MANIFEST)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
