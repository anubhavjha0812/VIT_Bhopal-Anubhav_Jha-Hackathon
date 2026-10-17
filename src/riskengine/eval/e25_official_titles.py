"""Small real, unlabelled 2010-2016 official-title sample. No article bodies or training.

uv run python -m riskengine.eval.e25_official_titles --dry-run  # counts; writes nothing
uv run python -m riskengine.eval.e25_official_titles --build    # cache, CSV and manifest

Only English archive indexes and title/date/link Search API fields are requested. Raw HTTP
responses stay in the ignored cache. A blocked host is stopped, never retried via a mirror,
different identity or proxy. Fed is skipped because the earlier source check returned 403.
The cue lists below are fixed sampling rules, never event labels. Official titles are formal
and differ in style from news. English screening is conservative and rule based, not a label
model: English source routes, Latin script and English title words; uncertain titles are dropped.
"""

import argparse
import hashlib
import json
import re
import time
import unicodedata
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from riskengine.eval.e25_extra import EXTRA_PATH
from riskengine.eval.e25_pool import FINAL_FILES, POOL_PATH, assert_clean, exclusions, remove_seen
from riskengine.eval.leaks import NEAR, norm
from riskengine.paths import DATA, ROOT

SEED, PER_CLASS = 25, 400
START, END = "2010-01-01", "2017-01-01"  # end exclusive; original publication dates only
CACHE = DATA / "cache/e25/official"
SAMPLE = DATA / "cache/e25/official_titles.csv"
MANIFEST = DATA / "outputs/e25_official_titles_manifest.json"
COLUMNS = ["id", "text", "source", "date", "url", "cue_class"]
INPUT_COLUMNS = ["text", "source", "date", "url"]
USER_AGENT = "E25OfficialTitlesResearch/1.0 (title-only noncommercial research)"
MIN_INTERVAL = 1.0  # one request per second, including robots and redirects
SOURCES = {
    "fed": {
        "attribution": "Board of Governors of the Federal Reserve System",
        "cue_class": "Macro",
        "licence": "Public domain Board material, except identified third-party material",
        "terms_url": "https://www.federalreserve.gov/disclaimer.htm",
        "conditions": "Cite the Board; exclude third-party material and protected seals/logos.",
        "date_field": "press-release URL date",
    },
    "ecb": {
        "attribution": "European Central Bank, Frankfurt am Main, Germany",
        "cue_class": "Macro",
        "licence": "ECB website copyright and free-use terms (not Creative Commons)",
        "terms_url": "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html",
        "conditions": "Cite ECB; reproduce accurately; disclose changes. For sold documents, "
        "tell buyers the information is available free from ECB. No framing. Exclude "
        "named-author papers, which need permission; only institutional press releases here.",
        "date_field": "press-release URL date",
    },
    "govuk_fco": {
        "attribution": "Foreign & Commonwealth Office, GOV.UK; Crown copyright",
        "cue_class": "Geo",
        "licence": "Open Government Licence v3.0, subject to its exclusions",
        "licence_url": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        "terms_url": "https://www.gov.uk/help/reuse-govuk-content",
        "conditions": "Contains public sector information licensed under the Open Government "
        "Licence v3.0. Retain attribution and licence link; no endorsement; excludes third-party "
        "rights, personal data, logos and other OGL exceptions.",
        "date_field": "Search API public_timestamp (update time, never before first "
        "publication, so a date <= 2016 is a safe upper bound)",
    },
}

# Fixed before fetching or viewing any sampled titles. Cues select rows, never assign labels.
MACRO_CUES = (
    r"\b(?:rate decisions?|(?:interest|policy|key(?: ECB)?) rates?|monetary polic(?:y|ies)"
    r"|inflation|GDP|gross domestic product|employment|unemployment|nonfarm payrolls?"
    r"|balance of payments|exchange rates?|oil prices?|financial stability)\b"
)
GEO_CUES = (
    r"\b(?:sanctions?|conflicts?|cease[ -]?fires?|military|missiles?|treat(?:y|ies)"
    r"|summits?|election observers?)\b"
)
# Country names alone do not qualify. This branch also requires a minister and statement cue.
COUNTRIES = (
    r"Afghanistan|Albania|Algeria|Argentina|Armenia|Australia|Austria|Azerbaijan|Bahrain"
    r"|Bangladesh|Belarus|Belgium|Bosnia(?: and Herzegovina)?|Brazil|Britain|Bulgaria|Burma"
    r"|Myanmar|Burundi|Cambodia|Cameroon|Canada|Central African Republic|Chad|Chile|China"
    r"|Colombia|Congo|Croatia|Cuba|Cyprus|Denmark|Egypt|Eritrea|Estonia|Ethiopia|Fiji"
    r"|Finland|France|Gambia|Georgia|Germany|Ghana|Greece|Guatemala|Guinea|Haiti|Honduras"
    r"|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|Ivory Coast|Japan"
    r"|Jordan|Kazakhstan|Kenya|Kosovo|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Liberia|Libya"
    r"|Lithuania|Macedonia|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Mauritania|Mauritius"
    r"|Mexico|Moldova|Mongolia|Montenegro|Morocco|Mozambique|Nepal|Netherlands|New Zealand"
    r"|Nicaragua|Niger|Nigeria|North Korea|DPRK|Norway|Oman|Pakistan|Palestine|Panama|Peru"
    r"|Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Saudi Arabia|Senegal|Serbia"
    r"|Sierra Leone|Singapore|Slovakia|Slovenia|Somalia|South Africa|South Korea|South Sudan"
    r"|Spain|Sri Lanka|Sudan|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand"
    r"|Tunisia|Turkey|Turkmenistan|Uganda|Ukraine|United Arab Emirates|United Kingdom"
    r"|United States|Uzbekistan|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe"
)
MINISTER = (
    r"\b(?:foreign (?:secretary|minister)|FCO minister|minister for "
    r"(?:the )?(?:Middle East|Asia|Africa|Europe))\b"
)
STATEMENT = r"\b(?:statements?|says?|comments?|condemns?|welcomes?|urges?|calls?|on)\b"
ENGLISH_WORDS = set(
    (
        "the of on in and to for with at from by as a an its is new following after over about "
        "announces publishes welcomes condemns statement statements calls says urges expresses "
        "extends agrees supports launches holds raises cuts decisions monetary policy inflation "
        "employment unemployment balance payments exchange rates oil prices financial stability "
        "sanctions conflict ceasefire military missile treaty summit election observers minister"
    ).split()
)
FOREIGN_WORDS = set(
    "le les des du une et avec pour sur dans aux der die das und zur zum fuer für mit "
    "el los las una del en con para por sobre il gli della che nel al alla di "
    "o os uma dos das em com pelo pela na no y yn wrth am gan ar".split()
)
EXTRA_EXCLUSIONS = {
    "E25 pool": POOL_PATH,
    "E25 extra": EXTRA_PATH,
    "R3 reserved texts": ROOT / "private_notes/annotation_uploads/task1/retest_set/to_label.csv",
    "credit held-out reserved texts": (
        ROOT / "private_notes/annotation_uploads/task1/credit_heldout/to_label.csv"
    ),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SourceUnavailable(RuntimeError):
    """Access, transport, robots, or schema failure; stop this source without a workaround."""


class Fetcher:
    """Polite, same-host HTTP fetch with raw response hashes and fail-closed robots checks."""

    def __init__(self, write: bool, cache: Path = CACHE, client=None, interval=MIN_INTERVAL):
        self.write, self.cache, self.interval = write, cache, interval
        self.client = client or httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT})
        self.last = 0.0
        self.robots: dict[str, RobotFileParser] = {}
        self.audit: dict[str, dict] = {}

    def raw(self, url: str) -> tuple[bytes, int, str]:
        key = sha(url.encode())
        meta_path, body_path = self.cache / f"{key}.json", self.cache / f"{key}.body"
        if meta_path.exists() and body_path.exists():
            meta, body = json.loads(meta_path.read_text()), body_path.read_bytes()
            if meta["url"] != url or meta["sha256"] != sha(body):
                raise SourceUnavailable("raw cache hash or URL mismatch")
        else:
            current = url
            for _ in range(6):
                time.sleep(max(0.0, self.interval - (time.monotonic() - self.last)))
                self.last = time.monotonic()
                try:
                    response = self.client.get(current, follow_redirects=False)
                except httpx.HTTPError as exc:
                    raise SourceUnavailable(f"{type(exc).__name__}: {exc}") from exc
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                target = urljoin(current, response.headers.get("location", ""))
                if urlsplit(target).netloc != urlsplit(url).netloc:
                    raise SourceUnavailable("redirect leaves the official source host")
                current = target
            else:
                raise SourceUnavailable("too many redirects")
            body = response.content
            meta = {
                "url": url,
                "final_url": current,
                "status": response.status_code,
                "sha256": sha(body),
                "bytes": len(body),
                "fetched_at_utc": datetime.now(UTC).isoformat(),
            }
            if self.write:
                self.cache.mkdir(parents=True, exist_ok=True)
                body_path.write_bytes(body)
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        self.audit[key] = meta
        return body, meta["status"], meta["final_url"]

    def get(self, url: str) -> bytes:
        host = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
        if host not in self.robots:
            body, status, _ = self.raw(host + "/robots.txt")
            if status not in (200, 404):
                raise SourceUnavailable(f"robots.txt HTTP {status}; source stopped")
            robot = RobotFileParser()
            robot.parse(body.decode("utf-8").splitlines() if status == 200 else [])
            self.robots[host] = robot
        robot = self.robots[host]
        if not robot.can_fetch(USER_AGENT, url):
            raise SourceUnavailable("robots.txt disallows this URL; source stopped")
        self.interval = max(self.interval, robot.crawl_delay(USER_AGENT) or 0)
        body, status, _ = self.raw(url)
        if status != 200:
            raise SourceUnavailable(f"HTTP {status}; source stopped (no bypass)")
        return body

    def close(self):
        self.client.close()


class ReleaseLinks(HTMLParser):
    """Read archive links only. Titles come from text nodes, dates from release URLs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.href, self.parts = "", []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href, self.parts = dict(attrs).get("href", ""), []

    def handle_data(self, data):
        if self.href:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, " ".join(" ".join(self.parts).split())))
            self.href, self.parts = "", []


def ecb_rows(body: bytes, base: str) -> list[dict]:
    parser = ReleaseLinks()
    parser.feed(body.decode("utf-8"))
    rows = []
    for href, title in parser.links:
        url = urljoin(base, href)
        match = re.search(
            r"/press/pr/date/(20\d{2})/html/(?:pr|ecb\.pr)(\d{6})[^/]*\.en\.html$", url
        )
        if match and urlsplit(url).netloc == "www.ecb.europa.eu":
            year, stamp = match.groups()
            date = f"{year}-{stamp[2:4]}-{stamp[4:6]}"
            if year[2:] == stamp[:2]:
                rows.append(dict(text=title, source="ecb", date=date, url=url))
    return rows


def fetch_ecb(fetcher: Fetcher) -> list[dict]:
    rows = []
    for year in range(2010, 2017):
        url = f"https://www.ecb.europa.eu/press/pr/date/{year}/html/index.en.html"
        found = ecb_rows(fetcher.get(url), url)
        if not found:
            # Public archive fragment used by the dynamic index, never tried after a block.
            url = f"https://www.ecb.europa.eu/press/pr/date/{year}/html/index_include.en.html"
            found = ecb_rows(fetcher.get(url), url)
        if not found:
            raise SourceUnavailable(f"ECB {year} index has no release links; schema check failed")
        rows.extend(found)
    return rows


def govuk_rows(payload: dict) -> list[dict]:
    rows = []
    for item in payload["results"]:
        # No first_published_at in the API; update time is never earlier than first publication.
        path = item.get("link", "")
        if not path.startswith("/government/news/") or re.search(r"\.[a-z]{2}(?:$|[/?])", path):
            continue
        rows.append(
            dict(
                text=item.get("title", ""),
                source="govuk_fco",
                date=item.get("public_timestamp"),
                url="https://www.gov.uk" + path,
            )
        )
    return rows


def fetch_govuk(fetcher: Fetcher) -> list[dict]:
    rows, start = [], 0
    while True:
        params = [
            ("filter_organisations[]", "foreign-commonwealth-office"),
            ("filter_format[]", "news_story"),
            ("filter_format[]", "press_release"),
            ("fields", "title"),
            ("fields", "link"),
            ("fields", "public_timestamp"),
            ("count", "500"),
            ("start", str(start)),
        ]
        url = "https://www.gov.uk/api/search.json?" + urlencode(params)
        payload = json.loads(fetcher.get(url))
        if not isinstance(payload.get("results"), list) or not isinstance(
            payload.get("total"), int
        ):
            raise SourceUnavailable("GOV.UK Search API schema changed")
        rows.extend(govuk_rows(payload))
        start += len(payload["results"])
        if start >= payload["total"]:
            break
        if not payload["results"]:
            raise SourceUnavailable("GOV.UK pagination ended before total")
    return rows


def collect(fetcher: Fetcher) -> tuple[pd.DataFrame, dict]:
    rows = []
    status = {
        "fed": {
            "status": "blocked",
            "rows": 0,
            "reason": "Prior archive request returned HTTP 403 (E25_REAL_SOURCES); "
            "skipped, no bypass.",
        }
    }
    for source, method in (("ecb", fetch_ecb), ("govuk_fco", fetch_govuk)):
        try:
            found = method(fetcher)
            rows.extend(found)
            status[source] = {"status": "fetched", "rows": len(found)}
        except (SourceUnavailable, ValueError, KeyError) as exc:
            # A partial/changed source is discarded rather than presented as a full archive.
            status[source] = {"status": "unavailable", "rows": 0, "reason": str(exc)}
    return pd.DataFrame(rows, columns=INPUT_COLUMNS), status


def english_title(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters or any("LATIN" not in unicodedata.name(c, "") for c in letters):
        return False
    words = set(re.findall(r"[a-zà-ž]+", text.lower()))
    # Require two English markers and no evidence of a translated title. False drops are OK.
    return len(words & ENGLISH_WORDS) >= 2 and not words & FOREIGN_WORDS


def cue_class(text: str, source: str) -> str:
    if source in ("fed", "ecb") and re.search(MACRO_CUES, text, re.I):
        return "Macro"
    if source == "govuk_fco":
        statement = all(
            re.search(p, text, re.I) for p in (MINISTER, STATEMENT, rf"\b(?:{COUNTRIES})\b")
        )
        if re.search(GEO_CUES, text, re.I) or statement:
            return "Geo"
    return ""


def eligible(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    f = raw[INPUT_COLUMNS].copy()
    count = {"raw_rows": len(f)}
    f = f.loc[f["text"].map(lambda t: isinstance(t, str) and len(t.strip()) >= 25).astype(bool)]
    count["after_min_length"] = len(f)
    dates = pd.to_datetime(f["date"], utc=True, errors="coerce", format="mixed")
    f = f[(dates >= START) & (dates < END)].copy()
    f["date"] = dates.loc[f.index].dt.strftime("%Y-%m-%d")
    count["after_original_date_window"] = len(f)
    f = f.loc[f["text"].map(english_title).astype(bool)].copy()
    count["after_english_screen"] = len(f)
    f["cue_class"] = [cue_class(t, s) for t, s in zip(f["text"], f["source"], strict=True)]
    f = f[f["cue_class"] != ""].sort_values(["date", "source", "url", "text"], kind="stable")
    count["after_cues"] = len(f)
    return f.reset_index(drop=True), count


def deduplicate(f: pd.DataFrame) -> pd.DataFrame:
    """Stable greedy normalized-exact and word 1-2-gram cosine >= 0.9 deduplication."""
    f = f.loc[~f["text"].map(norm).duplicated()].reset_index(drop=True)
    if len(f) < 2:
        return f
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(f["text"].map(norm))
    keep, rejected = [], set()
    for i in range(len(f)):
        if i in rejected:
            continue
        keep.append(i)
        scores = (matrix[i] @ matrix.T).tocoo()
        rejected.update(
            int(j)
            for j, score in zip(scores.col, scores.data, strict=True)
            if j > i and score >= NEAR
        )
    return f.iloc[keep].reset_index(drop=True)


def draw(f: pd.DataFrame, cap: int = PER_CLASS, seed: int = SEED) -> pd.DataFrame:
    f = f.sort_values(["date", "source", "url", "text"], kind="stable").reset_index(drop=True)
    rng, selected = np.random.default_rng(seed), []
    for group in ("Macro", "Geo"):
        positions = np.flatnonzero(f["cue_class"].eq(group).to_numpy())
        selected.extend(rng.permutation(positions)[:cap].tolist())
    return f.iloc[sorted(selected)].reset_index(drop=True)


def clean_sample(raw: pd.DataFrame, corpus: dict, cap=PER_CLASS, seed=SEED) -> tuple:
    f, audit = eligible(raw)
    f = deduplicate(f)
    audit["after_initial_dedup"] = len(f)
    passes = []
    # Existing helpers do not accept zero candidates or an empty reference corpus.
    corpus = {key: texts for key, texts in corpus.items() if texts}
    if len(f):
        f, info = remove_seen(f, corpus)
        passes.append(info)
    audit["before_cap"] = len(f)
    f = draw(f, cap, seed)
    # Refitting TF-IDF after shrinking can expose new near matches. Fail closed on instability.
    for _ in range(25):
        size = len(f)
        f = deduplicate(f)
        if len(f):
            f, info = remove_seen(f, corpus)
            passes.append(info)
        if len(f) == size:
            break
    else:
        raise RuntimeError("duplicate/exclusion checks did not stabilize")
    if len(f):
        assert_clean(f, corpus)
    if len(deduplicate(f)) != len(f):
        raise RuntimeError("sample contains near duplicates")
    f.insert(0, "id", [f"ot{k:06d}" for k in range(len(f))])
    audit["exclusion_passes"] = passes
    audit["sample_rows"] = len(f)
    return f[COLUMNS], audit


def load_exclusions() -> dict[str, list[str]]:
    from riskengine.eval.e24 import exclusion_corpus

    # Stable aliases preserve the existing exclusion policy without exposing source text.
    corpus = {
        f"fixed_exclusions_{i:02d}": texts
        for i, texts in enumerate(exclusions(exclusion_corpus(), FINAL_FILES).values())
    }
    for name, path in EXTRA_EXCLUSIONS.items():
        corpus[name] = pd.read_csv(path, usecols=["text"])["text"].astype(str).tolist()
    return corpus


def build(write: bool) -> dict:
    fetcher = Fetcher(write)
    try:
        raw, source_status = collect(fetcher)
    finally:
        fetcher.close()
    sample = pd.DataFrame(columns=COLUMNS)
    audit, cleared, error = {}, [], None
    if len(raw):
        try:
            corpus = load_exclusions()
            sample, audit = clean_sample(raw, corpus)
            cleared = sorted(corpus)
        except (OSError, ValueError, RuntimeError) as exc:
            error = f"Exclusion/build failed; no usable sample: {type(exc).__name__}: {exc}"
    else:
        error = "No source titles fetched. See source access failures; no usable sample built."
    csv = sample.to_csv(index=False, lineterminator="\n").encode("utf-8")
    info = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "blocked" if error else ("built" if len(sample) else "empty"),
        "error": error,
        "sample_rows": len(sample),
        "by_cue_class": {k: int(sample["cue_class"].eq(k).sum()) for k in ("Macro", "Geo")},
        "by_source": sample["source"].value_counts().to_dict(),
        "by_year": sample["date"].str[:4].value_counts().sort_index().to_dict(),
        "seed": SEED,
        "cap_per_cue_class": PER_CLASS,
        "window": [START, END],
        "cue_class_is_label": False,
        "labels_created": False,
        "training_run": False,
        "style_note": "Official press-release titles are formal; style differs from news.",
        "language_rule": "English source routes plus conservative Latin-script/English-word "
        "screen; uncertain/non-English titles dropped; not a general language detector.",
        "modifications": "Title-only extraction, HTML entity decoding and index whitespace "
        "normalization; no paraphrasing or translation. Date standardized to UTC calendar date.",
        "cues": {
            "Macro": MACRO_CUES,
            "Geo": GEO_CUES,
            "minister": MINISTER,
            "statement": STATEMENT,
            "countries": COUNTRIES,
        },
        "sources": {k: SOURCES[k] | source_status[k] for k in SOURCES},
        "sample_path": str(SAMPLE.relative_to(ROOT)),
        "sample_sha256": sha(csv),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "raw_cache_path": str(CACHE.relative_to(ROOT)),
        "fetch_policy": "One request per second; honor robots and longer crawl delays; "
        "stop each unavailable/blocked source; no access bypass. Dry-run writes nothing.",
        "raw_responses": fetcher.audit,
        "leak_check": "passed" if cleared else "not_run",
        "cleared_against": cleared,
        "audit": audit,
    }
    if write:
        if error and SAMPLE.exists() and len(pd.read_csv(SAMPLE)):
            raise RuntimeError("build failed; preserving the existing nonempty sample and manifest")
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE.write_bytes(csv)  # A blocked run writes a header only, explicitly not a real sample.
        MANIFEST.write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--build", action="store_true")
    args = parser.parse_args()
    info = build(write=args.build)
    # Counts/status only: no title text, URLs, exclusion texts or pseudo-labels in CLI output.
    print(
        json.dumps(
            {
                key: info[key]
                for key in (
                    "status",
                    "sample_rows",
                    "by_cue_class",
                    "by_source",
                    "sample_sha256",
                    "error",
                )
            },
            indent=2,
        )
    )
    if info["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
