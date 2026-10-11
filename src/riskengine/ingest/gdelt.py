"""Current GDELT headlines from DOC 2.0, with the latest GKG file as fallback."""

import html
import io
import re
import zipfile
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import httpx

from riskengine.ingest.linking import Linker

DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DATA_URL = "https://data.gdeltproject.org/gdeltv2/"
QUERY = (
    '(Apple OR Microsoft OR Google OR Nvidia OR Amazon OR Tesla OR Boeing OR JPMorgan '
    'OR "Meta Platforms" OR Facebook OR AMD OR Netflix OR Disney OR "Coca-Cola" '
    'OR Exxon OR "Bank of America" OR inflation OR "interest rates" OR "credit default" '
    'OR sanctions) sourcelang:english'
)
MARKET = re.compile(
    r"\b(inflation|interest rates?|central bank|federal reserve|recession|tariffs?|"
    r"sanctions?|sovereign debt|credit default|bond yields?|stock market|oil prices?)\b",
    re.IGNORECASE,
)
TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)


def article(title: str, url: str, stamp: str, domain: str, source: str) -> dict | None:
    """Normalise a headline; GDELT's timestamps are observation times, in UTC."""
    text = " ".join(html.unescape(title).split())
    if not text or urlsplit(url).scheme not in {"http", "https"}:
        return None
    try:
        observed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    return {
        "text": text,
        "url": url,
        "published_at": observed.isoformat(),
        "domain": domain or urlsplit(url).netloc,
        "source": "gdelt",
        "live_source": source,
    }


class GdeltClient:
    def __init__(self, linker: Linker, client: httpx.Client | None = None):
        self.linker = linker
        self.client = client or httpx.Client(timeout=45, follow_redirects=True)
        self.last_gkg: datetime | None = None

    def relevant(self, text: str) -> bool:
        return bool(self.linker.link(text) or MARKET.search(text))

    def doc(self) -> list[dict]:
        response = self.client.get(
            DOC_URL,
            params={
                "query": QUERY,
                "mode": "ArtList",
                "format": "json",
                "timespan": "24h",
                "maxrecords": 100,
                "sort": "DateDesc",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
            raise ValueError("GDELT DOC did not return an article list")
        rows = []
        for raw in payload["articles"]:
            row = article(
                raw.get("title", ""), raw.get("url", ""), raw.get("seendate", ""),
                raw.get("domain", ""), "gdelt_doc",
            )
            if row and self.relevant(row["text"]):
                rows.append(row)
        return rows

    def parse_gkg(self, content: bytes) -> list[dict]:
        rows = []
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            with archive.open(archive.namelist()[0]) as stream:
                for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
                    fields = line.rstrip("\r\n").split("\t")
                    if len(fields) < 27 or not (match := TITLE.search(fields[26])):
                        continue
                    title = html.unescape(match.group(1))
                    if not self.relevant(title):
                        continue
                    if not self.linker.link(title) and not re.search(
                        r"(?:^|;)(?:ECON|EPU|WB_)", fields[7]
                    ):
                        continue
                    row = article(
                        title, fields[4], f"{fields[1][:8]}T{fields[1][8:]}Z",
                        fields[3], "gdelt_gkg",
                    )
                    if row:
                        rows.append(row)
        return rows

    def gkg(self) -> list[dict]:
        response = self.client.get(DATA_URL + "lastupdate.txt")
        response.raise_for_status()
        filename = next(
            (line.split()[-1].rsplit("/", 1)[-1] for line in response.text.splitlines()
             if line.strip().endswith(".gkg.csv.zip")), None,
        )
        if filename is None:
            raise ValueError("GDELT update list has no GKG file")
        newest = datetime.strptime(filename.split(".")[0], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        for offset in range(9):
            stamp = newest - timedelta(minutes=15 * offset)
            if self.last_gkg and stamp <= self.last_gkg:
                return []
            response = self.client.get(DATA_URL + f"{stamp:%Y%m%d%H%M%S}.gkg.csv.zip")
            if response.status_code == 404:
                continue
            response.raise_for_status()
            rows = self.parse_gkg(response.content)
            self.last_gkg = stamp
            return rows
        raise RuntimeError("The latest GDELT GKG files are not yet available")

    def fetch(self) -> tuple[list[dict], str, str | None]:
        try:
            return self.doc(), "gdelt_doc", None
        except (httpx.HTTPError, ValueError) as exc:
            rows = self.gkg()
            return rows, "gdelt_gkg", f"DOC unavailable; using GKG fallback ({exc})"
