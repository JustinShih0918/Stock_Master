"""Polite HTTP fetching with raw-payload archiving and provenance records.

Every payload is written under ``data/raw/<source>/<for_date>/`` and can be registered in the
``source_fetch`` table (``record_fetch``) so stored rows point at the exact bytes they came from.
Requests are paced per host: TWSE blocks an IP that sends more than roughly one request a second.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from sqlalchemy import Engine
from tenacity import (Retrying, retry_if_exception, stop_after_attempt,
                      wait_exponential_jitter)

from stockmaster.settings import get_settings
from stockmaster.timeutil import now_taipei

log = logging.getLogger(__name__)

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36 stockmaster-research")
# Seconds between requests to the same host.
HOST_INTERVALS = {
    "www.twse.com.tw": 3.0,
    "openapi.twse.com.tw": 1.0,
    "mis.twse.com.tw": 1.0,
    "www.tpex.org.tw": 1.5,
    "mopsov.twse.com.tw": 3.0,
    "mops.twse.com.tw": 3.0,
    "isin.twse.com.tw": 2.0,
    "query1.finance.yahoo.com": 1.0,
}
DEFAULT_INTERVAL = 0.5


class FetchError(RuntimeError):
    pass


def _retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(e, (httpx.TransportError, httpx.TimeoutException))


@dataclass(frozen=True)
class FetchResult:
    source: str
    url: str
    params: dict | None
    status: int
    content: bytes
    content_type: str
    sha256: str
    raw_path: str | None
    fetched_at: datetime
    for_date: date | None

    def text(self, encoding: str | None = None) -> str:
        if encoding:
            return self.content.decode(encoding, errors="replace")
        m = re.search(r"charset=([\w-]+)", self.content_type or "")
        enc = m.group(1) if m else "utf-8"
        try:
            return self.content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content.decode("utf-8-sig"))


class Fetcher:
    def __init__(self, raw_dir: Path | None = None, client: httpx.Client | None = None,
                 archive: bool = True, timeout: float = 30.0):
        self.raw_dir = raw_dir or (get_settings().data_dir / "raw")
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True,
                                             headers={"User-Agent": USER_AGENT,
                                                      "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8"})
        self.archive = archive
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _pace(self, host: str) -> None:
        with self._lock:
            wait = HOST_INTERVALS.get(host, DEFAULT_INTERVAL) - (time.monotonic() - self._last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def request(self, method: str, source: str, url: str, *, params: dict | None = None,
                data: dict | None = None, for_date: date | None = None, name: str | None = None,
                headers: dict | None = None) -> FetchResult:
        host = urlsplit(url).hostname or ""
        resp = None
        for attempt in Retrying(retry=retry_if_exception(_retryable), reraise=True,
                                wait=wait_exponential_jitter(initial=3, max=60),
                                stop=stop_after_attempt(5)):
            with attempt:
                self._pace(host)
                resp = self.client.request(method, url, params=params, data=data, headers=headers)
                resp.raise_for_status()
        assert resp is not None
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()
        ctype = resp.headers.get("content-type", "")
        raw_path = self._save(source, for_date, name or _name_from(url), sha, content, ctype) \
            if self.archive else None
        return FetchResult(source, str(resp.url), params or data, resp.status_code, content, ctype, sha,
                           raw_path, now_taipei(), for_date)

    def get(self, source: str, url: str, **kw) -> FetchResult:
        return self.request("GET", source, url, **kw)

    def post(self, source: str, url: str, **kw) -> FetchResult:
        return self.request("POST", source, url, **kw)

    def _save(self, source: str, for_date: date | None, name: str, sha: str, content: bytes,
              ctype: str) -> str:
        ext = "json" if "json" in ctype else "html" if "html" in ctype else "csv" if "csv" in ctype else "bin"
        d = self.raw_dir / source / (for_date.isoformat() if for_date else now_taipei().date().isoformat())
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}_{sha[:12]}.{ext}"
        if not path.exists():
            path.write_bytes(content)
        return str(path)


def _name_from(url: str) -> str:
    tail = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1] or "index"
    return re.sub(r"[^\w.-]", "_", tail)[:60]


def record_fetch(engine: Engine, res: FetchResult, *, row_count: int | None = None,
                 is_stale: bool = False, is_synthetic: bool = False, note: str | None = None) -> int:
    """Register a fetched payload in ``source_fetch``; returns its id."""
    from stockmaster.data import schema
    with engine.begin() as conn:
        return conn.execute(schema.source_fetch.insert().values(
            source=res.source, endpoint=res.url.split("?", 1)[0], params=res.params,
            for_date=res.for_date, fetched_at=res.fetched_at, http_status=res.status,
            content_sha256=res.sha256, raw_path=res.raw_path, row_count=row_count,
            is_stale=is_stale, is_synthetic=is_synthetic, note=note)
            .returning(schema.source_fetch.c.id)).scalar_one()
