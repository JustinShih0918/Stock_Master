"""HTTP for issuer adapters: browser-like headers, and query-token redaction in httpx logs."""

from __future__ import annotations

import logging
import re

import httpx

from stockmaster.data.http import USER_AGENT

TIMEOUT = 30.0
_TOKEN = re.compile(r"((?:token|access_token|apikey|api_key|key|sig|signature)=)[^&\s'\"]+", re.I)


def redact(url) -> str:
    return _TOKEN.sub(r"\1<redacted>", str(url))


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        red = redact(msg)
        if red != msg:
            record.msg, record.args = red, ()
        return True


_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(f, _RedactFilter) for f in _httpx_logger.filters):
    _httpx_logger.addFilter(_RedactFilter())


def client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True,
                        headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-TW,zh;q=0.9"})


def get(c: httpx.Client, url: str, **kw) -> httpx.Response:
    r = c.get(url, **kw)
    r.raise_for_status()
    return r


def post(c: httpx.Client, url: str, **kw) -> httpx.Response:
    r = c.post(url, **kw)
    r.raise_for_status()
    return r
