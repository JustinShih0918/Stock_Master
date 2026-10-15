"""Shared types and helpers for active-ETF holdings adapters.

An adapter knows how to fetch one issuer's published holdings (PCF / 申購買回清單) and parse them
into ``EtfHolding`` lines. Parsers are pure (bytes -> (as_of, holdings)) so they can be tested on
saved payloads; they raise ``ValueError`` when the payload belongs to a different fund.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

from stockmaster.timeutil import TAIPEI, now_taipei


class HoldingsUnavailable(RuntimeError):
    def __init__(self, etf_code: str, reason: str):
        super().__init__(f"{etf_code}: {reason}")
        self.etf_code, self.reason = etf_code, reason


@dataclass(frozen=True)
class EtfHolding:
    code: str        # exchange code ('2330'), foreign ticker ('NVDA US'), or '' for cash lines
    name: str
    weight: float    # fraction of fund NAV, 0..1
    shares: int | None


@dataclass(frozen=True)
class EtfHoldingsSnapshot:
    etf_code: str
    as_of: date
    fetched_at: datetime
    source_url: str
    holdings: tuple[EtfHolding, ...]
    raw: bytes
    raw_content_type: str


Parser = Callable[[bytes, str], tuple[date, tuple[EtfHolding, ...]]]


@dataclass(frozen=True)
class Adapter:
    name: str
    codes: tuple[str, ...]
    fetch: Callable[[str, httpx.Client], EtfHoldingsSnapshot]
    official: bool = True


# --- parsing helpers ---------------------------------------------------------------------------

def num(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = re.sub(r"<[^>]+>", "", str(s)).replace(",", "").replace("%", "").replace("NT$", "") \
        .replace("TWD", "").replace("$", "").strip()
    if t in ("", "-", "--", "—", "N/A"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def pct(s) -> float | None:
    """'9.42%' or ' 9.42 ' -> 0.0942 (issuer pages publish weights in percent)."""
    v = num(s)
    return None if v is None else v / 100


def shares_of(s) -> int | None:
    v = num(s)
    return None if v is None else int(round(v))


_MS_DATE = re.compile(r"/Date\((-?\d+)\)/")


def parse_date(s) -> date:
    """Issuer date formats: 2026/10/01, 2026-10-01T00:00:00, 20261001, 115/10/01, 1151001,
    115年10月1日, 10/01/2026 (US order), /Date(1790697600000)/ (Taipei date of the epoch ms)."""
    t = str(s).strip()
    m = _MS_DATE.search(t)
    if m:
        return datetime.fromtimestamp(int(m.group(1)) / 1000, timezone.utc).astimezone(TAIPEI).date()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", t)
    if m:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = re.match(r"^(\d{2,4})\s*[/\-年.]\s*(\d{1,2})\s*[/\-月.]\s*(\d{1,2})", t)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    else:
        digits = re.sub(r"\D", "", t)
        if len(digits) == 8:
            y, mo, d = int(digits[:4]), int(digits[4:6]), int(digits[6:])
        elif len(digits) == 7:
            y, mo, d = int(digits[:3]), int(digits[3:5]), int(digits[5:])
        else:
            raise ValueError(f"unrecognised date {s!r}")
    return date(y + 1911 if y < 1911 else y, mo, d)


_TW_CODE = re.compile(r"\d{4}[A-Z]?|00\d{2,4}[A-Z]?|91\d{4}")
_FOREIGN = re.compile(r"[A-Z0-9][A-Z0-9./-]* [A-Z]{2}")
_NON_STOCK_NAMES = ("現金", "保證金", "期貨", "選擇權", "指數", "應收", "應付", "申贖", "附買回", "存款",
                    "Cash", "CASH", "Margin")


def is_equity_line(h: EtfHolding) -> bool:
    """A stock (TW-listed code or 'TICKER XX' foreign listing), not cash, margin, futures or options."""
    code = h.code.strip()
    if not code or any(w in h.name for w in _NON_STOCK_NAMES):
        return False
    return bool(_TW_CODE.fullmatch(code) or _FOREIGN.fullmatch(code))


def clean_code(code) -> str:
    """'2330 TT' -> '2330'; '  2330 ' -> '2330'; foreign tickers keep their exchange suffix."""
    c = re.sub(r"\s+", " ", str(code or "")).strip()
    if c.endswith(" TT"):
        c = c[:-3]
    return c


def holding(code, name, weight: float | None, shares: int | None = None) -> EtfHolding:
    """Build a line from a weight fraction (None -> 0), clamped to [0, 1]."""
    w = 0.0 if weight is None else float(weight)
    return EtfHolding(clean_code(code), re.sub(r"\s+", " ", str(name or "")).strip(),
                      max(0.0, min(1.0, w)), shares)


def require_fund(found, code: str) -> None:
    """Guard against an issuer returning another fund's page (wrong id, redirects)."""
    if str(found or "").strip().upper() != code.upper():
        raise ValueError(f"payload is for fund {found!r}, not {code}")


def snapshot(code: str, url: str, resp: httpx.Response, parse: Parser) -> EtfHoldingsSnapshot:
    raw = resp.content
    try:
        as_of, holdings = parse(raw, code)
    except Exception as e:  # noqa: BLE001 - any parse failure means the snapshot is unusable
        raise HoldingsUnavailable(code, f"cannot parse {url}: {type(e).__name__}: {e}") from e
    if not holdings:
        raise HoldingsUnavailable(code, f"no holdings in {url}")
    return EtfHoldingsSnapshot(code, as_of, now_taipei(), url, tuple(holdings), raw,
                               resp.headers.get("content-type", ""))


# --- HTML helpers ------------------------------------------------------------------------------

_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_TR = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)


def text(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def decode_html(raw: bytes) -> str:
    return _html.unescape(raw.decode("utf-8", errors="replace"))


def table_rows(html: str, header_must: tuple[str, ...]) -> list[list[str]]:
    """Data rows of the first table whose header row contains every string in ``header_must``."""
    for t in _TABLE.findall(html):
        rows = [[text(c) for c in _CELL.findall(tr)] for tr in _TR.findall(t)]
        rows = [r for r in rows if r]
        if rows and all(any(h in cell for cell in rows[0]) for h in header_must):
            return rows[1:]
    return []


def first_date(html: str, pattern: str) -> date:
    m = re.search(pattern, html)
    if not m:
        raise ValueError(f"date not found ({pattern})")
    return parse_date(m.group(1))


def http_fetch(url_for: Callable[[str], str | None], parse: Parser) -> Callable[[str, httpx.Client], EtfHoldingsSnapshot]:
    """Standard fetch: GET the issuer page for ``code`` and parse it. ``url_for`` returns None for
    codes whose official endpoint is not known; those fall through to the aggregator."""
    def fetch(code: str, client: httpx.Client) -> EtfHoldingsSnapshot:
        from stockmaster.data.sources.active_etf_issuers import _http
        url = url_for(code)
        if url is None:
            raise HoldingsUnavailable(code, "official endpoint not configured")
        return snapshot(code, url, _http.get(client, url), parse)
    return fetch


def no_url(code: str) -> None:
    return None
