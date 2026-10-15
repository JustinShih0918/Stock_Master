"""Intraday prices for the monitor (display only; never used for decisions or accounting).

TWSE MIS (official, ~5 s delay) first, Yahoo Finance chart API (unofficial) as fallback. Results
are cached for ``TTL`` seconds and fetched only around market hours on trading days."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, time as dtime

import httpx

from stockmaster.data.http import USER_AGENT
from stockmaster.timeutil import TAIPEI, now_taipei

log = logging.getLogger(__name__)
TTL = 20.0
OPEN, CLOSE_FETCH = dtime(8, 59), dtime(14, 30)
MIS = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"


def market_session(now: datetime | None = None, is_trading_day: bool = True) -> str:
    now = now or now_taipei()
    if not is_trading_day:
        return "closed"
    t = now.timetz().replace(tzinfo=None)
    if t < dtime(8, 30):
        return "pre"
    if t < dtime(9, 0):
        return "pre_open"
    if t <= dtime(13, 30):
        return "open"
    if t <= dtime(14, 30):
        return "after_hours"  # 盤後定價 / 零股盤後
    return "closed"


class LiveQuotes:
    def __init__(self):
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=8.0, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        self._mis_down_until = 0.0

    def get(self, codes: dict[str, str], is_trading_day: bool) -> dict[str, dict]:
        """``codes`` maps code -> market ('TWSE' | 'TPEx')."""
        now = now_taipei()
        t = now.timetz().replace(tzinfo=None)
        if not codes or not is_trading_day or not (OPEN <= t <= CLOSE_FETCH):
            return {}
        out, todo = {}, {}
        with self._lock:
            for c, m in codes.items():
                hit = self._cache.get(c)
                if hit and time.monotonic() - hit[0] < TTL:
                    out[c] = hit[1]
                else:
                    todo[c] = m
        if todo:
            fresh = self._mis(todo) if time.monotonic() > self._mis_down_until else {}
            for c, m in todo.items():
                if c not in fresh:
                    q = self._yahoo(c, m)
                    if q:
                        fresh[c] = q
            with self._lock:
                for c, q in fresh.items():
                    self._cache[c] = (time.monotonic(), q)
            out.update(fresh)
        return out

    def _mis(self, codes: dict[str, str]) -> dict[str, dict]:
        ex = "|".join(f"{'otc' if m == 'TPEx' else 'tse'}_{c}.tw" for c, m in codes.items())
        try:
            r = self._client.get(MIS, params={"ex_ch": ex, "json": "1", "delay": "0"})
            r.raise_for_status()
            arr = r.json().get("msgArray", [])
        except (httpx.HTTPError, ValueError) as e:
            log.info("MIS unavailable (%s); using Yahoo for 5 minutes", type(e).__name__)
            self._mis_down_until = time.monotonic() + 300
            return {}
        out = {}
        for m in arr:
            price = _f(m.get("z")) or _mid(m)
            if price is None:
                continue
            ts = datetime.fromtimestamp(int(m["tlong"]) / 1000, TAIPEI) if m.get("tlong") else now_taipei()
            out[m["c"]] = {"price": price, "prev_close": _f(m.get("y")), "time": ts.isoformat(timespec="seconds"),
                           "source": "TWSE MIS"}
        return out

    def _yahoo(self, code: str, market: str) -> dict | None:
        sym = f"{code}.{'TWO' if market == 'TPEx' else 'TW'}"
        try:
            r = self._client.get(YAHOO.format(sym=sym), params={"range": "1d", "interval": "1m"})
            r.raise_for_status()
            meta = r.json()["chart"]["result"][0]["meta"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return None
        price = meta.get("regularMarketPrice")
        if price is None:
            return None
        ts = datetime.fromtimestamp(meta.get("regularMarketTime", time.time()), TAIPEI)
        return {"price": float(price), "prev_close": meta.get("chartPreviousClose"),
                "time": ts.isoformat(timespec="seconds"), "source": "Yahoo（非官方）"}


def _f(x) -> float | None:
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _mid(m: dict) -> float | None:
    """No trade yet: midpoint of the best bid/ask."""
    b = _f((m.get("b") or "").split("_")[0])
    a = _f((m.get("a") or "").split("_")[0])
    return (a + b) / 2 if a and b else None
