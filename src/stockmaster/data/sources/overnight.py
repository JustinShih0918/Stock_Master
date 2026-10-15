"""Overnight US market data (Yahoo Finance chart API, an UNOFFICIAL source): Philadelphia
Semiconductor (^SOX), Nasdaq-100 (^NDX), the TSMC ADR (TSM) and USD/TWD. Stored as macro_daily
series ``yahoo:<symbol>`` keyed by the exchange-local trading date."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SYMBOLS = ("^SOX", "^NDX", "TSM", "TWD=X")
ADR_RATIO = 5  # one TSM ADR = 5 ordinary shares of 2330


def params() -> dict:
    return {"range": "1mo", "interval": "1d"}


def parse_chart(payload: dict) -> pd.Series:
    """Daily closes indexed by exchange-local date."""
    res = (payload.get("chart") or {}).get("result") or []
    if not res:
        return pd.Series(dtype=float)
    r = res[0]
    tz = ZoneInfo(r.get("meta", {}).get("exchangeTimezoneName") or "America/New_York")
    ts = r.get("timestamp") or []
    closes = (((r.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
    data = {datetime.fromtimestamp(t, tz).date(): c for t, c in zip(ts, closes) if c is not None}
    return pd.Series(data, dtype=float).sort_index()


def summarize(series: dict[str, pd.Series], tsmc_close: float | None) -> dict:
    """Latest close and 1-day return per symbol, plus the TSM ADR premium over 2330 when the
    USD/TWD rate and the 2330 close are known."""
    out: dict = {}
    for sym, s in series.items():
        s = s.dropna().sort_index()
        if len(s) < 2:
            continue
        out[sym] = {"date": str(s.index[-1]), "close": float(s.iloc[-1]),
                    "ret_1d": float(s.iloc[-1] / s.iloc[-2] - 1)}
    tsm, fx = out.get("TSM"), out.get("TWD=X")
    if tsm and fx and tsmc_close:
        out["tsm_adr_premium"] = tsm["close"] * fx["close"] / (ADR_RATIO * tsmc_close) - 1
    return out
