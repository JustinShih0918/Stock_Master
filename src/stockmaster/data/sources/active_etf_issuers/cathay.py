"""國泰投信 (Cathay) PCF: three API responses stored together (assets, stock list, futures)."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, shares_of)

CODES = ("00400A",)


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


def parse(raw: bytes, code: str):
    d = json.loads(raw)
    assets = _j(d.get("GetETFAssets")) or {}
    stocks = _j(d["GetETFDetailStockList"])["result"]
    as_of = parse_date((assets.get("result") or {}).get("preDate") or d["SearchDate"])
    return as_of, tuple(holding(s["stockCode"], s["stockName"], pct(s["weights"]), shares_of(s.get("volumn")))
                        for s in stocks)


ADAPTERS = [Adapter("cathay", CODES, http_fetch(no_url, parse))]
