"""etfinfo.tw aggregator (UNOFFICIAL): one JSON format for every Taiwan ETF. Used only when the
issuer's own page cannot be fetched or parsed."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, parse_date, pct,
                                                              require_fund, shares_of)

URL = "https://etfinfo.tw/api/etf/{code}"
# Domestic and foreign active equity ETFs listed on TWSE (2026-10); the aggregator serves any code.
CODES = ("00400A", "00401A", "00402A", "00403A", "00404A", "00405A", "00406A", "00407A", "00408A",
         "00409A", "00410A", "00411A", "00980A", "00981A", "00982A", "00983A", "00984A", "00985A",
         "00986A", "00987A", "00988A", "00989A", "00990A", "00991A", "00992A", "00993A", "00994A",
         "00995A", "00996A", "00997A", "00998A", "00999A")


def parse(raw: bytes, code: str):
    d = json.loads(raw)
    h = d["holdings"]
    require_fund(h.get("etfCode") or (d.get("info") or {}).get("code"), code)
    lines = tuple(holding(x["code"], x.get("name"), pct(x.get("weight")), shares_of(x.get("shares")))
                  for x in h["holdings"] if x.get("assetType") in (None, "stock", "etf"))  # untyped -> is_equity_line decides
    return parse_date(h["snapshotDate"]), lines


ADAPTERS = [Adapter("etfinfo-aggregator-fallback", CODES, http_fetch(lambda c: URL.format(code=c), parse),
                    official=False)]
