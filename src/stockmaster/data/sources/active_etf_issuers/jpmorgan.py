"""摩根投信 (J.P. Morgan AM) fund data JSON (PCF equity holdings)."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

CODES = ("00401A",)


def parse(raw: bytes, code: str):
    f = json.loads(raw)["fundData"]
    require_fund(f["shareClass"].get("exchangeTicker"), code)
    eq = f["holdings"]["pcfEquityHoldings"]
    lines = tuple(holding(h.get("securityTicker"), h.get("securityDescription"), pct(h.get("marketValuePercent")),
                          shares_of(h.get("shares"))) for h in eq["data"])
    return parse_date(eq["effectiveDate"]), lines


ADAPTERS = [Adapter("jpmorgan", CODES, http_fetch(no_url, parse))]
