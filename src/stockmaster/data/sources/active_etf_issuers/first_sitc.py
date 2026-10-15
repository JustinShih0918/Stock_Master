"""第一金投信 (First SITC) PCF: ASP.NET {d: "<json rows>"}; group 1 = stocks."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

FUND_IDS = {"00994A": "182"}
CODES = ("00994A", "00408A")


def parse(raw: bytes, code: str):
    rows = json.loads(json.loads(raw)["d"])
    require_fund(rows[0].get("fundid") if rows else None, FUND_IDS.get(code.upper(), ""))
    stocks = [r for r in rows if str(r.get("group")) == "1"]
    return parse_date(stocks[0]["sdate"]), tuple(holding(r["A"], r["B"], pct(r["C"]), shares_of(r["D"]))
                                                 for r in stocks)


ADAPTERS = [Adapter("first_sitc", CODES, http_fetch(no_url, parse))]
