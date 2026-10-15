"""復華投信 (Fuh Hwa) PCF JSON."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

CODES = ("00991A", "00998A", "00409A")


def parse(raw: bytes, code: str):
    r = json.loads(raw)["result"][0]
    require_fund(r.get("etf002"), code)
    lines = tuple(holding(x["stockid"], x["stockname"], pct(x["prate_addaccint"]), shares_of(x.get("qshare")))
                  for x in r["detail"] if x.get("ftype") == "股票")
    return parse_date(r["dDate"]), lines


ADAPTERS = [Adapter("fuh_hwa", CODES, http_fetch(no_url, parse))]
