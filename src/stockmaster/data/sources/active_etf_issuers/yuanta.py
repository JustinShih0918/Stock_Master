"""元大投信 (Yuanta) PCF JSON."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

CODES = ("00990A",)


def parse(raw: bytes, code: str):
    d = json.loads(raw)
    require_fund(d["PCF"].get("markcd"), code)
    lines = tuple(holding(s["code"], s["name"], pct(s["weights"]), shares_of(s.get("qty")))
                  for s in d["FundWeights"]["StockWeights"])
    return parse_date(d["PCF"]["trandate"]), lines


ADAPTERS = [Adapter("yuanta", CODES, http_fetch(no_url, parse))]
