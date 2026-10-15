"""中國信託投信 (CTBC) PCF JSON."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

CODES = ("00995A", "00406A")


def parse(raw: bytes, code: str):
    d = json.loads(raw)["Data"]
    require_fund(d["Fund"][0].get("ETF_ID"), code)
    stock = next(g for g in d["FundAssetsDetail"] if g.get("Code") == "STOCK")
    lines = tuple(holding(r["code_"], r["name_"], pct(r["weights_"]), shares_of(r.get("qty_"))) for r in stock["Data"])
    return parse_date(d["FundAssets"][0]["資料日期"]), lines


ADAPTERS = [Adapter("ctbc", CODES, http_fetch(no_url, parse))]
