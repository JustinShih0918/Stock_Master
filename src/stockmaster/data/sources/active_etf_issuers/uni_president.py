"""統一投信 (Uni-President, ezmoney) PCF JSON."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

CODES = ("00981A", "00988A", "00403A", "00411A")


def parse(raw: bytes, code: str):
    d = json.loads(raw)
    require_fund((d.get("fund") or {}).get("sStockNo"), code)
    st = next(a for a in d["asset"] if a.get("AssetCode") == "ST")
    det = st["Details"]
    return parse_date(det[0]["TranDate"]), tuple(
        holding(x["DetailCode"], x["DetailName"], pct(x["NavRate"]), shares_of(x.get("Share"))) for x in det)


ADAPTERS = [Adapter("uni_president", CODES, http_fetch(no_url, parse))]
