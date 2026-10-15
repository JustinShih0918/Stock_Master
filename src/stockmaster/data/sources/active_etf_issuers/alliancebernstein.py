"""聯博投信 (AllianceBernstein) PCF JSON."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

CODES = ("00404A",)


def parse(raw: bytes, code: str):
    d = json.loads(raw)
    require_fund((d.get("otherAsset") or d.get("fundAssetTotal") or {}).get("secID"), code)
    eq = next(g for g in d["domesticHoldings"] if g["holdingCategory"] == "holdings-section-equity")
    lines = tuple(holding(h["holdingCode"], h["holding"], pct(h["holdingPerc"]), shares_of(h.get("holdingShares")))
                  for h in eq["holdings"])
    return parse_date(eq["asOfDate"]), lines


ADAPTERS = [Adapter("alliancebernstein", CODES, http_fetch(no_url, parse))]
