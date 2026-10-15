"""群益投信 (Capital) PCF JSON. The payload names the fund only by its long name."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, shares_of)

CODES = ("00982A", "00992A")


def parse(raw: bytes, code: str):
    d = json.loads(raw)["data"]
    lines = tuple(holding(s["stocNo"], s["stocName"], pct(s["weight"]), shares_of(s.get("share")))
                  for s in d["stocks"])
    return parse_date(d["pcf"]["date2"]), lines


ADAPTERS = [Adapter("capital", CODES, http_fetch(no_url, parse))]
