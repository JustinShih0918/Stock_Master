"""野村投信 (Nomura) PCF JSON ({Entries: {FundID, Data: {FundAsset, Table: [...]}}}). The same
vendor format is used by Allianz, whose FundID is an internal id rather than the ETF code."""

import json

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, http_fetch, holding, no_url, parse_date,
                                                              pct, require_fund, shares_of)

CODES = ("00980A", "00985A", "00999A")


def parse_entries(raw: bytes, code: str, fund_id: str | None):
    d = json.loads(raw)["Entries"]
    require_fund(d.get("FundID"), fund_id or "")
    data = d["Data"]
    for t in data["Table"]:
        cols = [c.get("Name") or "" for c in t.get("Columns") or []]
        if "股票代號" in cols:
            ic, iname, ish, iw = (cols.index("股票代號"), cols.index("股票名稱"), cols.index("股數"),
                                  cols.index("權重(%)"))
            lines = tuple(holding(r[ic], r[iname], pct(r[iw]), shares_of(r[ish])) for r in t["Rows"])
            return parse_date(t.get("NavDate") or data["FundAsset"]["NavDate"]), lines
    raise ValueError("no stock table")


def parse(raw: bytes, code: str):
    return parse_entries(raw, code, code)


ADAPTERS = [Adapter("nomura", CODES, http_fetch(no_url, parse))]
