"""MOPS monthly revenue (每月營業收入彙總表). Amounts are in NT$ thousands, as published."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

from stockmaster.data.sources._parse import integer, num, strip_html

MARKET_DIR = {"TWSE": "sii", "TPEx": "otc"}
COLS = ["code", "name", "ym", "revenue", "revenue_prev_month", "revenue_last_year", "mom_pct", "yoy_pct",
        "cum_revenue", "cum_revenue_last_year", "cum_yoy_pct", "note"]
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)


def url(market: str, year: int, month: int) -> str:
    """Domestic companies (``_0``); the page is Big5-encoded."""
    return f"https://mopsov.twse.com.tw/nas/t21/{MARKET_DIR[market]}/t21sc03_{year - 1911}_{month}_0.html"


def parse_monthly_revenue(html: str, market: str, year: int, month: int) -> pd.DataFrame:
    ym = date(year, month, 1)
    rows = []
    for tr in _ROW.findall(html):
        cells = [strip_html(c) for c in _CELL.findall(tr)]
        if len(cells) < 10 or not re.fullmatch(r"\d{4}[0-9A-Z]?", cells[0]):
            continue
        rows.append({"code": cells[0], "name": cells[1], "ym": ym, "revenue": integer(cells[2]),
                     "revenue_prev_month": integer(cells[3]), "revenue_last_year": integer(cells[4]),
                     "mom_pct": num(cells[5]), "yoy_pct": num(cells[6]), "cum_revenue": integer(cells[7]),
                     "cum_revenue_last_year": integer(cells[8]), "cum_yoy_pct": num(cells[9]),
                     "note": (cells[10] if len(cells) > 10 and cells[10] not in ("-", "") else None)})
    return pd.DataFrame(rows, columns=COLS).drop_duplicates("code")
