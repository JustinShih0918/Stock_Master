"""TWSE ISIN registry (isin.twse.com.tw class query): the instrument master for every listed
security, with market, security type, industry and listing date. Pages are Big5-encoded."""

from __future__ import annotations

import re

from stockmaster.data.sources._parse import strip_html

URL = "https://isin.twse.com.tw/isin/class_main.jsp"
MARKET_PARAM = {"TWSE": "1", "TPEx": "2"}
# issuetype: 1 = 股票 (common stock), I = ETF
ISSUE_TYPES = {"stock": "1", "etf": "I"}
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
MARKETS = {"上市": "TWSE", "上櫃": "TPEx", "興櫃": "ESB"}


def params(market: str, issue: str) -> dict:
    return {"owncode": "", "stockname": "", "isincode": "", "market": MARKET_PARAM[market],
            "issuetype": ISSUE_TYPES[issue], "industry_code": "", "Page": "1", "chklike": "Y"}


def decode(raw: bytes) -> str:
    for enc in ("big5", "cp950"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp950", errors="replace")


def parse_isin_html(raw: bytes | str) -> list[dict]:
    """Rows: code, name, isin, market ('TWSE' | 'TPEx' | raw label), type, industry, listed
    ('YYYY/MM/DD' as published), cfi."""
    html = decode(raw) if isinstance(raw, bytes) else raw
    out = []
    for tr in _ROW.findall(html):
        cells = [strip_html(c) for c in _CELL.findall(tr)]
        if len(cells) < 9 or not cells[0].strip().isdigit():
            continue
        _, isin, code, name, market, typ, industry, listed, cfi = cells[:9]
        out.append({"code": code.strip(), "name": name.strip(), "isin": isin.strip(),
                    "market": MARKETS.get(market.strip(), market.strip()), "type": typ.strip(),
                    "industry": industry.strip() or None, "listed": listed.strip(), "cfi": cfi.strip()})
    return out
