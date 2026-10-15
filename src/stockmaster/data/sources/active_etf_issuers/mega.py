"""兆豐投信 (Mega) product page (HTML; holdings are div rows, four cells each)."""

import re

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, decode_html, first_date, holding,
                                                              http_fetch, pct, shares_of, text)

PAGE_IDS = {"00996A": "23"}
CODES = tuple(PAGE_IDS)
URL = "https://www.megafunds.com.tw/MEGA/etf/etf_product.aspx?id={id}"
_ROW = re.compile(r'class="fund-info content-list-1">(.*?)(?=class="fund-info|<!--|</section|id="fund_content)', re.S)
_CELL = re.compile(r'<div class="fund-content[^"]*">(.*?)</div>', re.S)


def parse(raw: bytes, code: str):
    html = decode_html(raw)
    if f"{code} TT" not in html:
        raise ValueError(f"page is not for {code}")
    lines = []
    for block in _ROW.findall(html):
        cells = [text(c) for c in _CELL.findall(block)]
        if len(cells) >= 4:
            lines.append(holding(cells[0], cells[1], pct(cells[3]), shares_of(cells[2])))
    return first_date(html, r"資料來源：兆豐投信，\s*(\d{4}/\d{1,2}/\d{1,2})"), tuple(lines)


ADAPTERS = [Adapter("mega", CODES, http_fetch(lambda c: URL.format(id=PAGE_IDS[c]) if c in PAGE_IDS else None,
                                              parse))]
