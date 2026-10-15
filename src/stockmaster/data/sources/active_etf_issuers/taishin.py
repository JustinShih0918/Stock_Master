"""台新投信 (Taishin) PCF page (HTML)."""

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, decode_html, first_date, holding,
                                                              http_fetch, pct, shares_of, table_rows)

CODES = ("00987A", "00986A")
URL = "https://www.tsit.com.tw/ETF/Home/Pcf/{code}"


def parse(raw: bytes, code: str):
    html = decode_html(raw)
    if f"({code})" not in html:
        raise ValueError(f"page is not for {code}")
    rows = table_rows(html, ("股數", "持股權重"))  # the futures table has 口數
    return (first_date(html, r"(\d{4}/\d{1,2}/\d{1,2})\s*每基數實際申購總價金"),
            tuple(holding(r[0], r[1], pct(r[3]), shares_of(r[2])) for r in rows if len(r) >= 4))


ADAPTERS = [Adapter("taishin", CODES, http_fetch(lambda c: URL.format(code=c), parse))]
