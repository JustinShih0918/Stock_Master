"""凱基投信 (KGI) PCF page (HTML)."""

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, decode_html, first_date, holding,
                                                              http_fetch, no_url, pct, shares_of, table_rows)

CODES = ("00407A",)


def parse(raw: bytes, code: str):
    html = decode_html(raw)
    if f"({code})" not in html:
        raise ValueError(f"page is not for {code}")
    rows = table_rows(html, ("股票代號", "權重"))
    return (first_date(html, r"\((\d{4}/\d{1,2}/\d{1,2})\)\s*每受益權單位淨資產價值"),
            tuple(holding(r[0], r[1], pct(r[3]), shares_of(r[2])) for r in rows if len(r) >= 4))


ADAPTERS = [Adapter("kgi", CODES, http_fetch(no_url, parse))]
