"""永豐投信 (SinoPac) PCF page (HTML)."""

from stockmaster.data.sources.active_etf_issuers.base import (Adapter, decode_html, first_date, holding,
                                                              http_fetch, pct, shares_of, table_rows)

CODES = ("00410A",)
URL = "https://sitc.sinopac.com/SinopacEtfs/Etfs/SinglePcf/{code}"


def parse(raw: bytes, code: str):
    html = decode_html(raw)
    if f"代碼：{code}" not in html and f"SinglePcf/{code}" not in html:
        raise ValueError(f"page is not for {code}")
    rows = table_rows(html, ("證券代碼", "權重"))
    return (first_date(html, r"資料日期：\s*(\d{4}/\d{1,2}/\d{1,2})"),
            tuple(holding(r[0], r[1], pct(r[3]), shares_of(r[2])) for r in rows if len(r) >= 4))


ADAPTERS = [Adapter("sinopac", CODES, http_fetch(lambda c: URL.format(code=c), parse))]
