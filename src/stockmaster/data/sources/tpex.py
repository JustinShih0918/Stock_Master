"""TPEx (上櫃) daily reports: quotes with published average price and next-day limits, institutional
flows, margin balances and valuation. Parsers mirror ``twse`` and return the same columns."""

from __future__ import annotations

from datetime import date

import pandas as pd

from stockmaster.data.sources._parse import (field_index, integer, is_stock_code, num, strip_html)
from stockmaster.data.sources.twse import MARGIN_COLS, QUOTE_COLS

BASE = "https://www.tpex.org.tw/www/zh-tw"
URLS = {
    "quotes": f"{BASE}/afterTrading/dailyQuotes",
    "t86": f"{BASE}/insti/dailyTrade",
    "margin": f"{BASE}/margin/balance",
    "valuation": f"{BASE}/afterTrading/peQryDate",
}


def params(kind: str, d: date) -> dict:
    ymd = d.strftime("%Y/%m/%d")
    return {
        "quotes": {"date": ymd, "type": "AL", "response": "json"},
        "t86": {"date": ymd, "type": "Daily", "sect": "AL", "response": "json"},
        "margin": {"date": ymd, "response": "json"},
        "valuation": {"date": ymd, "response": "json"},
    }[kind]


def has_data(payload) -> bool:
    tables = payload.get("tables") or []
    return any(t.get("data") for t in tables) or bool(payload.get("aaData"))


def _tables(payload: dict) -> list[dict]:
    if "tables" in payload:
        return [t for t in payload["tables"] if t and t.get("fields")]
    if "aaData" in payload:  # pre-2024 site
        return [{"fields": None, "data": payload["aaData"]}]
    return []


def parse_quotes(payload: dict, d: date) -> pd.DataFrame:
    rows = []
    for t in _tables(payload):
        f = t["fields"]
        if f is not None:
            ix = {k: field_index(f, *n) for k, n in {
                "close": ("收盤",), "chg": ("漲跌",), "open": ("開盤",), "high": ("最高",),
                "low": ("最低",), "avg": ("均價",), "vol": ("成交股數",), "value": ("成交金額(元)",),
                "trades": ("成交筆數",), "ref": ("次日參考價",), "up": ("次日漲停價",),
                "dn": ("次日跌停價",)}.items()}
        else:  # aaData layout (17 columns)
            ix = dict(close=2, chg=3, open=4, high=5, low=6, avg=7, vol=8, value=9, trades=10,
                      ref=14, up=15, dn=16)
        for r in t["data"]:
            code = strip_html(r[0])
            if not is_stock_code(code):
                continue
            close, change = num(r[ix["close"]]), num(r[ix["chg"]])
            vol, value = integer(r[ix["vol"]]), integer(r[ix["value"]])
            rows.append({
                "code": code, "name": strip_html(r[1]), "date": d, "market": "TPEx",
                "open": num(r[ix["open"]]), "high": num(r[ix["high"]]), "low": num(r[ix["low"]]),
                "close": close, "change": change,
                "prev_close": round(close - change, 4) if (close is not None and change is not None) else None,
                "volume_sh": vol, "value_ntd": value, "trades": integer(r[ix["trades"]]),
                "vwap": (value / vol) if (vol and value) else None,
                "vwap_published": num(r[ix["avg"]]),
                "next_ref_price": num(r[ix["ref"]]), "next_limit_up": num(r[ix["up"]]),
                "next_limit_down": num(r[ix["dn"]]), "halted": close is None})
    return pd.DataFrame(rows, columns=QUOTE_COLS).drop_duplicates("code")


def parse_institutional(payload: dict, d: date) -> pd.DataFrame:
    """Column groups (buy, sell, net): 外資及陸資(不含外資自營商) 2-4, 外資自營商 5-7,
    外資及陸資合計 8-10, 投信 11-13, 自營商(自行買賣) 14-16, 自營商(避險) 17-19, 自營商合計 20-22,
    三大法人合計 23."""
    cols = ["code", "date", "foreign_net", "itrust_net", "dealer_self_net", "dealer_hedge_net", "total_net"]
    rows = []
    for t in _tables(payload):
        for r in t["data"]:
            code = strip_html(r[0])
            if not is_stock_code(code) or len(r) < 24:
                continue
            rows.append({"code": code, "date": d, "foreign_net": integer(r[4]), "itrust_net": integer(r[13]),
                         "dealer_self_net": integer(r[16]), "dealer_hedge_net": integer(r[19]),
                         "total_net": integer(r[23])})
    return pd.DataFrame(rows, columns=cols)


def parse_margin(payload: dict, d: date) -> pd.DataFrame:
    """Layout: code, name, 前資餘額, 資買, 資賣, 現償, 資餘額, 資屬證金, 資使用率, 資限額,
    前券餘額, 券賣, 券買, 券償, 券餘額, ..."""
    rows = []
    for t in _tables(payload):
        for r in t["data"]:
            code = strip_html(r[0])
            if not is_stock_code(code) or len(r) < 15:
                continue
            rows.append({"code": code, "date": d, "margin_prev": integer(r[2]), "margin_buy": integer(r[3]),
                         "margin_sell": integer(r[4]), "margin_balance": integer(r[6]),
                         "short_prev": integer(r[10]), "short_sell": integer(r[11]),
                         "short_buy": integer(r[12]), "short_balance": integer(r[14])})
    return pd.DataFrame(rows, columns=MARGIN_COLS)


def parse_valuation(payload: dict, d: date) -> pd.DataFrame:
    cols = ["code", "date", "pe", "pb", "div_yield"]
    rows = []
    for t in _tables(payload):
        f = t["fields"]
        i_pe = field_index(f, "本益比") if f else 2
        i_dy = field_index(f, "殖利率(%)") if f else 5
        i_pb = field_index(f, "股價淨值比") if f else 6
        for r in t["data"]:
            code = strip_html(r[0])
            if is_stock_code(code):
                rows.append({"code": code, "date": d, "pe": num(r[i_pe]), "pb": num(r[i_pb]),
                             "div_yield": num(r[i_dy])})
    return pd.DataFrame(rows, columns=cols)
