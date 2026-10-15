"""TWSE (上市) daily reports: quotes and indices (MI_INDEX), institutional flows (T86), margin
(MI_MARGN), valuation (BWIBBU_d), ex-rights forecast (TWT48U) and the holiday schedule.

Parsers take the decoded JSON payload and return DataFrames; fetching lives in ``data.ingest``.
TWSE changed several layouts over the years, so columns are located by field name, not position.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stockmaster.data.sources._parse import (field_index, integer, is_stock_code, num, roc_date,
                                             sign_of, strip_html)
from stockmaster.domain import limit_prices

BASE = "https://www.twse.com.tw/rwd/zh"
URLS = {
    "quotes": f"{BASE}/afterTrading/MI_INDEX",
    "t86": f"{BASE}/fund/T86",
    "margin": f"{BASE}/marginTrading/MI_MARGN",
    "valuation": f"{BASE}/afterTrading/BWIBBU_d",
    "ex_rights": f"{BASE}/exRight/TWT48U",
    "holidays": "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule",
}
TAIEX = "發行量加權股價指數"
QUOTE_COLS = ["code", "name", "date", "market", "open", "high", "low", "close", "change", "prev_close",
              "volume_sh", "value_ntd", "trades", "vwap", "vwap_published", "next_ref_price",
              "next_limit_up", "next_limit_down", "halted"]


def params(kind: str, d: date | None = None) -> dict:
    ymd = d.strftime("%Y%m%d") if d else None
    return {
        "quotes": {"date": ymd, "type": "ALLBUT0999", "response": "json"},
        "t86": {"date": ymd, "selectType": "ALLBUT0999", "response": "json"},
        "margin": {"date": ymd, "selectType": "ALL", "response": "json"},
        "valuation": {"date": ymd, "selectType": "ALL", "response": "json"},
        "ex_rights": {"response": "json"},
        "holidays": {},
    }[kind]


def has_data(payload) -> bool:
    """TWSE answers holidays and not-yet-published days with stat != OK and no tables."""
    if isinstance(payload, list):
        return bool(payload)
    return str(payload.get("stat", "")).upper() == "OK"


def _tables(payload: dict) -> list[dict]:
    """Normalise the 2023+ ``tables`` list and the older ``fieldsN``/``dataN`` keys."""
    if "tables" in payload:
        return [t for t in payload["tables"] if t]
    out = []
    if "fields" in payload and "data" in payload:
        out.append({"title": payload.get("title"), "fields": payload["fields"], "data": payload["data"]})
    for k in sorted(payload):
        if k.startswith("fields") and k != "fields":
            n = k[len("fields"):]
            out.append({"title": payload.get(f"subtitle{n}") or payload.get(f"title{n}"),
                        "fields": payload[k], "data": payload.get(f"data{n}", [])})
    return out


def _find_table(payload: dict, first_field: tuple[str, ...], must_have: str | None = None) -> dict | None:
    for t in _tables(payload):
        fields = [strip_html(f) for f in (t.get("fields") or [])]
        if fields and fields[0] in first_field and (must_have is None or must_have in fields):
            return t
    return None


# --- quotes, indices, breadth --------------------------------------------------------------------

def parse_quotes(payload: dict, d: date) -> pd.DataFrame:
    t = _find_table(payload, ("證券代號",), "收盤價")
    if t is None:
        return pd.DataFrame(columns=QUOTE_COLS)
    f = t["fields"]
    ix = {k: field_index(f, *names) for k, names in {
        "code": ("證券代號",), "name": ("證券名稱",), "vol": ("成交股數",), "trades": ("成交筆數",),
        "value": ("成交金額",), "open": ("開盤價",), "high": ("最高價",), "low": ("最低價",),
        "close": ("收盤價",), "sign": ("漲跌(+/-)",), "chg": ("漲跌價差",)}.items()}
    rows = []
    for r in t["data"]:
        code = strip_html(r[ix["code"]])
        if not is_stock_code(code):
            continue
        close = num(r[ix["close"]])
        vol, value = integer(r[ix["vol"]]), integer(r[ix["value"]])
        sgn, mag = sign_of(r[ix["sign"]]), num(r[ix["chg"]])
        change = None if (sgn is None or mag is None) else sgn * mag
        prev_close = round(close - change, 4) if (close is not None and change is not None) else None
        ref = close  # next day's reference price; ex-rights adjustments come from corp_action
        up = dn = None
        if ref:
            u, l_ = limit_prices(ref)
            up, dn = float(u), float(l_)
        rows.append({
            "code": code, "name": strip_html(r[ix["name"]]), "date": d, "market": "TWSE",
            "open": num(r[ix["open"]]), "high": num(r[ix["high"]]), "low": num(r[ix["low"]]),
            "close": close, "change": change, "prev_close": prev_close, "volume_sh": vol,
            "value_ntd": value, "trades": integer(r[ix["trades"]]),
            "vwap": (value / vol) if (vol and value) else None, "vwap_published": None,
            "next_ref_price": ref, "next_limit_up": up, "next_limit_down": dn,
            "halted": close is None})
    return pd.DataFrame(rows, columns=QUOTE_COLS)


def normalize_index_name(name: str) -> str:
    n = strip_html(name).replace("数", "數").replace(" ", "")
    return TAIEX if n in ("發行量加權股價指數", "加權股價指數") else n


def parse_index_and_breadth(payload: dict, d: date) -> tuple[pd.DataFrame, dict]:
    rows = []
    for t in _tables(payload):
        fields = [strip_html(x) for x in (t.get("fields") or [])]
        if not fields or fields[0] not in ("指數",):  # price indices only (not 報酬指數)
            continue
        i_close, i_sign, i_pct = (field_index(fields, "收盤指數"), field_index(fields, "漲跌(+/-)"),
                                  field_index(fields, "漲跌百分比(%)"))
        for r in t["data"]:
            sgn = sign_of(r[i_sign]) if i_sign is not None else 1
            pct = num(r[i_pct]) if i_pct is not None else None
            if pct is not None and sgn is not None and pct > 0 and sgn < 0:
                pct = -pct
            rows.append({"index_name": normalize_index_name(r[0]), "date": d,
                         "close": num(r[i_close]), "change_pct": pct})
    idx = pd.DataFrame(rows, columns=["index_name", "date", "close", "change_pct"])
    idx = idx.drop_duplicates("index_name", keep="first")

    breadth: dict = {}
    t = _find_table(payload, ("類型",))
    if t is not None:
        keys = {"上漲": "up", "下跌": "down", "持平": "flat", "未成交": "untraded", "無比價": "no_compare"}
        for r in t["data"]:
            label = strip_html(r[0])
            for zh, en in keys.items():
                if label.startswith(zh):
                    whole, stocks = strip_html(r[1]), strip_html(r[2]) if len(r) > 2 else ""
                    breadth[en] = _count(stocks)
                    breadth[f"{en}_all"] = _count(whole)
                    limit = _paren(stocks)
                    if limit is not None:
                        breadth[f"{en}_limit"] = limit
    return idx, breadth


def _count(s: str) -> int | None:
    v = num(s.split("(")[0])
    return None if v is None else int(v)


def _paren(s: str) -> int | None:
    if "(" in s:
        v = num(s.split("(", 1)[1].rstrip(")"))
        return None if v is None else int(v)
    return None


# --- institutional flows -----------------------------------------------------------------------

def parse_institutional(payload: dict, d: date) -> pd.DataFrame:
    """三大法人買賣超 (shares). ``foreign_net`` excludes foreign dealers (外資自營商), matching the
    TPEx 外資及陸資(不含外資自營商) column. Before 2014-12 dealers were not split: all of it is
    reported as ``dealer_self_net`` and hedge is 0."""
    cols = ["code", "date", "foreign_net", "itrust_net", "dealer_self_net", "dealer_hedge_net", "total_net"]
    t = _find_table(payload, ("證券代號",), "三大法人買賣超股數")
    if t is None:
        return pd.DataFrame(columns=cols)
    f = t["fields"]
    i_foreign = field_index(f, "外陸資買賣超股數(不含外資自營商)", "外資買賣超股數")
    i_itrust = field_index(f, "投信買賣超股數")
    i_self = field_index(f, "自營商買賣超股數(自行買賣)", "自營商買賣超股數")
    i_hedge = field_index(f, "自營商買賣超股數(避險)")
    i_total = field_index(f, "三大法人買賣超股數")
    rows = []
    for r in t["data"]:
        code = strip_html(r[0])
        if not is_stock_code(code):
            continue
        rows.append({"code": code, "date": d, "foreign_net": integer(r[i_foreign]),
                     "itrust_net": integer(r[i_itrust]), "dealer_self_net": integer(r[i_self]),
                     "dealer_hedge_net": integer(r[i_hedge]) if i_hedge is not None else 0,
                     "total_net": integer(r[i_total])})
    return pd.DataFrame(rows, columns=cols)


# --- margin and valuation ----------------------------------------------------------------------

MARGIN_COLS = ["code", "date", "margin_buy", "margin_sell", "margin_prev", "margin_balance",
               "short_sell", "short_buy", "short_prev", "short_balance"]


def parse_margin(payload: dict, d: date) -> pd.DataFrame:
    """融資融券彙總 (lots). Layout: code, name, 融資 買進/賣出/現金償還/前日餘額/今日餘額/限額,
    融券 買進/賣出/現券償還/前日餘額/今日餘額/限額, 資券互抵, 註記."""
    t = _find_table(payload, ("代號", "股票代號"))
    if t is None:
        return pd.DataFrame(columns=MARGIN_COLS)
    rows = []
    for r in t["data"]:
        code = strip_html(r[0])
        if not is_stock_code(code) or len(r) < 13:
            continue
        rows.append({"code": code, "date": d, "margin_buy": integer(r[2]), "margin_sell": integer(r[3]),
                     "margin_prev": integer(r[5]), "margin_balance": integer(r[6]),
                     "short_buy": integer(r[8]), "short_sell": integer(r[9]),
                     "short_prev": integer(r[11]), "short_balance": integer(r[12])})
    return pd.DataFrame(rows, columns=MARGIN_COLS)


def parse_valuation(payload: dict, d: date) -> pd.DataFrame:
    cols = ["code", "date", "pe", "pb", "div_yield"]
    t = _find_table(payload, ("證券代號",), "本益比")
    if t is None:
        return pd.DataFrame(columns=cols)
    f = t["fields"]
    i_pe, i_pb, i_dy = field_index(f, "本益比"), field_index(f, "股價淨值比"), field_index(f, "殖利率(%)")
    rows = [{"code": strip_html(r[0]), "date": d, "pe": num(r[i_pe]), "pb": num(r[i_pb]),
             "div_yield": num(r[i_dy])} for r in t["data"] if is_stock_code(strip_html(r[0]))]
    return pd.DataFrame(rows, columns=cols)


# --- corporate actions and calendar ------------------------------------------------------------

def parse_ex_rights_calendar(payload: dict) -> pd.DataFrame:
    """除權除息預告表. ``kind`` is the exchange label: 息 (cash), 權 (stock) or 權息 (both).
    Cash dividends not yet announced ('待公告') are None."""
    cols = ["ex_date", "code", "name", "kind", "cash_dividend", "stock_dividend_ratio"]
    t = _find_table(payload, ("除權除息日期",))
    if t is None:
        return pd.DataFrame(columns=cols)
    f = t["fields"]
    i_code, i_name, i_kind = field_index(f, "股票代號"), field_index(f, "名稱"), field_index(f, "除權息")
    i_cash, i_stock = field_index(f, "現金股利"), field_index(f, "無償配股率")
    rows = []
    for r in t["data"]:
        try:
            ex = roc_date(r[0])
        except ValueError:
            continue
        rows.append({"ex_date": ex, "code": strip_html(r[i_code]), "name": strip_html(r[i_name]),
                     "kind": strip_html(r[i_kind]), "cash_dividend": num(r[i_cash]),
                     "stock_dividend_ratio": num(r[i_stock])})
    return pd.DataFrame(rows, columns=cols)


def parse_holidays(payload: list[dict]) -> pd.DataFrame:
    rows = [{"date": roc_date(r["Date"]), "name": strip_html(r.get("Name", "")),
             "description": strip_html(r.get("Description", ""))} for r in payload]
    return pd.DataFrame(rows, columns=["date", "name", "description"])


def is_trading_notice(name: str) -> bool:
    """Schedule rows that announce a trading day (first/last trading day around holidays) rather
    than a closure."""
    return "開始交易" in name or "最後交易" in name
