"""Active ETFs: the listed set (TWSE ISIN registry + MIS asset classes) and their holdings, fetched
from the issuer first and the etfinfo.tw aggregator second.

The competition's Active Share rule compares our portfolio with the top-10 holdings of every
*reference* active ETF: domestic Taiwan-equity active ETFs listed on or before the reference-list
date (rules.active_share.reference_list_date).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date

import httpx

from stockmaster.data.sources import isin
from stockmaster.data.sources.active_etf_issuers import _http
from stockmaster.data.sources.active_etf_issuers import registry as _registry
from stockmaster.data.sources.active_etf_issuers.base import (EtfHoldingsSnapshot, HoldingsUnavailable,
                                                              is_equity_line, parse_date)

log = logging.getLogger(__name__)
MIS_ALL_ETF = "https://mis.twse.com.tw/stock/data/all_etf.txt"

__all__ = ["ActiveEtf", "HoldingsUnavailable", "fetch_holdings", "holdings_weights", "list_active_etfs",
           "parse_isin_html", "parse_mis_classes", "reference_etfs", "registry", "supported_etfs"]

parse_isin_html = isin.parse_isin_html
registry = _registry

ISSUERS = (("國泰", "國泰投信"), ("摩根", "摩根投信"), ("安聯", "安聯投信"), ("統一", "統一投信"),
           ("聯博", "聯博投信"), ("富邦", "富邦投信"), ("中信", "中國信託投信"), ("凱基", "凱基投信"),
           ("第一金", "第一金投信"), ("復華", "復華投信"), ("永豐", "永豐投信"), ("野村", "野村投信"),
           ("群益", "群益投信"), ("台新", "台新投信"), ("元大", "元大投信"), ("兆豐", "兆豐投信"),
           ("貝萊德", "貝萊德投信"))


@dataclass(frozen=True)
class ActiveEtf:
    code: str
    name: str
    issuer: str
    market: str
    listed_date: date | None
    asset_class: str  # tw_equity | foreign_equity | bond


def _issuer(name: str) -> str:
    for token, issuer in ISSUERS:
        if token in name:
            return issuer
    return "unknown"


def parse_mis_classes(raw: bytes) -> dict[str, str]:
    """MIS all_etf feed: code -> category ('1' = domestic equity; other values = foreign/bond)."""
    d = json.loads(raw.decode("utf-8-sig"))
    out = {}
    for block in d.get("a1", []):
        for m in block.get("msgArray", []) or []:
            out[m["a"]] = str(m.get("k", ""))
    return out


def _to_active_etf(row: dict, mis_class: str | None) -> ActiveEtf:
    code = row["code"]
    if code.endswith("D"):
        asset_class = "bond"
    elif mis_class == "1":
        asset_class = "tw_equity"
    else:
        asset_class = "foreign_equity"
    listed = parse_date(row["listed"]) if row.get("listed") else None
    return ActiveEtf(code, row["name"], _issuer(row["name"]), row.get("market", ""), listed, asset_class)


def list_active_etfs(client: httpx.Client | None = None) -> list[ActiveEtf]:
    """Active ETFs (codes ending in A or D) on TWSE and TPEx. Without the MIS feed, every active
    equity ETF is classed foreign (fail-safe: nothing is wrongly added to the reference list)."""
    own = client is None
    client = client or _http.client()
    try:
        rows = []
        for market in ("TWSE", "TPEx"):
            r = _http.get(client, isin.URL, params=isin.params(market, "etf"))
            rows += [x | {"market": market} for x in isin.parse_isin_html(r.content)]
        try:
            classes = parse_mis_classes(_http.get(client, MIS_ALL_ETF).content)
        except (httpx.HTTPError, ValueError) as e:
            log.warning("MIS ETF classes unavailable (%s); asset classes default to foreign", e)
            classes = {}
        return [_to_active_etf(r, classes.get(r["code"])) for r in rows if r["code"][-1:] in ("A", "D")]
    finally:
        if own:
            client.close()


def reference_etfs(etfs: list[ActiveEtf], as_of: date | None = None) -> list[ActiveEtf]:
    if as_of is None:
        from stockmaster.config import load_rules
        as_of = load_rules().active_share.reference_list_date
    return [e for e in etfs if e.asset_class == "tw_equity" and e.listed_date is not None
            and e.listed_date <= as_of]


def supported_etfs() -> dict[str, str]:
    """code -> adapter name; codes only the aggregator covers are labelled as fallback."""
    official, fallback = registry()
    out = {code: a.name for code, a in fallback.items()}
    out.update({code: a.name for code, a in official.items()})
    return dict(sorted(out.items()))


def fetch_holdings(code: str, client: httpx.Client | None = None) -> EtfHoldingsSnapshot:
    code = code.strip().upper()
    official, fallback = registry()
    chain = [a for a in (official.get(code), fallback.get(code)) if a is not None]
    if not chain:
        raise HoldingsUnavailable(code, "no adapter for this ETF")
    own = client is None
    client = client or _http.client()
    failures: list[str] = []
    try:
        for i, adapter in enumerate(chain):
            if i > 0:
                log.warning("%s: %s; using aggregator fallback %s", code, failures[-1], adapter.name)
            try:
                return adapter.fetch(code, client)
            except HoldingsUnavailable as e:
                failures.append(f"{adapter.name}: {e.reason}")
            except httpx.HTTPError as e:
                failures.append(f"{adapter.name}: {type(e).__name__}")
    finally:
        if own:
            client.close()
    raise HoldingsUnavailable(code, "; ".join(failures))


def holdings_weights(s: EtfHoldingsSnapshot) -> dict[str, float]:
    """Stock weights (fractions of NAV) with cash, margin, futures and options removed and
    duplicate lines summed; zero-weight lines are dropped."""
    out: dict[str, float] = {}
    for h in s.holdings:
        if is_equity_line(h) and h.weight > 0:
            out[h.code] = out.get(h.code, 0.0) + h.weight
    return out
