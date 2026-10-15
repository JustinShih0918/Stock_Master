"""Active-ETF list and holdings adapters. Offline: parses real raw responses saved as fixtures.

Fixtures under tests/fixtures/active_etf/ are REAL issuer responses captured on 2026-10-02
(Asia/Taipei); they are not synthetic (00996A.html is a trimmed excerpt). Network tests are marked
``network`` and run only with SM_NETWORK_TESTS=1.
"""

from __future__ import annotations

import importlib
import logging
import os
import warnings
from datetime import date
from pathlib import Path

import httpx
import pytest

from stockmaster.data.sources import active_etf as A
from stockmaster.data.sources.active_etf_issuers import registry
from stockmaster.data.sources.active_etf_issuers._http import redact
from stockmaster.data.sources.active_etf_issuers.base import (Adapter, EtfHolding,
                                                              EtfHoldingsSnapshot, is_equity_line,
                                                              num, parse_date, pct, snapshot)
from stockmaster.timeutil import now_taipei

FIXTURES = Path(__file__).parent / "fixtures" / "active_etf"
ISSUERS_PKG = "stockmaster.data.sources.active_etf_issuers"

# The 'network' marker is not registered in pyproject.toml (outside this module's scope), so
# build it once with the unknown-mark warning silenced; skipped unless SM_NETWORK_TESTS=1.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    _network_mark = pytest.mark.network
_skip_offline = pytest.mark.skipif(os.environ.get("SM_NETWORK_TESTS") != "1",
                                   reason="network test; set SM_NETWORK_TESTS=1 to run")


def network(f):
    return _network_mark(_skip_offline(f))


# (adapter module, ETF code, fixture file, as_of, #equity lines, top stock code, its weight,
#  domestic TW-equity fund?)
CASES = [
    ("alliancebernstein", "00404A", "00404A.json", date(2026, 10, 1), 51, "2330", 0.23104555, True),
    ("allianz", "00984A", "00984A.json", date(2026, 10, 1), 94, "2330", 0.0424, True),
    ("capital", "00982A", "00982A.json", date(2026, 10, 1), 55, "2330", 0.088907, True),
    ("cathay", "00400A", "00400A.json", date(2026, 10, 1), 45, "2330", 0.0895, True),
    ("ctbc", "00995A", "00995A.json", date(2026, 10, 2), 56, "2330", 0.089, True),
    ("first_sitc", "00994A", "00994A.json", date(2026, 10, 1), 41, "2330", 0.1455, True),
    ("fubon", "00405A", "00405A.html", date(2026, 10, 1), 46, "3037", 0.089828, True),
    ("fuh_hwa", "00991A", "00991A.json", date(2026, 10, 1), 44, "2330", 0.13273, True),
    ("fuh_hwa", "00998A", "00998A.json", date(2026, 10, 1), 53, "SDLF LN", 0.04378, False),
    ("jpmorgan", "00401A", "00401A.json", date(2026, 10, 1), 56, "2330", 0.202181, True),
    ("kgi", "00407A", "00407A.html", date(2026, 10, 1), 50, "2330", 0.0926, True),
    ("mega", "00996A", "00996A.html", date(2026, 10, 2), 52, "2330", 0.0891, True),
    ("nomura", "00980A", "00980A.json", date(2026, 10, 1), 50, "2330", 0.0942, True),
    ("sinopac", "00410A", "00410A.html", date(2026, 10, 1), 29, "2330", 0.0585, True),
    ("taishin", "00987A", "00987A.html", date(2026, 10, 1), 29, "2330", 0.088901, True),
    ("uni_president", "00981A", "00981A.json", date(2026, 10, 1), 34, "2330", 0.099, True),
    ("uni_president", "00988A", "00988A.json", date(2026, 9, 30), 42, "LITE US", 0.0688, False),
    ("yuanta", "00990A", "00990A.json", date(2026, 10, 1), 47, "LITE US", 0.0763, False),
    ("aggregator_etfinfo", "00981A", "etfinfo_00981A.json", date(2026, 10, 1), 34, "2330", 0.099,
     True),
]


def _parse(module: str, etf: str, fixture: str):
    mod = importlib.import_module(f"{ISSUERS_PKG}.{module}")
    return mod.parse((FIXTURES / fixture).read_bytes(), etf)


def _snap(etf: str, as_of: date, holdings) -> EtfHoldingsSnapshot:
    return EtfHoldingsSnapshot(etf, as_of, now_taipei(), "file://fixture", tuple(holdings), b"", "")


@pytest.mark.parametrize("module, etf, fixture, as_of, n_eq, top_code, top_w, domestic", CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_fixture_parses(module, etf, fixture, as_of, n_eq, top_code, top_w, domestic):
    d, holdings = _parse(module, etf, fixture)
    assert d == as_of
    assert all(isinstance(h, EtfHolding) and 0 <= h.weight <= 1 for h in holdings)
    w = A.holdings_weights(_snap(etf, d, holdings))
    assert len(w) == n_eq
    top = max(w, key=w.get)
    assert top == top_code
    assert w[top] == pytest.approx(top_w, abs=1e-6)
    assert 0.85 <= sum(w.values()) <= 1.02
    if domestic:
        assert "2330" in w
        assert all(len(c) in (4, 5, 6) for c in w)


_NO_FUND_ID_IN_PAYLOAD = {"capital", "cathay"}   # payloads identify the fund only by request id


_GUARDED = [c for c in CASES if c[0] not in _NO_FUND_ID_IN_PAYLOAD]


@pytest.mark.parametrize("module, etf, fixture", [c[:3] for c in _GUARDED],
                         ids=[f"{c[0]}-{c[1]}" for c in _GUARDED])
def test_parser_rejects_payload_of_another_fund(module, etf, fixture):
    with pytest.raises(ValueError):
        _parse(module, "00000A", fixture)


def _fake_adapter(name: str, result, official: bool = True) -> Adapter:
    def fetch(code, client):
        if isinstance(result, Exception):
            raise result
        return _snap(code, date(2026, 10, 1), [EtfHolding("2330", "台積電", 0.1, 1)])
    return Adapter(name, ("00999A",), fetch, official)


@pytest.mark.parametrize("official_result, used", [
    (None, "official"),
    (A.HoldingsUnavailable("00999A", "boom"), "fallback"),
    (httpx.ConnectError("down"), "fallback"),
])
def test_fetch_holdings_dispatch_and_fallback(monkeypatch, caplog, official_result, used):
    off = _fake_adapter("official", official_result)
    fb = _fake_adapter("etfinfo-aggregator-fallback", None, official=False)
    calls = []
    monkeypatch.setattr(A, "registry", lambda: ({"00999A": _tracking(off, calls)},
                                                {"00999A": _tracking(fb, calls)}))
    with httpx.Client() as c:
        snap = A.fetch_holdings("00999a", client=c)
    assert snap.etf_code == "00999A"
    assert calls[-1] == used
    assert ("aggregator fallback" in caplog.text) == (used == "fallback")


def test_fetch_holdings_reports_every_failure(monkeypatch):
    off = _fake_adapter("official", A.HoldingsUnavailable("00999A", "parse error"))
    fb = _fake_adapter("fb", httpx.ReadTimeout("slow"), official=False)
    monkeypatch.setattr(A, "registry", lambda: ({"00999A": off}, {"00999A": fb}))
    with httpx.Client() as c, pytest.raises(A.HoldingsUnavailable) as e:
        A.fetch_holdings("00999A", client=c)
    assert "official: parse error" in e.value.reason and "fb: ReadTimeout" in e.value.reason


def _tracking(adapter: Adapter, calls: list[str]) -> Adapter:
    def fetch(code, client):
        calls.append("official" if adapter.official else "fallback")
        return adapter.fetch(code, client)
    return Adapter(adapter.name, adapter.codes, fetch, adapter.official)


def test_snapshot_wraps_parse_errors_and_empty_holdings():
    resp = httpx.Response(200, content=b"{}", headers={"content-type": "application/json"},
                          request=httpx.Request("GET", "https://example.test/x"))
    with pytest.raises(A.HoldingsUnavailable, match="cannot parse"):
        snapshot("00999A", "https://example.test/x", resp, lambda raw, code: ({}["missing"]))
    with pytest.raises(A.HoldingsUnavailable, match="no holdings"):
        snapshot("00999A", "https://example.test/x", resp,
                 lambda raw, code: (date(2026, 10, 1), ()))
    s = snapshot("00999A", "https://example.test/x", resp,
                 lambda raw, code: (date(2026, 10, 1), (EtfHolding("2330", "x", 0.1, 1),)))
    assert s.raw == b"{}" and s.raw_content_type == "application/json"
    assert s.fetched_at.utcoffset().total_seconds() == 8 * 3600


def test_query_tokens_are_redacted_in_logs(caplog):
    caplog.set_level(logging.INFO, logger="httpx")
    url = "https://api.example.test/etf?token=s3cr3t&FID=E0036"
    assert redact(url) == "https://api.example.test/etf?token=<redacted>&FID=E0036"
    logging.getLogger("httpx").info("HTTP Request: %s %s", "POST", httpx.URL(url))
    assert "s3cr3t" not in caplog.text and "token=<redacted>" in caplog.text


def test_every_official_adapter_has_a_fixture_case():
    official, _ = registry()
    tested = {a.name for c in CASES
              for a in importlib.import_module(f"{ISSUERS_PKG}.{c[0]}").ADAPTERS}
    assert {a.name for a in official.values()} <= tested


def test_supported_etfs_prefers_official_and_labels_fallback():
    s = A.supported_etfs()
    assert s["00980A"] == "nomura"
    assert all(v for v in s.values())
    official, _ = registry()
    for code, name in s.items():
        if code not in official:
            assert "fallback" in name


def test_fetch_holdings_unknown_code_raises():
    with pytest.raises(A.HoldingsUnavailable) as e:
        A.fetch_holdings("12345Z")
    assert e.value.etf_code == "12345Z"


def test_holdings_weights_drops_non_stock_lines_and_sums_duplicates():
    lines = [EtfHolding("2330", "台積電", 0.10, 1000), EtfHolding("2330", "台積電", 0.01, 100),
             EtfHolding("TX", "台灣證交所加權股價指數", 0.03, 62),
             EtfHolding("TXFJ6", "台指期貨", 0.02, 5), EtfHolding("", "現金", 0.05, None),
             EtfHolding("CASH_TWD", "Cash", 0.01, None), EtfHolding("00878", "國泰永續高股息", 0.02, 1),
             EtfHolding("2881A", "富邦特", 0.01, 1), EtfHolding("NVDA US", "NVIDIA", 0.04, 10),
             EtfHolding("1234", "保證金", 0.01, None)]
    w = A.holdings_weights(_snap("X", date(2026, 10, 1), lines))
    assert w == pytest.approx({"2330": 0.11, "00878": 0.02, "2881A": 0.01, "NVDA US": 0.04})


def test_is_equity_line():
    assert is_equity_line(EtfHolding("6488", "環球晶", 0.01, 1))
    assert not is_equity_line(EtfHolding("MXF", "小型台指", 0.01, 1))


@pytest.mark.parametrize("s, d", [("2026/10/01", date(2026, 10, 1)),
                                  ("2026-10-01T00:00:00", date(2026, 10, 1)),
                                  ("20261001", date(2026, 10, 1)), ("115/10/01", date(2026, 10, 1)),
                                  ("1151001", date(2026, 10, 1)), ("115年10月1日", date(2026, 10, 1))])
def test_parse_date(s, d):
    assert parse_date(s) == d


def test_number_helpers():
    assert num("1,234.5") == 1234.5
    assert num("--") is None
    assert pct("9.42%") == pytest.approx(0.0942)
    assert pct(" 9.42 ") == 0.0942


# --- list ------------------------------------------------------------------------------------

def test_parse_isin_html_and_mis_classes():
    rows = A.parse_isin_html((FIXTURES / "isin_twse.html").read_bytes())
    by = {r["code"]: r for r in rows}
    assert by["00981A"]["name"] == "主動統一台股增長"
    assert by["00981A"]["listed"] == "2025/05/27"
    classes = A.parse_mis_classes((FIXTURES / "mis_all_etf.json").read_bytes())
    assert classes["00981A"] == "1" and classes["00983A"] == "4"
    etf = A._to_active_etf(by["00981A"] | {"market": "TWSE"}, classes.get("00981A"))
    assert etf == A.ActiveEtf("00981A", "主動統一台股增長", "統一投信", "TWSE", date(2025, 5, 27),
                              "tw_equity")
    assert A._to_active_etf(by["00983A"] | {"market": "TWSE"}, "4").asset_class == "foreign_equity"
    assert A._to_active_etf(by["00982D"] | {"market": "TWSE"}, "4").asset_class == "bond"


def test_reference_etfs_filters_by_class_and_listing_date():
    etfs = [A.ActiveEtf("00981A", "x", "統一投信", "TWSE", date(2025, 5, 27), "tw_equity"),
            A.ActiveEtf("00412A", "y", "統一投信", "TWSE", date(2026, 9, 20), "tw_equity"),
            A.ActiveEtf("00983A", "z", "中國信託投信", "TWSE", date(2025, 6, 18), "foreign_equity")]
    assert [e.code for e in A.reference_etfs(etfs)] == ["00981A"]


def test_issuer_from_name():
    assert A._issuer("主動第一金台股優") == "第一金投信"
    assert A._issuer("兆豐主動台灣豐收") == "兆豐投信"
    assert A._issuer("主動不存在台灣") == "unknown"


# --- live ------------------------------------------------------------------------------------

@network
def test_live_list():
    etfs = A.list_active_etfs()
    ref = A.reference_etfs(etfs)
    assert len(ref) >= 20
    assert {"00980A", "00981A", "00982A"} <= {e.code for e in ref}


@network
@pytest.mark.parametrize("code", sorted(A.supported_etfs()))
def test_live_fetch(code):
    s = A.fetch_holdings(code)
    w = A.holdings_weights(s)
    assert s.fetched_at.tzinfo is not None
    assert (s.fetched_at.date() - s.as_of).days <= 7
    assert 0.5 <= sum(w.values()) <= 1.02
