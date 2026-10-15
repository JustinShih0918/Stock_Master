"""Offline parser tests on trimmed real payloads from 2026-10-01 (TWSE/TPEx) and 2026-08 (MOPS)."""

import json
from datetime import date
from pathlib import Path

import pytest

from stockmaster.data.sources import mops, tpex, twse
from stockmaster.domain import limit_prices

FX = Path(__file__).parent / "fixtures"
D = date(2026, 10, 1)


def load(rel):
    return json.loads((FX / rel).read_text(encoding="utf-8"))


def test_twse_quotes_and_vwap():
    q = twse.parse_quotes(load("twse/MI_INDEX_20261001.json"), D).set_index("code")
    r = q.loc["2330"]
    assert r.close == 2510 and r.open == 2490 and r.volume_sh == 20_666_092
    assert r.vwap == pytest.approx(51_702_445_610 / 20_666_092)
    assert r.change == 30 and r.prev_close == 2480
    assert q.loc["2883"].change == pytest.approx(-1.95)


def test_twse_index_and_breadth():
    idx, breadth = twse.parse_index_and_breadth(load("twse/MI_INDEX_20261001.json"), D)
    taiex = idx[idx.index_name == "發行量加權股價指數"]
    assert len(taiex) == 1 and taiex.iloc[0].close > 10000
    assert breadth  # up/down counts parsed


def test_twse_institutional():
    df = twse.parse_institutional(load("twse/T86_20261001.json"), D).set_index("code")
    r = df.loc["2330"]
    assert r.foreign_net == 3_090_730
    assert r.itrust_net == 596_450
    assert r.dealer_self_net == 249_100 and r.dealer_hedge_net == 353_428
    assert r.total_net == 4_289_708


def test_twse_margin_and_valuation():
    m = twse.parse_margin(load("twse/MI_MARGN_20261001.json"), D).set_index("code")
    assert m.loc["2330"].margin_balance == 30_658 and m.loc["2330"].short_balance == 20
    v = twse.parse_valuation(load("twse/BWIBBU_20261001.json"), D).set_index("code")
    assert v.loc["2330"].pe == pytest.approx(29.09) and v.loc["2330"].pb == pytest.approx(10.12)


def test_twse_ex_rights_calendar():
    cal = twse.parse_ex_rights_calendar(load("twse/TWT48U.json"))
    assert {"ex_date", "code", "kind", "cash_dividend"} <= set(cal.columns)
    assert cal.ex_date.min() >= date(2026, 10, 1)


def test_twse_holidays_include_oct26_makeup_day():
    h = twse.parse_holidays(load("twse/holidaySchedule_115.json"))
    assert date(2026, 10, 26) in set(h.date)
    assert date(2026, 10, 9) in set(h.date)


def test_tpex_quotes_limits_match_tick_rule():
    q = tpex.parse_quotes(load("tpex/dailyQuotes_20261001.json"), D).set_index("code")
    r = q.loc["5274"]
    assert r.vwap_published == pytest.approx(19035.23)
    assert r.vwap == pytest.approx(2_396_725_205 / 125_910)
    up, dn = limit_prices(r.next_ref_price)
    assert (float(up), float(dn)) == (r.next_limit_up, r.next_limit_down)


def test_tpex_institutional_margin_valuation():
    i = tpex.parse_institutional(load("tpex/dailyTrade_20261001.json"), D).set_index("code")
    r = i.loc["5274"]
    assert (r.foreign_net, r.itrust_net, r.dealer_self_net, r.dealer_hedge_net, r.total_net) == \
        (-15242, 802, 1205, 1374, -11861)
    m = tpex.parse_margin(load("tpex/margin_20261001.json"), D).set_index("code")
    assert m.loc["5274"].margin_balance == 137
    v = tpex.parse_valuation(load("tpex/peQryDate_20261001.json"), D).set_index("code")
    assert v.loc["5274"].pe == pytest.approx(127.01)


def test_mops_monthly_revenue():
    html = (FX / "mops/t21sc03_sii_115_8.html").read_text(encoding="utf-8")
    df = mops.parse_monthly_revenue(html, "TWSE", 2026, 8).set_index("code")
    r = df.loc["2330"]
    assert r.revenue == 514_805_337
    assert r.yoy_pct == pytest.approx(53.32)
    assert r.cum_yoy_pct == pytest.approx(39.26)


def test_holiday_schedule_trading_notices_are_open_days():
    h = twse.parse_holidays(load("twse/holidaySchedule_115.json")).set_index("date")
    for d in (date(2026, 1, 2), date(2026, 2, 11), date(2026, 2, 23)):
        assert twse.is_trading_notice(h.loc[d, "name"]), d
    for d in (date(2026, 2, 12), date(2026, 2, 17), date(2026, 10, 26)):
        assert not twse.is_trading_notice(h.loc[d, "name"]), d


def test_twse_institutional_pre_2017_layout():
    df = twse.parse_institutional(load("twse/T86_20150603_old_layout.json"), date(2015, 6, 3)).set_index("code")
    r = df.loc["2330"]
    assert r.foreign_net == -1_142_009 and r.itrust_net == -1_284_000
    assert r.dealer_self_net == -1_518_000 and r.dealer_hedge_net == -1_022_000
    assert r.total_net == -4_966_009


def test_taiex_name_variants_are_normalized():
    for n in ("發行量加權股價指數", "加權股價指數", " 加權股價指数 "):
        assert twse.normalize_index_name(n) == "發行量加權股價指數"
    assert twse.normalize_index_name("臺灣50指數") == "臺灣50指數"


def test_twse_institutional_pre_2014_12_layout():
    df = twse.parse_institutional(load("twse/T86_20141128_pre_dealer_split.json"), date(2014, 11, 28)).set_index("code")
    r = df.loc["2330"]
    assert (r.foreign_net, r.itrust_net, r.dealer_self_net, r.dealer_hedge_net, r.total_net) == \
        (10_563_856, -370_508, 119_000, 0, 10_312_348)
