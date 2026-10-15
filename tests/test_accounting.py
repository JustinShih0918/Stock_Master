from datetime import date
from decimal import Decimal

import pytest

from stockmaster.accounting.fees import make_fill
from stockmaster.accounting.ledger import LedgerError, apply_fills, run_day
from stockmaster.domain import Order, PortfolioState, Side, limit_prices
from stockmaster.execution.simulator import vwap_from_value


def test_buy_fee_hand_computed(rules):
    f = make_fill("2330", Side.BUY, 5, Decimal("100.50"), rules.fees)
    assert f.gross == Decimal("502500.00")
    assert f.commission == Decimal(716)  # 502,500 * 0.1425% = 716.06 -> floor
    assert f.tax == 0
    assert f.net_cash == Decimal("-503216.00")


def test_sell_fee_and_tax_hand_computed(rules):
    f = make_fill("2330", Side.SELL, 5, Decimal(110), rules.fees)
    assert f.gross == Decimal(550000)
    assert f.commission == Decimal(783)  # 783.75 -> floor
    assert f.tax == Decimal(1650)
    assert f.net_cash == Decimal(547567)


@pytest.mark.parametrize("prev, up, dn", [
    (100, "110", "90"),
    (2510, "2760", "2260"),       # tick 5 above 1000
    (19040, "20940", "17140"),    # matches TPEx NextLimitUp/Down for 5274 on 2026-10-01
    (9.99, "10.95", "9.00"),      # crosses the 10-dollar tick boundary
    (48.0, "52.8", "43.2"),
])
def test_limit_prices_tick_rounding(prev, up, dn):
    u, d = limit_prices(prev)
    assert (u, d) == (Decimal(up), Decimal(dn))


def test_vwap_from_value():
    # TWSE 2330 on 2026-10-01: TradeValue 51,702,445,610 / TradeVolume 20,666,092
    assert vwap_from_value(51702445610, 20666092) == Decimal("2501.80")


def test_run_day_applied_and_nav(rules):
    s0 = PortfolioState.initial(rules.initial_cash)
    r = run_day(s0, date(2026, 10, 27), [Order("2330", Side.BUY, 10)],
                vwaps={"2330": Decimal(1000)}, closes={"2330": 1010}, rules=rules)
    cost = Decimal(10_000_000) + Decimal(14250)
    assert r.state.cash == Decimal(rules.initial_cash) - cost
    assert r.valuation.nav == r.state.cash + Decimal(10_100_000)
    assert r.state.lots == {"2330": 10}


def test_voided_day_keeps_previous_portfolio(rules):
    s0 = PortfolioState(as_of=date(2026, 10, 27), cash=Decimal(500), lots={"2330": 2})
    r = run_day(s0, date(2026, 10, 28), [Order("2454", Side.BUY, 1)],
                vwaps={"2454": 1000}, closes={"2330": 1000}, rules=rules,
                applied=False, warning=True)
    assert r.state.lots == {"2330": 2}
    assert r.state.cash == Decimal(500)
    assert r.state.warnings == 1
    assert r.fills == []


def test_dividends_accrue_separately(rules):
    s0 = PortfolioState(as_of=date(2026, 10, 27), cash=Decimal(0), lots={"2330": 3})
    r = run_day(s0, date(2026, 10, 28), [Order("2330", Side.SELL, 3)],
                vwaps={"2330": 1000}, closes={"2330": 1000}, rules=rules,
                cash_dividends_per_share={"2330": Decimal(5)})
    # Entitlement from the previous close's 3 lots even though they were sold on the ex-date.
    assert r.state.accrued_dividends == Decimal(15000)
    assert r.valuation.nav == r.state.cash  # dividends excluded from organizer NAV
    assert r.valuation.nav_with_dividends == r.state.cash + Decimal(15000)


def test_overweight_days_counter(rules):
    s = PortfolioState(as_of=date(2026, 10, 27), cash=Decimal(800_000), lots={"2454": 1})
    r1 = run_day(s, date(2026, 10, 28), [], {}, {"2454": 1000}, rules)
    assert r1.state.overweight_days == {"2454": 1}  # 1.0M / 1.8M = 55%
    r2 = run_day(r1.state, date(2026, 10, 29), [], {}, {"2454": 1000}, rules)
    assert r2.state.overweight_days == {"2454": 2}


def test_ledger_rejects_short_and_day_trade(rules):
    s = PortfolioState(as_of=None, cash=Decimal(10_000_000), lots={"2330": 1})
    with pytest.raises(LedgerError):
        apply_fills(s, [make_fill("2330", Side.SELL, 2, 100, rules.fees)])
    with pytest.raises(LedgerError):
        apply_fills(s, [make_fill("2330", Side.BUY, 1, 100, rules.fees),
                        make_fill("2330", Side.SELL, 1, 100, rules.fees)])
