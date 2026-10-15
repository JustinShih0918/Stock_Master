from datetime import date
from decimal import Decimal

import pytest

from stockmaster.domain import Order, PortfolioState, Quote, Side
from stockmaster.risk import stress
from stockmaster.risk.active_share import (active_share_normalized, active_share_raw,
                                           active_share_report)
from stockmaster.risk.validator import ValidationContext, validate_orders

from .conftest import AS_OF, TRADE_DATE


def flat_quotes(codes, price=100.0):
    return {c: Quote(c, price, price * 1.1, price * 0.9, AS_OF) for c in codes}


@pytest.mark.parametrize("invest, ok_floor, ok_ceiling", [
    (907_000_000, True, True),
    (909_000_000, False, True),    # limit-up fills would overdraw cash
    (834_000_000, True, True),
    (831_000_000, True, False),    # limit-down day leaves cash >= 25%
])
def test_day1_feasible_band(rules, invest, ok_floor, ok_ceiling):
    """Day 1 from all cash: X <= 907.8M (cash >= 0) and X > 832.5M (cash < 25%)."""
    s = PortfolioState.initial(rules.initial_cash)
    q = flat_quotes(["A"], 100.0)
    orders = [Order("A", Side.BUY, int(invest / 100_000))]
    floor_ok = stress.worst_case_min_cash(s, orders, q, rules.fees) >= 0
    ceiling_ok = stress.worst_case_max_cash_ratio(s, orders, q, rules.fees) < 0.25
    assert (floor_ok, ceiling_ok) == (ok_floor, ok_ceiling)


def test_stressed_weight_formula():
    q = flat_quotes(["A", "B"], 100.0)
    lots = {"A": 1, "B": 9}
    w = stress.stressed_weight("A", lots, 0.0, q, rest_move=0.0)
    assert w == pytest.approx(110 / (110 + 900))


def test_active_share_formulas():
    p = {"A": 0.5, "B": 0.3, "C": 0.2}
    b = {"A": 0.4, "B": 0.1, "D": 0.5}
    assert active_share_raw(p, b, 10) == pytest.approx(0.5 * (0.1 + 0.2 + 0.2 + 0.5))
    assert active_share_normalized(p, b, 10) == pytest.approx(0.5)
    # identical portfolios -> 0
    assert active_share_raw(p, p) == 0 and active_share_normalized(p, p) == 0


def test_active_share_uses_top10_only():
    p = {f"S{i:02d}": 0.04 for i in range(25)}
    p["S00"] = 0.2
    b = dict(p)
    r = active_share_report(p, {"ETF": b})[0]
    assert r.value == pytest.approx(0.0)
    assert len(r.overlap) == 10


def _ctx(rules, strategy, state, quotes, universe, **kw):
    return ValidationContext(trade_date=TRADE_DATE, state=state, quotes=quotes,
                             universe=frozenset(universe), rules=rules, strategy=strategy, **kw)


def test_validator_flags_structural_violations(rules, strategy, universe, quotes):
    s = PortfolioState(as_of=AS_OF, cash=Decimal(10**9), lots={"2330": 10})
    orders = [Order("2330", Side.SELL, 20), Order("9999", Side.BUY, 1),
              Order("2454", Side.BUY, 1), Order("2454", Side.SELL, 1)]
    q = dict(quotes)
    q["9999"] = Quote("9999", 10, 11, 9, AS_OF)
    rep = validate_orders(orders, _ctx(rules, strategy, s, q, universe.codes))
    failed = {r.rule_id for r in rep.violations}
    assert {"R_NO_SHORT", "R_UNIVERSE", "R_NO_BUY_SELL_SAME", "R_ONE_ORDER_PER_STOCK"} <= failed


def _equal_portfolio(rules, n, price=100.0, invest=0.95):
    codes = [f"{1000 + i}" for i in range(n)]
    lots_each = int(rules.initial_cash * invest / n / (price * 1000))
    cash = rules.initial_cash - lots_each * n * price * 1000
    return codes, PortfolioState(as_of=AS_OF, cash=Decimal(cash), lots={c: lots_each for c in codes})


@pytest.mark.parametrize("n, ok", [(19, False), (20, True), (30, True), (31, False)])
def test_holding_count_bounds(rules, strategy, n, ok):
    codes, s = _equal_portfolio(rules, n)
    rep = validate_orders([], _ctx(rules, strategy, s, flat_quotes(codes), codes))
    assert next(r for r in rep.results if r.rule_id == "R_COUNT").passed is ok


def test_buy_into_capped_name_is_blocked(rules, strategy):
    codes, s = _equal_portfolio(rules, 20)
    lots = dict(s.lots)
    lots[codes[0]] = 1010  # ~10.6% of NAV, over the 10% cap
    s = s.with_(lots=lots, cash=Decimal(60_000_000))
    rep = validate_orders([Order(codes[0], Side.BUY, 1)],
                          _ctx(rules, strategy, s, flat_quotes(codes), codes))
    assert any(r.rule_id == "R_CAP_ADD" for r in rep.violations)


def test_drift_must_be_trimmed_after_grace(rules, strategy):
    codes, s = _equal_portfolio(rules, 20)
    lots = dict(s.lots)
    lots[codes[0]] = 1100
    s = s.with_(lots=lots, cash=Decimal(40_000_000), overweight_days={codes[0]: 2})
    rep = validate_orders([], _ctx(rules, strategy, s, flat_quotes(codes), codes))
    assert any(r.rule_id == "R_CAP_DRIFT" for r in rep.violations)
    # A fresh drift (1 day) is tolerated.
    s1 = s.with_(overweight_days={codes[0]: 1})
    rep1 = validate_orders([], _ctx(rules, strategy, s1, flat_quotes(codes), codes))
    assert not any(r.rule_id == "R_CAP_DRIFT" for r in rep1.violations)


def _halt_swap(rules):
    """22 names at ~4.3% each and ~5.2% cash; sell all of one name, buy 300 lots each of two others."""
    codes, s = _equal_portfolio(rules, 22)
    orders = [Order(codes[0], Side.SELL, s.lots[codes[0]]), Order(codes[1], Side.BUY, 300),
              Order(codes[2], Side.BUY, 300)]
    return codes, s, orders


def test_halted_sell_case_is_worst_case_minus_largest_sale(rules):
    codes, s, orders = _halt_swap(rules)
    q = flat_quotes(codes)
    worst = stress.worst_case_min_cash(s, orders, q, rules.fees)
    sale = 431 * 1000 * 90.0 * (1 - rules.fees.commission_rate - rules.fees.sell_tax_rate)
    assert stress.halted_sell_min_cash(s, orders, q, rules.fees) == pytest.approx(worst - sale, abs=2)
    assert stress.halted_sell_min_cash(s, [], q, rules.fees) == pytest.approx(float(s.cash))


def test_swap_funded_by_one_sale_is_blocked(rules, strategy):
    """2327 on 2019-11-12: the sale of 2327 funded a swap's buys, passed the worst-case check, then
    2327 was halted all day and cash closed negative."""
    codes, s, orders = _halt_swap(rules)
    rep = validate_orders(orders, _ctx(rules, strategy, s, flat_quotes(codes), codes))
    by_id = {r.rule_id: r for r in rep.results}
    assert by_id["R_CASH_MIN_WORST"].passed
    assert not by_id["R_CASH_MIN_HALT"].passed
    assert rep.metrics["halted_sell_cash_min_pct"] < 0


def test_active_share_rule_blocks_index_hugging(rules, strategy):
    codes, s = _equal_portfolio(rules, 20)
    etf = {c: 1 / 20 for c in codes}  # benchmark identical to us
    rep = validate_orders([], _ctx(rules, strategy, s, flat_quotes(codes), codes,
                                   etf_holdings={"00999A": etf}))
    assert any(r.rule_id == "R_ACTIVE_SHARE" for r in rep.violations)


def test_missing_etf_data_is_reported_not_silent(rules, strategy):
    codes, s = _equal_portfolio(rules, 20)
    rep = validate_orders([], _ctx(rules, strategy, s, flat_quotes(codes), codes))
    assert rep.passed
    assert any(r.rule_id == "R_ACTIVE_SHARE" for r in rep.soft_failures)


def test_quote_date_must_precede_trade_date(rules, strategy):
    codes, s = _equal_portfolio(rules, 20)
    q = flat_quotes(codes)
    q[codes[0]] = Quote(codes[0], 100, 110, 90, date(2026, 10, 27))
    rep = validate_orders([], _ctx(rules, strategy, s, q, codes))
    assert any(r.rule_id == "R_QUOTE_DATE" for r in rep.violations)
