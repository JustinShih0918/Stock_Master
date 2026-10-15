from decimal import Decimal

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stockmaster.domain import PortfolioState, Side
from stockmaster.portfolio.order_builder import build_orders
from stockmaster.portfolio.repair import repair_targets
from stockmaster.risk.validator import ValidationContext, validate_orders

from .conftest import AS_OF, TRADE_DATE, ranked_codes

HARD_CASH_AND_CAPS = {"R_CASH_MIN_WORST", "R_CASH_MIN_HALT", "R_CASH_MAX_WORST", "R_CAP_ADD", "R_NO_SHORT",
                      "R_NO_BUY_SELL_SAME", "R_UNIVERSE"}


def _ctx(rules, strategy, state, quotes, universe, etf=None):
    return ValidationContext(trade_date=TRADE_DATE, state=state, quotes=quotes,
                             universe=frozenset(universe.codes), rules=rules, strategy=strategy,
                             etf_holdings=etf or {})


def _day1_targets(universe):
    codes = [c for c in ranked_codes(universe) if c != "2330"][:23]
    targets = {"2330": 0.20}
    for c in codes:
        targets[c] = (0.87 - 0.20) / len(codes)
    return targets


def test_day1_build_from_cash_passes_validation(rules, strategy, universe, quotes, cash_state):
    res = build_orders(cash_state, _day1_targets(universe), quotes, rules, strategy)
    assert all(o.side is Side.BUY for o in res.orders)
    rep = validate_orders(res.orders, _ctx(rules, strategy, cash_state, quotes, universe))
    assert rep.passed, rep.summary()
    assert 20 <= rep.metrics["n_holdings"] <= 30
    assert rep.metrics["worst_cash_min_pct"] >= strategy.buffers.cash_min_worst


def test_overinvested_day1_targets_are_shrunk(rules, strategy, universe, quotes, cash_state):
    targets = {c: w * (0.99 / 0.87) for c, w in _day1_targets(universe).items()}
    res = build_orders(cash_state, targets, quotes, rules, strategy)
    rep = validate_orders(res.orders, _ctx(rules, strategy, cash_state, quotes, universe))
    assert rep.passed, rep.summary()


def test_underinvested_day1_targets_are_topped_up(rules, strategy, universe, quotes, cash_state):
    targets = {c: w * (0.70 / 0.87) for c, w in _day1_targets(universe).items()}
    res = build_orders(cash_state, targets, quotes, rules, strategy)
    rep = validate_orders(res.orders, _ctx(rules, strategy, cash_state, quotes, universe))
    assert rep.passed, rep.summary()


def test_add_cap_respected_for_oversized_targets(rules, strategy, universe, quotes, cash_state):
    targets = _day1_targets(universe)
    targets["2330"] = 0.40
    first = next(c for c in targets if c != "2330")
    targets[first] = 0.30
    res = build_orders(cash_state, targets, quotes, rules, strategy)
    rep = validate_orders(res.orders, _ctx(rules, strategy, cash_state, quotes, universe))
    assert not any(r.rule_id == "R_CAP_ADD" for r in rep.violations), rep.summary()
    assert rep.metrics["weights"]["2330"] <= strategy.buffers.add_cap("2330") + 1e-9


def test_builder_keeps_cash_if_the_largest_sale_is_halted(rules, strategy):
    """Exit one ~4.3% name and add to two others: buys are cut until cash stays >= 0 even if the
    exit's sale does not fill."""
    from stockmaster.domain import Quote
    from stockmaster.risk import stress
    codes = [f"{1000 + i}" for i in range(22)]
    q = {c: Quote(c, 100.0, 110.0, 90.0, AS_OF) for c in codes}
    state = PortfolioState(as_of=AS_OF, cash=Decimal(1_000_000_000 - 431 * 22 * 100_000),
                           lots={c: 431 for c in codes})
    targets = {c: 0.0431 for c in codes[3:]} | {codes[1]: 0.0731, codes[2]: 0.0731}
    res = build_orders(state, targets, q, rules, strategy)
    assert sum(o.lots for o in res.orders if o.side is Side.BUY) < 600
    assert stress.halted_sell_min_cash(state, res.orders, q, rules.fees) >= 0
    rep = validate_orders(res.orders, _ctx(rules, strategy, state, q, type("U", (), {"codes": codes})))
    assert not {r.rule_id for r in rep.violations} & HARD_CASH_AND_CAPS, rep.summary()


def test_builder_shrinks_a_full_core_rotation_into_the_cash_band(rules, strategy):
    """Swap all 8 core names (~70% of NAV traded): the buys may not lean on the largest sale, which
    alone would leave worst-case cash over the ceiling; the builder trades part of the rotation."""
    from stockmaster.domain import Quote
    from stockmaster.risk import stress
    old_core, new_core = [f"{1100 + i}" for i in range(8)], [f"{1200 + i}" for i in range(8)]
    sats = [f"{1300 + i}" for i in range(13)]
    codes = ["2330"] + old_core + new_core + sats
    q = {c: Quote(c, 100.0, 110.0, 90.0, AS_OF) for c in codes}
    lots = {"2330": 2000} | {c: 840 for c in old_core} | {c: 75 for c in sats}
    cash = 1_000_000_000 - sum(lots.values()) * 100_000
    state = PortfolioState(as_of=AS_OF, cash=Decimal(cash), lots=lots)
    targets = {"2330": 0.20} | {c: 0.0875 for c in new_core} | {c: 0.0075 for c in sats}
    res = build_orders(state, targets, q, rules, strategy)
    assert any("rotation shrunk" in n for n in res.notes)
    assert stress.halted_sell_min_cash(state, res.orders, q, rules.fees) >= 0
    rep = validate_orders(res.orders, _ctx(rules, strategy, state, q, type("U", (), {"codes": codes})))
    assert not {r.rule_id for r in rep.violations} & HARD_CASH_AND_CAPS, rep.summary()
    assert rep.metrics["turnover_one_way"] > 0.2  # most of the rotation still happens today


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_property_builder_never_breaks_cash_or_caps(rules, strategy, universe, quotes, data):
    codes = ranked_codes(universe)
    held = data.draw(st.lists(st.sampled_from(codes), min_size=20, max_size=28, unique=True))
    lots = {c: data.draw(st.integers(1, 40)) for c in held}
    invested = sum(n * 1000 * quotes[c].ref_price for c, n in lots.items())
    cash = Decimal(max(0, int(invested * data.draw(st.floats(0.02, 0.2)))))
    state = PortfolioState(as_of=AS_OF, cash=cash, lots=lots)
    tgt_codes = data.draw(st.lists(st.sampled_from(codes), min_size=20, max_size=28, unique=True))
    raw = [data.draw(st.floats(0.001, 1.0)) for _ in tgt_codes]
    invest = data.draw(st.floats(0.8, 1.05))
    targets = {c: x / sum(raw) * invest for c, x in zip(tgt_codes, raw)}
    res = build_orders(state, targets, quotes, rules, strategy)
    rep = validate_orders(res.orders, _ctx(rules, strategy, state, quotes, universe))
    failed = {r.rule_id for r in rep.violations} & HARD_CASH_AND_CAPS
    # The cash ceiling can be infeasible only if no targeted name has headroom; the builder notes it.
    if "R_CASH_MAX_WORST" in failed:
        assert any("ceiling" in n or "narrow" in n for n in res.notes)
        failed.discard("R_CASH_MAX_WORST")
    assert not failed, rep.summary()
    sides = {}
    for o in res.orders:
        assert o.lots > 0
        assert sides.setdefault(o.code, o.side) is o.side


def test_repair_fixes_count_caps_and_active_share(rules, strategy, universe, quotes, cash_state):
    codes = ranked_codes(universe)
    hug = codes[:10]
    proposal = {c: 0.087 for c in hug}  # 10 names only, at the add-cap
    proposal["2330"] = 0.35  # over the 2330 cap
    etf = {c: 0.1 for c in hug}
    rep = repair_targets(proposal, state=cash_state, quotes=quotes, universe=universe.codes,
                         ranking=codes, etf_holdings={"00999A": etf}, rules=rules,
                         strategy=strategy, cash_target=1 - strategy.buffers.day1_invest_target)
    w = rep.weights
    assert len(w) >= strategy.buffers.holdings_target_min
    assert w["2330"] <= strategy.buffers.add_cap("2330") + 1e-9
    assert sum(w.values()) == __import__("pytest").approx(strategy.buffers.day1_invest_target, abs=1e-4)
    assert rep.active_share_min >= strategy.buffers.active_share_target
    res = build_orders(cash_state, w, quotes, rules, strategy)
    vrep = validate_orders(res.orders, _ctx(rules, strategy, cash_state, quotes, universe,
                                            etf={"00999A": etf}))
    assert vrep.passed, vrep.summary()
