"""Competition simulator on a SYNTHETIC panel: the rules must never be broken."""

import pytest

from stockmaster.backtest.compsim import run_compsim
from stockmaster.backtest.engine import FeatureCache, baseline_policy, equal_weight_policy
from stockmaster.backtest.signals import ic_table

from .test_features import synthetic_panel


@pytest.fixture(scope="module")
def cache(universe):
    return FeatureCache(synthetic_panel(sorted(universe.codes), n_days=200))


def test_compsim_zero_violations(cache, universe, rules, strategy):
    policies = {
        "baseline_adaptive": baseline_policy(strategy, rules, universe),
        "baseline_aggressive": baseline_policy(strategy, rules, universe, "aggressive"),
        "equal_weight": equal_weight_policy(strategy, rules),
    }
    summary, rows, raw = run_compsim(cache, policies, n_windows=3, universe=universe, rules=rules,
                                     strategy=strategy)
    assert len(rows) == 9
    assert summary["hard_violations"].sum() == 0, [r.hard_violations for v in raw.values() for r in v]
    assert summary.loc[["baseline_adaptive", "baseline_aggressive"], "close_breaches"].sum() == 0
    for r in raw["baseline_adaptive"]:
        assert len(r.nav) == 24
        assert r.daily["n_holdings"].between(20, 30).all()
        assert (r.daily["cash_pct"] < 0.25).all() and (r.daily["cash_pct"] >= 0).all()


def test_ic_table_runs(cache):
    dates = list(cache.panel.close.index[150:160])
    t = ic_table(cache, dates, features=["ret_5d", "itrust_5d"], horizons=(1, 5))
    assert not t.empty


def test_ranker_walk_forward_runs(cache):
    pytest.importorskip("lightgbm")
    from stockmaster.alpha.ranker import walk_forward
    days = list(cache.panel.close.index)[130:185]
    ev = walk_forward(cache, days, horizon=5, min_train_dates=20, retrain_every=10)
    s = ev.summary()
    assert s["ranker"]["n"] > 0 and "passes_gate" in s


def test_suspended_holding_does_not_crash_the_engine(universe, rules, strategy):
    """A held stock with no trades for several days (suspension) is valued at its last close and
    its orders go unfilled, instead of raising."""
    import numpy as np
    from stockmaster.backtest.engine import FeatureCache, baseline_policy, simulate
    panel = synthetic_panel(sorted(universe.codes), n_days=200)
    days = list(panel.close.index)
    target = "2330"
    for frame in (panel.close, panel.vwap, panel.value, panel.volume, panel.ref_price):
        frame.loc[days[160]:days[165], target] = np.nan
    cache = FeatureCache(panel)
    res = simulate(baseline_policy(strategy, rules, universe, "aggressive"), days[150:174], days[149],
                   cache, universe=universe, rules=rules, strategy=strategy)
    assert len(res.nav) == 24 and np.isfinite(res.nav).all()


def test_dividends_are_added_to_final_nav_only(cache, universe, rules, strategy):
    from stockmaster.backtest.engine import baseline_policy, simulate
    days = list(cache.panel.close.index)
    win, prev = days[150:174], days[149]
    pol = baseline_policy(strategy, rules, universe, "aggressive")
    base = simulate(pol, win, prev, cache, universe=universe, rules=rules, strategy=strategy)
    divs = {win[5]: {c: 5.0 for c in universe.codes}}  # every stock pays 5/share on day 6
    withd = simulate(pol, win, prev, cache, universe=universe, rules=rules, strategy=strategy, dividends=divs)
    assert withd.final_accrued_dividends > 0
    assert list(withd.nav) == list(base.nav)  # daily NAV and drawdown unchanged
    m0, m1 = base.metrics(rules.initial_cash), withd.metrics(rules.initial_cash)
    assert m1["return"] == pytest.approx(m0["return"] + withd.final_accrued_dividends / rules.initial_cash)
    assert m1["mdd"] == m0["mdd"]


def test_dividend_on_a_closure_day_is_credited_next_trading_day(cache, universe, rules, strategy):
    from datetime import timedelta
    from stockmaster.backtest.engine import baseline_policy, simulate
    days = list(cache.panel.close.index)
    win, prev = days[150:174], days[149]
    pol = baseline_policy(strategy, rules, universe, "aggressive")
    on_trading = simulate(pol, win, prev, cache, universe=universe, rules=rules, strategy=strategy,
                          dividends={win[5]: {"2330": 5.0}})
    off_day = [d for d in (win[5] - timedelta(days=k) for k in range(1, 4)) if d not in days and d > win[4]]
    if off_day:  # synthetic bdate index: weekends between trading days are closures
        on_closure = simulate(pol, win, prev, cache, universe=universe, rules=rules, strategy=strategy,
                              dividends={off_day[0]: {"2330": 5.0}})
        assert on_closure.final_accrued_dividends == on_trading.final_accrued_dividends > 0
