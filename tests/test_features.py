"""Feature, composite and baseline tests on a SYNTHETIC panel (random walk prices)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockmaster.alpha.composite import composite
from stockmaster.domain import PortfolioState, Quote
from stockmaster.features.compute import compute_features
from stockmaster.features.panel import Panel
from stockmaster.features.regime import market_regime
from stockmaster.portfolio.baseline import baseline_targets
from stockmaster.portfolio.order_builder import build_orders
from stockmaster.portfolio.repair import repair_targets
from stockmaster.risk.validator import ValidationContext, validate_orders

AS_OF = date(2026, 10, 23)


def synthetic_panel(codes, n_days=300, seed=7) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(AS_OF), periods=n_days).date
    rets = pd.DataFrame(rng.normal(0.0005, 0.02, (n_days, len(codes))), index=dates, columns=codes)
    close = 100 * (1 + rets).cumprod()
    ref = close.shift(1)
    vol = pd.DataFrame(rng.integers(1_000_000, 20_000_000, (n_days, len(codes))), index=dates,
                       columns=codes).astype(float)
    flows = pd.DataFrame(rng.normal(0, 200_000, (n_days, len(codes))), index=dates, columns=codes)
    idx = pd.Series(20000 * (1 + rets.mean(axis=1)).cumprod(), index=dates)
    rev_rows = []
    for c in codes:
        for k in range(15):
            ym = (pd.Timestamp(AS_OF) - pd.DateOffset(months=k + 1)).date().replace(day=1)
            rev_rows.append({"code": c, "ym": ym, "revenue": 1e6 * (1 + rng.random()),
                             "yoy_pct": rng.normal(10, 20), "mom_pct": 0.0,
                             "available_at": pd.Timestamp(ym) + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)})
    return Panel(as_of=AS_OF, close=close, ref_price=ref, vwap=close, value=vol * close, volume=vol,
                 itrust_net=flows, foreign_net=flows * 2, dealer_net=flows / 2,
                 margin_balance=(vol / 1000).cumsum() * 0.01, pe=close * 0 + 15, pb=close * 0 + 2,
                 div_yield=close * 0 + 3, index_close=idx, revenue=pd.DataFrame(rev_rows))


@pytest.fixture(scope="module")
def panel(universe):
    return synthetic_panel(sorted(universe.codes))


def test_features_shape_and_no_lookahead(panel):
    f = compute_features(panel)
    assert len(f) == len(panel.codes)
    assert f["ret_20d"].notna().all()
    assert f["rev_yoy_last"].notna().all()
    # Revenue for last month is not yet available on 10/23 (published by 11/10).
    assert (f["rev_last_ym"] <= date(2026, 9, 1)).all()


def test_composite_ranks_and_regime(panel):
    s = composite(compute_features(panel))
    assert s["score"].is_monotonic_decreasing
    reg = market_regime(panel)
    assert reg.label in {"risk_on", "neutral", "risk_off"}


@pytest.mark.parametrize("level", ["defensive", "normal", "aggressive"])
def test_baseline_day1_is_valid_after_repair(panel, universe, rules, strategy, level):
    feats = compute_features(panel)
    scores = composite(feats)
    quotes = {c: Quote.from_close(c, float(feats.loc[c, "close"]), AS_OF) for c in panel.codes}
    state = PortfolioState.initial(rules.initial_cash)
    invest = strategy.buffers.day1_invest_target
    w = baseline_targets(scores, state=state, quotes=quotes, universe=universe.codes,
                         risk_level=level, strategy=strategy, rules=rules, invest=invest,
                         beta=feats["beta_120d"])
    assert sum(w.values()) == pytest.approx(invest, abs=1e-6)
    assert strategy.buffers.holdings_target_min <= len(w) <= strategy.buffers.holdings_target_max
    etf = {"00999A": {c: 0.08 for c in list(scores.index[:10])}}
    rep = repair_targets(w, state=state, quotes=quotes, universe=universe.codes,
                         ranking=list(scores.index), etf_holdings=etf, rules=rules,
                         strategy=strategy, cash_target=1 - invest)
    res = build_orders(state, rep.weights, quotes, rules, strategy)
    v = validate_orders(res.orders, ValidationContext(
        trade_date=date(2026, 10, 27), state=state, quotes=quotes, universe=universe.codes,
        rules=rules, strategy=strategy, etf_holdings=etf))
    assert v.passed, v.summary()
