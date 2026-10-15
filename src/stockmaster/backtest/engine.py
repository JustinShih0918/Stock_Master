"""Event-driven backtest that applies the competition rules exactly as the organizer would.

Per trading day t:
    features as of t-1 -> policy target weights -> repair -> whole-lot orders -> validator
    -> (organizer view) hard violation => trades void + warning; else fill at VWAP(t)
    -> value at close(t) -> realized close-state checks (count, cash, caps).

The same ledger, order builder, validator and repair code paths as live trading are used.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from stockmaster.accounting.ledger import run_day
from stockmaster.alpha.composite import composite
from stockmaster.config import Rules, Strategy, Universe
from stockmaster.domain import PortfolioState, Quote
from stockmaster.features.compute import compute_features
from stockmaster.features.panel import Panel
from stockmaster.features.regime import Regime, market_regime
from stockmaster.portfolio.order_builder import build_orders
from stockmaster.portfolio.repair import repair_targets
from stockmaster.risk.validator import ValidationContext, validate_orders


@dataclass(frozen=True)
class DayInputs:
    trade_date: date
    as_of: date
    state: PortfolioState
    features: pd.DataFrame
    scores: pd.DataFrame
    regime: Regime
    quotes: dict[str, Quote]
    day_index: int
    days_total: int


Policy = Callable[[DayInputs], tuple[dict[str, float], float]]  # (weights, cash_target)


@dataclass
class SimResult:
    nav: pd.Series
    daily: pd.DataFrame
    warnings: int
    hard_violations: list[str] = field(default_factory=list)
    close_breaches: list[str] = field(default_factory=list)
    final_accrued_dividends: float = 0.0  # added to the final NAV only (competition rule)

    def metrics(self, initial: float) -> dict:
        nav = self.nav
        final = nav.iloc[-1] + self.final_accrued_dividends if len(nav) else initial
        ret = final / initial - 1 if len(nav) else 0.0
        peak = nav.cummax()
        mdd = float(((peak - nav) / peak).max()) if len(nav) else 0.0
        return {"return": float(ret), "mdd": mdd, "dividends": float(self.final_accrued_dividends) / initial,
                "turnover": float(self.daily["turnover"].sum()),
                "cost": float(self.daily["cost"].sum()), "warnings": self.warnings,
                "hard_violations": len(self.hard_violations), "close_breaches": len(self.close_breaches),
                "avg_holdings": float(self.daily["n_holdings"].mean()),
                "avg_cash_pct": float(self.daily["cash_pct"].mean())}


class FeatureCache:
    """Features/scores/regime per as_of date, computed once from a long panel."""

    def __init__(self, panel: Panel, lookback_rows: int = 300):
        self.panel, self.lookback = panel, lookback_rows
        self._cache: dict[date, tuple[pd.DataFrame, pd.DataFrame, Regime]] = {}

    def slice(self, as_of: date) -> Panel:
        p = self.panel

        def cut(df):
            return df.loc[:as_of].tail(self.lookback) if len(df) else df

        rev = p.revenue
        return Panel(as_of=as_of, close=cut(p.close), ref_price=cut(p.ref_price), vwap=cut(p.vwap),
                     value=cut(p.value), volume=cut(p.volume), itrust_net=cut(p.itrust_net),
                     foreign_net=cut(p.foreign_net), dealer_net=cut(p.dealer_net),
                     margin_balance=cut(p.margin_balance), pe=cut(p.pe), pb=cut(p.pb),
                     div_yield=cut(p.div_yield), index_close=p.index_close.loc[:as_of].tail(self.lookback),
                     revenue=rev)

    def get(self, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame, Regime]:
        if as_of not in self._cache:
            sl = self.slice(as_of)
            feats = compute_features(sl)
            self._cache[as_of] = (feats, composite(feats), market_regime(sl))
        return self._cache[as_of]


class Stepper:
    """Day-by-day competition simulation; ``simulate`` and the RL environment both drive it."""

    def __init__(self, cache: FeatureCache, days: list[date], prev_day: date, *, universe: Universe,
                 rules: Rules, strategy: Strategy, etf_holdings: dict | None = None,
                 initial: PortfolioState | None = None,
                 dividends: Mapping[date, Mapping[str, float]] | None = None):
        """``dividends``: ex-date -> {code: per-share value}. When given, holdings at the previous close
        accrue them and they are added to the final NAV (the organizer's rule); daily NAV and drawdown
        exclude them. None = price-only (the behaviour of every result before 2026-10-03)."""
        self.cache, self.days, self.universe = cache, days, universe
        self.dividends = dividends or {}
        self.rules, self.strategy = rules, strategy
        self.etf_holdings = etf_holdings or {}
        self.state = initial or PortfolioState.initial(rules.initial_cash)
        self.as_of, self.i = prev_day, 0
        self.rows: list[dict] = []
        self.navs: dict[date, float] = {}
        self.hard: list[str] = []
        self.breaches: list[str] = []

    @property
    def done(self) -> bool:
        return self.i >= len(self.days)

    def inputs(self) -> DayInputs:
        p = self.cache.panel
        feats, scores, regime = self.cache.get(self.as_of)
        hist = p.close.loc[:self.as_of]
        last, today = hist.ffill().iloc[-1], hist.iloc[-1]
        # A stock with no trade on as_of keeps its last close and is marked halted (not tradable).
        quotes = {c: Quote.from_close(c, float(last[c]), self.as_of, self.rules.price_limit_pct,
                                      halted=bool(pd.isna(today.get(c))))
                  for c in sorted(self.universe.codes) if c in last.index and pd.notna(last[c])}
        return DayInputs(self.days[self.i], self.as_of, self.state, feats, scores, regime, quotes,
                         self.i + 1, len(self.days))

    def step(self, x: DayInputs, weights: dict[str, float], cash_target: float) -> dict:
        rules, strategy, p, t = self.rules, self.strategy, self.cache.panel, x.trade_date
        state = self.state
        rep = repair_targets(weights, state=state, quotes=x.quotes, universe=self.universe.codes,
                             ranking=list(x.scores.index), etf_holdings=self.etf_holdings, rules=rules,
                             strategy=strategy, cash_target=cash_target)
        build = build_orders(state, rep.weights, x.quotes, rules, strategy)
        report = validate_orders(build.orders, ValidationContext(
            trade_date=t, state=state, quotes=x.quotes, universe=self.universe.codes, rules=rules,
            strategy=strategy, etf_holdings=self.etf_holdings))
        applied = report.passed
        if not applied:
            self.hard.append(f"{t}: " + "; ".join(r.rule_id for r in report.violations))
        last_prev = p.close.loc[:self.as_of].ffill().iloc[-1]
        vw = p.vwap.loc[t] if t in p.vwap.index else pd.Series(dtype=float)
        cl = p.close.loc[t]

        def fill_px(c):
            v = vw.get(c)
            if v is not None and pd.notna(v) and v > 0:
                return float(v)
            c_ = cl.get(c)
            return float(c_) if c_ is not None and pd.notna(c_) else None

        # Orders in stocks that did not trade on t cannot fill; they are dropped (logged in the row).
        fillable = [o for o in build.orders if fill_px(o.code) is not None]
        unfilled = sorted(o.code for o in build.orders if fill_px(o.code) is None)
        vwaps = {o.code: fill_px(o.code) for o in sorted(fillable, key=lambda o: o.code)}
        closes = {c: float(cl[c]) if pd.notna(cl.get(c)) else float(last_prev[c])
                  for c in sorted(set(state.lots) | set(vwaps))}
        # Ex-dates that fell on a market closure (e.g. typhoon days) are credited on the next trading day.
        divs: dict[str, float] = {}
        for d in sorted(k for k in self.dividends if self.as_of < k <= t):
            for c, v in self.dividends[d].items():
                divs[c] = divs.get(c, 0.0) + v
        res = run_day(state, t, fillable if applied else [], vwaps, closes, rules,
                      applied=applied, warning=not applied, cash_dividends_per_share=divs or None)
        v = res.valuation
        traded = sum(float(f.gross) for f in res.fills)
        cost = sum(float(f.commission + f.tax) for f in res.fills)
        if not rules.holdings.min_count <= v.n_holdings <= rules.holdings.max_count:
            self.breaches.append(f"{t}: holdings {v.n_holdings}")
        if v.cash < 0 or v.cash_pct >= rules.cash.max_exclusive:
            self.breaches.append(f"{t}: cash {v.cash_pct:.2%}")
        bought = {o.code for o in build.orders if o.side.value == "buy"} if applied else set()
        for c, w in v.weights.items():
            if w > rules.weight_caps.cap(c) and c in bought:
                self.breaches.append(f"{t}: {c} {w:.2%} after buying")
        row = {"date": t, "nav": float(v.nav), "cash_pct": v.cash_pct, "n_holdings": v.n_holdings,
               "turnover": traded / 2 / float(v.nav) if self.i else 0.0, "cost": cost,
               "applied": applied, "regime": x.regime.label, "unfilled": ",".join(unfilled)}
        self.rows.append(row)
        self.navs[t] = float(v.nav)
        self.state, self.as_of = res.state, t
        self.i += 1
        return row

    def result(self) -> SimResult:
        daily = pd.DataFrame(self.rows).set_index("date") if self.rows else pd.DataFrame()
        return SimResult(nav=pd.Series(self.navs), daily=daily, warnings=self.state.warnings,
                         hard_violations=self.hard, close_breaches=self.breaches,
                         final_accrued_dividends=float(self.state.accrued_dividends))


def simulate(policy: Policy, days: list[date], prev_day: date, cache: FeatureCache, *,
             universe: Universe, rules: Rules, strategy: Strategy,
             etf_holdings: dict[str, dict[str, float]] | None = None,
             initial: PortfolioState | None = None,
             dividends: Mapping[date, Mapping[str, float]] | None = None) -> SimResult:
    sim = Stepper(cache, days, prev_day, universe=universe, rules=rules, strategy=strategy,
                  etf_holdings=etf_holdings, initial=initial, dividends=dividends)
    while not sim.done:
        x = sim.inputs()
        weights, cash_target = policy(x)
        sim.step(x, weights, cash_target)
    return sim.result()


# --- reference policies -------------------------------------------------------------------------

def baseline_policy(strategy: Strategy, rules: Rules, universe: Universe,
                    risk_level: str | None = None) -> Policy:
    from stockmaster.portfolio.baseline import baseline_targets

    def policy(x: DayInputs) -> tuple[dict[str, float], float]:
        buf = strategy.buffers
        first = not x.state.held_codes()
        cash = (1 - buf.day1_invest_target) if first else (buf.cash_target_min + buf.cash_target_max) / 2
        level = risk_level or x.regime.suggested_risk_level
        if x.day_index > x.days_total - strategy.turnover.freeze_last_days and not first:
            nav = float(x.state.cash) + sum(n * rules.lot_size * x.quotes[c].ref_price
                                            for c, n in x.state.lots.items())
            return ({c: n * rules.lot_size * x.quotes[c].ref_price / nav for c, n in x.state.lots.items()},
                    float(x.state.cash) / nav)
        w = baseline_targets(x.scores, state=x.state, quotes=x.quotes, universe=universe.codes,
                             risk_level=level, strategy=strategy, rules=rules, invest=1 - cash,
                             beta=x.features.get("beta_120d"))
        return w, cash
    return policy


def equal_weight_policy(strategy: Strategy, rules: Rules, n: int = 24) -> Policy:
    def policy(x: DayInputs) -> tuple[dict[str, float], float]:
        buf = strategy.buffers
        cash = (1 - buf.day1_invest_target) if not x.state.held_codes() else 0.05
        top = [c for c in x.scores.index if c in x.quotes][:n]
        return {c: (1 - cash) / len(top) for c in top}, cash
    return policy


def summarize(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df
    agg = df.groupby("policy").agg(
        windows=("return", "size"), mean_ret=("return", "mean"), median_ret=("return", "median"),
        p10_ret=("return", lambda s: float(np.percentile(s, 10))),
        p90_ret=("return", lambda s: float(np.percentile(s, 90))),
        mean_mdd=("mdd", "mean"), mean_turnover=("turnover", "mean"),
        mean_cost=("cost", "mean"), hard_violations=("hard_violations", "sum"),
        close_breaches=("close_breaches", "sum"), warnings=("warnings", "sum"))
    return agg.sort_values("mean_ret", ascending=False)
