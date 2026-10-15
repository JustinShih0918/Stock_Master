"""Assemble the agent's ToolContext from already-loaded frames (no DB access here)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import pandas as pd

from stockmaster.agent.tools import ToolContext
from stockmaster.calendar import TradingCalendar
from stockmaster.config import Rules, Strategy, Universe
from stockmaster.domain import PortfolioState, Quote
from stockmaster.features.regime import Regime
from stockmaster.portfolio.baseline import baseline_targets
from stockmaster.portfolio.repair import repair_targets


def competition_day(trade_date: date, rules: Rules, cal: TradingCalendar | None
                    ) -> tuple[int | None, int | None]:
    if cal is None or not (rules.period.start <= trade_date <= rules.period.end):
        return None, None
    days = cal.trading_days(rules.period.start, rules.period.end)
    if trade_date not in days:
        return None, None
    i = days.index(trade_date)
    return i + 1, len(days) - i


def baseline_book(*, state: PortfolioState, quotes: Mapping[str, Quote], universe: Universe,
                  features: pd.DataFrame, scores: pd.DataFrame,
                  etf_holdings: Mapping[str, Mapping[str, float]], rules: Rules,
                  strategy: Strategy) -> dict[str, dict[str, float]]:
    """Repaired quant baseline per risk level."""
    buf = strategy.buffers
    cash = (1 - buf.day1_invest_target) if not state.held_codes() \
        else (buf.cash_target_min + buf.cash_target_max) / 2
    out = {}
    for level in strategy.risk_levels:
        raw = baseline_targets(scores, state=state, quotes=quotes, universe=universe.codes,
                               risk_level=level, strategy=strategy, rules=rules, invest=1 - cash,
                               beta=features.get("beta_120d"))
        rep = repair_targets(raw, state=state, quotes=quotes, universe=universe.codes,
                             ranking=list(scores.index), etf_holdings=etf_holdings, rules=rules,
                             strategy=strategy, cash_target=cash)
        out[level] = rep.weights
    return out


def build_tool_context(*, trade_date: date, as_of: date, state: PortfolioState,
                       quotes: Mapping[str, Quote], universe: Universe, features: pd.DataFrame,
                       scores: pd.DataFrame, regime: Regime,
                       etf_holdings: Mapping[str, Mapping[str, float]],
                       etf_asof: Mapping[str, date], rules: Rules, strategy: Strategy,
                       calendar: TradingCalendar | None = None, events: list[dict] | None = None,
                       stale: list[str] | None = None, reconciliation: str = "n/a",
                       sectors: Mapping[str, str] | None = None,
                       overnight: dict | None = None) -> ToolContext:
    day_index, remaining = competition_day(trade_date, rules, calendar)
    baselines = baseline_book(state=state, quotes=quotes, universe=universe, features=features,
                              scores=scores, etf_holdings=etf_holdings, rules=rules, strategy=strategy)
    return ToolContext(trade_date=trade_date, as_of=as_of, day_index=day_index,
                       days_remaining=remaining, state=state, quotes=quotes, universe=universe,
                       features=features, scores=scores, regime=regime, baselines=baselines,
                       etf_holdings=etf_holdings, etf_asof=etf_asof, rules=rules, strategy=strategy,
                       events=events or [], stale=stale or [], reconciliation=reconciliation,
                       sectors=sectors or {}, overnight=overnight or {})
