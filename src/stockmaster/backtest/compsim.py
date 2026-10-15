"""Competition simulator: many random 24-trading-day windows under the exact rules.

The pass criterion before go-live is zero hard-rule violations and zero close-state breaches in
every window, including forced stress windows (large market drops)."""

from __future__ import annotations

import random
from datetime import date

import pandas as pd

from stockmaster.backtest.engine import (FeatureCache, Policy, SimResult, simulate, summarize)
from stockmaster.config import Rules, Strategy, Universe

STRESS_STARTS = [date(2020, 3, 2), date(2024, 7, 25), date(2025, 3, 31)]


def windows(trading_days: list[date], n: int, length: int = 24, warmup: int = 130,
            seed: int = 2026, forced: list[date] | None = None) -> list[tuple[date, list[date]]]:
    """(prev_day, days) pairs. ``warmup`` days of history precede every window."""
    eligible = list(range(warmup, len(trading_days) - length))
    rng = random.Random(seed)
    starts = sorted(rng.sample(eligible, min(n, len(eligible))))
    for f in forced or []:
        idx = next((i for i, d in enumerate(trading_days) if d >= f), None)
        if idx is not None and warmup <= idx < len(trading_days) - length and idx not in starts:
            starts.append(idx)
    return [(trading_days[i - 1], trading_days[i:i + length]) for i in sorted(starts)]


def run_compsim(cache: FeatureCache, policies: dict[str, Policy], *, n_windows: int,
                universe: Universe, rules: Rules, strategy: Strategy,
                etf_holdings: dict | None = None, seed: int = 2026,
                progress=None) -> tuple[pd.DataFrame, list[dict], dict[str, list[SimResult]]]:
    days = [d for d in cache.panel.close.index if cache.panel.close.loc[d].notna().sum() > 50]
    wins = windows(days, n_windows, seed=seed, forced=STRESS_STARTS)
    rows, raw = [], {k: [] for k in policies}
    for wi, (prev, ds) in enumerate(wins):
        for name, pol in policies.items():
            r = simulate(pol, ds, prev, cache, universe=universe, rules=rules, strategy=strategy,
                         etf_holdings=etf_holdings)
            raw[name].append(r)
            rows.append({"policy": name, "start": ds[0], **r.metrics(rules.initial_cash)})
        if progress:
            progress(wi + 1, len(wins))
    return summarize(rows), rows, raw
