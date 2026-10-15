"""Deterministic quant portfolio: core-satellite construction from composite scores.

Used three ways: as the reference the LLM starts from, as the deterministic fallback when the LLM
fails, and as the baseline shadow account that measures what the LLM adds.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from stockmaster.alpha.composite import zscore
from stockmaster.config import Rules, Strategy
from stockmaster.domain import PortfolioState, Quote

TSMC = "2330"
SATELLITE_MAX = 0.025
HOLD_BONUS = 0.25  # score bonus for current holdings (hysteresis against churn)


def baseline_targets(scores: pd.DataFrame, *, state: PortfolioState, quotes: Mapping[str, Quote],
                     universe: frozenset[str], risk_level: str, strategy: Strategy, rules: Rules,
                     invest: float, n_target: int | None = None,
                     beta: pd.Series | None = None) -> dict[str, float]:
    """Target weights (fraction of NAV) for the risk level. ``scores`` needs a ``score`` column."""
    rl = strategy.risk_levels[risk_level]
    buf = strategy.buffers
    n_target = n_target or (buf.holdings_target_min + buf.holdings_target_max) // 2

    tradable = [c for c in scores.index if c in universe and c in quotes and not quotes[c].halted]
    adj = scores.loc[tradable, "score"].astype(float).copy()
    held = state.held_codes()
    adj[adj.index.isin(held)] += HOLD_BONUS
    if beta is not None:
        tilt = (rl.beta_target - 1.0) * 1.5
        adj += tilt * zscore(beta.reindex(adj.index)).fillna(0.0) * 0.5
    ranked = [c for c in adj.sort_values(ascending=False).index if c != TSMC]
    raw_rank = {c: i + 1 for i, c in enumerate(scores.loc[tradable, "score"].sort_values(ascending=False).index)}
    exit_rank = strategy.turnover.exit_rank

    w: dict[str, float] = {}
    if TSMC in tradable:
        w[TSMC] = rl.tsmc_weight
    core = ranked[:rl.core_names]
    for c in core:
        w[c] = rl.core_weight
    n_sat = max(0, n_target - len(w))
    # Satellites: keep current holdings that are still within exit_rank, then add the best new names.
    kept = [c for c in ranked[rl.core_names:] if c in held and raw_rank.get(c, 999) <= exit_rank]
    fresh = [c for c in ranked[rl.core_names:] if c not in kept]
    sats = (kept + fresh)[:n_sat]

    remaining = invest - sum(w.values())
    per_sat = remaining / len(sats) if sats else 0.0
    if per_sat < buf.min_position_weight and sats:
        # Not enough room for satellites: shrink core and 2330 proportionally.
        need = buf.min_position_weight * len(sats)
        scale = (invest - need) / sum(w.values())
        w = {c: x * scale for c, x in w.items()}
        per_sat = buf.min_position_weight
    elif per_sat > SATELLITE_MAX and sats:
        # Too much room: add satellites up to the target band max, then lift the core.
        extra = ranked[rl.core_names + n_sat:rl.core_names + n_sat
                       + max(0, buf.holdings_target_max - n_target)]
        while extra and (invest - sum(w.values())) / (len(sats) + 1) > SATELLITE_MAX:
            sats.append(extra.pop(0))
        per_sat = min(SATELLITE_MAX, (invest - sum(w.values())) / len(sats))
        leftover = invest - sum(w.values()) - per_sat * len(sats)
        if leftover > 0 and core:
            room = {c: buf.add_cap(c) - w[c] for c in core}
            total_room = sum(max(0.0, r) for r in room.values())
            for c in core:
                w[c] += leftover * max(0.0, room[c]) / total_room if total_room else 0.0
    for c in sats:
        w[c] = per_sat
    return _partial_rebalance(w, state, quotes, rules, strategy)


def _partial_rebalance(target: dict[str, float], state: PortfolioState, quotes: Mapping[str, Quote],
                       rules: Rules, strategy: Strategy) -> dict[str, float]:
    """Move only part of the way from current weights to the target (turnover control).
    Exits are full; new positions enter at least at the minimum position weight."""
    speed = strategy.turnover.rebalance_speed
    if speed >= 1.0 or not state.held_codes():
        return target
    nav = float(state.cash) + sum(n * rules.lot_size * quotes[c].ref_price
                                  for c, n in state.lots.items() if c in quotes)
    cur = {c: n * rules.lot_size * quotes[c].ref_price / nav for c, n in state.lots.items() if c in quotes}
    out = {}
    for c, t in target.items():
        now = cur.get(c, 0.0)
        out[c] = now + speed * (t - now) if now > 0 else max(speed * t, strategy.buffers.min_position_weight)
    total, goal = sum(out.values()), sum(target.values())
    return {c: x * goal / total for c, x in out.items()} if total > 0 else target


def ranking(scores: pd.DataFrame, quotes: Mapping[str, Quote], universe: frozenset[str]) -> list[str]:
    """Codes ordered best-first, tradable only — used by repair to fill or rebalance."""
    return [c for c in scores.index if c in universe and c in quotes and not quotes[c].halted]
