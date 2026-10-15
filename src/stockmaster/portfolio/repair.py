"""Deterministic projection of target weights onto the rule-feasible set.

Used when a model proposal fails validation (after the bounded LLM repair rounds) and by the
deterministic fallback. Every change is recorded in ``notes`` so the report can explain it as a
風控調整 (risk-control adjustment).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from stockmaster.config import Rules, Strategy
from stockmaster.domain import PortfolioState, Quote
from stockmaster.risk.active_share import active_share_report, top_n

SATELLITE_WEIGHT = 0.015
AS_STEP = 0.005


@dataclass(frozen=True)
class RepairResult:
    weights: dict[str, float]
    notes: list[str] = field(default_factory=list)
    active_share_min: float | None = None


def current_weights(state: PortfolioState, quotes: Mapping[str, Quote],
                    lot_size: int) -> tuple[dict[str, float], float]:
    values = {c: n * lot_size * quotes[c].ref_price for c, n in state.lots.items() if n > 0}
    nav = float(state.cash) + sum(values.values())
    return {c: v / nav for c, v in values.items()}, nav


def repair_targets(weights: Mapping[str, float], *, state: PortfolioState,
                   quotes: Mapping[str, Quote], universe: frozenset[str], ranking: Sequence[str],
                   etf_holdings: Mapping[str, Mapping[str, float]], rules: Rules,
                   strategy: Strategy, cash_target: float,
                   locked: frozenset[str] = frozenset()) -> RepairResult:
    buf = strategy.buffers
    notes: list[str] = []
    cur, nav = current_weights(state, quotes, rules.lot_size)
    halted_held = {c for c in state.held_codes() if c in quotes and quotes[c].halted}
    frozen = set(locked) | halted_held

    w: dict[str, float] = {}
    for c, x in weights.items():
        if x <= 0:
            continue
        if c not in universe:
            notes.append(f"移除非標的池個股 {c}")
            continue
        if c not in quotes or (quotes[c].halted and c not in state.lots):
            notes.append(f"移除無報價或暫停交易個股 {c}")
            continue
        w[c] = float(x)
    for c in sorted(frozen):
        if c in cur:
            w[c] = cur[c]

    def limit(c: str) -> float:
        """Max target weight for c given the add-cap / drift rules."""
        if c in frozen:
            return cur.get(c, 0.0)
        cap = rules.weight_caps.cap(c)
        cw = cur.get(c, 0.0)
        if cw >= cap:  # drifted over the cap: may only trim
            days = state.overweight_days.get(c, 0)
            return buf.drift_trim_target(c) if days >= buf.drift_trim_by_grace_day else cw
        return max(buf.add_cap(c), min(cw, cap)) if cw > buf.add_cap(c) else buf.add_cap(c)

    def one_lot_weight(c: str) -> float:
        return rules.lot_size * quotes[c].ref_price / nav

    def eligible_new(c: str) -> bool:
        return (c in universe and c in quotes and not quotes[c].halted and c not in w
                and one_lot_weight(c) <= buf.add_cap(c))

    # 1) caps
    for c in list(w):
        lim = limit(c)
        if w[c] > lim + 1e-12:
            notes.append(f"{c} 權重 {w[c]:.2%} 超過上限，調降至 {lim:.2%}")
            w[c] = lim

    # 2) minimum position size
    for c in list(w):
        if c not in frozen and w[c] < buf.min_position_weight:
            w[c] = max(buf.min_position_weight, min(one_lot_weight(c), limit(c)))

    # 3) holding count (official bounds; target band is preferred when adding)
    lo, hi = rules.holdings.min_count, rules.holdings.max_count
    target_lo = max(lo, buf.holdings_target_min)
    if len(w) < target_lo:
        for c in ranking:
            if len(w) >= target_lo:
                break
            if eligible_new(c):
                w[c] = max(SATELLITE_WEIGHT, one_lot_weight(c))
                notes.append(f"持股檔數不足，加入 {c}")
    if len(w) > hi:
        removable = sorted((x, c) for c, x in w.items() if c not in frozen)
        while len(w) > hi and removable:
            _, c = removable.pop(0)
            del w[c]
            notes.append(f"持股檔數過多，移除 {c}")

    # 4) scale to the invested target with capped water-filling
    invest = 1.0 - cash_target
    w = _scale_to(w, invest, limit, frozen)

    # 5) Active Share
    as_min = None
    if etf_holdings:
        etf_top_union = set()
        for h in etf_holdings.values():
            etf_top_union |= set(top_n(h, rules.active_share.top_n))
        for _ in range(400):
            report = active_share_report(w, etf_holdings, rules.active_share.top_n)
            as_min = report[0].value
            if as_min >= buf.active_share_target:
                break
            overlap = [c for c in report[0].overlap if c not in frozen
                       and w[c] - AS_STEP >= buf.min_position_weight]
            if not overlap:
                notes.append(f"Active Share {as_min:.1%} vs {report[0].etf}: no reducible overlap")
                break
            victim = max(overlap, key=lambda c: (w[c], c))
            recipients = [c for c in ranking if c in w and c not in etf_top_union
                          and c not in frozen and w[c] + AS_STEP <= limit(c)]
            if not recipients:
                new = next((c for c in ranking if eligible_new(c) and c not in etf_top_union), None)
                if new is None or len(w) >= hi:
                    notes.append(f"Active Share {as_min:.1%}: no recipient for reallocated weight")
                    break
                w[new] = 0.0
                recipients = [new]
            w[victim] -= AS_STEP
            w[recipients[0]] += AS_STEP
        else:
            notes.append("Active Share repair did not converge")
        if as_min is not None and as_min < buf.active_share_target:
            notes.append(f"Active Share 最低 {as_min:.1%}，低於目標 {buf.active_share_target:.0%}")
        elif as_min is not None:
            notes.append(f"Active Share 最低 {as_min:.1%}")

    return RepairResult(weights={c: x for c, x in w.items() if x > 0}, notes=notes,
                        active_share_min=as_min)


def _scale_to(w: dict[str, float], invest: float, limit, frozen) -> dict[str, float]:
    w = dict(w)
    for _ in range(50):
        total = sum(w.values())
        gap = invest - total
        if abs(gap) < 1e-6:
            break
        free = [c for c in w if c not in frozen and (gap < 0 or w[c] < limit(c) - 1e-12)]
        base = sum(w[c] for c in free)
        if not free or base <= 0:
            break
        f = 1 + gap / base
        for c in free:
            w[c] = min(w[c] * f, limit(c))
    return w
