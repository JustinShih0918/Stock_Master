"""Convert target weights into whole-lot orders that respect caps and the worst-case cash band.

The builder never invents names: it only trades codes that are held or targeted. Holding-count and
Active Share problems are fixed upstream in ``portfolio.repair``; the validator has the final say.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from stockmaster.config import Rules, Strategy
from stockmaster.domain import Order, PortfolioState, Quote, Side
from stockmaster.risk import stress

MAX_ITER = 2000
# Caps are sized against NAV net of a full-swap fee drag (0.585% round trip), so that post-trade
# weights measured after fees never exceed the cap through rounding.
FEE_DRAG = 0.006


@dataclass(frozen=True)
class BuildResult:
    orders: list[Order]
    target_lots: dict[str, int]
    notes: list[str] = field(default_factory=list)


def _orders_from(state: PortfolioState, target: Mapping[str, int]) -> list[Order]:
    orders = []
    for code in sorted(set(state.lots) | set(target)):
        delta = target.get(code, 0) - state.lots.get(code, 0)
        if delta > 0:
            orders.append(Order(code, Side.BUY, delta))
        elif delta < 0:
            orders.append(Order(code, Side.SELL, -delta))
    return orders


def build_orders(state: PortfolioState, targets: Mapping[str, float], quotes: Mapping[str, Quote],
                 rules: Rules, strategy: Strategy, *, no_trade_band: float | None = None,
                 frozen: frozenset[str] = frozenset()) -> BuildResult:
    """``targets`` maps code -> weight of NAV at reference prices; the rest is cash.

    ``frozen`` names keep their current lots (e.g. halted stocks).
    """
    buf, lot, fees = strategy.buffers, rules.lot_size, rules.fees
    band = strategy.turnover.no_trade_band if no_trade_band is None else no_trade_band
    notes: list[str] = []

    nav = float(state.cash) + sum(n * lot * quotes[c].ref_price for c, n in state.lots.items())
    if nav <= 0:
        raise ValueError("non-positive NAV")

    def lot_value(code: str) -> float:
        return lot * quotes[code].ref_price

    cap_nav = nav * (1 - FEE_DRAG)

    def add_cap_lots(code: str) -> int:
        return math.floor(buf.add_cap(code) * cap_nav / lot_value(code))

    tgt: dict[str, int] = {}
    for code in sorted(set(state.held_codes()) | {c for c, w in targets.items() if w > 0}):
        cur = state.lots.get(code, 0)
        if code in frozen or quotes[code].halted:
            tgt[code] = cur
            if quotes[code].halted:
                notes.append(f"{code} halted: position left unchanged")
            continue
        w_t = max(targets.get(code, 0.0), 0.0)
        if w_t == 0:
            tgt[code] = 0
            continue
        w_cur = cur * lot_value(code) / nav
        if cur > 0 and abs(w_t - w_cur) < band:
            tgt[code] = cur
            continue
        want = max(1, int(round(w_t * nav / lot_value(code))))
        if want > cur:
            if w_cur >= rules.weight_caps.cap(code):
                notes.append(f"{code} at {w_cur:.2%} is over its cap; no buying allowed")
                want = cur
            else:
                cap_lots = add_cap_lots(code)
                if want > cap_lots:
                    if cap_lots <= cur:
                        notes.append(f"{code} capped: already at add-cap")
                        want = cur
                    else:
                        notes.append(f"{code} target {w_t:.2%} capped at add-cap {buf.add_cap(code):.2%}")
                        want = cap_lots
        tgt[code] = want

    # Drift trims for names that have been over the cap for too long.
    trim_limit: dict[str, int] = {}  # mandatory sells: never sell less than this
    for code, days in state.overweight_days.items():
        if days >= buf.drift_trim_by_grace_day and code in tgt and code not in frozen:
            limit = math.floor(buf.drift_trim_target(code) * cap_nav / lot_value(code))
            trim_limit[code] = limit
            if tgt[code] > limit:
                notes.append(f"{code} drift trim to {buf.drift_trim_target(code):.2%}")
                tgt[code] = limit

    def sell_ceiling(code: str) -> int:
        """Most lots ``code`` may keep when a sell is cut back (never undoing a drift trim)."""
        return min(state.lots.get(code, 0), trim_limit.get(code, state.lots.get(code, 0)))

    dropped = [c for c, n in tgt.items() if n == 0 and targets.get(c, 0) > 0]
    if dropped:
        notes.append(f"targeted but zero lots after caps: {dropped}")

    # --- worst-case cash floor: shrink buys ----------------------------------------------------
    need, need_halt = buf.cash_min_worst * nav, buf.cash_min_halted_sell * nav

    def floor_shortfall(orders) -> float:
        """Cash missing under the worse of: all fills adverse; the largest sell not filling."""
        worst = stress.worst_case_min_cash(state, orders, quotes, fees, lot, buf.sell_proceeds_fund_buys)
        halt = stress.halted_sell_min_cash(state, orders, quotes, fees, lot)
        return max(need - worst, need_halt - halt)

    def fit_floor() -> bool:
        for _ in range(MAX_ITER):
            shortfall = floor_shortfall(_orders_from(state, tgt))
            if shortfall <= 0:
                return True
            buys = [(c, tgt[c] - state.lots.get(c, 0)) for c in tgt if tgt[c] > state.lots.get(c, 0)]
            # Keep at least one lot of a brand-new name so the holding count is preserved.
            reducible = [(c, d) for c, d in buys if d > (1 if state.lots.get(c, 0) == 0 else 0)]
            if not reducible:
                new_names = [c for c, d in buys if state.lots.get(c, 0) == 0]
                if not new_names:
                    notes.append("cannot satisfy the worst-case cash floor by reducing buys")
                    return False
                drop = min(new_names, key=lambda c: (targets.get(c, 0.0), c))
                notes.append(f"dropped new position {drop}: worst-case cash floor")
                tgt[drop] = 0
                continue
            code, delta = max(reducible, key=lambda x: x[1] * quotes[x[0]].limit_up)
            per_lot = lot * quotes[code].limit_up * (1 + fees.commission_rate)
            floor_lots = 1 if state.lots.get(code, 0) == 0 else 0
            step = min(delta - floor_lots, max(1, math.ceil(shortfall / per_lot)))
            tgt[code] -= step
        notes.append("cash-floor loop did not converge")
        return False

    fit_floor()

    # --- worst-case cash ceiling: add to underweight names -------------------------------------
    step_lots = lambda c: max(1, int(0.001 * nav / lot_value(c)))  # noqa: E731
    shrunk = False
    for _ in range(MAX_ITER):
        orders = _orders_from(state, tgt)
        ratio = stress.worst_case_max_cash_ratio(state, orders, quotes, fees, lot)
        if ratio < buf.cash_max_worst:
            break
        # Prefer cancelling sells (free), then buying the most underweight names with headroom.
        sell_reduce = [c for c in tgt if tgt[c] < sell_ceiling(c) and targets.get(c, 0) > 0]
        candidates = []
        for c in tgt:
            if c in frozen or quotes[c].halted or tgt[c] <= 0:
                continue
            if tgt[c] >= state.lots.get(c, 0) and tgt[c] < add_cap_lots(c) and \
                    state.lots.get(c, 0) * lot_value(c) / nav < rules.weight_caps.cap(c):
                gap = targets.get(c, 0) - tgt[c] * lot_value(c) / nav
                candidates.append((gap, c))
        move = None
        if sell_reduce:
            code = sell_reduce[0]
            move = (code, min(tgt[code] + step_lots(code), sell_ceiling(code)))
        elif candidates:
            code = max(candidates)[1]
            move = (code, min(tgt[code] + step_lots(code), add_cap_lots(code)))
        if move is not None:
            code, new = move
            old, tgt[code] = tgt[code], new
            if floor_shortfall(_orders_from(state, tgt)) <= 0:
                continue
            tgt[code] = old
        # The floor and the ceiling conflict (e.g. a rotation whose buys may not lean on its largest
        # sale): shrink the rotation instead - sell less of the largest sale, then buy less to match.
        n_after = sum(1 for n in tgt.values() if n > 0)
        sells = [(c, (state.lots.get(c, 0) - tgt[c]) * lot_value(c)) for c in tgt
                 if tgt[c] < sell_ceiling(c) and (tgt[c] > 0 or n_after < rules.holdings.max_count)]
        if not sells:
            notes.append("cannot bring worst-case cash under the ceiling: no headroom" if move is None
                         else "cash band too narrow: floor and ceiling conflict")
            break
        code = max(sells, key=lambda x: (x[1], x[0]))[0]
        tgt[code] = min(tgt[code] + step_lots(code), sell_ceiling(code))
        if not shrunk:
            notes.append("rotation shrunk to fit the worst-case cash band; the rest can follow on a later day")
            shrunk = True
        if not fit_floor():
            break
    else:
        notes.append("cash-ceiling loop did not converge")

    tgt = {c: n for c, n in tgt.items() if n > 0}
    return BuildResult(orders=_orders_from(state, tgt), target_lots=tgt, notes=notes)
