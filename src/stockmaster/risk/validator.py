"""Deterministic validation of a day's orders against the competition rules plus internal buffers.

Every model proposal passes through ``validate_orders`` before anything is recorded or submitted.
Hard failures block the orders; soft failures are reported and logged. The validator is pure:
it reads a ``ValidationContext`` and returns a ``ValidationReport``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from stockmaster.config import Rules, Strategy
from stockmaster.domain import Order, PortfolioState, Quote, Side
from stockmaster.risk import stress
from stockmaster.risk.active_share import active_share_report

EPS = 1e-9


class Severity(StrEnum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    severity: Severity
    passed: bool
    measured: float | None = None
    limit: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class ValidationContext:
    trade_date: date
    state: PortfolioState
    quotes: Mapping[str, Quote]
    universe: frozenset[str]
    rules: Rules
    strategy: Strategy
    etf_holdings: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    stale_codes: frozenset[str] = frozenset()
    turnover_budget: float | None = None  # defaults to strategy.turnover.daily_budget


@dataclass(frozen=True)
class ValidationReport:
    results: list[RuleResult]
    metrics: dict

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity is Severity.HARD)

    @property
    def violations(self) -> list[RuleResult]:
        return [r for r in self.results if r.severity is Severity.HARD and not r.passed]

    @property
    def soft_failures(self) -> list[RuleResult]:
        return [r for r in self.results if r.severity is Severity.SOFT and not r.passed]

    def summary(self) -> str:
        bad = self.violations + self.soft_failures
        if not bad:
            return "all checks passed"
        return "; ".join(f"[{r.severity}] {r.rule_id}: {r.detail}" for r in bad)


def validate_orders(orders: Sequence[Order], ctx: ValidationContext) -> ValidationReport:
    rules, buf = ctx.rules, ctx.strategy.buffers
    lot = rules.lot_size
    res: list[RuleResult] = []
    add = res.append
    state = ctx.state

    # --- structural rules -------------------------------------------------------------------
    counts = Counter(o.code for o in orders)
    dup = sorted(c for c, n in counts.items() if n > 1)
    sides: dict[str, set[Side]] = {}
    for o in orders:
        sides.setdefault(o.code, set()).add(o.side)
    both = sorted(c for c, s in sides.items() if len(s) > 1)
    add(RuleResult("R_NO_BUY_SELL_SAME", Severity.HARD, not both,
                   detail=f"buy and sell of the same stock: {both}" if both else ""))
    add(RuleResult("R_ONE_ORDER_PER_STOCK", Severity.HARD, not dup,
                   detail=f"multiple orders for {dup}" if dup else ""))

    outside = sorted({o.code for o in orders} - ctx.universe)
    held_outside = sorted(state.held_codes() - ctx.universe)
    add(RuleResult("R_UNIVERSE", Severity.HARD, not outside and not held_outside,
                   detail=f"not in the 150-stock universe: {outside + held_outside}"
                   if outside or held_outside else ""))

    shorts = [f"{o.code} sell {o.lots} > held {state.lots.get(o.code, 0)}"
              for o in orders if o.side is Side.SELL and o.lots > state.lots.get(o.code, 0)]
    add(RuleResult("R_NO_SHORT", Severity.HARD, not shorts, detail="; ".join(shorts)))

    needed = {o.code for o in orders} | state.held_codes()
    missing = sorted(c for c in needed if c not in ctx.quotes)
    add(RuleResult("R_QUOTES", Severity.HARD, not missing,
                   detail=f"no reference quote for {missing}" if missing else ""))
    if missing:
        return ValidationReport(res, {})

    halted = sorted(o.code for o in orders if ctx.quotes[o.code].halted)
    add(RuleResult("R_HALTED", Severity.HARD, not halted,
                   detail=f"orders on halted stocks: {halted}" if halted else ""))
    stale = sorted({o.code for o in orders} & ctx.stale_codes)
    add(RuleResult("R_STALE", Severity.HARD, not stale,
                   detail=f"orders on stocks with stale data: {stale}" if stale else ""))
    latest = max(ctx.quotes[c].as_of for c in needed) if needed else None
    bad_dates = sorted(c for c in needed if ctx.quotes[c].as_of >= ctx.trade_date)
    old_orders = sorted(o.code for o in orders if ctx.quotes[o.code].as_of != latest)
    add(RuleResult("R_QUOTE_DATE", Severity.HARD, not bad_dates and not old_orders,
                   detail=(f"reference price not before trade date: {bad_dates}; " if bad_dates else "")
                   + (f"orders priced off an older close than {latest}: {old_orders}" if old_orders else "")))
    if any(r.severity is Severity.HARD and not r.passed for r in res):
        return ValidationReport(res, {})

    # --- post-trade state at reference prices ----------------------------------------------
    lots_after = stress.post_trade_lots(state, orders)
    cash_ref = stress.reference_cash(state, orders, ctx.quotes, rules.fees, lot)
    weights_before, nav_before = stress.reference_weights(state.lots, float(state.cash), ctx.quotes, lot)
    weights_after, nav_ref = stress.reference_weights(lots_after, cash_ref, ctx.quotes, lot)
    n_after = sum(1 for n in lots_after.values() if n > 0)

    add(RuleResult("R_COUNT", Severity.HARD,
                   rules.holdings.min_count <= n_after <= rules.holdings.max_count, n_after,
                   detail=f"{n_after} holdings after trades, allowed "
                          f"{rules.holdings.min_count}-{rules.holdings.max_count}"))
    add(RuleResult("R_COUNT_TARGET", Severity.SOFT,
                   buf.holdings_target_min <= n_after <= buf.holdings_target_max, n_after,
                   detail=f"{n_after} holdings, target {buf.holdings_target_min}-{buf.holdings_target_max}"))

    # --- cash bounds under worst-case fills and closes ---------------------------------------
    min_cash = stress.worst_case_min_cash(state, orders, ctx.quotes, rules.fees, lot,
                                          buf.sell_proceeds_fund_buys)
    min_cash_pct = min_cash / nav_ref if nav_ref > 0 else -1.0
    add(RuleResult("R_CASH_MIN_WORST", Severity.HARD, min_cash_pct >= buf.cash_min_worst - EPS,
                   min_cash_pct, buf.cash_min_worst,
                   detail=f"worst-case cash {min_cash_pct:.2%} of NAV "
                          f"(buys at limit-up, sells at limit-down); need >= {buf.cash_min_worst:.2%}"))
    halt_cash = stress.halted_sell_min_cash(state, orders, ctx.quotes, rules.fees, lot)
    halt_cash_pct = halt_cash / nav_ref if nav_ref > 0 else -1.0
    add(RuleResult("R_CASH_MIN_HALT", Severity.HARD, halt_cash_pct >= buf.cash_min_halted_sell - EPS,
                   halt_cash_pct, buf.cash_min_halted_sell,
                   detail=f"worst-case cash {halt_cash_pct:.2%} of NAV if the largest sell does not fill "
                          f"(halted stock); need >= {buf.cash_min_halted_sell:.2%}"))
    max_ratio = stress.worst_case_max_cash_ratio(state, orders, ctx.quotes, rules.fees, lot)
    add(RuleResult("R_CASH_MAX_WORST", Severity.HARD, max_ratio < buf.cash_max_worst, max_ratio,
                   buf.cash_max_worst,
                   detail=f"worst-case cash ratio {max_ratio:.2%}; official limit "
                          f"{rules.cash.max_exclusive:.0%}, internal {buf.cash_max_worst:.0%}"))

    # --- weight caps ------------------------------------------------------------------------
    bought = {o.code for o in orders if o.side is Side.BUY}
    stress_cash = stress.adverse_fill_cash(state, orders, ctx.quotes, rules.fees,
                                           buf.stress_fill_premium, lot)
    cap_fail, stress_rows = [], {}
    for code in sorted(bought):
        cap = rules.weight_caps.cap(code)
        if weights_before.get(code, 0.0) >= cap - EPS:
            cap_fail.append(f"{code} already at {weights_before[code]:.2%} >= cap {cap:.0%}; any buy is 加碼")
        if weights_after.get(code, 0.0) > buf.add_cap(code) + EPS:
            cap_fail.append(f"{code} {weights_after[code]:.2%} > add-cap {buf.add_cap(code):.2%}")
        sw = stress.stressed_weight(code, lots_after, stress_cash, ctx.quotes,
                                    buf.stress_rest_of_portfolio, lot)
        stress_rows[code] = sw
        if sw > cap + EPS:
            cap_fail.append(f"{code} stressed weight {sw:.2%} > cap {cap:.0%}")
    add(RuleResult("R_CAP_ADD", Severity.HARD, not cap_fail, detail="; ".join(cap_fail)))

    drift_fail = []
    for code, n in lots_after.items():
        if code in bought or n <= 0:
            continue
        cap = rules.weight_caps.cap(code)
        days = state.overweight_days.get(code, 0)
        w = weights_after.get(code, 0.0)
        if days >= buf.drift_trim_by_grace_day and w > buf.drift_trim_target(code) + EPS:
            drift_fail.append(f"{code} overweight for {days} days, still {w:.2%} after trades; "
                              f"trim to {buf.drift_trim_target(code):.2%}")
        elif days >= rules.drift_grace_trading_days and w > cap:
            drift_fail.append(f"{code} beyond the {rules.drift_grace_trading_days}-day grace period")
    add(RuleResult("R_CAP_DRIFT", Severity.HARD, not drift_fail, detail="; ".join(drift_fail)))

    # --- Active Share vs reference active ETFs ------------------------------------------------
    as_min, as_binding = None, None
    if ctx.etf_holdings:
        report = active_share_report(weights_after, ctx.etf_holdings, rules.active_share.top_n)
        as_min, as_binding = report[0].value, report[0].etf
        add(RuleResult("R_ACTIVE_SHARE", Severity.HARD, as_min >= buf.active_share_hard_floor,
                       as_min, buf.active_share_hard_floor,
                       detail=f"min Active Share {as_min:.1%} vs {as_binding} "
                              f"(overlap {list(report[0].overlap)})"))
        add(RuleResult("R_ACTIVE_SHARE_TARGET", Severity.SOFT, as_min >= buf.active_share_target,
                       as_min, buf.active_share_target,
                       detail=f"min Active Share {as_min:.1%} below target {buf.active_share_target:.0%}"))
    else:
        add(RuleResult("R_ACTIVE_SHARE", Severity.SOFT, False,
                       detail="no active-ETF holdings available; Active Share NOT verified"))

    # --- turnover (soft) ----------------------------------------------------------------------
    traded = sum(o.lots * lot * ctx.quotes[o.code].ref_price for o in orders)
    turnover = traded / nav_ref / 2 if nav_ref > 0 else 0.0  # one-way
    budget = ctx.turnover_budget if ctx.turnover_budget is not None else ctx.strategy.turnover.daily_budget
    first_day = state.as_of is None
    add(RuleResult("R_TURNOVER", Severity.SOFT, first_day or turnover <= budget + EPS, turnover, budget,
                   detail=f"one-way turnover {turnover:.2%} vs budget {budget:.2%}"))

    signed_notional = sum(o.lots * lot * ctx.quotes[o.code].ref_price * (1 if o.side is Side.BUY else -1)
                          for o in orders)
    est_cost = (float(state.cash) - signed_notional) - cash_ref
    metrics = {
        "nav_ref": nav_ref,
        "nav_before_ref": nav_before,
        "cash_ref": cash_ref,
        "cash_ref_pct": cash_ref / nav_ref if nav_ref > 0 else None,
        "worst_cash_min_pct": min_cash_pct,
        "halted_sell_cash_min_pct": halt_cash_pct,
        "worst_cash_max_ratio": max_ratio,
        "n_holdings": n_after,
        "weights": weights_after,
        "stressed_weights": stress_rows,
        "turnover_one_way": turnover,
        "est_cost": est_cost,
        "active_share_min": as_min,
        "active_share_binding_etf": as_binding,
    }
    return ValidationReport(res, metrics)
