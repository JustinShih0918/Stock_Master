"""Target weights -> odd-lot share orders for the retail account, and the deterministic checks every
order list must pass. Pure functions: no I/O, no model calls.

Order of operations: ``proposal_problems`` (what the model is told to fix) -> ``project`` (the
deterministic repair if it does not) -> ``size`` (shares, minimum order size, minimum holding
period, worst-case cash, orders-per-day cap) -> ``checks`` (final verdict, recorded with the run).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from stockmaster.domain import Order, PortfolioState, Quote, Side
from stockmaster.retail.config import RetailRules
from stockmaster.risk.validator import RuleResult, Severity

EPS = 1e-9


@dataclass(frozen=True)
class SizingContext:
    trade_date: date
    state: PortfolioState
    quotes: Mapping[str, Quote]
    rules: RetailRules
    eligible: frozenset[str]
    held_days: Mapping[str, int] = field(default_factory=dict)  # trading days each holding has been held

    @property
    def nav_ref(self) -> float:
        return float(self.state.cash) + sum(n * self.quotes[c].ref_price for c, n in self.state.lots.items()
                                            if c in self.quotes)


@dataclass
class Plan:
    orders: list[Order]
    target_shares: dict[str, int]
    notes: list[str]
    checks: list[RuleResult]
    metrics: dict

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity is Severity.HARD)


def commission(notional: float, rules: RetailRules) -> float:
    if notional <= 0:
        return 0.0
    return max(math.floor(notional * rules.fees.commission_rate), rules.fees.min_commission)


def sell_cost(code: str, notional: float, rules: RetailRules) -> float:
    return commission(notional, rules) + math.floor(notional * rules.fees.sell_tax(code))


# --- proposal checks and projection --------------------------------------------------------------

def proposal_problems(weights: Mapping[str, float], ctx: SizingContext) -> list[str]:
    """Rule problems in a proposed weight vector, phrased for the model to fix."""
    p, out = ctx.rules.portfolio, []
    held = ctx.state.held_codes()
    bad = sorted(c for c in weights if c not in ctx.eligible and c not in held)
    if bad:
        out.append(f"不在可交易名單（流動性/價格/上市天數不足或不存在）：{bad}")
    unquoted = sorted(c for c in weights if c not in ctx.quotes)
    if unquoted:
        out.append(f"沒有參考價：{unquoted}")
    if sum(1 for w in weights.values() if w > 0) > p.max_positions:
        out.append(f"持股檔數上限 {p.max_positions} 檔，提案 {sum(1 for w in weights.values() if w > 0)} 檔")
    over = {c: round(w * 100, 1) for c, w in weights.items() if w > p.max_weight + EPS}
    if over:
        out.append(f"單一持股上限 {p.max_weight:.0%}，超過：{over}")
    total = sum(max(w, 0) for w in weights.values())
    if total > 1 - p.cash_buffer_pct + 1e-6:
        out.append(f"總投資比重 {total:.1%} 超過上限 {1 - p.cash_buffer_pct:.0%}（需保留 {p.cash_buffer_pct:.0%} 現金）")
    return out


def project(weights: Mapping[str, float], ctx: SizingContext) -> tuple[dict[str, float], list[str]]:
    """Deterministic repair: drop untradeable names, clip to the per-name cap, keep the largest
    ``max_positions`` and scale the total down to the investable share."""
    p, notes = ctx.rules.portfolio, []
    held = ctx.state.held_codes()
    w = {c: max(0.0, float(x)) for c, x in weights.items()
         if (c in ctx.eligible or c in held) and c in ctx.quotes and x > 0}
    dropped = sorted(set(weights) - set(w))
    if dropped:
        notes.append(f"移除不可交易或零權重標的：{dropped}")
    for c in list(w):
        if w[c] > p.max_weight:
            notes.append(f"{c} 權重 {w[c]:.1%} 降至上限 {p.max_weight:.0%}")
            w[c] = p.max_weight
    if len(w) > p.max_positions:
        keep = sorted(w, key=lambda c: (-w[c], c))[:p.max_positions]
        notes.append(f"持股檔數超過 {p.max_positions}，保留權重最高者：{keep}")
        w = {c: w[c] for c in keep}
    cap = 1 - p.cash_buffer_pct
    total = sum(w.values())
    if total > cap:
        w = {c: x * cap / total for c, x in w.items()}
        notes.append(f"總權重 {total:.1%} 等比例縮至 {cap:.0%}")
    return w, notes


# --- sizing --------------------------------------------------------------------------------------

def _orders(state: PortfolioState, target: Mapping[str, int]) -> list[Order]:
    out = []
    for c in sorted(set(state.lots) | set(target)):
        d = target.get(c, 0) - state.lots.get(c, 0)
        if d > 0:
            out.append(Order(c, Side.BUY, d))
        elif d < 0:
            out.append(Order(c, Side.SELL, -d))
    return out


def worst_case_cash(state: PortfolioState, orders: list[Order], quotes: Mapping[str, Quote],
                    rules: RetailRules) -> float:
    """Cash after the day if every buy fills at limit-up and every sell at limit-down (fees included)."""
    cash = float(state.cash)
    for o in orders:
        q = quotes[o.code]
        if o.side is Side.BUY:
            notional = o.lots * q.limit_up
            cash -= notional + commission(notional, rules)
        else:
            notional = o.lots * q.limit_down
            cash += notional - sell_cost(o.code, notional, rules)
    return cash


def size(weights: Mapping[str, float], ctx: SizingContext) -> Plan:
    rules, p = ctx.rules, ctx.rules.portfolio
    state, quotes, nav = ctx.state, ctx.quotes, ctx.nav_ref
    notes: list[str] = []
    tgt: dict[str, int] = {}
    for c in sorted(state.held_codes() | {c for c, w in weights.items() if w > 0}):
        cur, q = state.lots.get(c, 0), quotes.get(c)
        if q is None:
            tgt[c] = cur
            notes.append(f"{c} 無報價，部位不動")
            continue
        if q.halted:
            tgt[c] = cur
            if weights.get(c, 0) * nav / q.ref_price != cur:
                notes.append(f"{c} 停止交易或前一日無成交，部位不動")
            continue
        want = math.floor(weights.get(c, 0.0) * nav / q.ref_price)
        if want < cur and ctx.held_days.get(c, 10**6) < p.min_hold_days:
            notes.append(f"{c} 持有 {ctx.held_days.get(c)} 日 < 最短持有 {p.min_hold_days} 日，暫不賣出")
            want = cur
        trade_value = abs(want - cur) * q.ref_price
        if want > 0 and cur > 0 and trade_value < p.min_order_ntd:
            want = cur  # adjustment too small to be worth a minimum commission
        if cur == 0 and 0 < want and want * q.ref_price < p.min_order_ntd:
            notes.append(f"{c} 目標金額 NT${want * q.ref_price:,.0f} 低於最小下單金額 NT${p.min_order_ntd:,.0f}，不買")
            want = 0
        tgt[c] = want

    # Too many holdings after blocked sells: drop the smallest new buys.
    def n_after() -> int:
        return sum(1 for n in tgt.values() if n > 0)
    while n_after() > p.max_positions:
        new = [c for c in tgt if tgt[c] > 0 and state.lots.get(c, 0) == 0]
        if not new:
            break
        drop = min(new, key=lambda c: (weights.get(c, 0.0), c))
        notes.append(f"持股檔數上限：放棄新買進 {drop}")
        tgt[drop] = 0

    # Worst-case cash >= 0: trim buys one share at a time, smallest weight first.
    for _ in range(100000):
        orders = _orders(state, tgt)
        if worst_case_cash(state, orders, quotes, rules) >= 0:
            break
        buys = [o for o in orders if o.side is Side.BUY]
        if not buys:
            break
        o = min(buys, key=lambda o: (weights.get(o.code, 0.0), o.code))
        tgt[o.code] -= 1
        if state.lots.get(o.code, 0) == 0 and tgt[o.code] * quotes[o.code].ref_price < p.min_order_ntd:
            tgt[o.code] = 0
            notes.append(f"現金不足（含漲停價與手續費），取消買進 {o.code}")

    # Orders-per-day cap: keep every sell, then the largest buys.
    orders = _orders(state, tgt)
    if len(orders) > p.max_orders_per_day:
        sells = [o for o in orders if o.side is Side.SELL]
        buys = sorted((o for o in orders if o.side is Side.BUY),
                      key=lambda o: -o.lots * quotes[o.code].ref_price)
        keep_buys = buys[:max(0, p.max_orders_per_day - len(sells))]
        for o in buys[len(keep_buys):]:
            tgt[o.code] = state.lots.get(o.code, 0)
            notes.append(f"每日委託上限 {p.max_orders_per_day} 筆：延後買進 {o.code}")
        orders = _orders(state, tgt)

    tgt = {c: n for c, n in tgt.items() if n > 0}
    checks, metrics = evaluate(orders, ctx)
    return Plan(orders, tgt, notes, checks, metrics)


# --- final checks --------------------------------------------------------------------------------

def evaluate(orders: list[Order], ctx: SizingContext) -> tuple[list[RuleResult], dict]:
    rules, p, state, quotes = ctx.rules, ctx.rules.portfolio, ctx.state, ctx.quotes
    res: list[RuleResult] = []
    add = res.append
    sides: dict[str, set] = {}
    for o in orders:
        sides.setdefault(o.code, set()).add(o.side)
    both = sorted(c for c, s in sides.items() if len(s) > 1)
    add(RuleResult("R_NO_BUY_SELL_SAME", Severity.HARD, not both, detail=f"同日買賣同一標的：{both}" if both else ""))
    shorts = [o.code for o in orders if o.side is Side.SELL and o.lots > state.lots.get(o.code, 0)]
    add(RuleResult("R_NO_SHORT", Severity.HARD, not shorts, detail=f"賣出超過持股：{shorts}" if shorts else ""))
    missing = sorted({o.code for o in orders} - set(quotes))
    add(RuleResult("R_QUOTES", Severity.HARD, not missing, detail=f"無報價：{missing}" if missing else ""))
    if missing:
        return res, {}
    outside = sorted(o.code for o in orders if o.side is Side.BUY and o.code not in ctx.eligible)
    add(RuleResult("R_UNIVERSE", Severity.HARD, not outside, detail=f"買進不在可交易名單：{outside}" if outside else ""))
    halted = sorted(o.code for o in orders if quotes[o.code].halted)
    add(RuleResult("R_HALTED", Severity.HARD, not halted, detail=f"停止交易標的有委託：{halted}" if halted else ""))
    add(RuleResult("R_ORDERS_PER_DAY", Severity.HARD, len(orders) <= p.max_orders_per_day, len(orders),
                   p.max_orders_per_day, detail=f"{len(orders)} 筆委託"))

    lots = dict(state.lots)
    for o in orders:
        lots[o.code] = lots.get(o.code, 0) + (o.lots if o.side is Side.BUY else -o.lots)
    lots = {c: n for c, n in lots.items() if n > 0}
    fees = 0.0
    cash_ref = float(state.cash)
    traded = 0.0
    for o in orders:
        notional = o.lots * quotes[o.code].ref_price
        traded += notional
        if o.side is Side.BUY:
            fee = commission(notional, rules)
            cash_ref -= notional + fee
        else:
            fee = sell_cost(o.code, notional, rules)
            cash_ref += notional - fee
        fees += fee
    values = {c: n * quotes[c].ref_price for c, n in lots.items() if c in quotes}
    nav_after = cash_ref + sum(values.values())
    weights = {c: v / nav_after for c, v in values.items()} if nav_after > 0 else {}
    n_after = len(lots)
    add(RuleResult("R_POSITIONS", Severity.HARD, n_after <= p.max_positions, n_after, p.max_positions,
                   detail=f"交易後持股 {n_after} 檔"))
    bought = {o.code for o in orders if o.side is Side.BUY}
    over = {c: round(weights.get(c, 0), 4) for c in bought if weights.get(c, 0) > p.max_weight + 0.01}
    add(RuleResult("R_WEIGHT", Severity.HARD, not over, max(weights.values(), default=0.0), p.max_weight,
                   detail=f"買進後超過單檔上限：{over}" if over else ""))
    worst = worst_case_cash(state, orders, quotes, rules)
    add(RuleResult("R_CASH_WORST", Severity.HARD, worst >= -EPS, worst, 0.0,
                   detail=f"最差情境（買進漲停、賣出跌停）現金 NT${worst:,.0f}"))
    small = sorted(o.code for o in orders if o.side is Side.BUY
                   and o.lots * quotes[o.code].ref_price < p.min_order_ntd)
    add(RuleResult("R_MIN_ORDER", Severity.SOFT, not small, detail=f"低於最小下單金額：{small}" if small else ""))
    nav0 = ctx.nav_ref
    fee_pct = fees / nav0 if nav0 > 0 else 0.0
    add(RuleResult("R_FEE_DRAG", Severity.SOFT, fee_pct <= 0.01 + EPS, fee_pct, 0.01,
                   detail=f"預估手續費與稅 NT${fees:,.0f}（淨值 {fee_pct:.2%}）"))
    metrics = {"nav_ref": nav0, "cash_ref": cash_ref, "cash_ref_pct": cash_ref / nav_after if nav_after > 0 else None,
               "n_holdings": n_after, "weights": weights, "est_cost": fees, "fee_pct": fee_pct,
               "worst_cash": worst, "turnover_one_way": traded / nav0 / 2 if nav0 > 0 else 0.0}
    return res, metrics
