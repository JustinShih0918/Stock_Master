"""Portfolio ledger reproducing the organizer's accounting.

NAV = sum(lots * 1000 * close) + cash (rule 五). Cash dividends are accrued separately and only
added at the end of the round (rule 六(十二) 註1), so ``Valuation.nav`` excludes them.
If a day's submission violates a rule, the organizer voids the day's trades and values the previous
portfolio (rule 六(十四)); ``run_day(..., applied=False)`` reproduces that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from stockmaster.config import Rules
from stockmaster.domain import ZERO, Fill, Order, PortfolioState, Side
from stockmaster.execution.simulator import simulate_fills


class LedgerError(ValueError):
    pass


@dataclass(frozen=True)
class Valuation:
    date: date
    cash: Decimal
    holdings_value: Decimal
    nav: Decimal  # organizer NAV, excludes accrued dividends
    accrued_dividends: Decimal
    weights: dict[str, float]
    n_holdings: int

    @property
    def cash_pct(self) -> float:
        return float(self.cash / self.nav) if self.nav else 0.0

    @property
    def nav_with_dividends(self) -> Decimal:
        return self.nav + self.accrued_dividends


@dataclass(frozen=True)
class DayResult:
    state: PortfolioState
    valuation: Valuation
    fills: list[Fill] = field(default_factory=list)
    applied: bool = True


def apply_fills(state: PortfolioState, fills: Sequence[Fill]) -> PortfolioState:
    lots = dict(state.lots)
    cash = state.cash
    seen: dict[str, Side] = {}
    for f in fills:
        if f.code in seen and seen[f.code] is not f.side:
            raise LedgerError(f"day-trading is prohibited: both buy and sell of {f.code}")
        seen[f.code] = f.side
        held = lots.get(f.code, 0)
        if f.side is Side.SELL:
            if f.lots > held:
                raise LedgerError(f"short sale: selling {f.lots} lots of {f.code}, holding {held}")
            lots[f.code] = held - f.lots
        else:
            lots[f.code] = held + f.lots
        cash += f.net_cash
    lots = {c: n for c, n in lots.items() if n > 0}
    return state.with_(lots=lots, cash=cash)


def value(state: PortfolioState, closes: Mapping[str, float | Decimal], on: date,
          lot_size: int = 1000) -> Valuation:
    values: dict[str, Decimal] = {}
    for code, n in state.lots.items():
        if code not in closes:
            raise LedgerError(f"no close price for held stock {code} on {on}")
        values[code] = Decimal(str(closes[code])) * n * lot_size
    holdings = sum(values.values(), ZERO)
    nav = holdings + state.cash
    weights = {c: float(v / nav) for c, v in values.items()} if nav else {}
    return Valuation(date=on, cash=state.cash, holdings_value=holdings, nav=nav,
                     accrued_dividends=state.accrued_dividends, weights=weights,
                     n_holdings=len(values))


def run_day(state: PortfolioState, trade_date: date, orders: Sequence[Order],
            vwaps: Mapping[str, Decimal | float], closes: Mapping[str, float | Decimal],
            rules: Rules, *, applied: bool = True,
            cash_dividends_per_share: Mapping[str, Decimal | float] | None = None,
            warning: bool = False) -> DayResult:
    """Advance the ledger by one trading day.

    ``cash_dividends_per_share`` lists stocks whose ex-dividend date is ``trade_date``; entitlement
    belongs to lots held at the previous close (before today's fills).
    """
    if state.as_of is not None and trade_date <= state.as_of:
        raise LedgerError(f"trade_date {trade_date} is not after last valuation {state.as_of}")

    accrued = state.accrued_dividends
    for code, dps in (cash_dividends_per_share or {}).items():
        accrued += Decimal(str(dps)) * state.lots.get(code, 0) * rules.lot_size

    fills: list[Fill] = []
    after = state
    if applied and orders:
        fills = simulate_fills(orders, vwaps, rules)
        after = apply_fills(state, fills)

    after = after.with_(accrued_dividends=accrued, as_of=trade_date,
                        warnings=state.warnings + (1 if warning else 0))
    val = value(after, closes, trade_date, rules.lot_size)

    overweight = {}
    for code, w in val.weights.items():
        if w > rules.weight_caps.cap(code):
            overweight[code] = state.overweight_days.get(code, 0) + 1
    after = after.with_(overweight_days=overweight)
    return DayResult(state=after, valuation=val, fills=fills, applied=applied)
