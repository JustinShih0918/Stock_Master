"""Core value types shared by accounting, risk, portfolio construction, execution and the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum

ZERO = Decimal(0)


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


# TWSE/TPEx stock tick sizes (升降單位) by price level.
_TICKS: tuple[tuple[Decimal, Decimal], ...] = (
    (Decimal(10), Decimal("0.01")),
    (Decimal(50), Decimal("0.05")),
    (Decimal(100), Decimal("0.1")),
    (Decimal(500), Decimal("0.5")),
    (Decimal(1000), Decimal(1)),
)
_TOP_TICK = Decimal(5)


def tick_size(price: Decimal) -> Decimal:
    for upper, tick in _TICKS:
        if price < upper:
            return tick
    return _TOP_TICK


def limit_prices(prev_close: float | Decimal, limit_pct: float = 0.10) -> tuple[Decimal, Decimal]:
    """Daily limit-up / limit-down prices from the previous close, rounded inward to a valid tick."""
    p = Decimal(str(prev_close))
    pct = Decimal(str(limit_pct))
    up_raw = p * (1 + pct)
    dn_raw = p * (1 - pct)
    up_tick = tick_size(up_raw)
    dn_tick = tick_size(dn_raw)
    up = (up_raw / up_tick).to_integral_value(rounding=ROUND_FLOOR) * up_tick
    dn = (dn_raw / dn_tick).to_integral_value(rounding=ROUND_CEILING) * dn_tick
    return up, dn


@dataclass(frozen=True)
class Quote:
    """Reference prices known when a decision for ``trade_date`` is made (after the T-1 close)."""

    code: str
    ref_price: float  # previous trading day's close
    limit_up: float
    limit_down: float
    as_of: date  # date of ref_price
    halted: bool = False

    @classmethod
    def from_close(cls, code: str, close: float, as_of: date, limit_pct: float = 0.10,
                   halted: bool = False, limit_base: float | None = None) -> Quote:
        """``limit_base`` is the exchange reference price for the trade date when it differs from
        the close (ex-dividend / ex-rights days); price limits are computed from it."""
        up, dn = limit_prices(limit_base if limit_base else close, limit_pct)
        return cls(code, float(close), float(up), float(dn), as_of, halted)


@dataclass(frozen=True)
class Order:
    code: str
    side: Side
    lots: int

    def __post_init__(self) -> None:
        if not isinstance(self.lots, int) or self.lots <= 0:
            raise ValueError(f"order lots must be a positive int, got {self.lots!r} for {self.code}")


@dataclass(frozen=True)
class Fill:
    code: str
    side: Side
    lots: int
    price: Decimal
    gross: Decimal
    commission: Decimal
    tax: Decimal

    @property
    def net_cash(self) -> Decimal:
        """Signed cash impact: negative for buys, positive for sells."""
        if self.side is Side.BUY:
            return -(self.gross + self.commission)
        return self.gross - self.commission - self.tax


@dataclass(frozen=True)
class PortfolioState:
    """End-of-day state after valuation on ``as_of`` (None before the first trading day)."""

    as_of: date | None
    cash: Decimal
    lots: dict[str, int] = field(default_factory=dict)
    accrued_dividends: Decimal = ZERO
    # Consecutive trading days a name has closed above its official cap (price drift).
    overweight_days: dict[str, int] = field(default_factory=dict)
    warnings: int = 0

    def held_codes(self) -> frozenset[str]:
        return frozenset(c for c, n in self.lots.items() if n > 0)

    def with_(self, **changes) -> PortfolioState:
        return replace(self, **changes)

    @classmethod
    def initial(cls, cash: int | Decimal) -> PortfolioState:
        return cls(as_of=None, cash=Decimal(cash))
