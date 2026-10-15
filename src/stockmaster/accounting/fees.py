"""Commission and securities-tax calculation, per order and per trading day."""

from __future__ import annotations

from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

from stockmaster.config import Fees
from stockmaster.domain import Fill, Side


def _round(amount: Decimal, mode: str) -> Decimal:
    if mode == "floor":
        return amount.quantize(Decimal(1), rounding=ROUND_FLOOR)
    if mode == "round":
        return amount.quantize(Decimal(1), rounding=ROUND_HALF_UP)
    if mode == "none":
        return amount
    raise ValueError(f"unknown fee rounding mode {mode!r}")


def make_fill(code: str, side: Side, lots: int, price: Decimal | float, fees: Fees,
              lot_size: int = 1000) -> Fill:
    price = price if isinstance(price, Decimal) else Decimal(str(price))
    gross = price * lots * lot_size
    commission = _round(gross * Decimal(str(fees.commission_rate)), fees.rounding)
    commission = max(commission, Decimal(str(fees.min_commission)))
    tax = Decimal(0)
    if side is Side.SELL:
        tax = _round(gross * Decimal(str(fees.sell_tax(code))), fees.rounding)
    return Fill(code=code, side=side, lots=lots, price=price, gross=gross,
                commission=commission, tax=tax)


def cash_delta(side: Side, lots: int, price: float, fees: Fees, lot_size: int = 1000) -> float:
    """Float version of ``make_fill(...).net_cash`` for stress tests (no rounding)."""
    gross = price * lots * lot_size
    if side is Side.BUY:
        return -gross * (1 + fees.commission_rate)
    return gross * (1 - fees.commission_rate - fees.sell_tax_rate)
