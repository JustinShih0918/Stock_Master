"""Paper execution: every order fills in full at the trading day's VWAP (competition rule 六(四)(五))."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from stockmaster.accounting.fees import make_fill
from stockmaster.config import Rules
from stockmaster.domain import Fill, Order


class MissingFillPrice(KeyError):
    pass


def vwap_from_value(value_ntd: float | Decimal, volume_shares: float | Decimal,
                    decimals: int | None = 2) -> Decimal:
    """VWAP = traded value / traded shares. Rounded to ``decimals`` (None keeps full precision)."""
    value = Decimal(str(value_ntd))
    volume = Decimal(str(volume_shares))
    if volume <= 0:
        raise ValueError("no volume traded, VWAP undefined")
    v = value / volume
    return v if decimals is None else v.quantize(Decimal(1).scaleb(-decimals))


def simulate_fills(orders: Sequence[Order], vwaps: Mapping[str, Decimal | float],
                   rules: Rules) -> list[Fill]:
    fills = []
    for o in orders:
        if o.code not in vwaps:
            raise MissingFillPrice(o.code)
        fills.append(make_fill(o.code, o.side, o.lots, vwaps[o.code], rules.fees, rules.lot_size))
    return fills
