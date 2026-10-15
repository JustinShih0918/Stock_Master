"""Worst-case scenarios for a set of orders, given only information known before the open.

Fills happen at an unknown VWAP and valuation at an unknown close, both bounded by the daily price
limits. The scenarios here bound end-of-day cash and post-trade weights over that range.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from stockmaster.accounting.fees import cash_delta
from stockmaster.config import Fees
from stockmaster.domain import Order, PortfolioState, Quote, Side


def post_trade_lots(state: PortfolioState, orders: Sequence[Order]) -> dict[str, int]:
    lots = dict(state.lots)
    for o in orders:
        lots[o.code] = lots.get(o.code, 0) + (o.lots if o.side is Side.BUY else -o.lots)
    return {c: n for c, n in lots.items() if n != 0}


def _cash_after(state: PortfolioState, orders: Sequence[Order], buy_px, sell_px, fees: Fees,
                lot_size: int, include_sells: bool = True) -> float:
    cash = float(state.cash)
    for o in orders:
        if o.side is Side.BUY:
            cash += cash_delta(Side.BUY, o.lots, buy_px(o.code), fees, lot_size)
        elif include_sells:
            cash += cash_delta(Side.SELL, o.lots, sell_px(o.code), fees, lot_size)
    return cash


def reference_cash(state: PortfolioState, orders: Sequence[Order], quotes: Mapping[str, Quote],
                   fees: Fees, lot_size: int = 1000) -> float:
    px = lambda c: quotes[c].ref_price  # noqa: E731
    return _cash_after(state, orders, px, px, fees, lot_size)


def worst_case_min_cash(state: PortfolioState, orders: Sequence[Order],
                        quotes: Mapping[str, Quote], fees: Fees, lot_size: int = 1000,
                        sell_proceeds_fund_buys: bool = True) -> float:
    """Lowest possible end-of-day cash: buys fill at limit-up, sells at limit-down."""
    return _cash_after(state, orders, lambda c: quotes[c].limit_up, lambda c: quotes[c].limit_down,
                       fees, lot_size, include_sells=sell_proceeds_fund_buys)


def halted_sell_min_cash(state: PortfolioState, orders: Sequence[Order],
                         quotes: Mapping[str, Quote], fees: Fees, lot_size: int = 1000) -> float:
    """Lowest end-of-day cash if the largest sell does not fill at all (the stock is halted after the
    decision, as 2327 was for a whole day on 2019-11-12), with buys at limit-up and the other sells at
    limit-down. Sell proceeds settle the same day, so without this case a swap can overdraw cash."""
    sells = [cash_delta(Side.SELL, o.lots, quotes[o.code].limit_down, fees, lot_size)
             for o in orders if o.side is Side.SELL]
    worst = worst_case_min_cash(state, orders, quotes, fees, lot_size, sell_proceeds_fund_buys=True)
    return worst - max(sells, default=0.0)


def worst_case_max_cash_ratio(state: PortfolioState, orders: Sequence[Order],
                              quotes: Mapping[str, Quote], fees: Fees,
                              lot_size: int = 1000) -> float:
    """Highest possible cash/NAV at the close: sells fill at limit-up, buys at limit-down and every
    holding closes at limit-down."""
    cash = _cash_after(state, orders, lambda c: quotes[c].limit_down,
                       lambda c: quotes[c].limit_up, fees, lot_size)
    lots = post_trade_lots(state, orders)
    holdings = sum(n * lot_size * quotes[c].limit_down for c, n in lots.items())
    nav = cash + holdings
    return cash / nav if nav > 0 else float("inf")


def adverse_fill_cash(state: PortfolioState, orders: Sequence[Order], quotes: Mapping[str, Quote],
                      fees: Fees, premium: float, lot_size: int = 1000) -> float:
    """End-of-day cash when buys fill ``premium`` above and sells ``premium`` below the reference
    close. Used with ``stressed_weight``: a fill at limit-up would also mean a limit-up close, so
    pairing worst-case fills with falling closes would describe an impossible day."""
    return _cash_after(state, orders, lambda c: quotes[c].ref_price * (1 + premium),
                       lambda c: quotes[c].ref_price * (1 - premium), fees, lot_size)


def reference_weights(lots: Mapping[str, int], cash: float, quotes: Mapping[str, Quote],
                      lot_size: int = 1000) -> tuple[dict[str, float], float]:
    values = {c: n * lot_size * quotes[c].ref_price for c, n in lots.items() if n > 0}
    nav = cash + sum(values.values())
    return ({c: v / nav for c, v in values.items()} if nav > 0 else {}), nav


def stressed_weight(code: str, lots: Mapping[str, int], cash: float,
                    quotes: Mapping[str, Quote], rest_move: float, lot_size: int = 1000) -> float:
    """Weight of ``code`` at the close if it closes limit-up while every other holding moves by
    ``rest_move`` from its reference price, with end-of-day cash ``cash``."""
    own = lots.get(code, 0) * lot_size * quotes[code].limit_up
    rest = sum(n * lot_size * quotes[c].ref_price * (1 + rest_move)
               for c, n in lots.items() if c != code and n > 0)
    nav = cash + own + rest
    return own / nav if nav > 0 else float("inf")
