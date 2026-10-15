"""Position book from applied fills: average cost (fees included), realized P&L and the date each
open position was opened. Shared by the decision pipeline (holding periods) and the monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from sqlalchemy import Engine, select

from stockmaster.data import schema


@dataclass
class Lot:
    code: str
    shares: int = 0
    cost: float = 0.0          # remaining cost basis incl. buy commissions
    opened: date | None = None
    realized: float = 0.0      # realized P&L (net of all fees and taxes) from sells
    fees: float = 0.0          # commissions + taxes paid on this code
    buys: int = 0
    sells: int = 0

    @property
    def avg_cost(self) -> float | None:
        return self.cost / self.shares if self.shares else None


@dataclass
class Book:
    lots: dict[str, Lot] = field(default_factory=dict)

    @property
    def realized(self) -> float:
        return sum(lot.realized for lot in self.lots.values())

    @property
    def fees(self) -> float:
        return sum(lot.fees for lot in self.lots.values())

    def open_positions(self) -> dict[str, Lot]:
        return {c: lot for c, lot in self.lots.items() if lot.shares > 0}


def _f(x) -> float:
    return float(x) if isinstance(x, Decimal) else float(x or 0)


def position_book(engine: Engine, account_id: str, before: date | None = None) -> Book:
    """Replay applied fills (in date order; sells before buys within a day) through ``before``
    (exclusive)."""
    f = schema.fill
    q = select(f.c.date, f.c.code, f.c.side, f.c.lots, f.c.gross, f.c.commission, f.c.tax) \
        .where(f.c.account_id == account_id).where(f.c.status == "applied")
    if before is not None:
        q = q.where(f.c.date < before)
    with engine.connect() as conn:
        rows = conn.execute(q.order_by(f.c.date, f.c.id)).all()
    book = Book()
    rows = sorted(rows, key=lambda r: (r.date, 0 if r.side == "sell" else 1))
    for r in rows:
        lot = book.lots.setdefault(r.code, Lot(r.code))
        gross, comm, tax = _f(r.gross), _f(r.commission), _f(r.tax)
        lot.fees += comm + tax
        if r.side == "buy":
            if lot.shares == 0:
                lot.opened, lot.cost = r.date, 0.0
            lot.shares += r.lots
            lot.cost += gross + comm
            lot.buys += 1
        else:
            n = min(r.lots, lot.shares)
            basis = lot.cost * n / lot.shares if lot.shares else 0.0
            lot.realized += gross - comm - tax - basis
            lot.cost -= basis
            lot.shares -= n
            lot.sells += 1
            if lot.shares == 0:
                lot.cost, lot.opened = 0.0, None
    return book
