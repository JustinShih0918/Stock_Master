"""TWSE trading calendar: weekdays minus the official holiday schedule, plus manual overrides
(e.g. typhoon closures announced the evening before, or make-up trading days)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass(frozen=True)
class TradingCalendar:
    holidays: frozenset[date]
    # date -> is_open; wins over weekday/holiday logic
    overrides: dict[date, bool] = field(default_factory=dict)

    @classmethod
    def from_dates(cls, holidays: Iterable[date], overrides: dict[date, bool] | None = None):
        return cls(frozenset(holidays), dict(overrides or {}))

    def is_open(self, d: date) -> bool:
        if d in self.overrides:
            return self.overrides[d]
        return d.weekday() < 5 and d not in self.holidays

    def next_open(self, d: date) -> date:
        """First trading day strictly after d."""
        x = d + timedelta(days=1)
        while not self.is_open(x):
            x += timedelta(days=1)
        return x

    def prev_open(self, d: date) -> date:
        """Last trading day strictly before d."""
        x = d - timedelta(days=1)
        while not self.is_open(x):
            x -= timedelta(days=1)
        return x

    def trading_days(self, start: date, end: date) -> list[date]:
        """Trading days in [start, end]."""
        out, x = [], start
        while x <= end:
            if self.is_open(x):
                out.append(x)
            x += timedelta(days=1)
        return out

    def count_between(self, start: date, end: date) -> int:
        return len(self.trading_days(start, end))
