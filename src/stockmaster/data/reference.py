"""Reference data loaders used at decision time: the trading calendar, the latest complete price
date, and per-stock reference quotes (previous close and next-day price limits)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date

from sqlalchemy import Engine, func, select

from stockmaster.calendar import TradingCalendar
from stockmaster.data import schema
from stockmaster.domain import Quote

log = logging.getLogger(__name__)
# A date counts as having prices only when at least this many stocks are stored (guards against
# a half-finished ingest being mistaken for the latest close).
MIN_ROWS_PER_DAY = 100


def load_calendar(engine: Engine) -> TradingCalendar:
    """Weekdays minus TWSE holidays, with manual overrides (typhoon closures, make-up days).
    Schedule rows that announce trading (開始交易日 / 最後交易日) are stored as open days."""
    t, o = schema.trading_calendar, schema.calendar_override
    with engine.connect() as conn:
        holidays = [r.date for r in conn.execute(select(t.c.date).where(t.c.is_open.is_(False)))]
        overrides = {r.date: r.is_open for r in conn.execute(select(o.c.date, o.c.is_open))}
    if not holidays:
        log.warning("trading_calendar is empty: run `sm sync-ref`; using weekdays only")
    return TradingCalendar.from_dates(holidays, overrides)


def latest_price_date(engine: Engine, before: date) -> date | None:
    """Most recent date strictly before ``before`` with a complete set of closes."""
    t = schema.price_daily
    with engine.connect() as conn:
        return conn.execute(
            select(t.c.date).where(t.c.date < before).where(t.c.close.is_not(None))
            .group_by(t.c.date).having(func.count() >= MIN_ROWS_PER_DAY)
            .order_by(t.c.date.desc()).limit(1)).scalar()


def _ex_adjustments(engine: Engine, trade_date: date, codes: list[str]) -> dict[str, tuple[float, float]]:
    """code -> (cash dividend per share, stock dividend ratio) going ex on ``trade_date``."""
    t = schema.corp_action
    out: dict[str, tuple[float, float]] = {}
    with engine.connect() as conn:
        for r in conn.execute(select(t.c.code, t.c.cash_dividend, t.c.stock_dividend_ratio)
                              .where(t.c.ex_date == trade_date).where(t.c.code.in_(codes))):
            cash, ratio = out.get(r.code, (0.0, 0.0))
            out[r.code] = (cash + float(r.cash_dividend or 0), ratio + float(r.stock_dividend_ratio or 0))
    return out


def quotes_for(engine: Engine, codes: Iterable[str], as_of: date, rules, *,
               trade_date: date | None = None) -> dict[str, Quote]:
    """Reference quotes from the close of ``as_of``. A stock with no close that day (suspended or
    untraded) is quoted at its last close and marked halted, so nothing trades it blind.
    On ex-rights days the price limits are computed from the ex-rights reference price."""
    codes = sorted(set(codes))
    if not codes:
        return {}
    t = schema.price_daily
    limit_pct = getattr(rules, "price_limit_pct", 0.10)
    with engine.connect() as conn:
        rows = {r.code: r for r in conn.execute(
            select(t.c.code, t.c.close, t.c.next_ref_price, t.c.halted)
            .where(t.c.date == as_of).where(t.c.code.in_(codes)))}
        missing = [c for c in codes if c not in rows or rows[c].close is None]
        last: dict[str, tuple[date, float]] = {}
        if missing:
            sub = select(t.c.code, func.max(t.c.date).label("d")).where(t.c.code.in_(missing)) \
                .where(t.c.date <= as_of).where(t.c.close.is_not(None)).group_by(t.c.code).subquery()
            for r in conn.execute(select(t.c.code, t.c.date, t.c.close)
                                  .join(sub, (sub.c.code == t.c.code) & (sub.c.d == t.c.date))):
                last[r.code] = (r.date, r.close)
    adj = _ex_adjustments(engine, trade_date, codes) if trade_date else {}
    out: dict[str, Quote] = {}
    for c in codes:
        r = rows.get(c)
        if r is not None and r.close is not None:
            close, d, halted = float(r.close), as_of, bool(r.halted)
            base = float(r.next_ref_price) if r.next_ref_price else close
        elif c in last:
            d, close = last[c]
            close, halted, base = float(close), True, float(close)
        else:
            continue
        if c in adj:
            cash, ratio = adj[c]
            # TPEx already publishes the ex-rights reference price; TWSE's next_ref_price is the close.
            if abs(base - close) < 1e-9:
                base = (close - cash) / (1 + ratio) if (cash or ratio) else close
        out[c] = Quote.from_close(c, close, d, limit_pct, halted=halted, limit_base=base)
    return out


def instrument_names(engine: Engine, codes: Iterable[str] | None = None) -> dict[str, str]:
    t = schema.instrument
    q = select(t.c.code, t.c.name)
    if codes is not None:
        q = q.where(t.c.code.in_(list(codes)))
    with engine.connect() as conn:
        return {r.code: r.name for r in conn.execute(q)}
