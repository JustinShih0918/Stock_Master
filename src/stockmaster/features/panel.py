"""Wide (date x code) panels of market data loaded from the database for feature computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import Engine, select

from stockmaster.data import schema

TAIEX = "發行量加權股價指數"


@dataclass
class Panel:
    """All frames are indexed by date (ascending) with one column per stock code."""

    as_of: date
    close: pd.DataFrame
    ref_price: pd.DataFrame  # exchange reference price (previous close, ex-dividend adjusted)
    vwap: pd.DataFrame
    value: pd.DataFrame  # traded value NT$
    volume: pd.DataFrame  # shares
    itrust_net: pd.DataFrame  # shares
    foreign_net: pd.DataFrame
    dealer_net: pd.DataFrame
    margin_balance: pd.DataFrame  # lots
    pe: pd.DataFrame
    pb: pd.DataFrame
    div_yield: pd.DataFrame
    index_close: pd.Series  # TAIEX
    revenue: pd.DataFrame = field(default_factory=pd.DataFrame)  # long: code, ym, revenue, yoy_pct, available_at

    @property
    def codes(self) -> list[str]:
        return list(self.close.columns)

    def adj_returns(self) -> pd.DataFrame:
        """Daily price returns vs the exchange reference price (dividend-gap free)."""
        r = self.close / self.ref_price - 1
        fallback = self.close.pct_change(fill_method=None)
        return r.where(self.ref_price.notna(), fallback)


def _wide(df: pd.DataFrame, col: str, codes: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=codes, dtype=float)
    w = df.pivot_table(index="date", columns="code", values=col, aggfunc="last")
    return w.reindex(columns=codes).sort_index().astype(float)


def revenue_available_at(ym: date) -> pd.Timestamp:
    """Point-in-time convention for history: a month's revenue is known on the 10th of the next
    month (the statutory deadline)."""
    nxt = (pd.Timestamp(ym) + pd.offsets.MonthBegin(1))
    return nxt + pd.Timedelta(days=9)


def load_panel(engine: Engine, codes: list[str], as_of: date, lookback_days: int = 420) -> Panel:
    start = as_of - timedelta(days=lookback_days)

    def q(table, cols):
        t = table
        stmt = select(*[t.c[c] for c in cols]).where(t.c.code.in_(codes)) \
            .where(t.c.date >= start).where(t.c.date <= as_of)
        with engine.connect() as conn:
            return pd.read_sql(stmt, conn)

    px = q(schema.price_daily, ["code", "date", "close", "prev_close", "vwap", "value_ntd", "volume_sh"])
    fl = q(schema.inst_flow_daily, ["code", "date", "itrust_net", "foreign_net", "dealer_self_net",
                                    "dealer_hedge_net"])
    if not fl.empty:
        fl["dealer_net"] = fl["dealer_self_net"].fillna(0) + fl["dealer_hedge_net"].fillna(0)
    mg = q(schema.margin_daily, ["code", "date", "margin_balance"])
    va = q(schema.valuation_daily, ["code", "date", "pe", "pb", "div_yield"])

    t = schema.index_daily
    with engine.connect() as conn:
        idx = pd.read_sql(select(t.c.date, t.c.close).where(t.c.index_name == TAIEX)
                          .where(t.c.date >= start).where(t.c.date <= as_of), conn)
        r = schema.monthly_revenue
        rev = pd.read_sql(select(r.c.code, r.c.ym, r.c.revenue, r.c.yoy_pct, r.c.mom_pct,
                                 r.c.first_seen_at).where(r.c.code.in_(codes))
                          .where(r.c.ym >= as_of - timedelta(days=800)), conn)
    if not rev.empty:
        pit = rev["ym"].map(revenue_available_at)
        seen = pd.to_datetime(rev["first_seen_at"], utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
        rev["available_at"] = pd.concat([pit, seen], axis=1).min(axis=1)
    return Panel(
        as_of=as_of,
        close=_wide(px, "close", codes), ref_price=_wide(px, "prev_close", codes),
        vwap=_wide(px, "vwap", codes), value=_wide(px, "value_ntd", codes),
        volume=_wide(px, "volume_sh", codes),
        itrust_net=_wide(fl, "itrust_net", codes), foreign_net=_wide(fl, "foreign_net", codes),
        dealer_net=_wide(fl, "dealer_net", codes),
        margin_balance=_wide(mg, "margin_balance", codes),
        pe=_wide(va, "pe", codes), pb=_wide(va, "pb", codes), div_yield=_wide(va, "div_yield", codes),
        index_close=idx.set_index("date")["close"].sort_index() if not idx.empty else pd.Series(dtype=float),
        revenue=rev,
    )
