"""The retail account's tradeable universe, rebuilt daily from every listed security in the
database: stocks and (optionally) ETFs on TWSE and TPEx that pass price, liquidity and history
filters. Leveraged/inverse products are excluded by name."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import Engine, func, select

from stockmaster.data import schema
from stockmaster.retail.config import RetailRules


@dataclass(frozen=True)
class Universe:
    as_of: date
    info: pd.DataFrame  # index code; name, market, industry, close, adv20, n_days

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(self.info.index)

    def name(self, code: str) -> str:
        return str(self.info["name"].get(code, code)) if code in self.info.index else code


def _is_etf(code: str, industry: str | None) -> bool:
    return code.startswith("00") or industry == "ETF"


def eligible_universe(engine: Engine, as_of: date, rules: RetailRules) -> Universe:
    u = rules.universe
    p, i = schema.price_daily, schema.instrument
    with engine.connect() as conn:
        dates = [r[0] for r in conn.execute(
            select(p.c.date).where(p.c.date <= as_of).group_by(p.c.date)
            .having(func.count() >= 100).order_by(p.c.date.desc()).limit(20))]
        if not dates:
            return Universe(as_of, pd.DataFrame(columns=["name", "market", "industry", "close", "adv20", "n_days"]))
        liq = pd.read_sql(select(p.c.code, func.avg(p.c.value_ntd).label("adv20"))
                          .where(p.c.date.in_(dates)).group_by(p.c.code), conn).set_index("code")
        hist = pd.read_sql(select(p.c.code, func.count().label("n_days"))
                           .where(p.c.date <= as_of).where(p.c.close.is_not(None)).group_by(p.c.code),
                           conn).set_index("code")
        last = pd.read_sql(select(p.c.code, p.c.close, p.c.halted).where(p.c.date == as_of), conn).set_index("code")
        inst = pd.read_sql(select(i.c.code, i.c.name, i.c.market, i.c.industry)
                           .where(i.c.market.in_(u.markets)), conn).set_index("code")
    df = inst.join(last, how="inner").join(liq, how="left").join(hist, how="left")
    keep = (df["close"].notna() & ~df["halted"].fillna(False).astype(bool)
            & (df["close"] >= u.min_price)
            & (df["adv20"].fillna(0) >= u.min_adv20_ntd)
            & (df["n_days"].fillna(0) >= u.min_history_days))
    if not u.include_etfs:
        keep &= ~pd.Series([_is_etf(c, ind) for c, ind in zip(df.index, df["industry"])], index=df.index)
    if u.exclude_name_keywords:
        bad = df["name"].fillna("").map(lambda n: any(k in n for k in u.exclude_name_keywords))
        keep &= ~bad
    out = df[keep][["name", "market", "industry", "close", "adv20", "n_days"]].sort_index()
    return Universe(as_of, out)
