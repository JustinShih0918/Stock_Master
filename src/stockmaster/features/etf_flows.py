"""Active-ETF flow features from daily holdings snapshots: how many reference active ETFs added to
or cut each stock (share counts) between the latest snapshot and one ~N trading days earlier.

History starts when snapshot archiving started (2026-10-02), so these features are shown to the
agent but carry no weight in the composite until enough history exists to validate them."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import Engine, select

from stockmaster.data import schema


def etf_flow_features(engine: Engine, as_of: date, lookback_days: int = 7) -> pd.DataFrame:
    h, e = schema.etf_holding, schema.etf
    with engine.connect() as conn:
        df = pd.read_sql(select(h.c.etf_code, h.c.as_of, h.c.code, h.c.shares, h.c.weight)
                         .where(h.c.as_of <= as_of)
                         .where(h.c.as_of >= as_of - timedelta(days=lookback_days + 10))
                         .where(h.c.etf_code.in_(select(e.c.code).where(e.c.in_reference_list.is_(True)))),
                         conn)
    cols = ["etf_adds", "etf_cuts", "etf_new", "etf_holders", "etf_weight_sum"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out: dict[str, dict[str, float]] = {}
    for etf, g in df.groupby("etf_code"):
        dates = sorted(g["as_of"].unique())
        last = dates[-1]
        base_candidates = [d for d in dates if d <= last - timedelta(days=lookback_days // 2)]
        cur = g[g.as_of == last].set_index("code")
        for code, r in cur.iterrows():
            o = out.setdefault(code, dict.fromkeys(cols, 0.0))
            o["etf_holders"] += 1
            o["etf_weight_sum"] += float(r["weight"])
        if not base_candidates:
            continue
        prev = g[g.as_of == base_candidates[-1]].set_index("code")
        for code in set(cur.index) | set(prev.index):
            o = out.setdefault(code, dict.fromkeys(cols, 0.0))
            now_sh = cur["shares"].get(code) if code in cur.index else 0
            was_sh = prev["shares"].get(code) if code in prev.index else 0
            now_sh, was_sh = float(now_sh or 0), float(was_sh or 0)
            if was_sh == 0 and now_sh > 0:
                o["etf_new"] += 1
            if now_sh > was_sh * 1.02:
                o["etf_adds"] += 1
            elif now_sh < was_sh * 0.98:
                o["etf_cuts"] += 1
    return pd.DataFrame.from_dict(out, orient="index")[cols]
