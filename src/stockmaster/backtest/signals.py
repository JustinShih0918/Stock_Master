"""Signal study: rank information coefficient (Spearman) of each feature against forward returns
measured the competition way (enter at next day's VWAP, mark at a later close)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stockmaster.backtest.engine import FeatureCache

HORIZONS = (1, 5, 10, 20)


def forward_returns(cache: FeatureCache, as_of: date, horizon: int) -> pd.Series:
    p = cache.panel
    idx = list(p.close.index)
    i = idx.index(as_of)
    if i + horizon >= len(idx):
        return pd.Series(dtype=float)
    entry = p.vwap.iloc[i + 1].fillna(p.close.iloc[i + 1])
    exit_ = p.close.iloc[i + horizon]
    # Dividend-gap-free path: chain reference-price returns between entry and exit days.
    r = p.close.iloc[i + 1:i + horizon + 1] / p.ref_price.iloc[i + 1:i + horizon + 1]
    chained = r.iloc[1:].prod() if horizon > 1 else pd.Series(1.0, index=p.close.columns)
    first_leg = p.close.iloc[i + 1] / entry
    out = first_leg * chained - 1
    return out.where(exit_.notna())


def ic_table(cache: FeatureCache, dates: list[date], features: list[str] | None = None,
             horizons=HORIZONS) -> pd.DataFrame:
    rows = []
    for d in dates:
        feats, scores, _ = cache.get(d)
        cols = features or [c for c in feats.columns if feats[c].dtype.kind in "fi"] + ["score"]
        table = feats.join(scores[["score"]])
        for h in horizons:
            fwd = forward_returns(cache, d, h)
            if fwd.empty:
                continue
            for c in cols:
                x = table[c].astype(float)
                ok = x.notna() & fwd.notna()
                if ok.sum() < 30:
                    continue
                rows.append({"date": d, "feature": c, "h": h,
                             "ic": x[ok].rank().corr(fwd[ok].rank())})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["feature", "h"])["ic"]
    out = pd.DataFrame({"mean_ic": g.mean(), "ic_std": g.std(), "n": g.size()})
    out["t_stat"] = out["mean_ic"] / (out["ic_std"] / np.sqrt(out["n"]))
    return out.reset_index().pivot(index="feature", columns="h", values=["mean_ic", "t_stat"])
