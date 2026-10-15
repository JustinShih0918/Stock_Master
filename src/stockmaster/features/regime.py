"""Market regime summary: trend, volatility, breadth and aggregate institutional flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stockmaster.features.panel import Panel


@dataclass(frozen=True)
class Regime:
    as_of: str
    taiex: float | None
    taiex_ret_1d: float | None
    taiex_ret_5d: float | None
    taiex_ret_20d: float | None
    taiex_vs_ma20: float | None
    taiex_vs_ma60: float | None
    taiex_vol_20d: float | None
    universe_up_ratio_1d: float | None
    universe_above_ma20: float | None
    foreign_net_5d_bn: float | None  # NT$ billion, universe aggregate
    itrust_net_5d_bn: float | None
    label: str  # risk_on | neutral | risk_off
    suggested_risk_level: str  # aggressive | normal | defensive

    def to_dict(self) -> dict:
        return asdict(self)


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def market_regime(p: Panel) -> Regime:
    idx = p.index_close.dropna()
    ret = idx.pct_change()
    last = idx.iloc[-1] if len(idx) else None

    def ret_n(n):
        return idx.iloc[-1] / idx.iloc[-1 - n] - 1 if len(idx) > n else None

    vs20 = last / idx.tail(20).mean() - 1 if len(idx) >= 20 else None
    vs60 = last / idx.tail(60).mean() - 1 if len(idx) >= 60 else None
    vol20 = ret.tail(20).std() * np.sqrt(250) if len(ret) >= 20 else None

    r = p.adj_returns()
    up = (r.iloc[-1] > 0).mean() if len(r) else None
    above = (p.close.iloc[-1] > p.close.tail(20).mean()).mean() if len(p.close) >= 20 else None
    px = p.close.reindex(p.foreign_net.index).ffill()
    fnet = (p.foreign_net * px).tail(5).sum().sum() / 1e9 if len(p.foreign_net) else None
    inet = (p.itrust_net * px).tail(5).sum().sum() / 1e9 if len(p.itrust_net) else None

    score = 0
    score += 1 if vs20 is not None and vs20 > 0 else -1 if vs20 is not None else 0
    score += 1 if vs60 is not None and vs60 > 0 else -1 if vs60 is not None else 0
    score += 1 if above is not None and above > 0.55 else -1 if above is not None and above < 0.45 else 0
    score += 1 if fnet is not None and fnet > 0 else -1 if fnet is not None and fnet < 0 else 0
    if vol20 is not None and vol20 > 0.30:
        score -= 1
    label = "risk_on" if score >= 2 else "risk_off" if score <= -2 else "neutral"
    level = {"risk_on": "aggressive", "neutral": "normal", "risk_off": "defensive"}[label]
    return Regime(as_of=str(p.as_of), taiex=_f(last), taiex_ret_1d=_f(ret_n(1)),
                  taiex_ret_5d=_f(ret_n(5)), taiex_ret_20d=_f(ret_n(20)), taiex_vs_ma20=_f(vs20),
                  taiex_vs_ma60=_f(vs60), taiex_vol_20d=_f(vol20), universe_up_ratio_1d=_f(up),
                  universe_above_ma20=_f(above), foreign_net_5d_bn=_f(fnet),
                  itrust_net_5d_bn=_f(inet), label=label, suggested_risk_level=level)


def sector_summary(features: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    df = features.join(sectors.rename("sector"))
    return df.groupby("sector").agg(n=("close", "size"), ret_5d=("ret_5d", "median"),
                                    ret_20d=("ret_20d", "median"), itrust_5d=("itrust_5d", "sum"),
                                    foreign_5d=("foreign_5d", "sum")).sort_values("ret_20d", ascending=False)
