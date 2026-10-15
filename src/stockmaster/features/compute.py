"""Per-stock feature table at a decision date, computed only from data available by then.

Every feature is a raw, interpretable quantity; ``alpha.composite`` turns them into z-scores.
NaN means "not enough data" (e.g. recently listed stocks) and is handled downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockmaster.features.panel import Panel

TRADING_DAYS_YEAR = 250


def _window_sum(df: pd.DataFrame, n: int) -> pd.Series:
    return df.tail(n).sum(min_count=max(1, n // 2))


def _cum_return(r: pd.DataFrame, start: int, end: int = 0) -> pd.Series:
    """Compounded return over rows [-start, -end) (end=0 means through the last row)."""
    window = r.iloc[-start:len(r) - end] if end else r.iloc[-start:]
    valid = window.notna().sum() >= max(1, int(0.8 * (start - end)))
    return ((1 + window.fillna(0)).prod() - 1).where(valid)


def technical(p: Panel) -> pd.DataFrame:
    r = p.adj_returns()
    close = p.close
    out = pd.DataFrame(index=p.codes)
    out["close"] = close.ffill().iloc[-1]
    out["ret_1d"] = r.iloc[-1]
    out["ret_5d"] = _cum_return(r, 5)
    out["ret_20d"] = _cum_return(r, 20)
    out["ret_60d"] = _cum_return(r, 60)
    out["mom_6m_skip1w"] = _cum_return(r, 126, 5)
    out["mom_3m_skip1w"] = _cum_return(r, 63, 5)
    adj_level = (1 + r.fillna(0)).cumprod()
    hi = adj_level.tail(TRADING_DAYS_YEAR).max()
    out["dist_52w_high"] = (adj_level.iloc[-1] / hi - 1).where(r.notna().sum() >= 120)
    out["vol_20d"] = r.tail(20).std() * np.sqrt(TRADING_DAYS_YEAR)
    out["vol_60d"] = r.tail(60).std() * np.sqrt(TRADING_DAYS_YEAR)
    out["adv20_value"] = p.value.tail(20).mean()
    if len(p.index_close) > 60:
        m = p.index_close.pct_change().reindex(r.index)
        rr = r.tail(120)
        mm = m.tail(120)
        cov = rr.apply(lambda s: s.cov(mm))
        out["beta_120d"] = cov / mm.var()
    else:
        out["beta_120d"] = np.nan
    out["n_obs"] = r.notna().sum()
    return out


def flows(p: Panel) -> pd.DataFrame:
    """Institutional net buying as a share of average daily traded value."""
    out = pd.DataFrame(index=p.codes)
    px = p.close.reindex(p.itrust_net.index).ffill()
    adv = p.value.tail(20).mean()
    for name, df in (("itrust", p.itrust_net), ("foreign", p.foreign_net), ("dealer", p.dealer_net)):
        val = df * px
        for n in (1, 5, 20):
            out[f"{name}_{n}d"] = _window_sum(val, n) / adv
    # 投信 buying streak: consecutive most-recent days with net buying
    pos = (p.itrust_net > 0).astype(int)
    streak = pos.iloc[::-1].cumprod().sum()
    out["itrust_streak"] = streak.reindex(p.codes)
    return out


def margin(p: Panel) -> pd.DataFrame:
    out = pd.DataFrame(index=p.codes)
    mb = p.margin_balance
    adv_lots = p.volume.tail(20).mean() / 1000
    if len(mb) >= 6:
        out["margin_chg_5d"] = (mb.iloc[-1] - mb.iloc[-6]) / adv_lots
    else:
        out["margin_chg_5d"] = np.nan
    return out


def valuation(p: Panel) -> pd.DataFrame:
    out = pd.DataFrame(index=p.codes)
    for col in ("pe", "pb", "div_yield"):
        df = getattr(p, col)
        out[col] = df.ffill().iloc[-1] if len(df) else np.nan
    out["earnings_yield"] = 1 / out["pe"].where(out["pe"] > 0)
    return out


def revenue(p: Panel, decision_time: pd.Timestamp) -> pd.DataFrame:
    """Monthly revenue momentum using only months available at ``decision_time`` (naive Taipei)."""
    out = pd.DataFrame(index=p.codes, columns=["rev_yoy_last", "rev_yoy_3m", "rev_accel",
                                               "rev_new_high_12m", "rev_last_ym"], dtype=object)
    if p.revenue.empty:
        return out.astype({c: float for c in out.columns if c != "rev_last_ym"})
    rev = p.revenue[p.revenue["available_at"] <= decision_time].sort_values("ym")
    rows = {}
    for code, g in rev.groupby("code"):
        g = g.tail(15)
        yoy = g["yoy_pct"].astype(float)
        last = yoy.iloc[-1] if len(yoy) else np.nan
        prev3 = yoy.iloc[-4:-1].mean() if len(yoy) >= 4 else np.nan
        new_high = float(g["revenue"].iloc[-1] >= g["revenue"].tail(12).max()) if len(g) >= 12 else np.nan
        rows[code] = {"rev_yoy_last": last, "rev_yoy_3m": yoy.tail(3).mean(),
                      "rev_accel": last - prev3, "rev_new_high_12m": new_high,
                      "rev_last_ym": g["ym"].iloc[-1]}
    df = pd.DataFrame.from_dict(rows, orient="index")
    out.update(df)
    num = ["rev_yoy_last", "rev_yoy_3m", "rev_accel", "rev_new_high_12m"]
    out[num] = out[num].astype(float)
    return out


def compute_features(p: Panel, decision_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per code. ``decision_time`` defaults to 20:00 on ``p.as_of`` (evening Run A)."""
    if decision_time is None:
        decision_time = pd.Timestamp(p.as_of) + pd.Timedelta(hours=20)
    parts = [technical(p), flows(p), margin(p), valuation(p), revenue(p, decision_time)]
    df = pd.concat(parts, axis=1)
    df.index.name = "code"
    return df
