"""Gated LightGBM ranker: learns a cross-sectional score from the same features.

Walk-forward only: a model trained on dates < D scores date D. It replaces the hand-weighted
composite only if its out-of-sample rank IC beats the composite's on the same dates (and the
competition simulator agrees). Targets are forward returns measured the competition way
(enter at next day's VWAP, mark at the close ``horizon`` days later), cross-sectionally ranked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stockmaster.backtest.engine import FeatureCache
from stockmaster.backtest.signals import forward_returns

FEATURES = ["itrust_5d", "itrust_20d", "foreign_5d", "foreign_20d", "dealer_5d", "itrust_streak",
            "rev_accel", "rev_yoy_3m", "rev_yoy_last", "rev_new_high_12m", "mom_6m_skip1w",
            "mom_3m_skip1w", "dist_52w_high", "ret_5d", "ret_20d", "ret_60d", "vol_20d", "vol_60d",
            "beta_120d", "margin_chg_5d", "earnings_yield", "pb", "div_yield"]


@dataclass
class RankerEval:
    dates: list[date]
    ranker_ic: pd.Series
    composite_ic: pd.Series

    def summary(self) -> dict:
        def stats(s):
            s = s.dropna()
            return {"mean_ic": float(s.mean()), "t": float(s.mean() / (s.std() / np.sqrt(len(s)))) if len(s) > 1 else np.nan,
                    "n": int(len(s))}
        return {"ranker": stats(self.ranker_ic), "composite": stats(self.composite_ic),
                "passes_gate": bool(self.ranker_ic.mean() > self.composite_ic.mean())}


def dataset(cache: FeatureCache, dates: list[date], horizon: int) -> pd.DataFrame:
    frames = []
    for d in dates:
        feats, scores, _ = cache.get(d)
        fwd = forward_returns(cache, d, horizon)
        if fwd.empty:
            continue
        x = feats.reindex(columns=FEATURES).astype(float)
        # Cross-sectional rank-normalisation keeps the model regime-agnostic.
        x = x.rank(pct=True) - 0.5
        frame = x.assign(date=d, target=fwd.rank(pct=True) - 0.5, composite=scores["score"])
        frames.append(frame.dropna(subset=["target"]))
    return pd.concat(frames) if frames else pd.DataFrame()


def walk_forward(cache: FeatureCache, dates: list[date], horizon: int = 10, min_train_dates: int = 60,
                 retrain_every: int = 20, embargo: int | None = None) -> RankerEval:
    import lightgbm as lgb

    embargo = horizon if embargo is None else embargo  # avoid overlapping target windows
    data = dataset(cache, dates, horizon)
    all_dates = sorted(data["date"].unique())
    r_ic, c_ic = {}, {}
    model = None
    for i, d in enumerate(all_dates):
        if i < min_train_dates + embargo:
            continue
        if model is None or (i - min_train_dates) % retrain_every == 0:
            train = data[data["date"].isin(all_dates[:i - embargo])]
            model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15,
                                      min_child_samples=50, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
            model.fit(train[FEATURES].fillna(0.0), train["target"])
        test = data[data["date"] == d]
        pred = pd.Series(model.predict(test[FEATURES].fillna(0.0)), index=test.index)
        r_ic[d] = pred.rank().corr(test["target"].rank())
        c_ic[d] = test["composite"].rank().corr(test["target"].rank())
    return RankerEval(dates=list(r_ic), ranker_ic=pd.Series(r_ic), composite_ic=pd.Series(c_ic))
