"""Cross-sectional composite score from features.

Weights are explicit priors for 1-4 week horizons in Taiwan large/mid caps; the signal IC/decay
study (backtest.signals) checks them before they are trusted. Positive weight = higher is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# v1 priors. The active weights come from config/strategy.yaml (alpha.weights) via active_weights().
DEFAULT_WEIGHTS: dict[str, float] = {
    "itrust_5d": 0.15,
    "itrust_20d": 0.10,
    "foreign_5d": 0.10,
    "rev_accel": 0.15,
    "rev_yoy_3m": 0.05,
    "rev_new_high_12m": 0.05,
    "mom_6m_skip1w": 0.12,
    "dist_52w_high": 0.08,
    "ret_5d": -0.05,          # short-term reversal
    "margin_chg_5d": -0.05,   # retail leverage spikes are contrarian
    "earnings_yield": 0.05,
    "vol_60d": -0.05,
}

FACTOR_GROUPS = {
    "flow": ["itrust_5d", "itrust_20d", "foreign_5d"],
    "revenue": ["rev_accel", "rev_yoy_3m", "rev_new_high_12m"],
    "momentum": ["mom_6m_skip1w", "dist_52w_high"],
    "reversal": ["ret_5d", "margin_chg_5d"],
    "value_quality": ["earnings_yield", "vol_60d"],
}


def active_weights() -> dict[str, float]:
    from stockmaster.config import load_strategy
    return dict(load_strategy().alpha.weights) or dict(DEFAULT_WEIGHTS)


def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    scale = mad if mad and np.isfinite(mad) and mad > 0 else s.std()
    if not scale or not np.isfinite(scale):
        return pd.Series(0.0, index=s.index)
    return ((s - med) / scale).clip(-clip, clip)


def factor_zscores(features: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or active_weights()
    z = pd.DataFrame(index=features.index)
    for col in weights:
        if col in features:
            z[col] = zscore(features[col]).fillna(0.0)  # missing -> neutral
        else:
            z[col] = 0.0
    return z


def composite(features: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Return z-scores per factor, per group, and the composite ``score`` (sorted, best first)."""
    weights = weights or active_weights()
    z = factor_zscores(features, weights)
    out = pd.DataFrame(index=features.index)
    for g, cols in FACTOR_GROUPS.items():
        ws = {c: weights.get(c, 0.0) for c in cols}
        denom = sum(abs(w) for w in ws.values()) or 1.0
        out[f"g_{g}"] = sum(z[c] * w for c, w in ws.items()) / denom
    total = sum(abs(w) for w in weights.values()) or 1.0
    out["score"] = sum(z[c] * w for c, w in weights.items()) / total
    out = out.join(z.add_prefix("z_"))
    return out.sort_values("score", ascending=False)
