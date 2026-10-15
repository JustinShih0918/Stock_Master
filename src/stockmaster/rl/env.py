"""Gymnasium environment: a 24-trading-day competition episode under the exact rules.

The policy does not pick stocks. Each day it chooses a weighting of four factor sleeves
(flow, revenue, momentum, value/quality) and a risk level; the deterministic baseline constructor,
repair, order builder and validator turn that into orders. This keeps the action space small and
the learned behaviour explainable ("RL meta-allocator").
"""

from __future__ import annotations

import math
from datetime import date

import gymnasium as gym
import numpy as np
import pandas as pd

from stockmaster.backtest.engine import DayInputs, FeatureCache, Stepper
from stockmaster.config import Rules, Strategy, Universe
from stockmaster.portfolio.baseline import baseline_targets

SLEEVES = ("g_flow", "g_revenue", "g_momentum", "g_value_quality")
LEVELS = ("defensive", "normal", "aggressive")
N_OBS = 18


def sleeve_weights(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action[:len(SLEEVES)], dtype=float) * 2.0
    e = np.exp(a - a.max())
    return e / e.sum()


def risk_level(action: np.ndarray) -> str:
    r = float(action[len(SLEEVES)])
    return "defensive" if r < -1 / 3 else "aggressive" if r > 1 / 3 else "normal"


def rescore(scores: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    s = sum(scores[g] * wi for g, wi in zip(SLEEVES, w)) + 0.1 * scores["g_reversal"]
    out = scores.copy()
    out["score"] = s
    return out.sort_values("score", ascending=False)


def observation(x: DayInputs, nav_hist: list[float], initial: float) -> np.ndarray:
    r = x.regime
    f = x.features

    def v(val, scale=1.0):
        return 0.0 if val is None or not np.isfinite(val) else float(np.clip(val / scale, -5, 5))

    sleeve_ic = []
    for g in SLEEVES:
        a, b = x.scores[g], f["ret_5d"].reindex(x.scores.index)
        ok = a.notna() & b.notna()
        sleeve_ic.append(float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 20 else 0.0)
    nav = nav_hist[-1] if nav_hist else initial
    peak = max(nav_hist) if nav_hist else initial
    obs = [v(r.taiex_ret_1d, 0.02), v(r.taiex_ret_5d, 0.05), v(r.taiex_ret_20d, 0.10),
           v(r.taiex_vs_ma20, 0.05), v(r.taiex_vs_ma60, 0.10), v(r.taiex_vol_20d, 0.3),
           v(r.universe_up_ratio_1d) - 0.5, v(r.universe_above_ma20) - 0.5,
           v(r.foreign_net_5d_bn, 100), v(r.itrust_net_5d_bn, 20),
           x.day_index / x.days_total, v(nav / initial - 1, 0.1), v(1 - nav / peak, 0.1),
           float(x.state.cash) / max(nav, 1.0)] + [0.0 if math.isnan(c) else c for c in sleeve_ic]
    return np.asarray(obs, dtype=np.float32)


class CompetitionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cache: FeatureCache, starts: list[tuple[date, list[date]]], *,
                 universe: Universe, rules: Rules, strategy: Strategy,
                 etf_holdings: dict | None = None, violation_penalty: float = 1.0, seed: int = 0):
        super().__init__()
        self.cache, self.starts = cache, starts
        self.universe, self.rules, self.strategy = universe, rules, strategy
        self.etf, self.penalty = etf_holdings or {}, violation_penalty
        self.observation_space = gym.spaces.Box(-5, 5, (N_OBS,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (len(SLEEVES) + 1,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.sim: Stepper | None = None
        self.navs: list[float] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(options["window"]) if options and "window" in options else int(self._rng.integers(len(self.starts)))
        prev, days = self.starts[idx]
        self.sim = Stepper(self.cache, days, prev, universe=self.universe, rules=self.rules,
                           strategy=self.strategy, etf_holdings=self.etf)
        self.navs = []
        self._x = self.sim.inputs()
        return observation(self._x, self.navs, self.rules.initial_cash), {"start": str(days[0])}

    def step(self, action):
        x, buf = self._x, self.strategy.buffers
        level = risk_level(action)
        scores = rescore(x.scores, sleeve_weights(action))
        first = not x.state.held_codes()
        cash = (1 - buf.day1_invest_target) if first else (buf.cash_target_min + buf.cash_target_max) / 2
        weights = baseline_targets(scores, state=x.state, quotes=x.quotes, universe=self.universe.codes,
                                   risk_level=level, strategy=self.strategy, rules=self.rules,
                                   invest=1 - cash, beta=x.features.get("beta_120d"))
        n_breaches = len(self.sim.breaches)
        prev_nav = self.navs[-1] if self.navs else float(self.rules.initial_cash)
        row = self.sim.step(x, weights, cash)
        self.navs.append(row["nav"])
        reward = 100 * math.log(row["nav"] / prev_nav)
        reward -= self.penalty * ((not row["applied"]) + (len(self.sim.breaches) - n_breaches))
        terminated = self.sim.done
        if not terminated:
            self._x = self.sim.inputs()
        obs = observation(self._x, self.navs, self.rules.initial_cash)
        return obs, float(reward), terminated, False, {"nav": row["nav"], "level": level,
                                                       "applied": row["applied"]}
