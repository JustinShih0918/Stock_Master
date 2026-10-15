"""RL environment smoke test on a SYNTHETIC panel."""

import numpy as np
import pytest

from stockmaster.backtest.compsim import windows
from stockmaster.backtest.engine import FeatureCache
from stockmaster.rl.env import N_OBS, CompetitionEnv, risk_level, sleeve_weights

from .test_features import synthetic_panel

gym = pytest.importorskip("gymnasium")


def test_action_mapping():
    w = sleeve_weights(np.array([1, 0, 0, 0, 0.9]))
    assert w.sum() == pytest.approx(1) and w.argmax() == 0
    assert risk_level(np.array([0, 0, 0, 0, 0.9])) == "aggressive"
    assert risk_level(np.array([0, 0, 0, 0, -0.9])) == "defensive"


def test_episode_runs_without_violations(universe, rules, strategy):
    cache = FeatureCache(synthetic_panel(sorted(universe.codes), n_days=180))
    days = list(cache.panel.close.index)
    env = CompetitionEnv(cache, windows(days, 2), universe=universe, rules=rules, strategy=strategy)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (N_OBS,) and np.isfinite(obs).all()
    steps, done = 0, False
    rng = np.random.default_rng(0)
    while not done:
        obs, reward, done, _, info = env.step(rng.uniform(-1, 1, 5).astype(np.float32))
        assert np.isfinite(reward) and info["applied"]
        steps += 1
    assert steps == 24
    res = env.sim.result()
    assert not res.hard_violations and not res.close_breaches
