"""Train and gate the RL meta-allocator.

Walk-forward: train on windows that end before ``split``; evaluate on windows after it against
the static adaptive baseline. The policy is only offered to the agent (get_rl_suggestion) if it
beats the baseline out of sample after costs on mean AND 10th-percentile return, with zero rule
violations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from stockmaster.backtest.compsim import windows
from stockmaster.backtest.engine import FeatureCache, baseline_policy, simulate
from stockmaster.config import Rules, Strategy, Universe
from stockmaster.rl.env import CompetitionEnv


@dataclass
class GateReport:
    passed: bool
    rl_mean: float
    rl_p10: float
    base_mean: float
    base_p10: float
    rl_violations: int
    n_eval: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def split_windows(cache: FeatureCache, split: date, n_train: int, n_eval: int, seed: int = 7):
    days = [d for d in cache.panel.close.index if cache.panel.close.loc[d].notna().sum() > 50]
    train_days = [d for d in days if d < split]
    eval_days = [d for d in days if d >= split]
    # Evaluation windows need warm-up history from before the split.
    warm = days[:days.index(eval_days[0])] if eval_days else []
    tr = windows(train_days, n_train, seed=seed)
    ev = windows(warm[-130:] + eval_days, n_eval, warmup=min(130, len(warm)), seed=seed + 1) if eval_days else []
    return tr, ev


def train(cache: FeatureCache, train_windows, *, universe: Universe, rules: Rules,
          strategy: Strategy, timesteps: int = 50_000, n_envs: int = 8, seed: int = 0,
          out_dir: Path | None = None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make(i):
        return lambda: CompetitionEnv(cache, train_windows, universe=universe, rules=rules,
                                      strategy=strategy, seed=seed + i)

    env = DummyVecEnv([make(i) for i in range(n_envs)])
    model = PPO("MlpPolicy", env, n_steps=48, batch_size=96, learning_rate=3e-4, gamma=0.99,
                ent_coef=0.01, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "ppo_meta_allocator")
    return model


def evaluate(model, cache: FeatureCache, eval_windows, *, universe: Universe, rules: Rules,
             strategy: Strategy) -> GateReport:
    rl_rets, base_rets, violations = [], [], 0
    base = baseline_policy(strategy, rules, universe)
    for i, (prev, days) in enumerate(eval_windows):
        env = CompetitionEnv(cache, eval_windows, universe=universe, rules=rules, strategy=strategy)
        obs, _ = env.reset(options={"window": i})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        res = env.sim.result()
        violations += len(res.hard_violations) + len(res.close_breaches)
        rl_rets.append(res.metrics(rules.initial_cash)["return"])
        b = simulate(base, days, prev, cache, universe=universe, rules=rules, strategy=strategy)
        base_rets.append(b.metrics(rules.initial_cash)["return"])
    rl_m, b_m = float(np.mean(rl_rets)), float(np.mean(base_rets))
    rl_p, b_p = float(np.percentile(rl_rets, 10)), float(np.percentile(base_rets, 10))
    return GateReport(passed=rl_m > b_m and rl_p >= b_p and violations == 0, rl_mean=rl_m, rl_p10=rl_p,
                      base_mean=b_m, base_p10=b_p, rl_violations=violations, n_eval=len(eval_windows))


def save_gate(report: GateReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
