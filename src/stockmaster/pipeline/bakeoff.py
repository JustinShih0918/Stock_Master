"""Model bake-off: same frozen inputs, several models, repeated runs. Ranks models on
reliability (valid decisions without fallback, repair rounds, invented codes, run-to-run
stability, latency, rate-limit errors) rather than on a few days of P&L."""

from __future__ import annotations

import itertools
import statistics
import time
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import Engine

from stockmaster.agent.llm_client import ChatModel, OpenAICompatModel
from stockmaster.alpha.composite import composite
from stockmaster.config import load_rules, load_strategy, load_universe
from stockmaster.data.reference import load_calendar, quotes_for
from stockmaster.domain import PortfolioState
from stockmaster.features.compute import compute_features
from stockmaster.features.panel import load_panel
from stockmaster.features.regime import market_regime
from stockmaster.pipeline.context import build_tool_context
from stockmaster.pipeline.decide import deviation, decide
from stockmaster.pipeline.run import load_etf_holdings


@dataclass
class Trial:
    model: str
    as_of: date
    scenario: str
    repeat: int
    source: str
    rounds: int
    repair_rounds: int
    tool_calls: int
    invented_codes: int
    latency_s: float
    rate_limited: int
    error: str | None
    weights: dict[str, float]
    deviation_from_baseline: float


def _contexts(engine: Engine, as_of_dates: list[date]):
    rules, strategy, universe = load_rules(), load_strategy(), load_universe()
    cal = load_calendar(engine)
    codes = sorted(universe.codes)
    for as_of in as_of_dates:
        panel = load_panel(engine, codes, as_of)
        feats = compute_features(panel)
        scores = composite(feats)
        quotes = quotes_for(engine, codes, as_of, rules)
        etf, etf_asof, stale = load_etf_holdings(engine, as_of)
        trade_date = cal.next_open(as_of)
        common = dict(trade_date=trade_date, as_of=as_of, quotes=quotes, universe=universe,
                      features=feats, scores=scores, regime=market_regime(panel), etf_holdings=etf,
                      etf_asof=etf_asof, rules=rules, strategy=strategy, calendar=cal, stale=stale)
        day1 = build_tool_context(state=PortfolioState.initial(rules.initial_cash), **common)
        yield as_of, "day1", day1
        # Mid-competition: holding yesterday's 'normal' baseline (a realistic rebalance day).
        from stockmaster.portfolio.order_builder import build_orders
        b = build_orders(day1.state, day1.baselines["normal"], quotes, rules, strategy)
        spent = sum(o.lots * rules.lot_size * quotes[o.code].ref_price * (1 + rules.fees.commission_rate)
                    for o in b.orders)
        from decimal import Decimal
        held = PortfolioState(as_of=as_of, cash=Decimal(int(rules.initial_cash - spent)),
                              lots=dict(b.target_lots))
        yield as_of, "rebalance", build_tool_context(state=held, **common)


def run_bakeoff(engine: Engine, model_specs: list[str], as_of_dates: list[date], repeats: int = 3
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = load_strategy()
    trials: list[Trial] = []
    contexts = list(_contexts(engine, as_of_dates))
    for spec, (as_of, scenario, ctx0), rep in itertools.product(model_specs, contexts, range(repeats)):
        provider, _, model_id = spec.partition(":")
        model: ChatModel = OpenAICompatModel(provider, model_id)
        ctx = build_tool_context(trade_date=ctx0.trade_date, as_of=ctx0.as_of, state=ctx0.state,
                                 quotes=ctx0.quotes, universe=ctx0.universe, features=ctx0.features,
                                 scores=ctx0.scores, regime=ctx0.regime, etf_holdings=ctx0.etf_holdings,
                                 etf_asof=ctx0.etf_asof, rules=ctx0.rules, strategy=strategy,
                                 stale=ctx0.stale)
        t0 = time.monotonic()
        out = decide(ctx, [model], run_id=f"bakeoff-{spec}-{as_of}-{scenario}-{rep}",
                     account_id="bakeoff", strategy_version=strategy.strategy_version,
                     input_sha256="bakeoff")
        a = out.attempts[0] if out.attempts else None
        log = a.log if a else []
        invented = 0
        for e in log:
            if e["type"] == "submit" and e.get("decision"):
                invented += sum(p["code"] not in ctx.universe.codes for p in e["decision"]["positions"])
        f = out.final
        trials.append(Trial(
            model=spec, as_of=as_of, scenario=scenario, repeat=rep, source=f.decision_source,
            rounds=a.rounds if a else 0, repair_rounds=a.repair_rounds if a else 0,
            tool_calls=sum(e["type"] == "tool" for e in log), invented_codes=invented,
            latency_s=time.monotonic() - t0,
            rate_limited=sum("RateLimit" in (e.get("error") or "") for e in log),
            error=a.error if a else "no attempt",
            weights=f.post_trade.get("weights", {}),
            deviation_from_baseline=deviation(f.post_trade.get("weights", {}),
                                              ctx.baselines.get(f.risk_level, {}))))
    df = pd.DataFrame([t.__dict__ for t in trials])
    return df, score_trials(df)


def _stability(group: pd.DataFrame) -> float:
    """Mean pairwise L1 distance between repeated runs' weights (lower = more stable)."""
    dists = []
    for _, g in group.groupby(["as_of", "scenario"]):
        ws = list(g["weights"])
        dists += [deviation(a, b) for a, b in itertools.combinations(ws, 2)]
    return statistics.fmean(dists) if dists else float("nan")


def score_trials(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        rows.append({
            "model": model, "trials": len(g),
            "llm_decision_rate": float(g["source"].isin(["llm", "llm_repair"]).mean()),
            "first_try_rate": float((g["source"] == "llm").mean()),
            "fallback_rate": float(g["source"].isin(["fallback", "hold"]).mean()),
            "mean_repair_rounds": float(g["repair_rounds"].mean()),
            "invented_codes": int(g["invented_codes"].sum()),
            "mean_tool_calls": float(g["tool_calls"].mean()),
            "p50_latency_s": float(g["latency_s"].median()),
            "p95_latency_s": float(g["latency_s"].quantile(0.95)),
            "rate_limited": int(g["rate_limited"].sum()),
            "stability_l1": _stability(g),
            "mean_deviation": float(g["deviation_from_baseline"].mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["reliability_score"] = (out["llm_decision_rate"] * 50 + out["first_try_rate"] * 20
                                - out["invented_codes"] * 5 - out["rate_limited"]
                                - out["stability_l1"].fillna(1) * 10
                                - (out["p95_latency_s"] > 240) * 10)
    return out.sort_values("reliability_score", ascending=False)
