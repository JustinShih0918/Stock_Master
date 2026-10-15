"""Read-only market tools the agent can call, plus the dry-run checker.

Outputs are compact pipe-separated tables (weights in %, z-scores to 1 decimal) carrying
``as_of`` and stale flags so the model cites concrete, dated numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from stockmaster.agent.schemas import decision_json_schema
from stockmaster.config import Rules, Strategy, Universe
from stockmaster.domain import PortfolioState, Quote
from stockmaster.features.regime import Regime
from stockmaster.portfolio.order_builder import build_orders
from stockmaster.risk.active_share import active_share_report
from stockmaster.risk.validator import ValidationContext, validate_orders


@dataclass
class ToolContext:
    trade_date: date
    as_of: date
    day_index: int | None  # 1-based competition day, None outside the round
    days_remaining: int | None
    state: PortfolioState
    quotes: Mapping[str, Quote]
    universe: Universe
    features: pd.DataFrame
    scores: pd.DataFrame
    regime: Regime
    baselines: dict[str, dict[str, float]]
    etf_holdings: Mapping[str, Mapping[str, float]]
    etf_asof: Mapping[str, date]
    rules: Rules
    strategy: Strategy
    events: list[dict] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    reconciliation: str = "n/a"
    leaderboard: dict | None = None
    sectors: Mapping[str, str] = field(default_factory=dict)
    overnight: dict = field(default_factory=dict)
    checks_used: int = 0

    def name(self, code: str) -> str:
        try:
            return self.universe.name(code)
        except KeyError:
            return "?"

    def nav_ref(self) -> float:
        return float(self.state.cash) + sum(
            n * self.rules.lot_size * self.quotes[c].ref_price for c, n in self.state.lots.items())

    def current_weights(self) -> dict[str, float]:
        nav = self.nav_ref()
        return {c: n * self.rules.lot_size * self.quotes[c].ref_price / nav
                for c, n in self.state.lots.items() if n > 0}


def _fmt(x, kind: str = "f1") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if kind == "pct":
        return f"{x * 100:.1f}%"
    if kind == "pct2":
        return f"{x * 100:.2f}%"
    if kind == "f1":
        return f"{x:.1f}"
    if kind == "f2":
        return f"{x:.2f}"
    if kind == "int":
        return f"{int(x):,}"
    return str(x)


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join([" | ".join(header)] + [" | ".join(r) for r in rows])


def _header(ctx: ToolContext) -> str:
    stale = f" stale={ctx.stale}" if ctx.stale else ""
    return f"as_of={ctx.as_of} trade_date={ctx.trade_date}{stale}"


# --- tools --------------------------------------------------------------------------------------

def get_portfolio_state(ctx: ToolContext) -> str:
    nav = ctx.nav_ref()
    w = ctx.current_weights()
    lines = [_header(ctx),
             f"NAV(ref)={nav:,.0f} cash={float(ctx.state.cash):,.0f} ({float(ctx.state.cash) / nav:.2%}) "
             f"holdings={len(w)} warnings_used={ctx.state.warnings}/{ctx.rules.warnings_to_disqualify - 1} "
             f"reconciliation={ctx.reconciliation} day={ctx.day_index} days_remaining={ctx.days_remaining}"]
    if not w:
        lines.append("No holdings (all cash). Day 1 must end with 20-30 holdings and cash < 25%.")
        return "\n".join(lines)
    rows = []
    for c, x in sorted(w.items(), key=lambda kv: -kv[1]):
        cap = ctx.rules.weight_caps.cap(c)
        rows.append([c, ctx.name(c), str(ctx.state.lots[c]), _fmt(x, "pct2"),
                     _fmt(ctx.strategy.buffers.add_cap(c) - x, "pct2"),
                     str(ctx.state.overweight_days.get(c, 0)) if x > cap else "0",
                     _fmt(ctx.features.loc[c, "ret_5d"] if c in ctx.features.index else None, "pct"),
                     _fmt(ctx.scores.loc[c, "score"] if c in ctx.scores.index else None, "f2")])
    lines.append(_table(["code", "name", "lots", "weight", "add_cap_headroom", "overweight_days",
                         "ret_5d", "score"], rows))
    return "\n".join(lines)


def get_market_context(ctx: ToolContext) -> str:
    r = ctx.regime
    lines = [_header(ctx),
             f"TAIEX={_fmt(r.taiex, 'f1')} ret1d={_fmt(r.taiex_ret_1d, 'pct')} ret5d={_fmt(r.taiex_ret_5d, 'pct')} "
             f"ret20d={_fmt(r.taiex_ret_20d, 'pct')} vsMA20={_fmt(r.taiex_vs_ma20, 'pct')} "
             f"vsMA60={_fmt(r.taiex_vs_ma60, 'pct')} vol20d={_fmt(r.taiex_vol_20d, 'pct')}",
             f"universe: up_ratio_1d={_fmt(r.universe_up_ratio_1d, 'pct')} above_MA20={_fmt(r.universe_above_ma20, 'pct')} "
             f"foreign_net_5d=NT${_fmt(r.foreign_net_5d_bn, 'f1')}bn itrust_net_5d=NT${_fmt(r.itrust_net_5d_bn, 'f1')}bn",
             f"rule_based_regime={r.label} suggested_risk_level={r.suggested_risk_level}",
             f"competition: day={ctx.day_index} days_remaining={ctx.days_remaining}"]
    if ctx.sectors:
        sec = _sector_lines(ctx)
        if sec:
            lines.append("sectors (150-stock universe, median returns; flows = sum of %ADV):")
            lines += sec
    if ctx.overnight:
        parts = [f"{k} {v['date']} ret={v['ret_1d']:+.2%}" for k, v in ctx.overnight.items()
                 if isinstance(v, dict)]
        if "tsm_adr_premium" in ctx.overnight:
            parts.append(f"TSM_ADR_premium_vs_2330={ctx.overnight['tsm_adr_premium']:+.1%}")
        lines.append("overnight US (unofficial source): " + "; ".join(parts))
    if ctx.leaderboard:
        lines.append(f"leaderboard: {json.dumps(ctx.leaderboard, ensure_ascii=False)}")
    if ctx.events:
        lines.append("upcoming events:")
        lines += [f"- {e['date']}: {e['event']}" for e in ctx.events[:12]]
    return "\n".join(lines)


SORTABLE = {"score", "g_flow", "g_revenue", "g_momentum", "g_reversal", "g_value_quality",
            "itrust_5d", "itrust_20d", "foreign_5d", "rev_accel", "rev_yoy_last", "mom_6m_skip1w",
            "ret_5d", "ret_20d", "dist_52w_high", "vol_60d", "beta_120d"}


def screen_stocks(ctx: ToolContext, sort_by: str = "score", top_n: int = 20,
                  exclude_held: bool = False, ascending: bool = False) -> str:
    if sort_by not in SORTABLE:
        return f"error: sort_by must be one of {sorted(SORTABLE)}"
    top_n = max(1, min(int(top_n), 30))
    df = ctx.scores.join(ctx.features, how="left")
    df = df[[c in ctx.quotes and not ctx.quotes[c].halted for c in df.index]]
    if exclude_held:
        df = df[~df.index.isin(ctx.state.held_codes())]
    df = df.sort_values(sort_by, ascending=ascending).head(top_n)
    etf_count = _etf_top10_counts(ctx)
    rows = []
    for c, r in df.iterrows():
        rows.append([c, ctx.name(c), _fmt(r["score"], "f2"), _fmt(r.get("g_flow"), "f1"),
                     _fmt(r.get("g_revenue"), "f1"), _fmt(r.get("g_momentum"), "f1"),
                     _fmt(r.get("g_reversal"), "f1"), _fmt(r.get("ret_5d"), "pct"),
                     _fmt(r.get("ret_20d"), "pct"), _fmt(r.get("rev_yoy_last"), "f1"),
                     _fmt(r.get("itrust_5d"), "pct"), _fmt(r.get("vol_60d"), "pct"),
                     str(etf_count.get(c, 0)), "held" if c in ctx.state.held_codes() else ""])
    return _header(ctx) + f" sort_by={sort_by}\n" + _table(
        ["code", "name", "score", "z_flow", "z_rev", "z_mom", "z_rev(reversal)", "ret5d", "ret20d",
         "rev_yoy%", "itrust5d/ADV", "vol60d", "active_etf_top10_count", "status"], rows)


def get_stock_profiles(ctx: ToolContext, codes: list[str]) -> str:
    codes = [str(c) for c in codes][:8]
    out = [_header(ctx)]
    w = ctx.current_weights()
    etf_count = _etf_top10_counts(ctx)
    for c in codes:
        if c not in ctx.features.index:
            out.append(f"{c}: not in the 150-stock universe or no data")
            continue
        f = ctx.features.loc[c]
        s = ctx.scores.loc[c] if c in ctx.scores.index else {}
        q = ctx.quotes.get(c)
        out.append(
            f"[{c} {ctx.name(c)}] close={_fmt(f['close'], 'f2')} limit_up={q.limit_up if q else '-'} "
            f"halted={q.halted if q else '-'} weight_now={_fmt(w.get(c, 0.0), 'pct2')}\n"
            f"  returns: 1d={_fmt(f['ret_1d'], 'pct')} 5d={_fmt(f['ret_5d'], 'pct')} 20d={_fmt(f['ret_20d'], 'pct')} "
            f"60d={_fmt(f['ret_60d'], 'pct')} 6m(skip1w)={_fmt(f['mom_6m_skip1w'], 'pct')} "
            f"from_52w_high={_fmt(f['dist_52w_high'], 'pct')} vol60d={_fmt(f['vol_60d'], 'pct')} beta={_fmt(f['beta_120d'], 'f2')}\n"
            f"  flows(% of ADV): itrust 1d/5d/20d={_fmt(f['itrust_1d'], 'pct')}/{_fmt(f['itrust_5d'], 'pct')}/{_fmt(f['itrust_20d'], 'pct')} "
            f"streak={_fmt(f['itrust_streak'], 'int')} foreign 5d/20d={_fmt(f['foreign_5d'], 'pct')}/{_fmt(f['foreign_20d'], 'pct')} "
            f"dealer5d={_fmt(f['dealer_5d'], 'pct')} margin_chg5d/ADV={_fmt(f['margin_chg_5d'], 'pct')} "
            f"ADV20=NT${_fmt(f['adv20_value'] / 1e8 if pd.notna(f['adv20_value']) else None, 'f1')}億\n"
            f"  revenue: last={f.get('rev_last_ym')} yoy={_fmt(f['rev_yoy_last'], 'f1')}% yoy3m={_fmt(f['rev_yoy_3m'], 'f1')}% "
            f"accel={_fmt(f['rev_accel'], 'f1')}pp new12m_high={_fmt(f['rev_new_high_12m'], 'f1')}\n"
            f"  valuation: PE={_fmt(f['pe'], 'f1')} PB={_fmt(f['pb'], 'f2')} yield={_fmt(f['div_yield'], 'f2')}%\n"
            f"  score={_fmt(s.get('score') if len(s) else None, 'f2')} active_etf_top10_count={etf_count.get(c, 0)} "
            f"sector={ctx.sectors.get(c, '-')}\n"
            f"  active_ETFs(22 funds, ~5d share changes): holders={_fmt(f.get('etf_holders'), 'int')} "
            f"adds={_fmt(f.get('etf_adds'), 'int')} cuts={_fmt(f.get('etf_cuts'), 'int')} new={_fmt(f.get('etf_new'), 'int')} "
            f"avg_weight_in_holders={_fmt(f.get('etf_weight_sum') / f.get('etf_holders') if f.get('etf_holders') else None, 'pct')}")
        ev = [e for e in ctx.events if e.get("code") == c]
        if ev:
            out.append("  events: " + "; ".join(f"{e['date']} {e['event']}" for e in ev))
    return "\n".join(out)


def get_baseline_portfolio(ctx: ToolContext, risk_level: str = "normal") -> str:
    if risk_level not in ctx.baselines:
        return f"error: risk_level must be one of {sorted(ctx.baselines)}"
    w = ctx.baselines[risk_level]
    cur = ctx.current_weights()
    rows = [[c, ctx.name(c), _fmt(x, "pct2"), _fmt(cur.get(c, 0.0), "pct2"),
             _fmt(ctx.scores.loc[c, "score"] if c in ctx.scores.index else None, "f2")]
            for c, x in sorted(w.items(), key=lambda kv: -kv[1])]
    rep = active_share_report(w, ctx.etf_holdings, ctx.rules.active_share.top_n) if ctx.etf_holdings else []
    as_line = (f"min_active_share={rep[0].value:.1%} vs {rep[0].etf}" if rep else "active_share=unverified")
    turnover = sum(abs(w.get(c, 0) - cur.get(c, 0)) for c in set(w) | set(cur)) / 2
    return (_header(ctx) + f" risk_level={risk_level} invested={sum(w.values()):.1%} names={len(w)} "
            f"one_way_turnover_vs_now={turnover:.1%} {as_line}\n"
            + _table(["code", "name", "target", "now", "score"], rows))


def check_portfolio(ctx: ToolContext, positions: list[dict], target_cash_pct: float) -> str:
    ctx.checks_used += 1
    if ctx.checks_used > ctx.strategy.llm.max_check_calls:
        return "error: check_portfolio call budget exhausted; submit your decision now."
    targets = {}
    for p in positions:
        try:
            targets[str(p["code"])] = float(p["weight_pct"]) / 100
        except (KeyError, TypeError, ValueError):
            return f"error: each position needs code and weight_pct, got {p}"
    unknown = [c for c in targets if c not in ctx.quotes]
    if unknown:
        return f"error: unknown or untradable codes {unknown} (must be in the 150-stock universe)"
    res = build_orders(ctx.state, targets, ctx.quotes, ctx.rules, ctx.strategy)
    rep = validate_orders(res.orders, ValidationContext(
        trade_date=ctx.trade_date, state=ctx.state, quotes=ctx.quotes,
        universe=ctx.universe.codes, rules=ctx.rules, strategy=ctx.strategy,
        etf_holdings=ctx.etf_holdings))
    m = rep.metrics
    lines = [_header(ctx), f"result={'PASS' if rep.passed else 'FAIL'} orders={len(res.orders)} "
             f"holdings_after={m.get('n_holdings')} cash_after(ref)={_fmt(m.get('cash_ref_pct'), 'pct2')} "
             f"worst_case_cash={_fmt(m.get('worst_cash_min_pct'), 'pct2')} "
             f"cash_if_largest_sell_halted={_fmt(m.get('halted_sell_cash_min_pct'), 'pct2')} "
             f"worst_case_cash_ratio_max={_fmt(m.get('worst_cash_max_ratio'), 'pct2')} "
             f"turnover={_fmt(m.get('turnover_one_way'), 'pct')} est_cost=NT${_fmt(m.get('est_cost'), 'int')} "
             f"min_active_share={_fmt(m.get('active_share_min'), 'pct')} binding_etf={m.get('active_share_binding_etf')}"]
    feats = ctx.features
    wa = m.get("weights", {})
    if wa:
        beta = sum(wa[c] * (feats.loc[c, "beta_120d"] if c in feats.index and pd.notna(feats.loc[c, "beta_120d"]) else 1.0)
                   for c in wa)
        lines.append(f"portfolio_beta≈{beta:.2f}")
    for r in rep.violations + rep.soft_failures:
        lines.append(f"[{r.severity.upper()}] {r.rule_id}: {r.detail}")
    if res.notes:
        lines.append("builder notes: " + "; ".join(res.notes[:8]))
    return "\n".join(lines)


def _sector_lines(ctx: ToolContext, top: int = 5, bottom: int = 3) -> list[str]:
    f = ctx.features.join(pd.Series(ctx.sectors, name="sector"), how="left")
    f = f[f["sector"].notna()]
    if f.empty:
        return []
    g = f.groupby("sector").agg(n=("close", "size"), ret5=("ret_5d", "median"), ret20=("ret_20d", "median"),
                                itrust5=("itrust_5d", "mean"), foreign5=("foreign_5d", "mean"))
    g = g[g["n"] >= 2].sort_values("ret20", ascending=False)
    rows = list(g.head(top).iterrows()) + (list(g.tail(bottom).iterrows()) if len(g) > top + bottom else [])
    return [f"- {name}(n={int(r.n)}): ret5d={_fmt(r.ret5, 'pct')} ret20d={_fmt(r.ret20, 'pct')} "
            f"itrust5d={_fmt(r.itrust5, 'pct')} foreign5d={_fmt(r.foreign5, 'pct')}" for name, r in rows]


def _etf_top10_counts(ctx: ToolContext) -> dict[str, int]:
    from stockmaster.risk.active_share import top_n
    counts: dict[str, int] = {}
    for h in ctx.etf_holdings.values():
        for c in top_n(h, ctx.rules.active_share.top_n):
            counts[c] = counts.get(c, 0) + 1
    return counts


# --- registry -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str] | None  # None for submit_decision (handled by the harness)


def registry() -> dict[str, Tool]:
    tools = [
        Tool("get_portfolio_state", "目前持股、現金、權重、警告次數、對帳狀態與剩餘交易日。",
             {"type": "object", "properties": {}, "additionalProperties": False}, get_portfolio_state),
        Tool("get_market_context", "大盤趨勢/波動/漲跌家數、法人總買賣超、規則型市場狀態與近期事件。",
             {"type": "object", "properties": {}, "additionalProperties": False}, get_market_context),
        Tool("screen_stocks", "依因子排序篩選150檔標的（含綜合分數與因子z值）。",
             {"type": "object", "properties": {
                 "sort_by": {"type": "string", "enum": sorted(SORTABLE)},
                 "top_n": {"type": "integer", "minimum": 1, "maximum": 30},
                 "exclude_held": {"type": "boolean"},
                 "ascending": {"type": "boolean"}},
              "additionalProperties": False}, screen_stocks),
        Tool("get_stock_profiles", "最多8檔個股的報酬、法人籌碼、融資、月營收、評價與事件明細。",
             {"type": "object", "properties": {
                 "codes": {"type": "array", "items": {"type": "string"}, "maxItems": 8}},
              "required": ["codes"], "additionalProperties": False}, get_stock_profiles),
        Tool("get_baseline_portfolio", "量化基準投組（依風險等級），可作為起點再調整。",
             {"type": "object", "properties": {
                 "risk_level": {"type": "string", "enum": ["defensive", "normal", "aggressive"]}},
              "required": ["risk_level"], "additionalProperties": False}, get_baseline_portfolio),
        Tool("check_portfolio", "試算目標權重：換算整張下單、檢查所有比賽規則（現金、檔數、上限、Active Share）。",
             {"type": "object", "properties": {
                 "positions": {"type": "array", "items": {"type": "object", "properties": {
                     "code": {"type": "string"}, "weight_pct": {"type": "number"}},
                     "required": ["code", "weight_pct"]}},
                 "target_cash_pct": {"type": "number"}},
              "required": ["positions", "target_cash_pct"], "additionalProperties": False}, check_portfolio),
        Tool("submit_decision", "提交最終決策（結束本輪）。positions 必須包含所有要持有的股票；目前持股若不在 positions 中必須列在 exits。",
             decision_json_schema(), None),
    ]
    return {t.name: t for t in tools}


def openai_tool_specs(tools: Mapping[str, Tool]) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}} for t in tools.values()]


def run_tool(ctx: ToolContext, tool: Tool, args: dict) -> str:
    try:
        return tool.fn(ctx, **args)
    except TypeError as e:
        return f"error: bad arguments for {tool.name}: {e}"
