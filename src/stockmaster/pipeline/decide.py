"""Daily decision pipeline (DB-free core).

    context -> LLM chain (tools, bounded repair) -> deterministic repair -> fallback -> FinalDecision

The result is always a FinalDecision. ``decision_source`` records which stage produced it, and the
checks record every rule outcome, so a reviewer can see exactly why each order exists.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from stockmaster.agent.harness import PROMPT_VERSION, AgentResult, prompt_sha, run_agent
from stockmaster.agent.llm_client import ChatModel
from stockmaster.agent.schemas import (CheckRow, Decision, ExitItem, FinalDecision, FinalOrder,
                                       FinalTarget, Provenance)
from stockmaster.agent.tools import ToolContext
from stockmaster.domain import Order, Side
from stockmaster.portfolio.order_builder import BuildResult, build_orders
from stockmaster.portfolio.repair import repair_targets
from stockmaster.risk.validator import (Severity, ValidationContext, ValidationReport,
                                        validate_orders)


@dataclass
class Evaluation:
    build: BuildResult
    report: ValidationReport


@dataclass
class DecideOutcome:
    final: FinalDecision
    attempts: list[AgentResult] = field(default_factory=list)
    evaluation: Evaluation | None = None


def cash_target_for(ctx: ToolContext, requested_pct: float | None) -> float:
    buf = ctx.strategy.buffers
    if not ctx.state.held_codes():  # all cash: day-1 build
        return 1 - buf.day1_invest_target
    if requested_pct is None:
        return (buf.cash_target_min + buf.cash_target_max) / 2
    return min(max(requested_pct / 100, buf.cash_target_min), buf.cash_target_max)


def evaluate_weights(ctx: ToolContext, weights: dict[str, float]) -> Evaluation:
    build = build_orders(ctx.state, weights, ctx.quotes, ctx.rules, ctx.strategy)
    report = validate_orders(build.orders, ValidationContext(
        trade_date=ctx.trade_date, state=ctx.state, quotes=ctx.quotes, universe=ctx.universe.codes,
        rules=ctx.rules, strategy=ctx.strategy, etf_holdings=ctx.etf_holdings,
        stale_codes=frozenset()))
    return Evaluation(build, report)


def _scaled(d: Decision) -> dict[str, float]:
    w = d.weights()
    total = sum(w.values()) + d.target_cash_pct / 100
    return {c: x / total for c, x in w.items()} if total > 0 else w


def _feedback(ev: Evaluation) -> str:
    parts = [f"{r.rule_id}: {r.detail}" for r in ev.report.violations]
    if ev.build.notes:
        parts.append("builder: " + "; ".join(ev.build.notes[:6]))
    return " | ".join(parts) if parts else "ok"


def deviation(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a.get(c, 0.0) - b.get(c, 0.0)) for c in sorted(set(a) | set(b)))


def decide(ctx: ToolContext, models: Sequence[ChatModel], *, run_id: str, account_id: str,
           strategy_version: str, input_sha256: str, git_sha: str | None = None,
           extra_notes: str = "") -> DecideOutcome:
    attempts: list[AgentResult] = []
    chosen: tuple[str, Decision | None, dict[str, float], Evaluation, list[str], AgentResult | None] | None = None
    ranking = list(ctx.scores.index)

    def evaluate(d: Decision) -> tuple[bool, str]:
        ev = evaluate_weights(ctx, _scaled(d))
        return ev.report.passed, _feedback(ev)

    for model in models:
        res = run_agent(ctx, model, evaluate, extra_notes=extra_notes)
        attempts.append(res)
        if res.decision is None:
            continue
        w = _scaled(res.decision)
        if res.status == "accepted":
            src = "llm_repair" if res.repair_rounds else "llm"
            chosen = (src, res.decision, w, evaluate_weights(ctx, w), [], res)
            break
        # Rejected after the repair rounds: deterministic projection of the model's intent.
        rep = repair_targets(w, state=ctx.state, quotes=ctx.quotes, universe=ctx.universe.codes,
                             ranking=ranking, etf_holdings=ctx.etf_holdings, rules=ctx.rules,
                             strategy=ctx.strategy,
                             cash_target=cash_target_for(ctx, res.decision.target_cash_pct))
        ev = evaluate_weights(ctx, rep.weights)
        if ev.report.passed:
            chosen = ("det_repair", res.decision, rep.weights, ev, rep.notes, res)
            break

    if chosen is None:
        level = ctx.regime.suggested_risk_level
        base = ctx.baselines[level]
        ev = evaluate_weights(ctx, base)
        notes = [f"LLM 未產生可用決策，採用量化基準投組（{level}）"]
        if ev.report.passed:
            chosen = ("fallback", None, base, ev, notes, None)
        else:
            hold = evaluate_weights(ctx, ctx.current_weights())
            notes.append("基準投組未通過驗證：" + _feedback(ev))
            chosen = ("hold", None, ctx.current_weights(), hold, notes, None)

    source, decision, weights, ev, notes, agent = chosen
    final = _finalize(ctx, source, decision, weights, ev, notes, agent, attempts,
                      run_id=run_id, account_id=account_id, strategy_version=strategy_version,
                      input_sha256=input_sha256, git_sha=git_sha)
    return DecideOutcome(final=final, attempts=attempts, evaluation=ev)


def _finalize(ctx: ToolContext, source: str, decision: Decision | None, weights: dict[str, float],
              ev: Evaluation, notes: list[str], agent: AgentResult | None,
              attempts: list[AgentResult], **prov) -> FinalDecision:
    lot = ctx.rules.lot_size
    notes = list(notes)
    if decision:
        fixed_codes = []
        positions = []
        for p in decision.positions:
            text, changed = fact_check_rationale(ctx, p.code, p.rationale)
            positions.append(p.model_copy(update={"rationale": text}))
            if changed:
                fixed_codes.append(p.code)
        decision = decision.model_copy(update={"positions": positions})
        total = sum(p.target_weight_pct for p in decision.positions) + decision.target_cash_pct
        if abs(total - 100) > 0.5:
            notes.append(f"模型權重加現金合計 {total:.1f}%，已等比例調整為 100%")
        if fixed_codes:
            notes.append(f"理由中的營收月份已依資料更正：{', '.join(fixed_codes)}")
    pos_reason = {p.code: p for p in decision.positions} if decision else {}
    exit_reason = {e.code: e.rationale for e in decision.exits} if decision else {}
    m = ev.report.metrics
    w_after = m.get("weights", {})
    orders = []
    for o in ev.build.orders:
        q = ctx.quotes[o.code]
        if o.side is Side.SELL and o.code in exit_reason:
            why, src = exit_reason[o.code], "llm"
        elif o.code in pos_reason:
            why, src = pos_reason[o.code].rationale, "llm"
        elif source in ("fallback", "hold", "quant"):
            why, src = quant_rationale(ctx, o.code), "baseline"
        else:
            why, src = "風控調整：依比賽規則與內部風險限制自動調整", "risk_control"
        orders.append(FinalOrder(code=o.code, name=ctx.name(o.code), side=o.side.value, lots=o.lots,
                                 shares=o.lots * lot, ref_price=q.ref_price, limit_up=q.limit_up,
                                 limit_down=q.limit_down, est_amount=o.lots * lot * q.ref_price,
                                 rationale=why, rationale_source=src))
    targets = []
    for c, n in sorted(ev.build.target_lots.items(), key=lambda kv: -w_after.get(kv[0], 0.0)):
        p = pos_reason.get(c)
        targets.append(FinalTarget(code=c, name=ctx.name(c), target_weight=weights.get(c, 0.0),
                                   lots_after=n, weight_after_ref=w_after.get(c, 0.0),
                                   conviction=p.conviction if p else None,
                                   rationale=p.rationale if p else quant_rationale(ctx, c)))
    checks = [CheckRow(rule_id=r.rule_id, severity=str(r.severity), passed=r.passed,
                       measured=r.measured, limit=r.limit, detail=r.detail)
              for r in ev.report.results]
    level = decision.risk_level if decision else ctx.regime.suggested_risk_level
    dev = deviation(w_after, ctx.baselines.get(level, {}))
    checks.append(CheckRow(rule_id="R_DEVIATION", severity=str(Severity.SOFT),
                           passed=dev <= ctx.strategy.llm.deviation_budget, measured=dev,
                           limit=ctx.strategy.llm.deviation_budget,
                           detail=f"Σ|w − baseline({level})| = {dev:.1%}"))
    post = {k: v for k, v in m.items() if k not in ("weights", "stressed_weights")}
    post["weights"] = {c: round(x, 6) for c, x in w_after.items()}
    return FinalDecision(
        trade_date=ctx.trade_date, decision_source=source,
        regime=decision.regime if decision else ctx.regime.label, risk_level=level,
        market_view=decision.market_view if decision else _fallback_view(ctx),
        risk_notes=decision.risk_notes if decision else "依規則型市場狀態選擇風險等級；所有部位皆通過規則驗證。",
        targets=targets, exits=[ExitItem(code=c, rationale=r) for c, r in exit_reason.items()],
        orders=orders, post_trade=post, checks=checks,
        repair_notes=list(notes) + list(ev.build.notes), stale_inputs=list(ctx.stale),
        provenance=Provenance(
            run_id=prov["run_id"], account_id=prov["account_id"],
            provider=agent.provider if agent else None, model_id=agent.model_id if agent else None,
            prompt_version=f"{PROMPT_VERSION}:{prompt_sha()}" if agent else None,
            strategy_version=prov["strategy_version"], data_as_of=ctx.as_of,
            input_sha256=prov["input_sha256"], git_sha=prov["git_sha"],
            llm_attempts=[f"{a.provider}:{a.model_id}={a.status}"
                          + (f"({a.error})" if a.error else "") for a in attempts]))


FACTOR_LABELS = {
    "z_itrust_5d": "投信5日買超", "z_itrust_20d": "投信20日買超", "z_foreign_5d": "外資5日買超",
    "z_rev_accel": "月營收年增加速", "z_rev_yoy_3m": "近3月營收年增", "z_rev_new_high_12m": "營收創12月新高",
    "z_mom_6m_skip1w": "6個月動能", "z_dist_52w_high": "接近52週高點", "z_ret_5d": "短線反轉",
    "z_margin_chg_5d": "融資變化", "z_earnings_yield": "盈餘殖利率", "z_vol_60d": "波動度",
}


def quant_rationale(ctx: ToolContext, code: str, top: int = 3) -> str:
    """Deterministic reason from the stock's largest positive factor contributions."""
    from stockmaster.alpha.composite import active_weights
    if code not in ctx.scores.index:
        return "量化基準投組成分"
    row = ctx.scores.loc[code]
    weights = active_weights()
    contrib = {k: row[k] * weights.get(k[2:], 0.0) for k in FACTOR_LABELS if k in row.index}
    best = [k for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][:top]
    parts = [f"{FACTOR_LABELS[k]} z={row[k]:+.1f}" for k in best]
    return f"綜合分數 {row['score']:+.2f}（排名 {list(ctx.scores.index).index(code) + 1}/150）" + (
        "；" + "、".join(parts) if parts else "")


_REV_MONTH = re.compile(r"(\d{1,2})\s*月(?:份)?營收")


def fact_check_rationale(ctx: ToolContext, code: str, text: str) -> tuple[str, bool]:
    """Correct 'N 月營收' claims that contradict the latest revenue month in the data."""
    if code not in ctx.features.index or "rev_last_ym" not in ctx.features:
        return text, False
    ym = ctx.features.loc[code, "rev_last_ym"]
    month = getattr(ym, "month", None)
    if not month:
        return text, False
    fixed = _REV_MONTH.sub(lambda m: f"{month} 月營收" if int(m.group(1)) != month else m.group(0), text)
    return fixed, fixed != text


def _fallback_view(ctx: ToolContext) -> str:
    r = ctx.regime
    return (f"規則型市場狀態：{r.label}。加權指數 5 日報酬 {r.taiex_ret_5d or 0:.1%}、"
            f"20 日報酬 {r.taiex_ret_20d or 0:.1%}，相對 20 日均線 {r.taiex_vs_ma20 or 0:.1%}。")


def decide_quant(ctx: ToolContext, weights: dict[str, float], cash_target: float, *, strategy_label: str,
                 action: str, run_id: str, account_id: str, strategy_version: str, input_sha256: str,
                 git_sha: str | None = None) -> DecideOutcome:
    """Deterministic strategy decision with the backtest engine's path: repair -> orders -> validator."""
    rep = repair_targets(weights, state=ctx.state, quotes=ctx.quotes, universe=ctx.universe.codes,
                         ranking=list(ctx.scores.index), etf_holdings=ctx.etf_holdings, rules=ctx.rules,
                         strategy=ctx.strategy, cash_target=cash_target)
    ev = evaluate_weights(ctx, rep.weights)
    notes = [f"策略 {strategy_label}：{action}"] + [n for n in rep.notes if not n.startswith("Active Share 最低")]
    final = _finalize(ctx, "quant", None, rep.weights, ev, notes, None, [], run_id=run_id,
                      account_id=account_id, strategy_version=strategy_version,
                      input_sha256=input_sha256, git_sha=git_sha)
    final = final.model_copy(update={
        "market_view": f"量化策略 {strategy_label}（{action}）。" + _fallback_view(ctx),
        "risk_notes": "S10 集中結構：核心 10 檔（各 ≤8.5%）、衛星以最小部位補足 22 檔、現金 3%；"
                      "下單前經 repair → 整張下單 → 驗證器檢查，未通過則縮小步伐或維持持股。",
        "orders": [o.model_copy(update={"rationale_source": "quant"}) for o in final.orders]})
    return DecideOutcome(final=final, attempts=[], evaluation=ev)


def orders_from_final(final: FinalDecision) -> list[Order]:
    return [Order(o.code, Side(o.side), o.lots) for o in final.orders]
