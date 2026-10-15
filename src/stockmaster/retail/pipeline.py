"""Daily decision and settlement for the simulated retail account (whole market, odd lots).

decide: eligible universe -> features and composite score -> LLM proposal (or quant fallback) ->
deterministic projection and sizing -> checks -> report/trade book -> recorded run (same tables as
the competition accounts, so the monitor reads both the same way).

settle: after the close, fill the day's final orders at VWAP worsened by the configured slippage,
value the account at the close, and store the ledger day.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, select

from stockmaster.accounting.ledger import DayResult, run_day
from stockmaster.agent.llm_client import ChatModel, build_chain
from stockmaster.agent.schemas import CheckRow, FinalDecision, FinalOrder, FinalTarget, Provenance
from stockmaster.alpha.composite import composite
from stockmaster.data import schema
from stockmaster.data.reference import instrument_names, latest_price_date, load_calendar, quotes_for
from stockmaster.domain import Side
from stockmaster.features.compute import compute_features
from stockmaster.features.panel import load_panel
from stockmaster.features.regime import market_regime
from stockmaster.pipeline.accounts import (_dividends, _fillable, _prices, ensure_account, final_orders,
                                           load_state, save_day)
from stockmaster.pipeline.run import RunOutput, _features_sha, _should_supersede
from stockmaster.retail import agent as retail_agent
from stockmaster.retail.book import Book, position_book
from stockmaster.retail.config import RetailRules, load_retail_rules
from stockmaster.retail.report import write_artifacts
from stockmaster.retail.sizing import SizingContext, project, proposal_problems, size
from stockmaster.retail.universe import eligible_universe
from stockmaster.runs import recorder
from stockmaster.settings import get_settings
from stockmaster.timeutil import now_taipei

log = logging.getLogger(__name__)
LOOKBACK_DAYS = 400  # calendar days of history for 6-month momentum and the 52-week high


# --- inputs --------------------------------------------------------------------------------------

def held_days(book: Book, cal, trade_date: date) -> dict[str, int]:
    """Trading days each open position has been held before ``trade_date``."""
    out = {}
    for code, lot in book.open_positions().items():
        if lot.opened:
            out[code] = len(cal.trading_days(lot.opened, cal.prev_open(trade_date)))
    return out


def close_on_or_before(engine: Engine, code: str, d: date) -> tuple[date, float] | None:
    t = schema.price_daily
    with engine.connect() as conn:
        r = conn.execute(select(t.c.date, t.c.close).where(t.c.code == code).where(t.c.date <= d)
                         .where(t.c.close.is_not(None)).order_by(t.c.date.desc()).limit(1)).first()
    return (r.date, float(r.close)) if r else None


def benchmark_return(engine: Engine, code: str, start: date, as_of: date) -> float | None:
    """Price return of the benchmark from the close before ``start`` to ``as_of``."""
    a = close_on_or_before(engine, code, start - timedelta(days=1))
    b = close_on_or_before(engine, code, as_of)
    return b[1] / a[1] - 1 if a and b and a[0] < b[0] else None


def recent_decisions(engine: Engine, account_id: str, before: date, n: int = 5) -> list[dict]:
    s, d, r, ld = schema.submission, schema.decision, schema.run, schema.ledger_day
    with engine.connect() as conn:
        rows = conn.execute(select(s.c.trade_date, d.c.source, d.c.payload).join(d, d.c.id == s.c.decision_id)
                            .join(r, r.c.run_id == d.c.run_id).where(r.c.account_id == account_id)
                            .where(s.c.is_final.is_(True)).where(s.c.trade_date < before)
                            .order_by(s.c.trade_date.desc()).limit(n)).all()
        navs = {x.date: float(x.nav) for x in conn.execute(
            select(ld.c.date, ld.c.nav).where(ld.c.account_id == account_id))}
    out = []
    for row in rows:
        orders = [f"{'買' if o['side'] == 'buy' else '賣'}{o['code']}x{o['shares']}" for o in row.payload["orders"]]
        out.append({"date": row.trade_date, "source": row.source, "orders": orders or ["無交易"],
                    "nav": navs.get(row.trade_date), "view": (row.payload.get("market_view") or "")[:80]})
    return out


def quant_targets(scores: pd.DataFrame, feats: pd.DataFrame, eligible: set[str], held: set[str],
                  regime_label: str, nav: float, rules: RetailRules) -> tuple[dict[str, float], dict[str, str]]:
    """Deterministic fallback: keep holdings still ranked in the top 50, fill the remaining slots
    with the best-scored affordable names in a positive trend (6-month momentum, or the 20-day
    return when history is shorter), equal weight. Exposure is cut in a risk-off regime."""
    p = rules.portfolio
    exposure = {"risk_off": 0.5, "neutral": 0.8}.get(regime_label, 1.0) * (1 - p.cash_buffer_pct)
    ranked = [c for c in scores.index if c in eligible]
    rank = {c: i for i, c in enumerate(ranked)}
    keep = [c for c in held if rank.get(c, 10**6) < 50]
    max_price = p.max_weight * nav
    for c in ranked:
        if len(keep) >= p.max_positions:
            break
        f = feats.loc[c] if c in feats.index else None
        if c in keep or f is None or not (f.get("close", 0) <= max_price):
            continue
        trend = f.get("mom_6m_skip1w")
        if trend is None or pd.isna(trend):  # under ~6 months of history: use the 20-day return
            trend = f.get("ret_20d")
        if not (trend is not None and trend > 0):
            continue
        keep.append(c)
    if not keep:
        return {}, {}
    w = min(p.max_weight, exposure / len(keep))
    why = {c: f"量化備援：綜合分數排名第 {rank.get(c, 0) + 1}" for c in keep}
    return {c: w for c in keep}, why


def _fmt(x, kind: str = "f") -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if kind == "pct":
        return f"{x:+.1%}"
    if kind == "int":
        return f"{x:,.0f}"
    return f"{x:.2f}"


def _candidate_rows(codes: list[str], feats: pd.DataFrame, scores: pd.DataFrame, names: dict[str, str],
                    industry: dict[str, str]) -> str:
    lines = ["code | name | industry | price | score | ret5 | ret20 | mom6m | itrust5 | foreign5 | rev_yoy | pe | adv"]
    for c in codes:
        f = feats.loc[c] if c in feats.index else pd.Series(dtype=float)
        s = scores.loc[c, "score"] if c in scores.index else None
        lines.append(" | ".join([
            c, names.get(c, c), str(industry.get(c) or "—"), _fmt(f.get("close")), _fmt(s),
            _fmt(f.get("ret_5d"), "pct"), _fmt(f.get("ret_20d"), "pct"), _fmt(f.get("mom_6m_skip1w"), "pct"),
            _fmt(f.get("itrust_5d"), "pct"), _fmt(f.get("foreign_5d"), "pct"), _fmt(f.get("rev_yoy_last")),
            _fmt(f.get("pe")), _fmt((f.get("adv20_value") or 0) / 1e6, "int")]))
    return "\n".join(lines)


# --- decision ------------------------------------------------------------------------------------

def run_retail_decision(engine: Engine, *, rules: RetailRules | None = None, trade_date: date | None = None,
                        slot: str = "A", models: list[ChatModel] | None = None, use_llm: bool = True,
                        artifacts_root: Path | None = None) -> RunOutput:
    rules = rules or load_retail_rules()
    account_id = rules.account_id
    cal = load_calendar(engine)
    now = now_taipei()
    if trade_date is None:
        today = now.date()
        trade_date = today if (cal.is_open(today) and now.hour < 9) else cal.next_open(today)
    as_of = latest_price_date(engine, trade_date)
    if as_of is None:
        raise RuntimeError("no price data in the database; run `sm bootstrap` first")
    stale: list[str] = []
    if as_of != cal.prev_open(trade_date):
        stale.append(f"prices:last={as_of} expected={cal.prev_open(trade_date)}")

    ensure_account(engine, account_id, "retail", rules.initial_cash, rules.start_date,
                   note="simulated odd-lot account, whole TWSE/TPEx market")
    state = load_state(engine, account_id, rules, before=trade_date).state
    book = position_book(engine, account_id, before=trade_date)
    universe = eligible_universe(engine, as_of, rules)
    held = set(state.held_codes())
    codes = sorted(universe.codes | held)
    if len(universe.codes) < 50:
        stale.append(f"universe:only {len(universe.codes)} eligible names (history too short?)")

    panel = load_panel(engine, codes, as_of, lookback_days=LOOKBACK_DAYS)
    feats = compute_features(panel, decision_time=pd.Timestamp(now.replace(tzinfo=None)))
    scores = composite(feats)
    regime = market_regime(panel)
    quotes = quotes_for(engine, codes, as_of, rules, trade_date=trade_date)
    names = instrument_names(engine, codes)
    industry = universe.info["industry"].to_dict()
    eligible = frozenset(c for c in universe.codes if c in quotes and not quotes[c].halted)
    ctx = SizingContext(trade_date, state, quotes, rules, eligible, held_days(book, cal, trade_date))
    nav = ctx.nav_ref

    p = rules.portfolio
    affordable = [c for c in scores.index if c in eligible and quotes[c].ref_price <= p.max_weight * nav]
    candidates = affordable[:p.candidates_for_llm]

    # --- prompt ---------------------------------------------------------------------------------
    bench = benchmark_return(engine, rules.benchmark, rules.start_date, as_of) if as_of >= rules.start_date else None
    ret = nav / rules.initial_cash - 1
    account_txt = (f"淨值（參考價）NT${nav:,.0f}；現金 NT${float(state.cash):,.0f}；自 {rules.start_date} 起報酬 {ret:+.2%}"
                   f"（基準 {rules.benchmark} 同期 {_fmt(bench, 'pct')}）；已實現損益 NT${book.realized:,.0f}；"
                   f"累計手續費與稅 NT${book.fees:,.0f}")
    hold_lines = ["code | name | shares | avg_cost | price | unrealized | weight | held_days | score_rank"]
    rank = {c: i + 1 for i, c in enumerate(c for c in scores.index if c in eligible)}
    for c, n in sorted(state.lots.items()):
        lot = book.lots.get(c)
        px = quotes[c].ref_price if c in quotes else None
        avg = lot.avg_cost if lot else None
        hold_lines.append(" | ".join([c, names.get(c, c), str(n), _fmt(avg), _fmt(px),
                                      _fmt(px / avg - 1 if px and avg else None, "pct"),
                                      _fmt(n * px / nav if px else None, "pct"), str(ctx.held_days.get(c, "—")),
                                      str(rank.get(c, "未入選"))]))
    r = regime
    market_txt = (f"加權指數 {_fmt(r.taiex, 'int')}，1日 {_fmt(r.taiex_ret_1d, 'pct')}、5日 {_fmt(r.taiex_ret_5d, 'pct')}、"
                  f"20日 {_fmt(r.taiex_ret_20d, 'pct')}；相對20日均線 {_fmt(r.taiex_vs_ma20, 'pct')}、60日均線 "
                  f"{_fmt(r.taiex_vs_ma60, 'pct')}；20日年化波動 {_fmt(r.taiex_vol_20d, 'pct')}；"
                  f"規則判斷 {r.label}。可交易名單 {len(eligible)} 檔。")
    hist = recent_decisions(engine, account_id, trade_date)
    history_txt = "\n".join(f"- {h['date']} [{h['source']}] {'、'.join(h['orders'])}；當日收盤淨值 "
                            f"{_fmt(h['nav'], 'int')}" for h in hist) or "（尚無紀錄，這是第一個交易日）"
    warn_txt = ("\n⚠ 資料警示：" + "；".join(stale) + "\n") if stale else ""
    system = retail_agent.system_prompt(rules)
    user = retail_agent.user_prompt(
        trade_date=str(trade_date), as_of=str(as_of), account=account_txt, market=market_txt,
        holdings="\n".join(hold_lines) if state.lots else "（無持股，全部為現金）",
        n_candidates=str(len(candidates)), candidates=_candidate_rows(candidates, feats, scores, names, industry),
        history=history_txt, warnings=warn_txt)

    skipped: list[str] = []
    if use_llm and models is None:
        models, skipped = build_chain(rules.llm.chain or None)
    models = models if use_llm else []
    fsha = _features_sha(feats)
    manifest = {"trade_date": str(trade_date), "as_of": str(as_of), "account": account_id,
                "state": {"cash": str(state.cash), "lots": state.lots}, "features_sha": fsha,
                "universe_size": len(eligible), "candidates": candidates, "stale": stale,
                "models": [f"{m.provider}:{m.model_id}" for m in models], "skipped_models": skipped,
                "prompt": f"{retail_agent.PROMPT_VERSION}:{retail_agent.prompt_sha()}",
                "rules": rules.model_dump(mode="json")}
    input_sha = recorder.sha256_json(manifest)

    run_id = recorder.new_run_id()
    recorder.start_run(engine, run_id, kind="decide", account_id=account_id, trade_date=trade_date, slot=slot,
                       config_sha={"rules.retail.yaml": recorder.sha256_json(rules.model_dump(mode="json"))},
                       strategy_version=rules.strategy_version, input_manifest=manifest, features_sha=fsha)
    try:
        allowed = eligible | held

        def evaluate(d: retail_agent.RetailDecision) -> list[str]:
            w = d.weights()
            probs = proposal_problems(w, ctx)
            outside = sorted(c for c in w if c not in set(candidates) | held and c in allowed)
            if outside:
                probs.append(f"只能選擇候選名單或目前持股：{outside}")
            return probs

        attempts, decision, source, used = [], None, None, None
        for m in models:
            res = retail_agent.run_agent(m, system, user, evaluate, rules)
            attempts.append(res)
            if res.status == "accepted":
                decision, used = res.decision, res
                source = "llm" if res.repair_rounds == 0 else "llm_repair"
                break
            if res.decision is not None and decision is None:
                decision, used, source = res.decision, res, "det_repair"
        notes: list[str] = []
        if decision is not None:
            weights, notes = project(decision.weights(), ctx)
            if source != "det_repair":
                notes = []
            rationale = {t.code: t.rationale for t in decision.targets}
            conviction = {t.code: t.conviction for t in decision.targets}
            view, reg, risk, risk_notes = decision.market_view, decision.regime, decision.risk_level, decision.risk_notes
        else:
            weights, rationale = quant_targets(scores, feats, set(eligible), held, regime.label, nav, rules)
            conviction = {}
            source = "fallback" if use_llm else "quant"
            reasons = [a.error for a in attempts if a.error] or skipped or ["LLM 未啟用"]
            view = (f"未使用 LLM（{'；'.join(reasons)[:300]}）。採用量化規則：保留仍在前 50 名的持股，"
                    f"其餘以綜合分數最高、6 個月動能為正且買得起的標的補足，等權配置；市場狀態 {regime.label}。")
            reg, risk = regime.label, regime.suggested_risk_level
            risk_notes = "量化備援決策；綜合分數權重未在全市場驗證。"
        plan = size(weights, ctx)
        if not plan.passed:
            notes.append("風控檢查未通過，改為不交易：" + "；".join(c.detail for c in plan.checks
                                                         if c.severity.value == "hard" and not c.passed))
            plan, source = _hold_plan(ctx), "hold"
        final = _final_decision(trade_date, as_of, source, reg, risk, view, risk_notes, plan, weights, rationale,
                                conviction, names, quotes, notes, stale, run_id, account_id, used, attempts,
                                input_sha, rules)
        root = artifacts_root or get_settings().artifacts_dir
        art = write_artifacts(final, root / str(trade_date) / account_id / f"{slot}_{str(run_id)[:8]}")
        recorder.record_attempts(engine, run_id, attempts, trade_date)
        did = recorder.record_final(engine, run_id, final, art, account_id)
        hard_ok = all(c.passed for c in final.checks if c.severity == "hard")
        is_final = hard_ok and _should_supersede(engine, account_id, trade_date, final, slot)
        if is_final:
            recorder.mark_final(engine, did)
        recorder.finish_run(engine, run_id, status="ok" if hard_ok else "invalid",
                            provider=final.provenance.provider, model_id=final.provenance.model_id,
                            prompt_version=final.provenance.prompt_version)
        return RunOutput(final, art, str(run_id), did, is_final)
    except Exception as e:
        recorder.finish_run(engine, run_id, status="error", error=f"{type(e).__name__}: {e}"[:1000])
        raise


def _hold_plan(ctx: SizingContext):
    from stockmaster.retail.sizing import Plan, evaluate
    checks, metrics = evaluate([], ctx)
    return Plan([], dict(ctx.state.lots), [], checks, metrics)


def _final_decision(trade_date, as_of, source, regime, risk, view, risk_notes, plan, weights, rationale,
                    conviction, names, quotes, notes, stale, run_id, account_id, used, attempts, input_sha,
                    rules) -> FinalDecision:
    orders = []
    for o in plan.orders:
        q = quotes[o.code]
        if o.side is Side.BUY:
            why, src = rationale.get(o.code) or "建立/加碼部位", "llm" if source.startswith("llm") else "quant"
        else:
            why = (f"降低權重：{rationale[o.code]}" if o.code in plan.target_shares and o.code in rationale
                   else "不在目標持股中，出清")
            src = "llm" if source.startswith("llm") else "quant"
        if source in ("det_repair", "hold"):
            src = "risk_control"
        orders.append(FinalOrder(code=o.code, name=names.get(o.code, o.code), side=o.side.value, lots=o.lots,
                                 shares=o.lots * rules.lot_size, ref_price=q.ref_price, limit_up=q.limit_up,
                                 limit_down=q.limit_down, est_amount=o.lots * rules.lot_size * q.ref_price,
                                 rationale=why[:240], rationale_source=src))
    w_after = plan.metrics.get("weights", {})
    targets = [FinalTarget(code=c, name=names.get(c, c), target_weight=float(weights.get(c, 0.0)), lots_after=n,
                           weight_after_ref=float(w_after.get(c, 0.0)), conviction=conviction.get(c),
                           rationale=(rationale.get(c) or "")[:240]) for c, n in sorted(plan.target_shares.items())]
    checks = [CheckRow(rule_id=c.rule_id, severity=c.severity.value, passed=c.passed, measured=c.measured,
                       limit=c.limit, detail=c.detail) for c in plan.checks]
    llm_attempts = [f"{a.provider}:{a.model_id}={a.status}" + (f"({a.error[:80]})" if a.error and a.status != "accepted"
                                                             else "") for a in attempts]
    return FinalDecision(
        trade_date=trade_date, decision_source=source, regime=regime, risk_level=risk, market_view=view,
        risk_notes=risk_notes, targets=targets, exits=[], orders=orders, post_trade=plan.metrics, checks=checks,
        repair_notes=notes + plan.notes, stale_inputs=stale,
        provenance=Provenance(run_id=str(run_id), account_id=account_id,
                              provider=used.provider if used else None, model_id=used.model_id if used else None,
                              prompt_version=retail_agent.PROMPT_VERSION if used else None,
                              strategy_version=rules.strategy_version, data_as_of=as_of, input_sha256=input_sha,
                              git_sha=recorder.git_sha(), llm_attempts=llm_attempts))


# --- settlement ----------------------------------------------------------------------------------

def settle_retail(engine: Engine, trade_date: date, rules: RetailRules | None = None) -> DayResult | None:
    """Fill the final orders of ``trade_date`` at VWAP +/- slippage, value at the close, store the day.
    Returns None before the account's start date."""
    rules = rules or load_retail_rules()
    if trade_date < rules.start_date:
        return None
    account_id = rules.account_id
    ensure_account(engine, account_id, "retail", rules.initial_cash, rules.start_date)
    prev = load_state(engine, account_id, rules, before=trade_date)
    orders = final_orders(engine, account_id, trade_date)
    codes = set(prev.state.lots) | {o.code for o in orders}
    vwaps, closes = _prices(engine, trade_date, codes)
    orders, unfilled = _fillable(orders, vwaps)
    if unfilled:
        from stockmaster.ops.alerts import alert
        alert(f"[{account_id}] {trade_date}: {unfilled} did not trade; orders not filled", level="warning")
    slip = rules.fill.slippage_bps / 10000
    side = {o.code: o.side for o in orders}
    fill_px = {c: (Decimal(str(round(float(v) * (1 + slip if side.get(c) is Side.BUY else 1 - slip), 2))))
               for c, v in vwaps.items()}
    res = run_day(prev.state, trade_date, orders, fill_px, closes, rules,
                  cash_dividends_per_share=_dividends(engine, trade_date, prev.last_date))
    save_day(engine, account_id, res, "applied", lot_size=rules.lot_size)
    return res
