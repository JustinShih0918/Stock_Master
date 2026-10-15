"""DB-backed daily decision run: load inputs -> decide -> artifacts -> record -> mark final."""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, func, select

from stockmaster.agent.harness import PROMPT_VERSION, prompt_sha
from stockmaster.agent.llm_client import ChatModel, build_chain
from stockmaster.agent.schemas import FinalDecision
from stockmaster.alpha.composite import composite
from stockmaster.config import config_fingerprint, load_rules, load_strategy, load_universe
from stockmaster.data import schema
from stockmaster.data.reference import latest_price_date, load_calendar, quotes_for
from stockmaster.features.compute import compute_features
from stockmaster.features.panel import load_panel
from stockmaster.features.regime import market_regime
from stockmaster.pipeline.accounts import ensure_account, load_state, safe_state
from stockmaster.pipeline.context import build_tool_context
from stockmaster.pipeline.decide import decide, decide_quant
from stockmaster.reporting.consistency import check_consistency
from stockmaster.reporting.render import Artifacts, write_artifacts
from stockmaster.runs import recorder
from stockmaster.settings import get_settings
from stockmaster.timeutil import now_taipei

log = logging.getLogger(__name__)
ETF_MAX_AGE_DAYS = 7


@dataclass
class RunOutput:
    final: FinalDecision
    artifacts: Artifacts
    run_id: str
    decision_id: int
    is_final: bool


def load_etf_holdings(engine: Engine, as_of: date, reference_only: bool = True
                      ) -> tuple[dict[str, dict[str, float]], dict[str, date], list[str]]:
    """Latest holdings per active ETF on or before ``as_of``; flags stale snapshots."""
    h, e = schema.etf_holding, schema.etf
    with engine.connect() as conn:
        latest = select(h.c.etf_code, func.max(h.c.as_of).label("d")).where(h.c.as_of <= as_of) \
            .group_by(h.c.etf_code).subquery()
        q = select(h.c.etf_code, h.c.as_of, h.c.code, h.c.weight).join(
            latest, (latest.c.etf_code == h.c.etf_code) & (latest.c.d == h.c.as_of))
        if reference_only:
            refs = select(e.c.code).where(e.c.in_reference_list.is_(True))
            n_refs = conn.execute(select(func.count()).select_from(refs.subquery())).scalar()
            if n_refs:
                q = q.where(h.c.etf_code.in_(refs))
        df = pd.read_sql(q, conn)
    holdings: dict[str, dict[str, float]] = {}
    asof: dict[str, date] = {}
    stale: list[str] = []
    for etf_code, g in df.groupby("etf_code"):
        holdings[etf_code] = dict(zip(g["code"], g["weight"]))
        asof[etf_code] = g["as_of"].iloc[0]
        if (as_of - asof[etf_code]).days > ETF_MAX_AGE_DAYS:
            stale.append(f"etf_holdings:{etf_code}@{asof[etf_code]}")
    if not holdings:
        stale.append("etf_holdings:none")
    return holdings, asof, stale


def load_events(engine: Engine, trade_date: date, codes: set[str]) -> list[dict]:
    events = []
    t = schema.corp_action
    with engine.connect() as conn:
        rows = conn.execute(select(t.c.code, t.c.ex_date, t.c.kind, t.c.cash_dividend)
                            .where(t.c.ex_date >= trade_date)
                            .where(t.c.ex_date <= trade_date + timedelta(days=14))
                            .where(t.c.code.in_(list(codes)))).all()
    for r in rows:
        events.append({"date": str(r.ex_date), "code": r.code,
                       "event": f"{r.code} 除{r.kind}（現金股利 {r.cash_dividend or 0}）；除息金額由主辦方於賽末併入 NAV"})
    d = trade_date
    month_10th = date(d.year, d.month, 10)
    if d <= month_10th:
        events.append({"date": str(month_10th), "event": "上月營收公布截止日（各公司陸續公布）"})
    if date(2026, 10, 1) <= d <= date(2026, 11, 14):
        events.append({"date": "2026-11-14", "event": "第三季財報公布截止日"})
    if d <= date(2026, 11, 10):
        events.append({"date": "2026-11-10", "event": "10 月營收公布截止日（比賽期間主要催化劑）"})
    return sorted(events, key=lambda e: e["date"])


def load_sectors(engine: Engine) -> dict[str, str]:
    t = schema.instrument
    with engine.connect() as conn:
        return {r.code: r.industry for r in conn.execute(select(t.c.code, t.c.industry))
                if r.industry}


def load_overnight(engine: Engine, tsmc_close: float | None) -> dict:
    from stockmaster.data.sources import overnight
    m = schema.macro_daily
    with engine.connect() as conn:
        df = pd.read_sql(select(m.c.series_id, m.c.date, m.c.value)
                         .where(m.c.series_id.like("yahoo:%"))
                         .where(m.c.date >= now_taipei().date() - timedelta(days=15)), conn)
    if df.empty:
        return {}
    series = {sid.split(":", 1)[1]: g.set_index("date")["value"] for sid, g in df.groupby("series_id")}
    return overnight.summarize(series, tsmc_close)


def _input_manifest(**parts) -> tuple[dict, str]:
    manifest = {k: v for k, v in parts.items()}
    return manifest, recorder.sha256_json(manifest)


def _features_sha(features: pd.DataFrame) -> str:
    buf = io.BytesIO()
    features.sort_index().to_csv(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def day_inputs(ctx, day_index: int | None, days_total: int | None):
    """Engine-style DayInputs from the live ToolContext (for deterministic strategies)."""
    from stockmaster.backtest.engine import DayInputs
    return DayInputs(trade_date=ctx.trade_date, as_of=ctx.as_of, state=ctx.state, features=ctx.features,
                     scores=ctx.scores, regime=ctx.regime, quotes=dict(ctx.quotes),
                     day_index=day_index or 1, days_total=days_total)


def run_decision(engine: Engine, *, account_id: str, trade_date: date | None = None,
                 slot: str = "A", models: list[ChatModel] | None = None,
                 use_llm: bool = True, start_date: date | None = None,
                 artifacts_root: Path | None = None, strategy_name: str | None = None) -> RunOutput:
    rules, strategy, universe = load_rules(), load_strategy(), load_universe()
    cal = load_calendar(engine)
    now = now_taipei()
    if trade_date is None:
        today = now.date()
        trade_date = today if (cal.is_open(today) and now.hour < 9) else cal.next_open(today)
    as_of = latest_price_date(engine, trade_date)
    if as_of is None:
        raise RuntimeError("no price data in the database")
    stale: list[str] = []
    if as_of != cal.prev_open(trade_date):
        stale.append(f"prices:last={as_of} expected={cal.prev_open(trade_date)}")

    ensure_account(engine, account_id, "live" if account_id == "live" else "paper",
                   rules.initial_cash, start_date or trade_date)
    acct = load_state(engine, account_id, rules, before=trade_date)
    state = acct.state
    reconciliation = acct.day_status or "n/a"
    if acct.day_status == "unreconciled" and acct.last_date:
        state = safe_state(engine, account_id, acct.last_date, rules)
        stale.append(f"ledger:unreconciled@{acct.last_date}")

    codes = sorted(universe.codes)
    panel = load_panel(engine, codes, as_of)
    feats = compute_features(panel, decision_time=pd.Timestamp(now.replace(tzinfo=None)))
    from stockmaster.features.etf_flows import etf_flow_features
    feats = feats.join(etf_flow_features(engine, as_of), how="left")
    scores = composite(feats)
    regime = market_regime(panel)
    quotes = quotes_for(engine, codes, as_of, rules, trade_date=trade_date)
    stale_days = len(cal.trading_days(as_of, cal.prev_open(trade_date))) - 1
    if stale_days > 0:
        # Prices are k trading days old: the stress must cover k+1 days of limit moves.
        from stockmaster.domain import Quote
        up = (1 + rules.price_limit_pct) ** (stale_days + 1) - 1
        quotes = {c: Quote(q.code, q.ref_price, float(q.ref_price * (1 + up)),
                           float(q.ref_price * (1 - rules.price_limit_pct) ** (stale_days + 1)),
                           q.as_of, q.halted) for c, q in quotes.items()}
    missing_quotes = sorted(set(codes) - set(quotes))
    if missing_quotes:
        stale.append(f"quotes_missing:{','.join(missing_quotes[:10])}")
    etf, etf_asof, etf_stale = load_etf_holdings(engine, as_of)
    stale += etf_stale

    ctx = build_tool_context(trade_date=trade_date, as_of=as_of, state=state, quotes=quotes,
                             universe=universe, features=feats, scores=scores, regime=regime,
                             etf_holdings=etf, etf_asof=etf_asof, rules=rules, strategy=strategy,
                             calendar=cal, events=load_events(engine, trade_date, set(codes)),
                             stale=stale, reconciliation=reconciliation, sectors=load_sectors(engine),
                             overnight=load_overnight(engine, quotes["2330"].ref_price if "2330" in quotes else None))

    skipped: list[str] = []
    if strategy_name:
        models, use_llm = [], False
    if models is None:
        models, skipped = build_chain() if use_llm else ([], [])
    fsha = _features_sha(feats)
    manifest, input_sha = _input_manifest(
        trade_date=str(trade_date), as_of=str(as_of), account=account_id,
        state={"cash": str(state.cash), "lots": state.lots, "warnings": state.warnings},
        features_sha=fsha, universe_sha=universe.sha256,
        etf_asof={k: str(v) for k, v in etf_asof.items()}, stale=stale,
        models=[f"{m.provider}:{m.model_id}" for m in models], skipped_models=skipped,
        prompt=f"{PROMPT_VERSION}:{prompt_sha()}", strategy=strategy_name)

    run_id = recorder.new_run_id()
    recorder.start_run(engine, run_id, kind="decide", account_id=account_id, trade_date=trade_date,
                       slot=slot, config_sha=config_fingerprint(), strategy_version=strategy.strategy_version,
                       input_manifest=manifest, features_sha=fsha)
    try:
        if strategy_name:
            from stockmaster.strategies import s10
            if ctx.day_index is not None:  # competition account: competition day / 24
                d_idx, d_tot = ctx.day_index, rules.period.expected_trading_days
            else:  # paper account: trading days since the account started, no end-of-round freeze
                acct_start = start_date or trade_date
                d_idx, d_tot = len(cal.trading_days(acct_start, trade_date)), None
            env = s10.Env(rules=rules, strategy=strategy, universe=universe.codes, etf_holdings=etf)
            w, c_t, action = s10.decide(day_inputs(ctx, d_idx, d_tot), s10.VARIANTS[strategy_name], env)
            out = decide_quant(ctx, w, c_t, strategy_label=f"{strategy_name} day {d_idx}", action=action,
                               run_id=str(run_id), account_id=account_id,
                               strategy_version=f"{strategy.strategy_version}+{strategy_name}",
                               input_sha256=input_sha, git_sha=recorder.git_sha())
        else:
            out = decide(ctx, models, run_id=str(run_id), account_id=account_id,
                         strategy_version=strategy.strategy_version, input_sha256=input_sha,
                         git_sha=recorder.git_sha())
        final = out.final
        root = artifacts_root or get_settings().artifacts_dir
        art_dir = root / str(trade_date) / account_id / f"{slot}_{str(run_id)[:8]}"
        art = write_artifacts(final, art_dir)
        problems = check_consistency(final, art.report_md.read_text(encoding="utf-8"),
                                     art.trade_book_csv.read_text(encoding="utf-8-sig"))
        if problems:
            raise RuntimeError(f"report/trade-book inconsistency: {problems}")
        recorder.record_attempts(engine, run_id, out.attempts, trade_date)
        did = recorder.record_final(engine, run_id, final, art, account_id)
        hard_ok = all(c.passed for c in final.checks if c.severity == "hard")
        is_final = _should_supersede(engine, account_id, trade_date, final, slot) and hard_ok
        if is_final:
            recorder.mark_final(engine, did)
        recorder.finish_run(engine, run_id, status="ok" if hard_ok else "invalid",
                            provider=final.provenance.provider, model_id=final.provenance.model_id,
                            prompt_version=final.provenance.prompt_version)
        return RunOutput(final, art, str(run_id), did, is_final)
    except Exception as e:
        recorder.finish_run(engine, run_id, status="error", error=f"{type(e).__name__}: {e}"[:1000])
        raise


SOURCE_RANK = {"llm": 4, "quant": 4, "llm_repair": 3, "det_repair": 2, "fallback": 1, "hold": 0}


def _should_supersede(engine: Engine, account_id: str, trade_date: date, new: FinalDecision,
                      slot: str) -> bool:
    """A later run replaces the current final submission only if it is at least as good a source
    (e.g. a pre-open LLM run replaces an evening fallback, but a watchdog never replaces an LLM run)."""
    s, d, r = schema.submission, schema.decision, schema.run
    with engine.connect() as conn:
        cur = conn.execute(select(d.c.source).join(s, s.c.decision_id == d.c.id)
                           .join(r, r.c.run_id == d.c.run_id).where(r.c.account_id == account_id)
                           .where(s.c.trade_date == trade_date).where(s.c.is_final.is_(True))).first()
    if cur is None:
        return True
    if slot == "watchdog":
        return False
    return SOURCE_RANK.get(new.decision_source, 0) >= SOURCE_RANK.get(cur.source, 0)
