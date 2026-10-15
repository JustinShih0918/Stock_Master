"""Decision and settlement jobs for the scheduler. Every job checks the trading calendar and is
safe to re-run."""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select

from stockmaster.config import load_rules
from stockmaster.data import schema
from stockmaster.data.reference import load_calendar
from stockmaster.data.store import get_engine
from stockmaster.ops.alerts import alert
from stockmaster.pipeline.accounts import settle
from stockmaster.pipeline.run import run_decision
from stockmaster.timeutil import now_taipei

log = logging.getLogger(__name__)

COMPETITION_ACCOUNT = "live"
# account -> (uses LLM, first trade date). Paper accounts A/B-test model chains before the
# 10/23 code freeze; shadow_baseline measures what the LLM adds over the quant baseline.
ACCOUNTS: dict[str, tuple[bool, date]] = {
    "paper": (True, date(2026, 10, 5)),
    "paper_kimi": (True, date(2026, 10, 5)),
    "shadow_baseline": (False, date(2026, 10, 5)),
    # S10 top-6 research strategies (deterministic, no LLM); see research/preregistration_20261003.json
    "shadow_s10a": (False, date(2026, 10, 5)),
    "shadow_s10b": (False, date(2026, 10, 5)),
    "shadow_s10a_t10": (False, date(2026, 10, 5)),
    "shadow_s10b_t10": (False, date(2026, 10, 5)),
    COMPETITION_ACCOUNT: (True, date(2026, 10, 27)),
}
ACCOUNT_STRATEGIES: dict[str, str] = {
    "shadow_s10a": "s10a", "shadow_s10b": "s10b",
    "shadow_s10a_t10": "s10a_t10", "shadow_s10b_t10": "s10b_t10",
}
# Per-account provider:model chains; accounts not listed use the default chain
# (NVIDIA_MODEL, then strategy.yaml llm.fallback_models).
ACCOUNT_CHAINS: dict[str, str] = {
    "paper_kimi": "nvidia:moonshotai/kimi-k3,nvidia:nvidia/nemotron-3-ultra-550b-a55b,"
                  "nvidia:deepseek-ai/deepseek-v4.1-flash",
}


def _active(account: str, trade_date: date) -> bool:
    return trade_date >= ACCOUNTS[account][1]


def _final_source(account: str, trade_date: date) -> str | None:
    s, d, r = schema.submission, schema.decision, schema.run
    with get_engine().connect() as conn:
        row = conn.execute(select(d.c.source).join(s, s.c.decision_id == d.c.id)
                           .join(r, r.c.run_id == d.c.run_id).where(r.c.account_id == account)
                           .where(s.c.trade_date == trade_date).where(s.c.is_final.is_(True))).first()
    return row.source if row else None


def _decide(account: str, trade_date: date, slot: str, use_llm: bool) -> str:
    models = None
    if use_llm and account in ACCOUNT_CHAINS:
        from stockmaster.agent.llm_client import build_chain
        models, skipped = build_chain(ACCOUNT_CHAINS[account])
        if skipped:
            alert(f"[{account}] chain models unavailable: {skipped}", level="warning")
    out = run_decision(get_engine(), account_id=account, trade_date=trade_date, slot=slot,
                       use_llm=use_llm, start_date=ACCOUNTS[account][1], models=models,
                       strategy_name=ACCOUNT_STRATEGIES.get(account))
    f = out.final
    msg = (f"[{account}] {slot} {trade_date}: source={f.decision_source} orders={len(f.orders)} "
           f"holdings={f.post_trade.get('n_holdings')} final={out.is_final} report={out.artifacts.report_md}")
    hard_fail = [c.rule_id for c in f.checks if c.severity == "hard" and not c.passed]
    if hard_fail:
        alert(f"HARD RULE FAILURE {msg} {hard_fail}", level="error")
    elif f.decision_source not in ("llm", "llm_repair", "quant") and use_llm:
        alert(f"LLM not used: {msg} attempts={f.provenance.llm_attempts}", level="warning")
    elif f.stale_inputs:
        alert(f"stale inputs: {msg} {f.stale_inputs}", level="warning")
    return msg


def run_a() -> str:
    """Evening run for the next trading day. Runs only on the calendar day right before it, which
    is when the organizer's submission window (T-1 19:30 - T 08:55) is open."""
    from datetime import timedelta
    engine = get_engine()
    cal = load_calendar(engine)
    today = now_taipei().date()
    trade_date = cal.next_open(today)
    if trade_date != today + timedelta(days=1):
        return f"skip: next trading day {trade_date} is not tomorrow"
    return " | ".join(_decide(a, trade_date, "A", llm) for a, (llm, _) in ACCOUNTS.items()
                      if _active(a, trade_date))


OVERNIGHT_SHOCK = 0.03


def _overnight_shock() -> str | None:
    from stockmaster.pipeline.run import load_overnight
    o = load_overnight(get_engine(), None)
    big = [f"{k} {v['ret_1d']:+.1%}" for k, v in o.items()
           if isinstance(v, dict) and k in ("^SOX", "TSM") and abs(v["ret_1d"]) >= OVERNIGHT_SHOCK]
    return ", ".join(big) or None


def run_b() -> str:
    """Pre-open run: retry when Run A could not use the LLM, or re-decide after a large overnight
    US move (SOX / TSM ADR >= 3%). A Run B decision replaces Run A only if it validates."""
    cal = load_calendar(get_engine())
    today = now_taipei().date()
    if not cal.is_open(today):
        return f"{today} closed"
    shock = _overnight_shock()
    out = []
    for a, (llm, _) in ACCOUNTS.items():
        if not _active(a, today) or not llm:
            continue
        if _final_source(a, today) in ("llm", "llm_repair") and not shock:
            out.append(f"[{a}] keep Run A")
            continue
        out.append(_decide(a, today, "B", True) + (f" (overnight shock: {shock})" if shock else ""))
    return " | ".join(out)


def watchdog() -> str:
    """08:30 guarantee: every active account has a valid final submission for today."""
    cal = load_calendar(get_engine())
    today = now_taipei().date()
    if not cal.is_open(today):
        return f"{today} closed"
    out = []
    for a in ACCOUNTS:
        if _active(a, today) and _final_source(a, today) is None:
            alert(f"[{a}] no final submission at watchdog time; generating deterministic fallback",
                  level="error")
            out.append(_decide(a, today, "watchdog", False))
    return " | ".join(out) or "all accounts have a final submission"


def settle_today() -> str:
    engine = get_engine()
    cal = load_calendar(engine)
    today = now_taipei().date()
    if not cal.is_open(today):
        return f"{today} closed"
    rules = load_rules()
    out = []
    for a in ACCOUNTS:
        if not _active(a, today) or _final_source(a, today) is None:
            continue
        r = settle(engine, a, today, rules)
        v = r.valuation
        breaches = []
        if not rules.holdings.min_count <= v.n_holdings <= rules.holdings.max_count:
            breaches.append(f"holdings={v.n_holdings}")
        if v.cash < 0 or v.cash_pct >= rules.cash.max_exclusive:
            breaches.append(f"cash={v.cash_pct:.2%}")
        over = {c: w for c, w in v.weights.items() if w > rules.weight_caps.cap(c)}
        if over:
            breaches.append(f"over_cap={over}")
        msg = f"[{a}] settled {today}: NAV={v.nav:,.0f} cash={v.cash_pct:.2%} holdings={v.n_holdings}"
        if breaches:
            alert(f"{msg} CLOSE-STATE BREACH {breaches}", level="error")
        out.append(msg)
    return " | ".join(out)
