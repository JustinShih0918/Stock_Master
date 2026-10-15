"""Read-only queries behind the monitor: accounts, NAV history, holdings with cost basis, the
decision timeline (with model, timing and validation), fills and system health."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Engine, func, select

from stockmaster.data import schema
from stockmaster.features.panel import TAIEX
from stockmaster.pipeline.accounts import state_from_json
from stockmaster.retail.book import position_book
from stockmaster.settings import get_settings
from stockmaster.timeutil import TAIPEI

COMPETITION_LOT = 1000


def _iso(x) -> str | None:
    if x is None:
        return None
    if isinstance(x, datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=TAIPEI)
        return x.astimezone(TAIPEI).isoformat(timespec="seconds")
    return x.isoformat() if hasattr(x, "isoformat") else str(x)


def _num(x) -> float | None:
    if x is None:
        return None
    return float(x) if isinstance(x, (Decimal, int, float)) else x


def account_profile(kind: str) -> dict:
    """Lot size and benchmark per account kind (retail rules for the retail account)."""
    if kind == "retail":
        from stockmaster.retail.config import load_retail_rules
        r = load_retail_rules()
        return {"lot_size": r.lot_size, "benchmark": r.benchmark, "benchmark_label": f"{r.benchmark} 買進持有"}
    return {"lot_size": COMPETITION_LOT, "benchmark": TAIEX, "benchmark_label": "加權指數"}


# --- prices --------------------------------------------------------------------------------------

def latest_closes(engine: Engine, codes: list[str], on_or_before: date | None = None) -> dict[str, dict]:
    if not codes:
        return {}
    t = schema.price_daily
    q = select(t.c.code, func.max(t.c.date).label("d")).where(t.c.code.in_(codes)).where(t.c.close.is_not(None))
    if on_or_before:
        q = q.where(t.c.date <= on_or_before)
    sub = q.group_by(t.c.code).subquery()
    with engine.connect() as conn:
        rows = conn.execute(select(t.c.code, t.c.date, t.c.close, t.c.prev_close, t.c.market)
                            .join(sub, (sub.c.code == t.c.code) & (sub.c.d == t.c.date))).all()
    return {r.code: {"date": r.date, "close": r.close, "prev_close": r.prev_close, "market": r.market} for r in rows}


def benchmark_series(engine: Engine, benchmark: str, start: date) -> dict[date, float]:
    """Benchmark closes from the trading day before ``start`` onwards."""
    with engine.connect() as conn:
        if benchmark == TAIEX:
            t = schema.index_daily
            rows = conn.execute(select(t.c.date, t.c.close).where(t.c.index_name == TAIEX)
                                .where(t.c.date >= start - timedelta(days=10)).order_by(t.c.date)).all()
        else:
            t = schema.price_daily
            rows = conn.execute(select(t.c.date, t.c.close).where(t.c.code == benchmark)
                                .where(t.c.date >= start - timedelta(days=10)).where(t.c.close.is_not(None))
                                .order_by(t.c.date)).all()
    return {r.date: float(r.close) for r in rows if r.close is not None}


def instrument_info(engine: Engine, codes: list[str]) -> dict[str, dict]:
    if not codes:
        return {}
    t = schema.instrument
    with engine.connect() as conn:
        return {r.code: {"name": r.name, "industry": r.industry, "market": r.market}
                for r in conn.execute(select(t.c.code, t.c.name, t.c.industry, t.c.market).where(t.c.code.in_(codes)))}


# --- accounts ------------------------------------------------------------------------------------

def _ledger(engine: Engine, account_id: str) -> list:
    t = schema.ledger_day
    with engine.connect() as conn:
        return conn.execute(select(t).where(t.c.account_id == account_id).order_by(t.c.date)).all()


def _base_close(series: dict[date, float], first: date) -> float | None:
    before = [d for d in series if d < first]
    return series[max(before)] if before else None


def overview(engine: Engine) -> list[dict]:
    a = schema.account
    with engine.connect() as conn:
        accounts = conn.execute(select(a).order_by(a.c.kind.desc(), a.c.id)).all()
    out = []
    for acc in accounts:
        prof = account_profile(acc.kind)
        led = _ledger(engine, acc.id)
        start_cash = float(acc.start_cash)
        row = {"id": acc.id, "kind": acc.kind, "start_date": _iso(acc.start_date), "start_cash": start_cash,
               "note": acc.note, "benchmark": prof["benchmark_label"], "last_date": None, "nav": start_cash,
               "total_value": start_cash, "pnl": 0.0, "return": 0.0, "day_change": None, "n_holdings": 0,
               "cash_pct": 1.0, "benchmark_return": None, "last_decision": None}
        if led:
            last, prev = led[-1], (led[-2] if len(led) > 1 else None)
            nav, acc_div = float(last.nav), float(last.accrued_dividends or 0)
            row.update(last_date=_iso(last.date), nav=nav, total_value=nav + acc_div,
                       pnl=nav + acc_div - start_cash, n_holdings=last.n_holdings, cash_pct=last.cash_pct,
                       day_status=last.day_status, warnings=last.warnings)
            row["return"] = row["pnl"] / start_cash
            base_nav = float(prev.nav) if prev else start_cash
            row["day_change"] = nav - base_nav
            bench = benchmark_series(engine, prof["benchmark"], led[0].date)
            b0 = _base_close(bench, led[0].date)
            if b0 and last.date in bench:
                row["benchmark_return"] = bench[last.date] / b0 - 1
        row["last_decision"] = (decisions(engine, acc.id, limit=1) or [None])[0]
        out.append(row)
    return out


def nav_history(engine: Engine, account_id: str) -> dict:
    a = schema.account
    with engine.connect() as conn:
        acc = conn.execute(select(a).where(a.c.id == account_id)).first()
    if acc is None:
        return {"points": []}
    prof = account_profile(acc.kind)
    led = _ledger(engine, account_id)
    start_cash = float(acc.start_cash)
    bench = benchmark_series(engine, prof["benchmark"], led[0].date) if led else {}
    b0 = _base_close(bench, led[0].date) if led else None
    pts = [{"date": _iso(r.date), "nav": float(r.nav), "cash": float(r.cash),
            "return": (float(r.nav) + float(r.accrued_dividends or 0)) / start_cash - 1,
            "benchmark_return": (bench[r.date] / b0 - 1) if b0 and r.date in bench else None,
            "n_holdings": r.n_holdings, "status": r.day_status} for r in led]
    return {"account_id": account_id, "start_cash": start_cash, "start_date": _iso(acc.start_date),
            "benchmark": prof["benchmark_label"], "points": pts}


def holdings(engine: Engine, account_id: str) -> dict:
    a = schema.account
    with engine.connect() as conn:
        acc = conn.execute(select(a).where(a.c.id == account_id)).first()
    if acc is None:
        return {}
    prof = account_profile(acc.kind)
    lot = prof["lot_size"]
    led = _ledger(engine, account_id)
    start_cash = float(acc.start_cash)
    if not led:
        return {"account_id": account_id, "kind": acc.kind, "lot_size": lot, "as_of": None, "cash": start_cash,
                "nav": start_cash, "start_cash": start_cash, "rows": [], "realized": 0.0, "fees": 0.0,
                "accrued_dividends": 0.0, "pending": pending_orders(engine, account_id, None)}
    last = led[-1]
    state = state_from_json(last.state)
    codes = sorted(state.lots)
    closes = latest_closes(engine, codes, last.date)
    info = instrument_info(engine, codes)
    book = position_book(engine, account_id)
    rows = []
    nav = float(last.nav)
    for c in codes:
        n = state.lots[c]
        shares = n * lot
        px = closes.get(c, {})
        close = px.get("close")
        lt = book.lots.get(c)
        value = shares * close if close is not None else None
        cost = lt.cost if lt and lt.shares else None
        rows.append({"code": c, "name": info.get(c, {}).get("name", c), "industry": info.get(c, {}).get("industry"),
                     "market": info.get(c, {}).get("market") or px.get("market"), "shares": shares,
                     "avg_cost": (cost / shares) if cost else None, "cost": cost, "close": close,
                     "close_date": _iso(px.get("date")), "prev_close": px.get("prev_close"), "value": value,
                     "weight": value / nav if value is not None and nav else None,
                     "unrealized": (value - cost) if value is not None and cost else None,
                     "unrealized_pct": (value / cost - 1) if value is not None and cost else None,
                     "opened": _iso(lt.opened) if lt else None})
    return {"account_id": account_id, "kind": acc.kind, "lot_size": lot, "as_of": _iso(last.date),
            "cash": float(last.cash), "nav": nav, "start_cash": start_cash,
            "accrued_dividends": float(last.accrued_dividends or 0), "realized": book.realized, "fees": book.fees,
            "rows": rows, "pending": pending_orders(engine, account_id, last.date)}


def pending_orders(engine: Engine, account_id: str, after: date | None) -> list[dict]:
    """Final decisions for trade dates not yet settled (orders waiting for their fill day)."""
    s, d, r = schema.submission, schema.decision, schema.run
    q = (select(s.c.trade_date, d.c.payload, r.c.finished_at, r.c.model_id, d.c.source)
         .join(d, d.c.id == s.c.decision_id).join(r, r.c.run_id == d.c.run_id)
         .where(r.c.account_id == account_id).where(s.c.is_final.is_(True)))
    if after:
        q = q.where(s.c.trade_date > after)
    with engine.connect() as conn:
        rows = conn.execute(q.order_by(s.c.trade_date)).all()
    return [{"trade_date": _iso(x.trade_date), "decided_at": _iso(x.finished_at), "model": x.model_id,
             "source": x.source, "orders": [{k: o.get(k) for k in ("code", "name", "side", "shares", "ref_price",
                                                                    "est_amount", "rationale")}
                                            for o in x.payload.get("orders", [])]} for x in rows]


# --- decisions -----------------------------------------------------------------------------------

def decisions(engine: Engine, account_id: str, limit: int = 30) -> list[dict]:
    r, d, s = schema.run, schema.decision, schema.submission
    with engine.connect() as conn:
        runs = conn.execute(select(r).where(r.c.account_id == account_id).where(r.c.kind == "decide")
                            .order_by(r.c.started_at.desc()).limit(limit)).all()
        ids = [x.run_id for x in runs]
        finals = {x.run_id: x for x in conn.execute(
            select(d.c.id, d.c.run_id, d.c.source, d.c.payload, d.c.created_at, s.c.is_final, s.c.seq,
                   s.c.artifact_paths).outerjoin(s, s.c.decision_id == d.c.id)
            .where(d.c.run_id.in_(ids)).where(d.c.stage == "final"))} if ids else {}
    out = []
    for x in runs:
        f = finals.get(x.run_id)
        p = f.payload if f else {}
        dur = (x.finished_at - x.started_at).total_seconds() if x.finished_at and x.started_at else None
        out.append({
            "run_id": str(x.run_id), "decision_id": f.id if f else None, "trade_date": _iso(x.trade_date),
            "slot": x.slot, "status": x.status, "started_at": _iso(x.started_at), "finished_at": _iso(x.finished_at),
            "duration_s": dur, "provider": x.provider, "model_id": x.model_id, "prompt_version": x.prompt_version,
            "error": x.error, "source": f.source if f else None, "is_final": bool(f.is_final) if f else False,
            "market_view": p.get("market_view"), "risk_notes": p.get("risk_notes"), "regime": p.get("regime"),
            "risk_level": p.get("risk_level"),
            "orders": [{k: o.get(k) for k in ("code", "name", "side", "shares", "ref_price", "est_amount", "rationale",
                                               "rationale_source")} for o in p.get("orders", [])],
            "targets": p.get("targets", []), "failed_checks": [c for c in p.get("checks", []) if not c.get("passed")],
            "repair_notes": p.get("repair_notes", []), "stale_inputs": p.get("stale_inputs", []),
            "llm_attempts": (p.get("provenance") or {}).get("llm_attempts", []),
            "post_trade": {k: (p.get("post_trade") or {}).get(k) for k in ("est_cost", "cash_ref_pct", "n_holdings",
                                                                           "turnover_one_way")},
            "report": _artifact_rel((f.artifact_paths or {}).get("report_html")) if f else None})
    return out


def _artifact_rel(path: str | None) -> str | None:
    if not path:
        return None
    root = str(get_settings().artifacts_dir)
    return path[len(root):].lstrip("/") if path.startswith(root) else None


def run_detail(engine: Engine, run_id: str) -> dict:
    """Every LLM call (content, reasoning, tokens, latency) and every proposal of one run."""
    import uuid
    rid = uuid.UUID(run_id)
    lc, d = schema.llm_call, schema.decision
    with engine.connect() as conn:
        calls = conn.execute(select(lc).where(lc.c.run_id == rid).order_by(lc.c.seq)).all()
        props = conn.execute(select(d.c.id, d.c.stage, d.c.source, d.c.payload, d.c.created_at)
                             .where(d.c.run_id == rid).order_by(d.c.id)).all()
    return {"calls": [{"seq": c.seq, "provider": c.provider, "model_id": c.model_id, "status": c.status,
                       "started_at": _iso(c.started_at), "latency_ms": c.latency_ms, "prompt_tokens": c.prompt_tokens,
                       "completion_tokens": c.completion_tokens, "error": c.error,
                       "content": ((c.response or {}).get("content") or "")[:20000],
                       "reasoning": (c.reasoning or "")[:20000]} for c in calls],
            "proposals": [{"id": p.id, "stage": p.stage, "source": p.source, "created_at": _iso(p.created_at),
                           "accepted": (p.payload or {}).get("accepted"), "feedback": (p.payload or {}).get("feedback")}
                          for p in props if p.stage == "proposal"]}


def fills(engine: Engine, account_id: str, limit: int = 100) -> list[dict]:
    f = schema.fill
    with engine.connect() as conn:
        rows = conn.execute(select(f).where(f.c.account_id == account_id)
                            .order_by(f.c.date.desc(), f.c.id.desc()).limit(limit)).all()
    info = instrument_info(engine, sorted({r.code for r in rows}))
    return [{"date": _iso(r.date), "code": r.code, "name": info.get(r.code, {}).get("name", r.code), "side": r.side,
             "lots": r.lots, "price": _num(r.price), "gross": _num(r.gross), "commission": _num(r.commission),
             "tax": _num(r.tax), "net_cash": _num(r.net_cash), "status": r.status} for r in rows]


# --- system --------------------------------------------------------------------------------------

DATASETS = {
    "price_daily": schema.price_daily, "inst_flow_daily": schema.inst_flow_daily,
    "margin_daily": schema.margin_daily, "valuation_daily": schema.valuation_daily,
    "index_daily": schema.index_daily, "macro_daily": schema.macro_daily,
}


def system_status(engine: Engine) -> dict:
    out: dict = {"datasets": {}, "sources": [], "runs": [], "alerts": [], "heartbeat": None}
    with engine.connect() as conn:
        for name, t in DATASETS.items():
            last = conn.execute(select(func.max(t.c.date))).scalar()
            n = conn.execute(select(func.count()).select_from(t).where(t.c.date == last)).scalar() if last else 0
            out["datasets"][name] = {"last_date": _iso(last), "rows_last_date": n}
        h = schema.etf_holding
        out["datasets"]["etf_holding"] = {"last_date": _iso(conn.execute(select(func.max(h.c.as_of))).scalar()),
                                          "rows_last_date": None}
        sf = schema.source_fetch
        for r in conn.execute(select(sf.c.source, func.max(sf.c.fetched_at).label("last"), func.count().label("n"))
                              .group_by(sf.c.source).order_by(sf.c.source)):
            out["sources"].append({"source": r.source, "last_fetch": _iso(r.last), "fetches": r.n})
        r_ = schema.run
        for r in conn.execute(select(r_).order_by(r_.c.started_at.desc()).limit(25)):
            out["runs"].append({"run_id": str(r.run_id), "kind": r.kind, "account_id": r.account_id,
                                "trade_date": _iso(r.trade_date), "slot": r.slot, "status": r.status,
                                "started_at": _iso(r.started_at), "finished_at": _iso(r.finished_at),
                                "model_id": r.model_id, "error": r.error})
    s = get_settings()
    alerts = s.data_dir / "alerts.log"
    if alerts.exists():
        lines = alerts.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        out["alerts"] = [json.loads(x) for x in reversed(lines) if x.strip().startswith("{")]
    hb = s.data_dir / "scheduler_heartbeat.json"
    if hb.exists():
        out["heartbeat"] = json.loads(hb.read_text(encoding="utf-8"))
    return out
