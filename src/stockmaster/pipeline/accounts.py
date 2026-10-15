"""Account state persistence, paper settlement after the close, and reconciliation against the
organizer's published NAV (detects voided days)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import Engine, delete, func, select

from stockmaster.accounting.ledger import DayResult, run_day, value
from stockmaster.config import Rules
from stockmaster.data import schema
from stockmaster.data.store import upsert
from stockmaster.domain import Order, PortfolioState, Side
from stockmaster.execution.simulator import vwap_from_value


def state_to_json(s: PortfolioState) -> dict:
    return {"as_of": s.as_of.isoformat() if s.as_of else None, "cash": str(s.cash),
            "lots": s.lots, "accrued_dividends": str(s.accrued_dividends),
            "overweight_days": s.overweight_days, "warnings": s.warnings}


def state_from_json(d: dict) -> PortfolioState:
    return PortfolioState(as_of=date.fromisoformat(d["as_of"]) if d.get("as_of") else None,
                          cash=Decimal(d["cash"]), lots={k: int(v) for k, v in d["lots"].items()},
                          accrued_dividends=Decimal(d.get("accrued_dividends", "0")),
                          overweight_days={k: int(v) for k, v in d.get("overweight_days", {}).items()},
                          warnings=int(d.get("warnings", 0)))


def ensure_account(engine: Engine, account_id: str, kind: str, start_cash: int, start_date: date,
                   note: str = "") -> None:
    upsert(engine, schema.account, [{"id": account_id, "kind": kind, "start_cash": start_cash,
                                     "start_date": start_date, "note": note}], update=False)


@dataclass(frozen=True)
class AccountState:
    state: PortfolioState
    last_date: date | None
    day_status: str | None


def load_state(engine: Engine, account_id: str, rules: Rules, before: date | None = None) -> AccountState:
    t = schema.ledger_day
    q = select(t.c.date, t.c.state, t.c.day_status).where(t.c.account_id == account_id)
    if before is not None:
        q = q.where(t.c.date < before)
    with engine.connect() as conn:
        row = conn.execute(q.order_by(t.c.date.desc()).limit(1)).first()
    if row is None:
        return AccountState(PortfolioState.initial(rules.initial_cash), None, None)
    return AccountState(state_from_json(row.state), row.date, row.day_status)


def final_orders(engine: Engine, account_id: str, trade_date: date) -> list[Order]:
    s, d, r = schema.submission, schema.decision, schema.run
    with engine.connect() as conn:
        row = conn.execute(select(d.c.payload).join(s, s.c.decision_id == d.c.id)
                           .join(r, r.c.run_id == d.c.run_id)
                           .where(r.c.account_id == account_id).where(s.c.trade_date == trade_date)
                           .where(s.c.is_final.is_(True))).first()
    if row is None:
        return []
    return [Order(o["code"], Side(o["side"]), int(o["lots"])) for o in row.payload["orders"]]


def _prices(engine: Engine, on: date, codes: set[str]) -> tuple[dict, dict]:
    t = schema.price_daily
    with engine.connect() as conn:
        rows = conn.execute(select(t.c.code, t.c.close, t.c.value_ntd, t.c.volume_sh)
                            .where(t.c.date == on).where(t.c.code.in_(list(codes)))).all()
    vwaps = {r.code: vwap_from_value(r.value_ntd, r.volume_sh) for r in rows if r.volume_sh}
    closes = {r.code: r.close for r in rows if r.close is not None}
    missing = sorted(set(codes) - set(closes))
    if missing:
        # Suspended / untraded stocks are valued at their last close before ``on``.
        with engine.connect() as conn:
            last = select(t.c.code, func.max(t.c.date).label("d")).where(t.c.code.in_(missing)) \
                .where(t.c.date < on).where(t.c.close.is_not(None)).group_by(t.c.code).subquery()
            for r in conn.execute(select(t.c.code, t.c.close).join(
                    last, (last.c.code == t.c.code) & (last.c.d == t.c.date))).all():
                closes[r.code] = r.close
    return vwaps, closes


def _fillable(orders: list[Order], vwaps: dict) -> tuple[list[Order], list[str]]:
    """Orders whose stock traded on the day (has a VWAP); the rest cannot fill."""
    ok = [o for o in orders if o.code in vwaps]
    return ok, sorted(o.code for o in orders if o.code not in vwaps)


def _dividends(engine: Engine, on: date, after: date | None = None) -> dict[str, float]:
    """Per-share dividends going ex in (after, on]; ex-dates on market closures (typhoon days) are
    credited on the next trading day. ``after`` = the previous valuation date (None: only ``on``)."""
    t = schema.corp_action
    q = select(t.c.code, t.c.cash_dividend).where(t.c.ex_date <= on)
    q = q.where(t.c.ex_date > after) if after else q.where(t.c.ex_date == on)
    with engine.connect() as conn:
        rows = conn.execute(q).all()
    out: dict[str, float] = {}
    for r in rows:
        if r.cash_dividend:
            out[r.code] = out.get(r.code, 0.0) + float(r.cash_dividend)
    return out


def save_day(engine: Engine, account_id: str, res: DayResult, day_status: str,
             organizer_nav: Decimal | None = None, active_share: tuple[float, str] | None = None,
             lot_size: int = 1000) -> None:
    v, s = res.valuation, res.state
    with engine.begin() as conn:
        conn.execute(delete(schema.position_day).where(schema.position_day.c.account_id == account_id)
                     .where(schema.position_day.c.date == v.date))
        conn.execute(delete(schema.ledger_day).where(schema.ledger_day.c.account_id == account_id)
                     .where(schema.ledger_day.c.date == v.date))
        conn.execute(delete(schema.fill).where(schema.fill.c.account_id == account_id)
                     .where(schema.fill.c.date == v.date))
        conn.execute(schema.ledger_day.insert().values(
            account_id=account_id, date=v.date, cash=v.cash, holdings_value=v.holdings_value,
            nav=v.nav, accrued_dividends=v.accrued_dividends, cash_pct=v.cash_pct,
            n_holdings=v.n_holdings, day_status=day_status, warnings=s.warnings,
            organizer_nav=organizer_nav, nav_diff=(organizer_nav - v.nav) if organizer_nav else None,
            active_share_min=active_share[0] if active_share else None,
            active_share_etf=active_share[1] if active_share else None, state=state_to_json(s)))
        for code, n in s.lots.items():
            w = v.weights.get(code, 0.0)
            position_value = Decimal(str(w)) * v.nav
            conn.execute(schema.position_day.insert().values(
                account_id=account_id, date=v.date, code=code, lots=n,
                close=float(position_value / (n * lot_size)), value=position_value, weight=w,
                overweight_days=s.overweight_days.get(code, 0)))
        order_ids = {}
        if res.fills:
            o = schema.order_
            rows = conn.execute(select(o.c.id, o.c.code).where(o.c.account_id == account_id)
                                .where(o.c.trade_date == v.date).order_by(o.c.id.desc())).all()
            for r in rows:
                order_ids.setdefault(r.code, r.id)
        for f in res.fills:
            if f.code not in order_ids:
                continue
            conn.execute(schema.fill.insert().values(
                order_id=order_ids[f.code], account_id=account_id, date=v.date, code=f.code,
                side=f.side.value, lots=f.lots, price=f.price, gross=f.gross,
                commission=f.commission, tax=f.tax, net_cash=f.net_cash,
                status="applied" if res.applied else "voided"))


def settle(engine: Engine, account_id: str, trade_date: date, rules: Rules,
           applied: bool = True, warning: bool = False) -> DayResult:
    """Apply the day's final orders at the day's VWAP and value at the close."""
    prev = load_state(engine, account_id, rules, before=trade_date)
    orders = final_orders(engine, account_id, trade_date)
    codes = set(prev.state.lots) | {o.code for o in orders}
    vwaps, closes = _prices(engine, trade_date, codes)
    orders, unfilled = _fillable(orders, vwaps)
    if unfilled:
        from stockmaster.ops.alerts import alert
        alert(f"[{account_id}] {trade_date}: no VWAP (stock did not trade) for {unfilled}; orders not filled",
              level="error")
    res = run_day(prev.state, trade_date, orders, vwaps, closes, rules, applied=applied,
                  cash_dividends_per_share=_dividends(engine, trade_date, prev.last_date), warning=warning)
    save_day(engine, account_id, res, "applied" if applied else "voided", lot_size=rules.lot_size)
    return res


def reconcile(engine: Engine, account_id: str, trade_date: date, organizer_nav: Decimal,
              rules: Rules, tolerance: Decimal | None = None) -> str:
    """Compare the organizer's NAV with our applied and voided versions of the day.
    Returns 'applied', 'voided' (state rewritten, warning added) or 'unreconciled'."""
    prev = load_state(engine, account_id, rules, before=trade_date)
    orders = final_orders(engine, account_id, trade_date)
    codes = set(prev.state.lots) | {o.code for o in orders}
    vwaps, closes = _prices(engine, trade_date, codes)
    orders, _ = _fillable(orders, vwaps)
    divs = _dividends(engine, trade_date, prev.last_date)
    applied = run_day(prev.state, trade_date, orders, vwaps, closes, rules, cash_dividends_per_share=divs)
    voided = run_day(prev.state, trade_date, orders, vwaps, closes, rules, applied=False,
                     cash_dividends_per_share=divs, warning=True)
    tol = tolerance if tolerance is not None else Decimal(20000) + applied.valuation.nav * Decimal("0.00001")
    if abs(applied.valuation.nav - organizer_nav) <= tol:
        save_day(engine, account_id, applied, "applied", organizer_nav, lot_size=rules.lot_size)
        return "applied"
    if orders and abs(voided.valuation.nav - organizer_nav) <= tol:
        save_day(engine, account_id, voided, "voided", organizer_nav, lot_size=rules.lot_size)
        return "voided"
    save_day(engine, account_id, applied, "unreconciled", organizer_nav, lot_size=rules.lot_size)
    return "unreconciled"


def safe_state(engine: Engine, account_id: str, on: date, rules: Rules) -> PortfolioState:
    """When ``on`` is unreconciled we do not know whether its trades were applied. Decide from a
    state that is safe under both outcomes: the smaller lot count per stock and the lower cash."""
    prev = load_state(engine, account_id, rules, before=on)
    orders = final_orders(engine, account_id, on)
    codes = set(prev.state.lots) | {o.code for o in orders}
    vwaps, closes = _prices(engine, on, codes)
    orders, _ = _fillable(orders, vwaps)
    a = run_day(prev.state, on, orders, vwaps, closes, rules).state
    v = run_day(prev.state, on, orders, vwaps, closes, rules, applied=False).state
    lots = {c: min(a.lots.get(c, 0), v.lots.get(c, 0)) for c in set(a.lots) | set(v.lots)}
    return a.with_(lots={c: n for c, n in lots.items() if n > 0}, cash=min(a.cash, v.cash),
                   warnings=max(a.warnings, v.warnings + 1))


def nav_on(engine: Engine, account_id: str, on: date, rules: Rules) -> Decimal | None:
    st = load_state(engine, account_id, rules, before=None)
    if st.last_date != on:
        return None
    _, closes = _prices(engine, on, set(st.state.lots))
    return value(st.state, closes, on).nav
