"""Monitor web service: a read-only dashboard over the database (P&L, holdings, decisions, data
health). Run with ``sm monitor``; it never writes to the database and never places orders."""

from __future__ import annotations

import time
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from stockmaster.data.reference import load_calendar
from stockmaster.data.store import get_engine
from stockmaster.monitor import queries as Q
from stockmaster.monitor.live import LiveQuotes, market_session
from stockmaster.settings import get_settings
from stockmaster.timeutil import now_taipei

app = FastAPI(title="Stock Master monitor", docs_url="/api/docs", redoc_url=None)
_live = LiveQuotes()
_cal: dict = {"at": 0.0, "cal": None}


def _calendar():
    if _cal["cal"] is None or time.monotonic() - _cal["at"] > 600:
        _cal["cal"], _cal["at"] = load_calendar(get_engine()), time.monotonic()
    return _cal["cal"]


def _session() -> dict:
    now = now_taipei()
    cal = _calendar()
    trading = cal.is_open(now.date())
    return {"now": now.isoformat(timespec="seconds"), "trading_day": trading,
            "session": market_session(now, trading), "next_trading_day": cal.next_open(now.date()).isoformat()}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (resources.files("stockmaster.monitor") / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    with get_engine().connect() as conn:
        conn.exec_driver_sql("select 1")
    return {"ok": True, **_session()}


@app.get("/api/overview")
def overview() -> dict:
    return {"session": _session(), "accounts": Q.overview(get_engine())}


@app.get("/api/account/{account_id}")
def account(account_id: str) -> dict:
    h = Q.holdings(get_engine(), account_id)
    if not h:
        raise HTTPException(404, f"unknown account {account_id}")
    s = _session()
    live = _live.get({r["code"]: r.get("market") or "TWSE" for r in h["rows"]}, s["trading_day"])
    live_nav, day_pnl, any_live = h["cash"], 0.0, False
    for r in h["rows"]:
        q = live.get(r["code"])
        r["live"] = q
        if q:
            any_live = True
            r["live_value"] = r["shares"] * q["price"]
            if r.get("close") is not None:
                day_pnl += r["shares"] * (q["price"] - r["close"])
            if r.get("cost"):
                r["live_unrealized"] = r["live_value"] - r["cost"]
                r["live_unrealized_pct"] = r["live_value"] / r["cost"] - 1
        live_nav += r.get("live_value") or r.get("value") or 0.0
    h["live"] = {"available": any_live, "nav": live_nav if any_live else None,
                 "day_pnl": day_pnl if any_live else None}
    h["session"] = s
    return h


@app.get("/api/account/{account_id}/nav")
def nav(account_id: str) -> dict:
    return Q.nav_history(get_engine(), account_id)


@app.get("/api/account/{account_id}/decisions")
def decisions(account_id: str, limit: int = 30) -> list[dict]:
    return Q.decisions(get_engine(), account_id, limit=min(limit, 200))


@app.get("/api/run/{run_id}")
def run(run_id: str) -> dict:
    try:
        return Q.run_detail(get_engine(), run_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/account/{account_id}/fills")
def fills(account_id: str, limit: int = 100) -> list[dict]:
    return Q.fills(get_engine(), account_id, limit=min(limit, 1000))


@app.get("/api/system")
def system() -> dict:
    return {"session": _session(), **Q.system_status(get_engine())}


@app.get("/artifacts/{path:path}")
def artifact(path: str):
    root = get_settings().artifacts_dir.resolve()
    target = (root / path).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.exception_handler(Exception)
def _errors(_, exc: Exception):  # surface DB/config problems in the UI instead of a bare 500
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"[:500]})


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


def static_dir() -> Path:
    return Path(str(resources.files("stockmaster.monitor") / "static"))
