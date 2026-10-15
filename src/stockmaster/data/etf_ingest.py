"""Active-ETF list and daily holdings snapshots into ``etf`` / ``etf_holding``.

The reference list for Active Share comes from config/active_etf_reference.csv when present (the
organizer's official list); otherwise it is derived: domestic equity active ETFs listed on or
before rules.active_share.reference_list_date.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

from sqlalchemy import Engine, select

from stockmaster.data import schema
from stockmaster.data.sources import active_etf as A
from stockmaster.data.sources.active_etf_issuers import _http
from stockmaster.data.store import upsert
from stockmaster.settings import get_settings

log = logging.getLogger(__name__)
REFERENCE_FILE = "active_etf_reference.csv"


def configured_reference_codes() -> set[str] | None:
    path: Path = get_settings().config_dir / REFERENCE_FILE
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return {r["code"].strip().upper() for r in csv.DictReader(f) if r.get("code")}


def sync_etf_list(engine: Engine) -> dict:
    etfs = A.list_active_etfs()
    configured = configured_reference_codes()
    refs = configured if configured is not None else {e.code for e in A.reference_etfs(etfs)}
    rows = [{"code": e.code, "name": e.name, "issuer": e.issuer, "market": e.market,
             "asset_class": e.asset_class, "listed_date": e.listed_date, "in_reference_list": e.code in refs}
            for e in etfs]
    known = {e.code for e in etfs}
    # Reference ETFs missing from the ISIN pages (delisted, renamed) still get a row.
    rows += [{"code": c, "name": c, "issuer": None, "market": None, "asset_class": "tw_equity",
              "listed_date": None, "in_reference_list": True} for c in sorted(refs - known)]
    upsert(engine, schema.etf, rows)
    return {"listed": len(etfs), "reference": len(refs),
            "reference_source": "config" if configured is not None else "derived"}


def _reference_codes(engine: Engine) -> list[str]:
    e = schema.etf
    with engine.connect() as conn:
        codes = [r.code for r in conn.execute(select(e.c.code).where(e.c.in_reference_list.is_(True)))]
    return sorted(codes or (configured_reference_codes() or set()))


def _archive(snap) -> str:
    d = get_settings().data_dir / "raw" / "active_etf" / snap.as_of.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(snap.raw).hexdigest()
    ext = "json" if "json" in snap.raw_content_type else "html"
    path = d / f"{snap.etf_code}_{sha[:12]}.{ext}"
    if not path.exists():
        path.write_bytes(snap.raw)
    return str(path)


def snapshot_all(engine: Engine, codes: list[str] | None = None) -> dict:
    """Fetch and store today's holdings of every reference active ETF. One failure does not stop
    the rest; failures are returned (and stale snapshots are flagged at decision time)."""
    codes = codes or _reference_codes(engine)
    ok, failed = 0, {}
    with _http.client() as client:
        for code in codes:
            try:
                snap = A.fetch_holdings(code, client=client)
            except A.HoldingsUnavailable as e:
                failed[code] = e.reason[:200]
                continue
            weights = A.holdings_weights(snap)
            names = {h.code: h.name for h in snap.holdings}
            shares: dict[str, int] = {}
            for h in snap.holdings:
                if h.code in weights and h.shares is not None:
                    shares[h.code] = shares.get(h.code, 0) + h.shares
            sha = hashlib.sha256(snap.raw).hexdigest()
            with engine.begin() as conn:
                fid = conn.execute(schema.source_fetch.insert().values(
                    source="active_etf", endpoint=_http.redact(snap.source_url).split("?", 1)[0],
                    params={"etf": code}, for_date=snap.as_of, fetched_at=snap.fetched_at, http_status=200,
                    content_sha256=sha, raw_path=_archive(snap), row_count=len(weights),
                    note="aggregator fallback (unofficial)" if "etfinfo" in snap.source_url else None)
                    .returning(schema.source_fetch.c.id)).scalar_one()
            upsert(engine, schema.etf_holding, [
                {"etf_code": code, "as_of": snap.as_of, "code": c, "name": names.get(c), "weight": w,
                 "shares": shares.get(c), "source_fetch_id": fid} for c, w in weights.items()])
            ok += 1
    if failed:
        log.warning("active ETF snapshots failed: %s", failed)
    return {"ok": ok, "failed": failed}
