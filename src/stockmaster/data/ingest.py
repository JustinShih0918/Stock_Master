"""Idempotent ingestion of exchange data into Postgres, with a ``source_fetch`` record per payload.

A trading day is ingested in parts because TWSE/TPEx publish datasets at different times (quotes
around 14:30-15:00, institutional flows ~16:30, margin ~21:30). Every part can be re-run; rows are
upserted on their primary key.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sqlalchemy import Engine, func, select

from stockmaster.data import schema
from stockmaster.data.http import Fetcher, record_fetch
from stockmaster.data.sources import isin, mops, overnight, tpex, twse
from stockmaster.data.store import upsert
from stockmaster.timeutil import now_taipei, today_taipei

log = logging.getLogger(__name__)

DAILY_PARTS = ["twse_quotes", "tpex_quotes", "twse_t86", "tpex_t86", "twse_margin", "tpex_margin",
               "twse_valuation", "tpex_valuation"]
PRICE_COLS = ["code", "date", "market", "open", "high", "low", "close", "prev_close", "volume_sh",
              "value_ntd", "trades", "vwap", "vwap_published", "next_ref_price", "next_limit_up",
              "next_limit_down", "halted", "source_fetch_id"]

# part -> (exchange module, report kind, parser name, target table)
PARTS: dict[str, tuple] = {
    "twse_quotes": (twse, "quotes", "parse_quotes", schema.price_daily),
    "tpex_quotes": (tpex, "quotes", "parse_quotes", schema.price_daily),
    "twse_t86": (twse, "t86", "parse_institutional", schema.inst_flow_daily),
    "tpex_t86": (tpex, "t86", "parse_institutional", schema.inst_flow_daily),
    "twse_margin": (twse, "margin", "parse_margin", schema.margin_daily),
    "tpex_margin": (tpex, "margin", "parse_margin", schema.margin_daily),
    "twse_valuation": (twse, "valuation", "parse_valuation", schema.valuation_daily),
    "tpex_valuation": (tpex, "valuation", "parse_valuation", schema.valuation_daily),
}


@dataclass
class IngestReport:
    day: date
    rows: dict[str, int] = field(default_factory=dict)
    empty: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"{k}={v}" for k, v in self.rows.items()]
        if self.empty:
            parts.append(f"empty={','.join(self.empty)}")
        if self.errors:
            parts.append("errors=" + "; ".join(f"{k}: {v}" for k, v in self.errors.items()))
        return f"{self.day} " + " ".join(parts)


def records(df: pd.DataFrame, cols: Sequence[str]) -> list[dict]:
    """DataFrame -> list of dicts with plain Python values (NaN -> None)."""
    out = []
    for row in df.reindex(columns=list(cols)).itertuples(index=False, name=None):
        rec = {}
        for c, v in zip(cols, row):
            if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
                rec[c] = None
            elif hasattr(v, "item"):  # numpy scalar
                rec[c] = v.item()
            else:
                rec[c] = v
        out.append(rec)
    return out


def _market(module) -> str:
    return "TWSE" if module is twse else "TPEx"


def ingest_part(engine: Engine, fetcher: Fetcher, d: date, part: str) -> int:
    """Fetch, parse and store one dataset for one day. Returns rows stored (0 = not published)."""
    module, kind, parser, table = PARTS[part]
    res = fetcher.get(module.__name__.rsplit(".", 1)[-1], module.URLS[kind],
                      params=module.params(kind, d), for_date=d, name=f"{part}_{d:%Y%m%d}")
    payload = res.json()
    if not module.has_data(payload):
        record_fetch(engine, res, row_count=0, note="no data published")
        return 0
    df = getattr(module, parser)(payload, d)
    fid = record_fetch(engine, res, row_count=len(df))
    df["source_fetch_id"] = fid
    if part.endswith("_quotes"):
        _store_instruments(engine, df[["code", "name", "market"]])
        n = upsert(engine, table, records(df, PRICE_COLS))
        if module is twse:
            idx, breadth = twse.parse_index_and_breadth(payload, d)
            idx["source_fetch_id"] = fid
            upsert(engine, schema.index_daily, records(idx, ["index_name", "date", "close", "change_pct",
                                                             "source_fetch_id"]))
            if breadth:
                upsert(engine, schema.breadth_daily, [{"market": "TWSE", "date": d, "payload": breadth,
                                                       "source_fetch_id": fid}])
        return n
    cols = [c.name for c in table.columns]
    return upsert(engine, table, records(df, cols))


def _store_instruments(engine: Engine, df: pd.DataFrame) -> None:
    """New codes seen in quotes enter the instrument master (names/industry refined by sync)."""
    rows = [{"code": r.code, "name": r.name or r.code, "market": r.market}
            for r in df.drop_duplicates("code").itertuples()]
    upsert(engine, schema.instrument, rows, update=False)


def ingest_day(engine: Engine, fetcher: Fetcher, d: date, parts: Sequence[str] | None = None) -> IngestReport:
    rep = IngestReport(d)
    for part in parts or DAILY_PARTS:
        if part not in PARTS:
            rep.errors[part] = "unknown part"
            continue
        try:
            n = ingest_part(engine, fetcher, d, part)
        except Exception as e:  # noqa: BLE001 - one failed dataset must not stop the others
            log.exception("ingest %s %s failed", part, d)
            rep.errors[part] = f"{type(e).__name__}: {e}"[:300]
            continue
        rep.rows[part] = n
        if n == 0:
            rep.empty.append(part)
    return rep


def missing_parts(engine: Engine, d: date, parts: Sequence[str]) -> list[str]:
    """Parts with no stored rows for ``d`` (per market)."""
    out = []
    p = schema.price_daily
    with engine.connect() as conn:
        for part in parts:
            module, _, _, table = PARTS[part]
            market = _market(module)
            codes = select(p.c.code).where(p.c.date == d).where(p.c.market == market)
            q = select(func.count()).select_from(table).where(table.c.date == d)
            q = q.where(table.c.market == market) if table is p else q.where(table.c.code.in_(codes))
            if not conn.execute(q).scalar():
                out.append(part)
    return out


def mark_closed_if_no_data(engine: Engine, rep: IngestReport) -> bool:
    """A past weekday on which neither exchange published quotes was a market closure (e.g. a
    typhoon day missing from the holiday schedule): record it as a calendar override."""
    quotes = {"twse_quotes", "tpex_quotes"}
    if rep.day >= today_taipei() or rep.errors or not quotes <= set(rep.empty):
        return False
    upsert(engine, schema.calendar_override, [{"date": rep.day, "is_open": False,
                                               "reason": "no quotes published by TWSE or TPEx"}], update=False)
    return True


# --- reference data ------------------------------------------------------------------------------

def sync_calendar(engine: Engine, fetcher: Fetcher) -> int:
    res = fetcher.get("twse", twse.URLS["holidays"], name="holidaySchedule")
    h = twse.parse_holidays(res.json())
    fid = record_fetch(engine, res, row_count=len(h))
    rows = [{"date": r.date, "is_open": twse.is_trading_notice(r.name), "name": r.name,
             "source": f"twse_openapi:{fid}"} for r in h.itertuples()]
    return upsert(engine, schema.trading_calendar, rows)


def sync_instruments(engine: Engine, fetcher: Fetcher) -> dict[str, int]:
    """Instrument master from the ISIN registry: every listed stock, ETF, preferred share and TDR
    on TWSE and TPEx (warrants excluded), with industry and listing date."""
    from stockmaster.data.sources._parse import is_stock_code
    keep_types = {"股票", "ETF", "特別股", "臺灣存託憑證(TDR)", "ETN"}
    out = {}
    for market, mode in isin.MODES.items():
        res = fetcher.get("isin", isin.MODE_URL, params={"strMode": mode}, name=f"isin_mode{mode}")
        rows = [r for r in isin.parse_mode_html(res.content)
                if r["market"] == market and r["type"] in keep_types and is_stock_code(r["code"])]
        record_fetch(engine, res, row_count=len(rows))
        recs = []
        for r in rows:
            try:
                listed = pd.Timestamp(r["listed"].replace("/", "-")).date() if r["listed"] else None
            except ValueError:
                listed = None
            recs.append({"code": r["code"], "name": r["name"], "market": market,
                         "industry": "ETF" if r["type"] in ("ETF", "ETN") else (r["industry"] or r["type"]),
                         "listed_date": listed})
        out[market] = upsert(engine, schema.instrument, recs)
    return out


def sync_universe(engine: Engine) -> int:
    """The competition's fixed 150-stock list."""
    from stockmaster.config import load_rules, load_universe
    rules, universe = load_rules(), load_universe()
    upsert(engine, schema.instrument, [{"code": i.code, "name": i.name, "market": i.market}
                                       for i in universe.instruments], update=False)
    return upsert(engine, schema.universe_member, [{"universe_id": rules.competition_id, "code": c,
                                                    "list_sha256": universe.sha256} for c in universe.codes])


def sync_active_etfs(engine: Engine) -> int:
    from stockmaster.data.etf_ingest import sync_etf_list
    return sync_etf_list(engine)


def sync_reference(engine: Engine, fetcher: Fetcher) -> dict:
    """Calendar, universe, instrument master and the active-ETF list. Each step is independent."""
    out: dict = {}
    steps: dict[str, Callable[[], object]] = {
        "calendar": lambda: sync_calendar(engine, fetcher),
        "universe": lambda: sync_universe(engine),
        "instruments": lambda: sync_instruments(engine, fetcher),
        "active_etfs": lambda: sync_active_etfs(engine),
    }
    for name, step in steps.items():
        try:
            out[name] = step()
        except Exception as e:  # noqa: BLE001
            log.exception("sync %s failed", name)
            out[name] = f"error: {type(e).__name__}: {e}"[:300]
    return out


def ingest_ex_rights(engine: Engine, fetcher: Fetcher) -> int:
    """Upcoming ex-dividend / ex-rights dates (TWSE forecast table). Announced cash dividends
    replace earlier 'to be announced' rows on later runs."""
    res = fetcher.get("twse", twse.URLS["ex_rights"], params=twse.params("ex_rights"), name="TWT48U")
    df = twse.parse_ex_rights_calendar(res.json())
    df["source_fetch_id"] = record_fetch(engine, res, row_count=len(df))
    return upsert(engine, schema.corp_action, records(df, ["code", "ex_date", "kind", "cash_dividend",
                                                           "stock_dividend_ratio", "source_fetch_id"]))


def ingest_monthly_revenue(engine: Engine, fetcher: Fetcher, year: int, month: int) -> dict[str, int]:
    out = {}
    for market in ("TWSE", "TPEx"):
        try:
            res = fetcher.get("mops", mops.url(market, year, month), name=f"t21sc03_{market}_{year}_{month}")
        except Exception as e:  # noqa: BLE001 - the page does not exist before the first filings
            out[market] = f"error: {type(e).__name__}"
            continue
        df = mops.parse_monthly_revenue(res.text("big5"), market, year, month)
        df["source_fetch_id"] = record_fetch(engine, res, row_count=len(df))
        df["first_seen_at"] = now_taipei()
        df["source"] = "mops"
        cols = [c.name for c in schema.monthly_revenue.columns]
        out[market] = upsert(engine, schema.monthly_revenue, records(df, cols), keep=("first_seen_at",))
    return out


def ingest_overnight(engine: Engine, fetcher: Fetcher) -> dict[str, int]:
    out = {}
    for sym in overnight.SYMBOLS:
        try:
            res = fetcher.get("yahoo", overnight.URL.format(symbol=sym), params=overnight.params(),
                              name=f"chart_{sym}")
            s = overnight.parse_chart(res.json())
            fid = record_fetch(engine, res, row_count=len(s), note="unofficial source")
            out[sym] = upsert(engine, schema.macro_daily, [
                {"series_id": f"yahoo:{sym}", "date": d, "value": float(v), "source_fetch_id": fid}
                for d, v in s.items()])
        except Exception as e:  # noqa: BLE001
            log.warning("overnight %s failed: %s", sym, e)
            out[sym] = 0
    return out
