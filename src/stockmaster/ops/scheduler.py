"""Unattended job schedule (Asia/Taipei). Data jobs are idempotent and re-run several times per
evening because TWSE/TPEx publish different datasets at different times."""

from __future__ import annotations

import logging
from collections.abc import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from stockmaster.data import ingest
from stockmaster.data.http import Fetcher
from stockmaster.data.reference import load_calendar
from stockmaster.data.store import get_engine
from stockmaster.timeutil import TAIPEI, today_taipei

log = logging.getLogger("stockmaster.scheduler")


def _guarded(name: str, fn: Callable[[], object]) -> Callable[[], None]:
    def job() -> None:
        try:
            log.info("job %s start", name)
            result = fn()
            log.info("job %s done: %s", name, result)
        except Exception:  # noqa: BLE001 - a failed job must not kill the scheduler
            log.exception("job %s failed", name)
    return job


def trading_day_ingest() -> str:
    engine, today = get_engine(), today_taipei()
    if not load_calendar(engine).is_open(today):
        return f"{today} is not a trading day"
    return str(ingest.ingest_day(engine, Fetcher(), today))


def reference_refresh() -> str:
    engine, fetcher = get_engine(), Fetcher()
    ref = ingest.sync_reference(engine, fetcher)
    ex = ingest.ingest_ex_rights(engine, fetcher)
    today = today_taipei()
    months = [(today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)]
    if today.day <= 12:  # previous month's revenue is still being published
        prev = months[0]
        months.append((prev[0], prev[1] - 1) if prev[1] > 1 else (prev[0] - 1, 12))
    rev = {f"{y}-{m:02d}": ingest.ingest_monthly_revenue(engine, fetcher, y, m) for y, m in months}
    return f"reference={ref} ex_rights={ex} revenue={rev}"


def overnight_refresh() -> str:
    return str(ingest.ingest_overnight(get_engine(), Fetcher()))


def active_etf_snapshot() -> str:
    from stockmaster.data.etf_ingest import snapshot_all
    return str(snapshot_all(get_engine()))


def _job(name: str) -> Callable[[], object]:
    def call() -> object:
        from stockmaster.ops import jobs
        return getattr(jobs, name)()
    return call


def build_scheduler(extra_jobs: dict[str, tuple[CronTrigger, Callable[[], object]]] | None = None
                    ) -> BlockingScheduler:
    sched = BlockingScheduler(timezone=TAIPEI)
    jobs: dict[str, tuple[CronTrigger, Callable[[], object]]] = {
        "ingest_1500": (CronTrigger(hour=15, minute=0, timezone=TAIPEI), trading_day_ingest),
        "ingest_1730": (CronTrigger(hour=17, minute=30, timezone=TAIPEI), trading_day_ingest),
        "ingest_2140": (CronTrigger(hour=21, minute=40, timezone=TAIPEI), trading_day_ingest),
        "reference_1900": (CronTrigger(hour=19, minute=0, timezone=TAIPEI), reference_refresh),
        "etf_1830": (CronTrigger(hour=18, minute=30, timezone=TAIPEI), active_etf_snapshot),
        "etf_2110": (CronTrigger(hour=21, minute=10, timezone=TAIPEI), active_etf_snapshot),
        "etf_0700": (CronTrigger(hour=7, minute=0, timezone=TAIPEI), active_etf_snapshot),
        "overnight_0645": (CronTrigger(hour=6, minute=45, timezone=TAIPEI), overnight_refresh),
        "overnight_2130": (CronTrigger(hour=21, minute=30, timezone=TAIPEI), overnight_refresh),
        "settle_1745": (CronTrigger(hour=17, minute=45, timezone=TAIPEI), _job("settle_today")),
        "run_a_2150": (CronTrigger(hour=21, minute=50, timezone=TAIPEI), _job("run_a")),
        "run_b_0715": (CronTrigger(hour=7, minute=15, timezone=TAIPEI), _job("run_b")),
        "watchdog_0830": (CronTrigger(hour=8, minute=30, timezone=TAIPEI), _job("watchdog")),
    }
    jobs.update(extra_jobs or {})
    for name, (trigger, fn) in jobs.items():
        sched.add_job(_guarded(name, fn), trigger, id=name, name=name, max_instances=1,
                      coalesce=True, misfire_grace_time=1800)
    return sched
