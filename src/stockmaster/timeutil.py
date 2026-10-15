"""Asia/Taipei time helpers. Exchange dates are Taipei calendar dates; timestamps are tz-aware."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")


def now_taipei() -> datetime:
    return datetime.now(TAIPEI)


def today_taipei() -> date:
    return now_taipei().date()


def at_taipei(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=TAIPEI)


def parse_roc_date(s: str) -> date:
    """Parse ROC-calendar dates used by TWSE/TPEx: '1151001', '115/10/01' or '115-10-01'."""
    s = s.strip().replace("/", "").replace("-", "")
    if len(s) == 7:
        return date(int(s[:3]) + 1911, int(s[3:5]), int(s[5:7]))
    if len(s) == 8:  # already Gregorian YYYYMMDD
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f"unrecognised ROC date: {s!r}")


def to_roc(d: date, sep: str = "/") -> str:
    return f"{d.year - 1911}{sep}{d.month:02d}{sep}{d.day:02d}"
