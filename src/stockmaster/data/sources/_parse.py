"""Shared helpers for exchange payloads: numbers with thousands separators, HTML-wrapped signs,
ROC dates."""

from __future__ import annotations

import re
from datetime import date

_TAG = re.compile(r"<[^>]+>")
_MISSING = {"", "-", "--", "---", "----", "x", "X", "N/A", "n/a", "除權息", "除息", "除權"}


def strip_html(s) -> str:
    return _TAG.sub("", str(s or "")).replace("&nbsp;", " ").strip()


def num(s) -> float | None:
    """'1,234.50' -> 1234.5; placeholders ('--', 'X', '') -> None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = strip_html(s).replace(",", "").replace("+", "").strip()
    if t in _MISSING:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def integer(s) -> int | None:
    v = num(s)
    return None if v is None else int(round(v))


def sign_of(html) -> int | None:
    """TWSE '<p style= color:red>+</p>' -> +1, '-' -> -1, 'X' (no comparison) -> None, blank -> 0."""
    t = strip_html(html)
    if "+" in t:
        return 1
    if "-" in t:
        return -1
    if "X" in t.upper():
        return None
    return 0


def roc_date(s: str) -> date:
    """'115年10月08日', '115/10/08', '1151008', '2026/10/08' -> date."""
    t = strip_html(s)
    m = re.match(r"^\s*(\d{2,4})\D+(\d{1,2})\D+(\d{1,2})", t)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    else:
        digits = re.sub(r"\D", "", t)
        if len(digits) == 7:
            y, mo, d = int(digits[:3]), int(digits[3:5]), int(digits[5:7])
        elif len(digits) == 8:
            y, mo, d = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
        else:
            raise ValueError(f"unrecognised date {s!r}")
    if y < 1911:
        y += 1911
    return date(y, mo, d)


def field_index(fields: list[str], *names: str) -> int | None:
    """Index of the first field equal to one of ``names`` (after stripping HTML/whitespace)."""
    clean = [strip_html(f).replace(" ", "") for f in fields]
    for n in names:
        n = n.replace(" ", "")
        if n in clean:
            return clean.index(n)
    return None


_SECURITY = re.compile(
    r"\d{4}[A-Z]?"        # common stock (2330), preferred / special share (2881A)
    r"|00\d{2,4}[A-Z]?"   # ETF / ETN (0050, 00878, 006208, 00679B, 00981A)
    r"|91\d{4}")          # TDR (910322)


def is_stock_code(code: str) -> bool:
    """Listed securities we keep: stocks, ETFs/ETNs, preferred shares and TDRs. Warrants and
    callable bull/bear certificates (other six-digit codes, e.g. 03xxxx-08xxxx, 7xxxxx) are dropped."""
    return bool(_SECURITY.fullmatch(code))
