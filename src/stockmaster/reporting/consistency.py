"""Audit that the decision report and the trade book describe exactly the same orders.
An inconsistency found by the organizer's spot check is a rule violation (六(十四)(6))."""

from __future__ import annotations

import csv
import io
import re

from stockmaster.agent.schemas import FinalDecision

_ROW = re.compile(r"^\|\s*(\d{4}[A-Z]?)\s*\|[^|]*\|\s*(買進|賣出)\s*\|\s*(\d+)\s*\|", re.M)


def report_orders(report_md: str) -> set[tuple[str, str, int]]:
    section = report_md.split("## 三、交易明細", 1)[-1].split("## 四", 1)[0]
    return {(c, "buy" if s == "買進" else "sell", int(n)) for c, s, n in _ROW.findall(section)}


def trade_book_orders(csv_text: str) -> set[tuple[str, str, int]]:
    rows = csv.DictReader(io.StringIO(csv_text.lstrip("﻿")))
    return {(r["證券代號"], "buy" if r["買賣別"] == "買進" else "sell", int(r["張數"])) for r in rows}


def check_consistency(f: FinalDecision, report_md: str, trade_book_csv: str) -> list[str]:
    truth = {(o.code, o.side, o.lots) for o in f.orders}
    problems = []
    rep, book = report_orders(report_md), trade_book_orders(trade_book_csv)
    if rep != truth:
        problems.append(f"report orders differ from decision: {sorted(rep ^ truth)}")
    if book != truth:
        problems.append(f"trade book differs from decision: {sorted(book ^ truth)}")
    if f.sha256() not in report_md:
        problems.append("report does not carry the decision sha256")
    return problems
