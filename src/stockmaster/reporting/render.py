"""Render the 決策報告 (decision report) and 交易書 (trade book) from one FinalDecision.

The official templates from the organizer may differ; the layout lives in templates/ so it can be
swapped without touching the decision logic. Both artifacts embed the FinalDecision sha256.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import jinja2

from stockmaster.agent.schemas import FinalDecision
from stockmaster.timeutil import now_taipei

SOURCE_LABELS = {
    "llm": "AI Agent 決策",
    "llm_repair": "AI Agent 決策（經系統回饋後修正）",
    "det_repair": "AI Agent 決策＋確定性風控修正",
    "fallback": "量化基準投組（LLM 不可用時之備援）",
    "hold": "維持現有投組（備援）",
    "quant": "量化策略（確定性規則，未使用 LLM）",
}


@dataclass(frozen=True)
class Artifacts:
    report_md: Path
    report_html: Path
    trade_book_csv: Path
    trade_book_json: Path
    decision_json: Path
    sha256: dict[str, str]


def _env() -> jinja2.Environment:
    loader = jinja2.FunctionLoader(
        lambda name: (resources.files("stockmaster.reporting") / "templates" / name).read_text(encoding="utf-8"))
    env = jinja2.Environment(loader=loader, autoescape=False, trim_blocks=True, lstrip_blocks=True,
                             undefined=jinja2.StrictUndefined)
    return env


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def _money(x) -> str:
    return "—" if x is None else f"{x:,.0f}"


def _num(x) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}" if abs(x) < 10 else f"{x:,.0f}"


def render_report_md(f: FinalDecision) -> str:
    tpl = _env().get_template("decision_report.md.j2")
    return tpl.render(f=f, source_label=SOURCE_LABELS.get(f.decision_source, f.decision_source),
                      n_buy=sum(o.side == "buy" for o in f.orders),
                      n_sell=sum(o.side == "sell" for o in f.orders),
                      pct=_pct, money=_money, num=_num, sha=f.sha256(),
                      generated_at=now_taipei().isoformat(timespec="seconds"))


def render_trade_book_csv(f: FinalDecision) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["交易日期", "證券代號", "證券名稱", "買賣別", "張數", "股數", "參考價", "決策雜湊"])
    for o in f.orders:
        w.writerow([f.trade_date.isoformat(), o.code, o.name, "買進" if o.side == "buy" else "賣出",
                    o.lots, o.shares, f"{o.ref_price:.2f}", f.sha256()])
    return buf.getvalue()


def render_trade_book_json(f: FinalDecision) -> str:
    return json.dumps({"trade_date": f.trade_date.isoformat(), "decision_sha256": f.sha256(),
                       "orders": [{"code": o.code, "name": o.name, "side": o.side, "lots": o.lots,
                                   "shares": o.shares} for o in f.orders]},
                      ensure_ascii=False, indent=2)


def _html(md_text: str, title: str) -> str:
    """Minimal Markdown-to-HTML (tables, headings, lists) without extra dependencies."""
    import html as h
    out, in_table, in_list = [], False, False
    for line in md_text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-:") for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{h.escape(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{h.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{h.escape(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{h.escape(line[level:].strip())}</h{level}>")
        elif line.strip():
            out.append(f"<p>{h.escape(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    css = ("body{font-family:'Noto Sans TC',sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
           "table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}"
           "th{background:#f3f3f3}")
    return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>{h.escape(title)}</title>"
            f"<style>{css}</style></head><body>{''.join(out)}</body></html>")


def write_artifacts(f: FinalDecision, out_dir: Path) -> Artifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_report_md(f)
    files = {
        "report_md": (out_dir / f"decision_report_{f.trade_date}.md", md),
        "report_html": (out_dir / f"decision_report_{f.trade_date}.html", _html(md, f"Agent 決策報告 {f.trade_date}")),
        "trade_book_csv": (out_dir / f"trade_book_{f.trade_date}.csv", render_trade_book_csv(f)),
        "trade_book_json": (out_dir / f"trade_book_{f.trade_date}.json", render_trade_book_json(f)),
        "decision_json": (out_dir / f"final_decision_{f.trade_date}.json",
                          json.dumps(f.model_dump(mode="json"), ensure_ascii=False, indent=2)),
    }
    shas = {}
    for key, (path, text) in files.items():
        data = text.encode("utf-8-sig" if path.suffix == ".csv" else "utf-8")
        path.write_bytes(data)
        shas[key] = hashlib.sha256(data).hexdigest()
    return Artifacts(**{k: p for k, (p, _) in files.items()}, sha256=shas)
