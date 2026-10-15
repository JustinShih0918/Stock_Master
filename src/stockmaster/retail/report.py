"""決策報告 and 交易書 for the retail account, rendered from one FinalDecision (odd lots: shares)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import jinja2

from stockmaster.agent.schemas import FinalDecision
from stockmaster.reporting.render import SOURCE_LABELS, Artifacts, _html
from stockmaster.timeutil import now_taipei

TEMPLATE = """# 小額帳戶決策報告（模擬交易）— {{ f.trade_date }}

| 項目 | 內容 |
|---|---|
| 帳戶 | {{ f.provenance.account_id }} |
| 交易日 | {{ f.trade_date }}（以當日成交均價模擬成交） |
| 資料截至 | {{ f.provenance.data_as_of }} 收盤 |
| 決策來源 | {{ source_label }} |
| 模型 | {{ f.provenance.provider or "—" }} / {{ f.provenance.model_id or "量化規則（無 LLM）" }} |
| 市場狀態 / 風險等級 | {{ f.regime }} / {{ f.risk_level }} |
| 報告產生時間 | {{ generated_at }} |

## 一、市場觀點
{{ f.market_view }}

## 二、交易明細
{% if f.orders %}
| 代號 | 名稱 | 買賣 | 股數 | 參考價 | 預估金額 | 理由 |
|---|---|---|---:|---:|---:|---|
{% for o in f.orders -%}
| {{ o.code }} | {{ o.name }} | {{ "買進" if o.side == "buy" else "賣出" }} | {{ o.shares }} | {{ "%.2f"|format(o.ref_price) }} | {{ "{:,.0f}".format(o.est_amount) }} | {{ o.rationale }} |
{% endfor %}
{% else %}
今日無交易。
{% endif %}

預估手續費與稅：NT$ {{ "{:,.0f}".format(f.post_trade.est_cost or 0) }}；交易後現金比率（參考價）：{{ pct(f.post_trade.cash_ref_pct) }}

## 三、交易後目標持股
{% if f.targets %}
| 代號 | 名稱 | 股數 | 權重 | 信心 | 理由 |
|---|---|---:|---:|:---:|---|
{% for t in f.targets -%}
| {{ t.code }} | {{ t.name }} | {{ t.lots_after }} | {{ pct(t.weight_after_ref) }} | {{ t.conviction or "—" }} | {{ t.rationale or "—" }} |
{% endfor %}
{% else %}
全部持有現金。
{% endif %}

## 四、風險與檢查
{{ f.risk_notes }}

| 規則 | 嚴重度 | 結果 | 說明 |
|---|---|:---:|---|
{% for c in f.checks -%}
| {{ c.rule_id }} | {{ c.severity }} | {{ "✔" if c.passed else "✘" }} | {{ c.detail }} |
{% endfor %}
{% if f.repair_notes %}

### 系統調整
{% for n in f.repair_notes -%}
- {{ n }}
{% endfor %}
{% endif %}

## 五、紀錄
- run_id：{{ f.provenance.run_id }}；策略版本：{{ f.provenance.strategy_version }}；Prompt：{{ f.provenance.prompt_version or "—" }}；程式：{{ f.provenance.git_sha or "—" }}
- 輸入資料雜湊：{{ f.provenance.input_sha256 }}
- LLM 嘗試紀錄：{{ f.provenance.llm_attempts | join("；") if f.provenance.llm_attempts else "—" }}
{% if f.stale_inputs %}- ⚠ 過時或缺漏資料：{{ f.stale_inputs | join("、") }}
{% endif %}
- 決策雜湊：{{ sha }}
"""


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def write_artifacts(f: FinalDecision, out_dir: Path) -> Artifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
    md = env.from_string(TEMPLATE).render(
        f=f, source_label=SOURCE_LABELS.get(f.decision_source, f.decision_source), pct=_pct, sha=f.sha256(),
        generated_at=now_taipei().isoformat(timespec="seconds"))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["交易日期", "證券代號", "證券名稱", "買賣別", "股數", "參考價", "決策雜湊"])
    for o in f.orders:
        w.writerow([f.trade_date.isoformat(), o.code, o.name, "買進" if o.side == "buy" else "賣出", o.shares,
                    f"{o.ref_price:.2f}", f.sha256()])
    book_json = json.dumps({"trade_date": f.trade_date.isoformat(), "decision_sha256": f.sha256(),
                            "orders": [{"code": o.code, "name": o.name, "side": o.side, "shares": o.shares}
                                       for o in f.orders]}, ensure_ascii=False, indent=2)
    files = {
        "report_md": (out_dir / f"decision_report_{f.trade_date}.md", md),
        "report_html": (out_dir / f"decision_report_{f.trade_date}.html", _html(md, f"小額帳戶決策報告 {f.trade_date}")),
        "trade_book_csv": (out_dir / f"trade_book_{f.trade_date}.csv", buf.getvalue()),
        "trade_book_json": (out_dir / f"trade_book_{f.trade_date}.json", book_json),
        "decision_json": (out_dir / f"final_decision_{f.trade_date}.json",
                          json.dumps(f.model_dump(mode="json"), ensure_ascii=False, indent=2)),
    }
    shas = {}
    for key, (path, text) in files.items():
        data = text.encode("utf-8-sig" if path.suffix == ".csv" else "utf-8")
        path.write_bytes(data)
        shas[key] = hashlib.sha256(data).hexdigest()
    return Artifacts(**{k: p for k, (p, _) in files.items()}, sha256=shas)
