"""ETF投資策略說明文件 generator. The competition requires this document to be produced by the AI
Agent, so the system only assembles verified facts (rules handling, architecture, backtest
statistics) and the model writes the document. Without a working model nothing is produced."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from stockmaster.agent.llm_client import ChatModel
from stockmaster.config import load_rules, load_strategy
from stockmaster.reporting.render import _html
from stockmaster.timeutil import now_taipei

PROMPT_NAME = "strategy_doc"


@dataclass
class StrategyDoc:
    markdown: str
    model: str
    facts: dict


def default_facts(compsim_summary: dict | None = None, signal_summary: dict | None = None,
                  template_outline: str | None = None) -> dict:
    rules, strategy = load_rules(), load_strategy()
    return {
        "competition": {"capital": rules.initial_cash, "period": [str(rules.period.start), str(rules.period.end)],
                        "universe": "2026/7/31 市值前 100 上市 + 前 50 上櫃，共 150 檔",
                        "fill": "當日成交均價", "valuation": "收盤價"},
        "architecture": [
            "資料層：證交所/櫃買中心每日行情（含成交金額以計算成交均價）、三大法人、融資融券、本益比、月營收、除權息、主動式 ETF 每日持股；所有原始資料存檔並記錄來源與時間。",
            "特徵與因子：投信/外資買賣超（占成交值比）、月營收年增與加速度、創新高、中期動能、52 週高點距離、短線反轉、融資變化、評價與波動度；橫斷面 z-score 綜合分數。",
            "量化基準投組：核心＋衛星結構（2330 約 18–22%、核心 7–9 檔、衛星 0.75–2.5%），依市場狀態選擇風險等級。",
            "AI Agent：大型語言模型以工具查詢市場、個股、基準投組並試算規則，提出目標權重與理由；確定性風控程序負責換算整張、驗證所有規則與壓力測試。",
            "風控：漲跌停極端價格下的現金壓力測試、加碼個股壓力上限（2330 22%、其他 8.75%）、持股 22–26 檔、Active Share ≥ 30%、違規自動修正與量化備援，確保每日必定提交合規決策。",
            "可重現性：每次決策記錄輸入資料雜湊、模型與 prompt 版本、工具呼叫、提案、驗證結果與最終委託；報告與交易書由同一決策物件產生。",
        ],
        "strategy_config": {"buffers": strategy.buffers.model_dump(), "turnover": strategy.turnover.model_dump(),
                            "risk_levels": {k: v.model_dump() for k, v in strategy.risk_levels.items()}},
        "backtest": compsim_summary or "（尚未提供）",
        "signals": signal_summary or "（尚未提供）",
        "template_outline": template_outline or "（使用預設章節）",
        "generated_at": now_taipei().isoformat(timespec="seconds"),
    }


def generate(model: ChatModel, facts: dict, out_dir: Path, max_tokens: int = 8000) -> StrategyDoc:
    system = (resources.files("stockmaster.agent.prompts") / "v1" / f"{PROMPT_NAME}.md").read_text(encoding="utf-8")
    user = "以下是本系統經驗證的事實資料（JSON），請據此撰寫文件：\n" + json.dumps(facts, ensure_ascii=False, indent=1, default=str)
    res = model.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         None, temperature=0.3, max_tokens=max_tokens)
    md = (res.content or "").strip()
    if len(md) < 500:
        raise RuntimeError("model returned an empty or too-short strategy document")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_taipei().strftime("%Y%m%d_%H%M%S")
    footer = f"\n\n---\n本文件由 AI Agent（{model.provider}:{model.model_id}）於 {facts['generated_at']} 依系統事實資料自動產生。\n"
    (out_dir / f"strategy_doc_{stamp}.md").write_text(md + footer, encoding="utf-8")
    (out_dir / f"strategy_doc_{stamp}.html").write_text(_html(md + footer, "ETF投資策略說明文件"), encoding="utf-8")
    (out_dir / f"strategy_doc_{stamp}.facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return StrategyDoc(markdown=md + footer, model=f"{model.provider}:{model.model_id}", facts=facts)
