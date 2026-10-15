"""LLM decision step for the retail account: one prompt with every input inline (no tool calls,
so any chat model works), a JSON answer, and at most ``max_repair_rounds`` rounds of rule feedback.

The model proposes target weights only. ``evaluate`` (deterministic, from ``sizing``) decides
whether the proposal is acceptable; the model never sizes orders.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from importlib import resources
from string import Template
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from stockmaster.agent.harness import AgentResult, extract_json
from stockmaster.agent.llm_client import ChatModel
from stockmaster.retail.config import RetailRules
from stockmaster.timeutil import now_taipei

PROMPT_VERSION = "retail_v1"


class RetailTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")
    code: str = Field(pattern=r"^[0-9]{4,6}[A-Z]?$")
    weight_pct: float = Field(ge=0, le=100)
    conviction: int = Field(default=3, ge=1, le=5)
    rationale: str = Field(default="", max_length=400)


class RetailDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    market_view: str = Field(max_length=1500)
    regime: Literal["risk_on", "neutral", "risk_off"] = "neutral"
    risk_level: Literal["defensive", "normal", "aggressive"] = "normal"
    targets: list[RetailTarget] = Field(default_factory=list, max_length=12)
    cash_pct: float = Field(default=0, ge=0, le=100)
    risk_notes: str = Field(default="", max_length=1000)

    @field_validator("targets")
    @classmethod
    def _unique(cls, v: list[RetailTarget]) -> list[RetailTarget]:
        codes = [t.code for t in v]
        if len(codes) != len(set(codes)):
            raise ValueError("duplicate codes in targets")
        return v

    def weights(self) -> dict[str, float]:
        return {t.code: t.weight_pct / 100 for t in self.targets if t.weight_pct > 0}


def _load(name: str) -> str:
    return (resources.files("stockmaster.agent.prompts") / PROMPT_VERSION / f"{name}.md").read_text(encoding="utf-8")


def prompt_sha() -> str:
    h = hashlib.sha256()
    for n in ("system", "user"):
        h.update(_load(n).encode())
    return h.hexdigest()[:16]


def system_prompt(rules: RetailRules) -> str:
    p, f = rules.portfolio, rules.fees
    return Template(_load("system")).substitute(
        initial_cash=f"{rules.initial_cash:,}", max_positions=p.max_positions, max_weight=f"{p.max_weight:.0%}",
        cash_buffer=f"{p.cash_buffer_pct:.0%}", min_order=f"{p.min_order_ntd:,.0f}", min_hold=p.min_hold_days,
        max_orders=p.max_orders_per_day, min_fee=f"{f.min_commission:,.0f}",
        slippage=f"{rules.fill.slippage_bps / 100:.2f}%")


def user_prompt(**parts: str) -> str:
    return Template(_load("user")).substitute(**parts)


Evaluate = Callable[[RetailDecision], list[str]]


def run_agent(model: ChatModel, system: str, user: str, evaluate: Evaluate, rules: RetailRules) -> AgentResult:
    res = AgentResult(status="failed", decision=None, provider=model.provider, model_id=model.model_id)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    knobs = rules.llm
    for _ in range(knobs.max_repair_rounds + 1):
        res.rounds += 1
        t0 = now_taipei()
        try:
            out = model.complete(messages, None, temperature=knobs.temperature, max_tokens=knobs.max_tokens)
        except Exception as e:  # noqa: BLE001 - a provider failure ends this model's attempt
            res.error = f"{type(e).__name__}: {e}"[:500]
            res.log.append({"type": "llm", "seq": res.rounds, "started_at": t0.isoformat(), "status": "error",
                            "error": res.error})
            res.status = "rejected" if res.decision is not None else "failed"
            return res
        res.log.append({"type": "llm", "seq": res.rounds, "started_at": t0.isoformat(), "status": "ok",
                        "latency_ms": out.latency_ms, "prompt_tokens": out.prompt_tokens,
                        "completion_tokens": out.completion_tokens, "finish_reason": out.finish_reason,
                        "content": out.content, "reasoning": out.reasoning, "tool_calls": []})
        messages.append({"role": "assistant", "content": out.content or ""})
        obj = extract_json(out.content or "")
        if obj is None:
            feedback = "沒有找到 JSON 物件。請只輸出一個符合格式的 JSON 物件。"
            res.log.append({"type": "submit", "accepted": False, "stage": "parse", "detail": feedback})
        else:
            try:
                d = RetailDecision.model_validate(obj)
            except ValidationError as e:
                feedback = "JSON 不符合格式：" + "; ".join(
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:8])
                res.log.append({"type": "submit", "accepted": False, "stage": "schema", "detail": feedback})
            else:
                problems = evaluate(d)
                res.decision = d
                res.last_feedback = "; ".join(problems) or "accepted"
                res.log.append({"type": "submit", "accepted": not problems, "stage": "rules",
                                "detail": res.last_feedback, "decision": d.model_dump(mode="json")})
                if not problems:
                    res.status = "accepted"
                    return res
                feedback = "系統驗證未通過：" + "；".join(problems)
        if res.rounds > knobs.max_repair_rounds:
            break
        res.repair_rounds += 1
        messages.append({"role": "user", "content": feedback + "\n請修正後重新輸出完整 JSON。"})
    res.status = "rejected" if res.decision is not None else "failed"
    res.error = res.error or res.last_feedback or "no valid decision"
    return res
